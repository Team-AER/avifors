"""Capacity-pooled residency: keep every model that fits loaded together.

A lane configured with `capacity_mib` uses PoolScheduler instead of the exclusive Scheduler. Each
model in the lane declares `memory_mib`. Models load on demand and stay resident while their
memory fits next to the others; a model that does not fit evicts idle residents, least recently
used first. A busy resident is never preempted mid-job: once it has held its lease (`max_hold`)
while another model waits for room, it stops taking new work, drains, and is released.

Lanes without a capacity keep the exclusive single-owner Scheduler (unchanged GPU behaviour).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .scheduler import Job, Rejected

if TYPE_CHECKING:
    from .config import Model

LOG = logging.getLogger(__name__)


@dataclass(eq=False)
class Resident:
    model: Model
    state: str = "loading"  # loading | ready | draining | releasing
    epoch: float = 0.0
    last_used: float = 0.0
    active: dict = field(default_factory=dict)  # Job -> Task
    task: asyncio.Task | None = None  # activation / release task
    reset: bool = False


class PoolScheduler:
    def __init__(self, config, lifecycle, capacity_mib):
        self.cfg, self.lifecycle, self.capacity = config, lifecycle, float(capacity_mib)
        self.pending: list[Job] = []
        self.residents: dict[str, Resident] = {}
        self.fault = None
        self.model_faults = {}
        self.fault_until = {}
        self.events = asyncio.Event()
        self.counts: Counter = Counter()
        self.model_stats = defaultdict(Counter)
        self.user_service = defaultdict(float)
        self.sequence = itertools.count()
        self.runner = None
        self.closing = False

    # ── public interface shared with Scheduler ─────────────────────────────

    @property
    def active(self):
        return {j: t for r in self.residents.values() for j, t in r.active.items()}

    async def start(self):
        await self.lifecycle.recover()
        self.runner = asyncio.create_task(self.loop())

    async def close(self):
        self.closing = True
        if self.runner:
            self.runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.runner
        for r in list(self.residents.values()):
            await self.abort(r, "broker shutting down")
            if r.task and not r.task.done():
                r.task.cancel()
                await asyncio.gather(r.task, return_exceptions=True)
        self.reject_pending(503, "broker shutting down")
        self.residents.clear()
        await self.lifecycle.recover()

    def reject_pending(self, status, message, model=None):
        for j in list(self.pending):
            if model is None or j.model == model:
                self.pending.remove(j)
                if not j.future.done():
                    j.future.set_exception(Rejected(status, message))

    def enqueue(self, model, user, limit, call, *, min_budget=0):
        if self.fault or self.closing:
            raise Rejected(503, "model pool unavailable")
        if model.id in self.model_faults:
            raise Rejected(503, "model activation failed")
        all_jobs = self.pending + list(self.active)
        if (
            len(all_jobs) >= self.cfg.max_pending
            or sum(j.model == model.id for j in all_jobs) >= model.max_pending
            or sum(j.user == user for j in all_jobs) >= limit
        ):
            self.counts["rejected"] += 1
            self.model_stats[model.id]["busy"] += 1
            raise Rejected(429, "inference queue is full")
        if not 0 <= min_budget < model.max_hold:
            raise ValueError("minimum execution budget must fit the model lease")
        now = time.monotonic()
        job = Job(next(self.sequence), model.id, user, now, now + model.queue_timeout, call,
                  asyncio.get_running_loop().create_future())
        job.min_budget = min_budget
        self.pending.append(job)
        self.events.set()
        return job

    def cancel(self, job):
        if job.cancelled:
            return
        job.cancelled = True
        if job in self.pending:
            self.pending.remove(job)
            job.future.cancel()
        elif job in self.active:
            job.future.cancel()  # drain within the original deadline, as Scheduler does
            self.counts["client_cancelled"] += 1
        self.events.set()

    def snapshot(self):
        now = time.monotonic()
        return {
            "policy": "pool",
            "state": "fault" if self.fault else ("ready" if self.residents else "unloaded"),
            "resident": sorted(self.residents),
            "residents": {
                k: {"state": r.state, "active": len(r.active), "memory_mib": r.model.memory_mib,
                    "idle_seconds": round(now - r.last_used, 1) if not r.active else 0}
                for k, r in sorted(self.residents.items())
            },
            "capacity_mib": self.capacity,
            "used_mib": self.used(),
            "sleeping": [],
            "active": len(self.active),
            "queued": len(self.pending),
            "queues": dict(Counter(j.model for j in self.pending)),
            "oldest_wait_seconds": max((now - j.queued for j in self.pending), default=0),
            "lease_remaining_seconds": 0,
            "fault": self.fault,
            "model_faults": self.model_faults,
            "counters": dict(self.counts),
            "model_stats": {k: dict(v) for k, v in self.model_stats.items()},
        }

    # ── internals ──────────────────────────────────────────────────────────

    def used(self):
        return sum(r.model.memory_mib for r in self.residents.values())

    def expire_pending(self):
        now = time.monotonic()
        for job in list(self.pending):
            if job.cancelled or now >= job.expires:
                self.pending.remove(job)
                if not job.future.done():
                    job.future.set_exception(Rejected(504, "queue deadline exceeded"))
                self.counts["queue_expired"] += 1

    async def run_job(self, resident, job):
        model = resident.model
        job.started = time.monotonic()
        self.counts["queue_seconds"] += job.started - job.queued
        try:
            async with asyncio.timeout_at(job.started + model.max_runtime):
                result = await job.call()
            if not job.future.done():
                job.future.set_result(result)
            self.counts["completed"] += 1
            self.model_stats[job.model]["requests"] += 1
        except TimeoutError:
            resident.reset = True
            self.counts["timeouts"] += 1
            self.model_stats[job.model]["failures"] += 1
            if not job.future.done():
                job.future.set_exception(Rejected(504, "execution deadline exceeded"))
        except asyncio.CancelledError:
            if not job.future.done():
                job.future.set_exception(Rejected(503, "worker interrupted"))
            raise
        except Exception as exc:
            resident.reset = True
            self.counts["failed"] += 1
            self.model_stats[job.model]["failures"] += 1
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            elapsed = time.monotonic() - job.started
            self.user_service[job.user] += elapsed
            self.model_stats[job.model]["last_duration_seconds"] = elapsed
            resident.active.pop(job, None)
            resident.last_used = time.monotonic()
            self.events.set()

    async def abort(self, resident, reason):
        jobs = list(resident.active)
        tasks = [resident.active[j] for j in jobs]
        for job, task in zip(jobs, tasks):
            if not job.future.done():
                job.future.set_exception(Rejected(503, reason))
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        resident.active.clear()

    def begin_activation(self, model):
        r = Resident(model, "loading", epoch=time.monotonic(), last_used=time.monotonic())
        self.residents[model.id] = r

        async def activate():
            try:
                async with asyncio.timeout(model.activation_timeout):
                    await self.lifecycle.activate(model)
                r.state, r.epoch, r.last_used = "ready", time.monotonic(), time.monotonic()
                self.model_faults.pop(model.id, None)
                self.counts["activations"] += 1
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("activation failed model=%s", model.id)
                self.model_faults[model.id] = "activation failed"
                self.fault_until[model.id] = time.monotonic() + 15
                self.reject_pending(503, "model activation failed", model.id)
                r.state = "releasing"
                await self.lifecycle.release(model, hard=True)
                self.residents.pop(model.id, None)
            finally:
                self.events.set()

        r.task = asyncio.create_task(activate())

    def begin_release(self, resident, hard=False):
        resident.state = "releasing"

        async def release():
            try:
                await self.abort(resident, "worker reset after cancellation or failure")
                await self.lifecycle.release(resident.model, hard=hard)
                self.counts["releases"] += 1
                self.residents.pop(resident.model.id, None)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("release failed model=%s; pool admission disabled", resident.model.id)
                self.fault = "worker release failed"
                self.reject_pending(503, "model pool fault")
            finally:
                self.events.set()

        resident.task = asyncio.create_task(release())

    def waiting_models(self):
        """Models with pending work that are not resident, oldest request first."""
        seen, out = set(), []
        for job in sorted(self.pending, key=lambda j: j.sequence):
            if job.model not in self.residents and job.model not in seen and job.model not in self.model_faults:
                seen.add(job.model)
                out.append(self.cfg.models[job.model])
        return out

    def make_room(self, model, now):
        """Start releasing idle residents (LRU) until `model` fits; drain leased-out busy ones.
        Returns True when the model fits now."""
        free = self.capacity - self.used()
        if model.memory_mib <= free:
            return True
        wanted = {j.model for j in self.pending}
        idle = sorted(
            (r for r in self.residents.values() if r.state == "ready" and not r.active),
            key=lambda r: (r.model.id in wanted, r.last_used),
        )
        for r in idle:
            if free >= model.memory_mib:
                break
            free += r.model.memory_mib
            self.counts["evictions"] += 1
            self.begin_release(r)
        if free < model.memory_mib:
            # Busy residents that have held their lease while others wait stop taking work.
            for r in self.residents.values():
                if r.state == "ready" and r.active and now - r.epoch >= r.model.max_hold:
                    r.state = "draining"
        return False

    async def loop(self):
        try:
            while True:
                self.events.clear()
                now = time.monotonic()
                for key in list(self.fault_until):
                    if now >= self.fault_until[key]:
                        self.model_faults.pop(key, None)
                        self.fault_until.pop(key)
                self.expire_pending()
                if self.fault:
                    self.reject_pending(503, "model pool fault")
                for r in list(self.residents.values()):
                    if r.state in ("ready", "draining") and r.reset and not r.active:
                        r.reset = False
                        self.begin_release(r, hard=True)
                    elif r.state == "draining" and not r.active:
                        self.counts["evictions"] += 1
                        self.begin_release(r)
                    elif (
                        r.state == "ready"
                        and not r.active
                        and not any(j.model == r.model.id for j in self.pending)
                        and now - r.last_used >= r.model.idle_timeout
                    ):
                        self.begin_release(r)
                # Dispatch to ready residents.
                for r in self.residents.values():
                    if r.state != "ready":
                        continue
                    if now - r.epoch >= r.model.max_hold and not r.active:
                        r.epoch = now  # renew an idle lease
                    while len(r.active) < r.model.concurrency:
                        matching = [j for j in self.pending if j.model == r.model.id]
                        if not matching:
                            break
                        job = min(matching, key=lambda j: (self.user_service[j.user], j.sequence))
                        self.pending.remove(job)
                        r.active[job] = asyncio.create_task(self.run_job(r, job))
                # Load waiting models that fit; make room for the oldest one that does not.
                if not self.fault:
                    for model in self.waiting_models():
                        if self.make_room(model, now):
                            self.begin_activation(model)
                        else:
                            break  # keep arrival order: do not let a smaller later model jump the queue
                try:
                    async with asyncio.timeout(0.05):
                        await self.events.wait()
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("model pool fault; admission disabled")
            self.fault = "pool supervisor fault"
            for r in list(self.residents.values()):
                await self.abort(r, "model pool fault")
            self.reject_pending(503, "model pool fault")
