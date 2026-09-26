from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

LOG = logging.getLogger(__name__)


class Rejected(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


@dataclass(eq=False)
class Job:
    sequence: int
    model: str
    user: str
    queued: float
    expires: float
    call: Callable[[], Awaitable]
    future: asyncio.Future
    task: asyncio.Task | None = None
    cancelled: bool = False
    started: float = 0


class Scheduler:
    def __init__(self, config, lifecycle):
        self.cfg, self.lifecycle = config, lifecycle
        self.pending: list[Job] = []
        self.active: dict[Job, asyncio.Task] = {}
        self.current = None
        self.state = "unloaded"
        self.epoch = 0.0
        self.last_finished = time.monotonic()
        self.fault = None
        self.model_faults = {}
        self.reset_required = False
        self.epoch_started = False
        self.preferred = None
        self.fault_until = {}
        self.events = asyncio.Event()
        self.counts = Counter()
        self.model_stats = defaultdict(Counter)
        self.service = defaultdict(float)
        self.user_service = defaultdict(float)
        self.sequence = itertools.count()
        self.runner = None
        self.closing = False

    async def start(self):
        await self.lifecycle.recover()
        self.runner = asyncio.create_task(self.loop())

    async def close(self):
        self.closing = True
        if self.runner:
            self.runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.runner
        await self.abort("broker shutting down")
        self.reject_pending(503, "broker shutting down")
        await self.lifecycle.recover()

    def reject_pending(self, status, message, model=None):
        for j in list(self.pending):
            if model is None or j.model == model:
                self.pending.remove(j)
                if not j.future.done():
                    j.future.set_exception(Rejected(status, message))

    def enqueue(self, model, user, limit, call):
        if self.fault or self.closing:
            raise Rejected(503, "GPU supervisor unavailable")
        all_jobs = self.pending + list(self.active)
        if (
            len(all_jobs) >= self.cfg.max_pending
            or sum(j.model == model.id for j in all_jobs) >= model.max_pending
            or sum(j.user == user for j in all_jobs) >= limit
        ):
            self.counts["rejected"] += 1
            self.model_stats[model.id]["busy"] += 1
            raise Rejected(429, "inference queue is full")
        now = time.monotonic()
        job = Job(
            next(self.sequence),
            model.id,
            user,
            now,
            now + model.queue_timeout,
            call,
            asyncio.get_running_loop().create_future(),
        )
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
            # Drain abandoned work within its original deadline. Resetting a
            # shared worker here lets one short client timeout kill other users.
            job.future.cancel()
            self.counts["client_cancelled"] += 1
        self.events.set()

    def choose(self):
        if not self.pending:
            return None
        candidates = self.pending
        if self.preferred and any(j.model == self.preferred for j in candidates):
            return self.preferred
        if self.current:
            other = [j for j in candidates if j.model != self.current.id]
            if other:
                candidates = other
        if self.cfg.policy == "fair":
            # Aging makes a long-waiting model eligible regardless of cost estimates.
            old = min(candidates, key=lambda j: j.sequence)
            if time.monotonic() - old.queued >= self.cfg.models[old.model].max_hold:
                return old.model
            return min(
                candidates,
                key=lambda j: (self.service[j.model] / self.cfg.models[j.model].weight, j.sequence),
            ).model
        return min(candidates, key=lambda j: j.sequence).model

    def next_job(self):
        matching = [j for j in self.pending if j.model == self.current.id]
        if not matching:
            return None
        if self.cfg.policy == "fifo" and self.pending[0].model != self.current.id:
            return None
        if self.cfg.policy == "fair":
            return min(matching, key=lambda j: (self.user_service[j.user], j.sequence))
        return matching[0]

    async def run_job(self, job):
        job.started = time.monotonic()
        self.counts["queue_seconds"] += job.started - job.queued
        try:
            end = min(self.epoch + self.current.max_hold, job.started + self.current.max_runtime)
            async with asyncio.timeout_at(end):
                result = await job.call()
            if not job.future.done():
                job.future.set_result(result)
            self.counts["completed"] += 1
            self.model_stats[job.model]["requests"] += 1
        except TimeoutError:
            self.reset_required = True
            self.counts["timeouts"] += 1
            self.model_stats[job.model]["failures"] += 1
            if not job.future.done():
                job.future.set_exception(Rejected(504, "GPU execution deadline exceeded"))
        except asyncio.CancelledError:
            if not job.future.done():
                job.future.set_exception(Rejected(503, "GPU worker interrupted"))
            raise
        except Exception as exc:
            self.reset_required = True
            self.counts["failed"] += 1
            self.model_stats[job.model]["failures"] += 1
            if not job.future.done():
                job.future.set_exception(exc)
        finally:
            elapsed = time.monotonic() - job.started
            self.user_service[job.user] += elapsed
            self.model_stats[job.model]["last_duration_seconds"] = elapsed
            self.active.pop(job, None)
            self.last_finished = time.monotonic()
            self.events.set()

    async def abort(self, reason):
        jobs = list(self.active)
        tasks = [self.active[j] for j in jobs]
        for job, task in zip(jobs, tasks):
            if not job.future.done():
                job.future.set_exception(Rejected(503, reason))
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.active.clear()

    async def release(self, hard=False):
        if self.current:
            model = self.current
            self.preferred = self.choose()
            self.state = "releasing"
            await self.lifecycle.release(model, hard=hard)
            self.service[model.id] += time.monotonic() - self.epoch
            self.current = None
            self.counts["releases"] += 1
        self.state = "unloaded"

    def expire_pending(self):
        now = time.monotonic()
        for job in list(self.pending):
            if job.cancelled or now >= job.expires:
                self.pending.remove(job)
                if not job.future.done():
                    job.future.set_exception(Rejected(504, "queue deadline exceeded"))
                self.counts["queue_expired"] += 1

    async def activate(self, model):
        task = asyncio.create_task(self.lifecycle.activate(model))
        try:
            while not task.done():
                self.expire_pending()
                if not any(j.model == model.id for j in self.pending):
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await self.release(hard=True)
                    return False
                await asyncio.wait({task}, timeout=0.025)
            await task
            return True
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

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
                if self.reset_required:
                    await self.abort("worker reset after cancellation or failure")
                    await self.release(hard=True)
                    self.reset_required = False
                if self.current:
                    elapsed = now - self.epoch
                    competitors = any(j.model != self.current.id for j in self.pending)
                    if elapsed >= self.current.max_hold:
                        if self.active:
                            self.counts["lease_expired"] += 1
                            await self.abort("GPU ownership deadline exceeded")
                            await self.release(hard=True)
                        elif competitors:
                            await self.release()
                        else:
                            self.service[self.current.id] += elapsed
                            self.epoch = now
                    if self.current and not self.active:
                        matching = any(j.model == self.current.id for j in self.pending)
                        switch = competitors and (self.epoch_started or not matching)
                        if self.cfg.policy == "fifo":
                            switch = competitors and self.pending[0].model != self.current.id
                        elif self.cfg.policy == "throughput":
                            switch = competitors and (
                                not matching or elapsed >= self.current.max_hold - self.cfg.drain_margin
                            )
                        if switch or (
                            not self.pending and now - self.last_finished >= self.current.idle_timeout
                        ):
                            await self.release()
                    if self.current:
                        drain = (
                            competitors
                            and self.epoch_started
                            and (
                                self.cfg.policy in {"latency", "fair"}
                                or elapsed >= self.current.max_hold - self.cfg.drain_margin
                            )
                        )
                        self.state = "draining" if drain else "ready"
                        while not drain and len(self.active) < self.current.concurrency:
                            job = self.next_job()
                            if not job:
                                break
                            self.pending.remove(job)
                            self.epoch_started = True
                            self.active[job] = asyncio.create_task(self.run_job(job))
                if not self.current and self.pending:
                    model = self.cfg.models[self.choose()]
                    self.current, self.state, self.epoch = model, "loading", time.monotonic()
                    self.epoch_started = False
                    self.preferred = None
                    try:
                        if not await self.activate(model):
                            continue
                        self.model_faults.pop(model.id, None)
                        self.counts["activations"] += 1
                        self.last_finished = time.monotonic()
                    except Exception:
                        LOG.exception("activation failed model=%s", model.id)
                        self.model_faults[model.id] = "activation failed"
                        self.fault_until[model.id] = time.monotonic() + 15
                        self.reject_pending(503, "model activation failed", model.id)
                        await self.release(hard=True)
                    continue
                try:
                    async with asyncio.timeout(0.05):
                        await self.events.wait()
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("GPU supervisor fault; admission disabled")
            self.fault = "worker release failed"
            self.state = "fault"
            await self.abort("GPU supervisor fault")
            self.reject_pending(503, "GPU supervisor fault")

    def snapshot(self):
        return {
            "policy": self.cfg.policy,
            "state": self.state,
            "resident": self.current.id if self.current else None,
            "sleeping": sorted(getattr(self.lifecycle, "sleeping", set())),
            "active": len(self.active),
            "queued": len(self.pending),
            "queues": dict(Counter(j.model for j in self.pending)),
            "oldest_wait_seconds": max((time.monotonic() - j.queued for j in self.pending), default=0),
            "lease_remaining_seconds": max(0, self.epoch + self.current.max_hold - time.monotonic())
            if self.current
            else 0,
            "fault": self.fault,
            "model_faults": self.model_faults,
            "counters": dict(self.counts),
            "model_stats": {k: dict(v) for k, v in self.model_stats.items()},
        }
