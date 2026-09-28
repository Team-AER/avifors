import asyncio
import time

import pytest

from avifors.config import Config, Model
from avifors.scheduler import Rejected, Scheduler


class FakeLifecycle:
    def __init__(self):
        self.owner = None
        self.events = []
        self.sleeping = set()
        self.fail_start = False
        self.fail_stop = False

    async def recover(self):
        self.owner = None

    async def activate(self, model):
        assert self.owner is None, "two concurrent GPU owners"
        if self.fail_start:
            raise RuntimeError("load failed")
        self.owner = model.id
        self.events.append(("start", model.id))

    async def release(self, model, hard=False):
        if self.fail_stop:
            raise RuntimeError("GPU still owned")
        self.events.append(("stop", model.id, hard))
        self.owner = None


def config(policy="latency"):
    ms = [
        Model(
            id=x,
            upstream="http://localhost:1",
            start=["true"],
            stop=["true"],
            max_hold=0.4,
            max_runtime=0.3,
            queue_timeout=2,
            idle_timeout=0.05,
            concurrency=2 if x == "text" else 1,
        )
        for x in ("text", "image")
    ]
    return Config(models={m.id: m for m in ms}, policy=policy, drain_margin=0.05)


async def result(delay=0.01, value="ok"):
    await asyncio.sleep(delay)
    return value


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("policy", ["latency", "fifo", "fair", "throughput"])
async def test_mixed_queue_has_single_owner_and_progress(policy):
    cfg, life = config(policy), FakeLifecycle()
    s = Scheduler(cfg, life)
    await s.start()
    try:
        jobs = [
            s.enqueue(cfg.models[m], "alice", 20, lambda: result())
            for m in ("text", "image", "text", "image")
        ]
        assert await asyncio.gather(*(j.future for j in jobs)) == ["ok"] * 4
        assert any(e[:2] == ("start", "image") for e in life.events)
        await until(lambda: s.current is None)
        assert life.owner is None
    finally:
        await s.close()


async def test_fifo_cannot_jump_image_for_later_text():
    cfg, life = config("fifo"), FakeLifecycle()
    s, order = Scheduler(cfg, life), []

    async def run(name):
        order.append(name)
        await asyncio.sleep(0.02)

    await s.start()
    try:
        jobs = [
            s.enqueue(cfg.models[m], "alice", 20, lambda n=n: run(n))
            for n, m in [("t1", "text"), ("i1", "image"), ("t2", "text")]
        ]
        await asyncio.gather(*(j.future for j in jobs))
        assert order == ["t1", "i1", "t2"]
    finally:
        await s.close()


async def test_latency_stops_replenishing_when_other_model_waits():
    cfg, life = config(), FakeLifecycle()
    cfg.models["text"].concurrency = 1
    s, gate, order = Scheduler(cfg, life), asyncio.Event(), []

    async def first():
        order.append("t1")
        await gate.wait()

    async def run(name):
        order.append(name)

    await s.start()
    try:
        a = s.enqueue(cfg.models["text"], "alice", 20, first)
        await until(lambda: bool(s.active))
        b = s.enqueue(cfg.models["text"], "alice", 20, lambda: run("t2"))
        c = s.enqueue(cfg.models["image"], "bob", 20, lambda: run("image"))
        gate.set()
        await asyncio.gather(a.future, b.future, c.future)
        assert order == ["t1", "image", "t2"]
    finally:
        await s.close()


async def test_hard_deadline_terminates_work_and_next_model_runs():
    cfg, life = config("throughput"), FakeLifecycle()
    cfg.models["text"].max_hold = 0.12
    s = Scheduler(cfg, life)
    await s.start()
    try:
        a = s.enqueue(cfg.models["text"], "alice", 20, lambda: result(5))
        b = s.enqueue(cfg.models["image"], "bob", 20, lambda: result())
        started = time.monotonic()
        with pytest.raises(Rejected):
            await a.future
        assert await b.future == "ok"
        assert time.monotonic() - started < 0.5
        assert ("stop", "text", True) in life.events
    finally:
        await s.close()


async def test_queue_limits_are_per_user_and_global():
    cfg, life = config(), FakeLifecycle()
    s = Scheduler(cfg, life)
    a = s.enqueue(cfg.models["text"], "alice", 1, result)
    with pytest.raises(Rejected, match="queue is full") as e:
        s.enqueue(cfg.models["image"], "alice", 1, result)
    assert e.value.status == 429
    b = s.enqueue(cfg.models["image"], "bob", 1, result)
    s.cancel(a)
    s.cancel(b)
    assert not s.pending


async def test_cancelled_active_request_drains_without_interrupting_peer():
    cfg, life = config(), FakeLifecycle()
    s = Scheduler(cfg, life)
    await s.start()
    try:
        a = s.enqueue(cfg.models["text"], "alice", 20, lambda: result(0.08))
        b = s.enqueue(cfg.models["text"], "bob", 20, lambda: result(0.1))
        await until(lambda: len(s.active) == 2)
        s.cancel(a)
        with pytest.raises(asyncio.CancelledError):
            await a.future
        assert await b.future == "ok"
        await until(lambda: s.current is None)
        assert ("stop", "text", True) not in life.events
    finally:
        await s.close()


async def test_abandoned_request_still_has_hard_execution_deadline():
    cfg, life = config(), FakeLifecycle()
    s = Scheduler(cfg, life)
    await s.start()
    try:
        a = s.enqueue(cfg.models["text"], "alice", 20, lambda: result(5))
        await until(lambda: bool(s.active))
        s.cancel(a)
        await until(lambda: s.current is None)
        assert s.counts["timeouts"] == 1
        assert ("stop", "text", True) in life.events
    finally:
        await s.close()


async def test_release_failure_fences_all_future_work():
    cfg, life = config(), FakeLifecycle()
    s = Scheduler(cfg, life)
    await s.start()
    a = s.enqueue(cfg.models["text"], "alice", 20, result)
    await a.future
    life.fail_stop = True
    await until(lambda: s.fault is not None)
    with pytest.raises(Rejected) as e:
        s.enqueue(cfg.models["image"], "bob", 20, result)
    assert e.value.status == 503
    life.fail_stop = False
    await s.close()


async def test_expired_queue_does_not_activate_model():
    cfg, life = config(), FakeLifecycle()
    cfg.models["image"].queue_timeout = 0.01
    s = Scheduler(cfg, life)
    a = s.enqueue(cfg.models["image"], "alice", 20, result)
    await asyncio.sleep(0.02)
    await s.start()
    try:
        with pytest.raises(Rejected, match="queue deadline"):
            await a.future
        assert not life.events
    finally:
        await s.close()


async def test_failed_activation_is_released():
    cfg, life = config(), FakeLifecycle()
    life.fail_start = True
    s = Scheduler(cfg, life)
    await s.start()
    try:
        a = s.enqueue(cfg.models["text"], "alice", 20, result)
        with pytest.raises(Rejected, match="activation"):
            await a.future
        await until(lambda: s.current is None)
        assert "text" in s.model_faults
        assert ("stop", "text", True) in life.events
    finally:
        await s.close()


async def test_queue_deadline_expires_during_slow_activation():
    cfg, life = config(), FakeLifecycle()
    cfg.models["text"].queue_timeout = 0.05

    async def slow(model):
        life.owner = model.id
        await asyncio.sleep(2)

    life.activate = slow
    s = Scheduler(cfg, life)
    await s.start()
    try:
        started = time.monotonic()
        j = s.enqueue(cfg.models["text"], "alice", 20, result)
        with pytest.raises(Rejected, match="queue deadline"):
            await j.future
        assert time.monotonic() - started < 0.3
        await until(lambda: s.current is None)
        assert life.owner is None
    finally:
        await s.close()


async def test_disconnected_only_waiter_cancels_loading():
    cfg, life = config(), FakeLifecycle()

    async def slow(model):
        life.owner = model.id
        await asyncio.sleep(2)

    life.activate = slow
    s = Scheduler(cfg, life)
    await s.start()
    try:
        j = s.enqueue(cfg.models["text"], "alice", 20, result)
        await until(lambda: life.owner == "text")
        s.cancel(j)
        await until(lambda: s.current is None)
        assert life.owner is None
    finally:
        await s.close()


async def test_recovery_stops_all_workers_before_global_memory_check(monkeypatch):
    from avifors import lifecycle

    cfg = config()
    cfg.release_check = ["check"]
    for m in cfg.models.values():
        m.stop = ["stop", m.id]
    owners = {"image"}
    calls = []

    async def command(argv, timeout):
        calls.append(argv)
        if argv[0] == "stop":
            owners.discard(argv[1])
        else:
            assert not owners

    monkeypatch.setattr(lifecycle, "execute", command)
    driver = lifecycle.Lifecycle(cfg, None)
    await driver.recover()
    assert calls[-1] == ["check"]
    assert calls.count(["check"]) == 1


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("latency", ["t1", "image", "t2"]),
        ("fifo", ["t1", "t2", "image"]),
        ("throughput", ["t1", "t2", "image"]),
    ],
)
async def test_policy_controls_replenishment_after_active_job(policy, expected):
    cfg, life = config(policy), FakeLifecycle()
    cfg.models["text"].concurrency = 1
    s, gate, order = Scheduler(cfg, life), asyncio.Event(), []

    async def first():
        order.append("t1")
        await gate.wait()

    async def run(name):
        order.append(name)

    await s.start()
    try:
        a = s.enqueue(cfg.models["text"], "alice", 20, first)
        await until(lambda: bool(s.active))
        b = s.enqueue(cfg.models["text"], "alice", 20, lambda: run("t2"))
        c = s.enqueue(cfg.models["image"], "bob", 20, lambda: run("image"))
        gate.set()
        await asyncio.gather(a.future, b.future, c.future)
        assert order == expected
    finally:
        await s.close()


async def test_checkpointed_work_requests_fresh_budget_before_start():
    cfg, life = config(), FakeLifecycle()
    s = Scheduler(cfg, life)
    await s.start()
    try:
        first = s.enqueue(cfg.models["text"], "alice", 10, lambda: result(0.25))
        await first.future
        # Only ~150ms remains; this chunk needs 250ms and must get a new epoch.
        second = s.enqueue(cfg.models["text"], "alice", 10, lambda: result(0.20), min_budget=0.25)
        assert await second.future == "ok"
        assert len([e for e in life.events if e == ("start", "text")]) == 2
        assert s.counts["timeouts"] == 0
    finally:
        await s.close()
