import asyncio

import pytest

from avifors.config import Config, Model
from avifors.pool import PoolScheduler
from avifors.scheduler import Rejected


class PoolLifecycle:
    """Fake lifecycle for a pooled lane: several owners allowed, never more memory than capacity."""

    def __init__(self, capacity):
        self.capacity, self.loaded, self.events = capacity, {}, []
        self.fail_start = set()

    async def recover(self):
        self.loaded.clear()

    async def activate(self, model):
        await asyncio.sleep(0.01)
        if model.id in self.fail_start:
            raise RuntimeError("load failed")
        self.loaded[model.id] = model.memory_mib
        assert sum(self.loaded.values()) <= self.capacity, "pool exceeded its capacity"
        self.events.append(("start", model.id))

    async def release(self, model, hard=False):
        self.loaded.pop(model.id, None)
        self.events.append(("stop", model.id, hard))


def pool_config(**overrides):
    ms = [
        Model(id=x, upstream="http://localhost:1", start=["true"], stop=["true"], lane="cpu", kind="decision",
              memory_mib=mem, max_hold=0.3, max_runtime=0.5, queue_timeout=2, idle_timeout=overrides.get("idle", 5),
              concurrency=2)
        for x, mem in (("a", 2000), ("b", 2000), ("c", 3000))
    ]
    capacity = overrides.get("capacity", 4500)
    return Config(models={m.id: m for m in ms}, lanes={"cpu": {"capacity_mib": capacity}})


async def result(delay=0.01, value="ok"):
    await asyncio.sleep(delay)
    return value


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


async def started(life):
    return [e[1] for e in life.events if e[0] == "start"]


async def test_models_that_fit_stay_resident_together():
    cfg, life = pool_config(), PoolLifecycle(4500)
    s = PoolScheduler(cfg, life, 4500)
    await s.start()
    try:
        jobs = [s.enqueue(cfg.models[m], "u", 20, lambda: result(0.05)) for m in ("a", "b", "a", "b")]
        assert await asyncio.gather(*(j.future for j in jobs)) == ["ok"] * 4
        assert set(life.loaded) == {"a", "b"}  # both kept loaded together
        assert not any(e[0] == "stop" for e in life.events)
        snap = s.snapshot()
        assert snap["resident"] == ["a", "b"] and snap["used_mib"] == 4000 and snap["policy"] == "pool"
        # more work for either model needs no reload
        await s.enqueue(cfg.models["a"], "u", 20, lambda: result()).future
        assert [e for e in life.events if e[0] == "start"] == [("start", "a"), ("start", "b")]
    finally:
        await s.close()


async def test_model_that_does_not_fit_evicts_least_recently_used_idle():
    # capacity 5000: a + b = 4000 resident; c (3000) needs one of them gone, and b is least recently used
    cfg, life = pool_config(capacity=5000), PoolLifecycle(5000)
    s = PoolScheduler(cfg, life, 5000)
    await s.start()
    try:
        await s.enqueue(cfg.models["a"], "u", 20, lambda: result()).future
        await s.enqueue(cfg.models["b"], "u", 20, lambda: result()).future
        await s.enqueue(cfg.models["a"], "u", 20, lambda: result()).future  # a is now most recently used
        assert await s.enqueue(cfg.models["c"], "u", 20, lambda: result()).future == "ok"
        assert ("stop", "b", False) in life.events and set(life.loaded) == {"a", "c"}
        assert not any(e[:2] == ("stop", "a") for e in life.events)
        assert s.snapshot()["counters"]["evictions"] == 1
    finally:
        await s.close()


async def test_busy_resident_is_not_preempted_and_drains_after_its_lease():
    cfg, life = pool_config(), PoolLifecycle(4500)
    s = PoolScheduler(cfg, life, 4500)
    await s.start()
    try:
        long_a = s.enqueue(cfg.models["a"], "u", 20, lambda: result(0.4))
        long_b = s.enqueue(cfg.models["b"], "u", 20, lambda: result(0.4))
        await until(lambda: len(s.active) == 2)
        c = s.enqueue(cfg.models["c"], "u", 20, lambda: result())
        assert await long_a.future == "ok" and await long_b.future == "ok"  # never cut short
        assert await c.future == "ok"
        assert "c" in life.loaded
    finally:
        await s.close()


async def test_idle_timeout_releases_and_activation_failure_is_isolated():
    cfg, life = pool_config(idle=0.05), PoolLifecycle(4500)
    life.fail_start = {"b"}
    s = PoolScheduler(cfg, life, 4500)
    await s.start()
    try:
        await s.enqueue(cfg.models["a"], "u", 20, lambda: result()).future
        failed = s.enqueue(cfg.models["b"], "u", 20, lambda: result())
        with pytest.raises(Rejected) as exc:
            await failed.future
        assert exc.value.status == 503
        with pytest.raises(Rejected):
            s.enqueue(cfg.models["b"], "u", 20, lambda: result())  # fault window
        assert await s.enqueue(cfg.models["a"], "u", 20, lambda: result()).future == "ok"  # others unaffected
        await until(lambda: not life.loaded)  # idle release
        assert s.fault is None
    finally:
        await s.close()


async def test_pool_admission_limits_and_cancellation():
    cfg, life = pool_config(), PoolLifecycle(4500)
    cfg.max_pending = 2
    s = PoolScheduler(cfg, life, 4500)
    await s.start()
    try:
        j1 = s.enqueue(cfg.models["a"], "u", 20, lambda: result(0.2))
        s.enqueue(cfg.models["a"], "u", 20, lambda: result(0.2))
        with pytest.raises(Rejected) as exc:
            s.enqueue(cfg.models["b"], "u", 20, lambda: result())
        assert exc.value.status == 429
        s.cancel(j1)
        assert j1.future.cancelled() or j1.future.done()
    finally:
        await s.close()


def test_lane_capacity_validation():
    from avifors.config import validate_lanes

    cfg = pool_config()
    validate_lanes(cfg)
    cfg.models["c"].memory_mib = 9000
    with pytest.raises(ValueError):
        validate_lanes(cfg)
    cfg = pool_config()
    cfg.lanes = {"cpu": {"capacity_mib": 4500, "extra": 1}}
    with pytest.raises(ValueError):
        validate_lanes(cfg)
