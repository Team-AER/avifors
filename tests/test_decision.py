import asyncio
import copy

import pytest
from aiohttp import web
from test_scheduler import FakeLifecycle, config, until

from avifors.config import Model, User
from avifors.decision_worker import TooLong, create_worker
from avifors.server import create_app

AUTH = {"Authorization": "Bearer test-key-alice-long"}
NIMBLE = {
    "model": "decider",
    "state": "Our checkout has returned 500 errors since 9am.",
    "questions": {
        "label": {
            "type": "choice",
            "instructions": "Which label fits this ticket?",
            "criteria": {"billing": None, "bug": None, "account": None},
        }
    },
}


class FakeEngine:
    """Stands in for LayaEngine: fixed probabilities, 10 tokens per question."""

    def __init__(self):
        self.calls = 0
        self.too_long = False

    def tokens(self, state, questions):
        if self.too_long:
            raise TooLong("prompt exceeds the model context (1024)")
        return 10 * len(questions)

    def score(self, state, questions):
        self.calls += 1
        out = {}
        for name, q in questions.items():
            if q["type"] == "choice":
                ks = list(q["criteria"])
                out[name] = {k: (0.9 if i == 1 else 0.1 / (len(ks) - 1)) for i, k in enumerate(ks)}
            elif q["type"] == "noul":
                out[name] = {"false": 0.2, "true": 0.8}
            else:
                n = len(q["criteria"])
                out[name] = {str(i): 1 / n for i in range(n)}
        return out


# ── worker ────────────────────────────────────────────────────────────────────


async def test_worker_answers_in_nimble_shape(aiohttp_client):
    client = await aiohttp_client(create_worker(FakeEngine(), "fake"))
    payload = NIMBLE | {
        "questions": NIMBLE["questions"]
        | {
            "urgent": {"type": "noul", "instructions": "Is it urgent?"},
            "impact": {"type": "score", "instructions": "Impact?", "criteria": ["low", "mid", "high"]},
        }
    }
    r = await client.post("/v1/systemone", json=payload)
    assert r.status == 200
    data = await r.json()
    assert set(data) == {"model", "answers", "usage"}
    assert data["model"] == "decider"
    label = data["answers"]["label"]
    assert set(label) == {"type", "choice", "probabilities", "confidence"}
    assert label["choice"] == "bug" and label["probabilities"]["bug"] == pytest.approx(0.9)
    assert data["answers"]["urgent"] == {"type": "noul", "noul": 0.8}
    impact = data["answers"]["impact"]
    assert set(impact) == {"type", "score", "legend", "probabilities", "confidence"}
    assert impact["score"] == pytest.approx(1.0) and impact["confidence"] == pytest.approx(0)
    assert data["usage"] == {"input_tokens": 30, "output_tokens": 3}


async def test_worker_errors_are_ollama_shaped(aiohttp_client):
    engine = FakeEngine()
    client = await aiohttp_client(create_worker(engine, "fake"))
    r = await client.post("/v1/systemone", json=NIMBLE | {"questions": {}})
    assert r.status == 400 and isinstance((await r.json())["error"], str)
    r = await client.post(
        "/v1/systemone",
        data=b'{"pad":"' + b"x" * (64 * 1024) + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert r.status == 413 and (await r.json()) == {"error": "request body must not exceed 64 KiB"}
    engine.too_long = True
    r = await client.post("/v1/systemone", json=NIMBLE)
    assert r.status == 400 and "context" in (await r.json())["error"]
    assert engine.calls == 0  # nothing scored for rejected requests


# ── broker ────────────────────────────────────────────────────────────────────


def decision_config(tmp_path, decider_url, text_url):
    c = config()
    c.store = tmp_path
    c.admin_key = "admin-secret"
    c.users = [User("alice", "test-key-alice-long", ["*"]), User("bob", "test-key-bob-long", ["text"])]
    for m in c.models.values():
        m.upstream = str(text_url).rstrip("/")
    c.models["decider"] = Model(
        id="decider",
        kind="decision",
        lane="cpu",
        upstream=str(decider_url).rstrip("/"),
        start=["true"],
        stop=["true"],
        max_hold=5,
        max_runtime=2,
        queue_timeout=2,
        idle_timeout=0.05,
    )
    return c


async def stack(aiohttp_client, aiohttp_server, tmp_path, text_delay=0.0):
    engine = FakeEngine()
    worker = await aiohttp_server(create_worker(engine, "fake"))

    async def chat(request):
        await asyncio.sleep(text_delay)
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", chat)
    text = await aiohttp_server(upstream)
    lifecycles = {"gpu": FakeLifecycle(), "cpu": FakeLifecycle()}
    cfg = decision_config(tmp_path, worker.make_url("/"), text.make_url("/"))
    client = await aiohttp_client(create_app(cfg, lifecycles))
    return client, lifecycles, engine


async def test_systemone_through_broker(aiohttp_client, aiohttp_server, tmp_path):
    client, life, engine = await stack(aiohttp_client, aiohttp_server, tmp_path)
    r = await client.post("/v1/systemone", json=NIMBLE, headers=AUTH)
    assert r.status == 200
    data = await r.json()
    assert data["answers"]["label"]["choice"] == "bug" and data["usage"]["input_tokens"] == 10
    assert ("start", "decider") in life["cpu"].events
    assert life["gpu"].events == []  # the CPU decision lane never touched the GPU slot
    await until(lambda: life["cpu"].owner is None)  # idle release
    r = await client.post("/backend/decider/v1/systemone", json=NIMBLE, headers=AUTH)
    assert r.status == 200


async def test_invalid_requests_never_wake_the_worker(aiohttp_client, aiohttp_server, tmp_path):
    client, life, engine = await stack(aiohttp_client, aiohttp_server, tmp_path)
    cases = [
        (NIMBLE | {"questions": {}}, 400),
        (NIMBLE | {"state": ""}, 400),
        (NIMBLE | {"questions": {"x": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}}, 400),
        (NIMBLE | {"model": "missing"}, 404),
        (NIMBLE | {"model": "text"}, 400),  # exists but is not a decision model
    ]
    for payload, status in cases:
        r = await client.post("/v1/systemone", json=payload, headers=AUTH)
        assert r.status == status, payload
        body = await r.json()
        assert set(body) == {"error"} and isinstance(body["error"], str)
    r = await client.post(
        "/v1/systemone",
        data=b'{"pad":"' + b"x" * (64 * 1024) + b'"}',
        headers=AUTH | {"Content-Type": "application/json"},
    )
    assert r.status == 413 and (await r.json()) == {"error": "request body must not exceed 64 KiB"}
    r = await client.post("/v1/systemone", json=NIMBLE)
    assert r.status == 401 and set(await r.json()) == {"error"}
    # A user whose allowlist excludes the model sees it as absent, as Ollama would.
    r = await client.post("/v1/systemone", json=NIMBLE, headers={"Authorization": "Bearer test-key-bob-long"})
    assert r.status == 404
    # Decision models serve only /v1/systemone.
    r = await client.post("/v1/chat/completions", json={"model": "decider", "messages": []}, headers=AUTH)
    assert r.status == 400
    assert life["cpu"].events == [] and life["gpu"].events == [] and engine.calls == 0


async def test_cpu_lane_does_not_evict_resident_gpu_model(aiohttp_client, aiohttp_server, tmp_path):
    client, life, _ = await stack(aiohttp_client, aiohttp_server, tmp_path, text_delay=0.2)
    text = asyncio.create_task(
        client.post("/v1/chat/completions", json={"model": "text", "messages": []}, headers=AUTH)
    )
    await until(lambda: life["gpu"].owner == "text")
    r = await client.post("/v1/systemone", json=NIMBLE, headers=AUTH)
    assert r.status == 200
    assert life["gpu"].owner == "text"  # still resident: the decision ran on its own lane
    assert not any(e[:2] == ("stop", "text") for e in life["gpu"].events)
    assert (await text).status == 200


async def test_catalog_state_and_metrics_include_decision_lane(aiohttp_client, aiohttp_server, tmp_path):
    client, _, _ = await stack(aiohttp_client, aiohttp_server, tmp_path)
    models = {m["id"]: m for m in (await (await client.get("/v1/models", headers=AUTH)).json())["data"]}
    assert models["decider"]["capabilities"] == ["systemone"]
    assert "capabilities" not in models["text"]
    state = await (await client.get("/admin/state", headers={"Authorization": "Bearer admin-secret"})).json()
    assert state["lanes"]["cpu"]["fault"] is None
    metrics = await (await client.get("/metrics", headers=AUTH)).text()
    assert 'avifors_lane_up{lane="cpu"} 1' in metrics
    assert (await client.get("/health", headers=AUTH)).status == 200


async def test_worker_failure_surfaces_as_ollama_error(aiohttp_client, aiohttp_server, tmp_path):
    client, _, engine = await stack(aiohttp_client, aiohttp_server, tmp_path)

    def boom(state, questions):
        raise RuntimeError("cuda died")

    engine.score = boom
    r = await client.post("/v1/systemone", json=NIMBLE, headers=AUTH)
    assert r.status == 500 and (await r.json()) == {"error": "decision scoring failed"}


def test_decision_config_validation():
    base = dict(id="d", upstream="http://127.0.0.1:1", start=["true"], stop=["true"], kind="decision")
    m = Model(**base)
    assert m.paths == ["/v1/systemone"] and m.lane == "gpu"
    with pytest.raises(ValueError):
        Model(**base | {"paths": ["/v1/chat/completions"]})
    with pytest.raises(ValueError):
        Model(**base | {"lane": "CPU lane"})
    copy.deepcopy(m)


async def test_pooled_lane_keeps_two_decision_models_resident(aiohttp_client, aiohttp_server, tmp_path):
    from test_pool import PoolLifecycle

    worker = await aiohttp_server(create_worker(FakeEngine(), "fake"))
    cfg = decision_config(tmp_path, worker.make_url("/"), "http://localhost:1")
    second = copy.deepcopy(cfg.models["decider"])
    second.id = "decider-2"
    cfg.models["decider-2"] = second
    for m in ("decider", "decider-2"):
        cfg.models[m].memory_mib = 1500
        cfg.models[m].idle_timeout = 5
    cfg.lanes = {"cpu": {"capacity_mib": 4000}}
    life = {"gpu": FakeLifecycle(), "cpu": PoolLifecycle(4000)}
    client = await aiohttp_client(create_app(cfg, life))
    for model in ("decider", "decider-2", "decider", "decider-2"):
        r = await client.post("/v1/systemone", json=NIMBLE | {"model": model}, headers=AUTH)
        assert r.status == 200 and (await r.json())["model"] == model
    assert set(life["cpu"].loaded) == {"decider", "decider-2"}
    assert [e for e in life["cpu"].events if e[0] == "stop"] == []
    state = await (await client.get("/admin/state", headers={"Authorization": "Bearer admin-secret"})).json()
    assert state["lanes"]["cpu"]["policy"] == "pool" and state["lanes"]["cpu"]["used_mib"] == 3000
