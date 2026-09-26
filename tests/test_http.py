import asyncio
import base64

from aiohttp import web
from test_scheduler import FakeLifecycle, config

from avifors.config import User
from avifors.server import create_app

AUTH = {"Authorization": "Bearer test-key-alice-long"}


def configured(tmp_path, url):
    c = config()
    c.store = tmp_path
    c.admin_key = "admin-secret"
    c.users = [User("alice", "test-key-alice-long", ["*"]), User("bob", "test-key-bob-long", ["text"])]
    for m in c.models.values():
        m.upstream = str(url).rstrip("/")
    c.models["image"].kind = "sdapi"
    c.models["image"].paths = ["/v1/images/generations"]
    return c


async def test_auth_catalog_and_admin_do_not_wake(aiohttp_client, tmp_path):
    life = FakeLifecycle()
    c = configured(tmp_path, "http://localhost:1")
    client = await aiohttp_client(create_app(c, life))
    assert (await client.get("/v1/models")).status == 401
    r = await client.get("/v1/models", headers=AUTH)
    assert len((await r.json())["data"]) == 2
    assert (await client.get("/admin/state", headers=AUTH)).status == 401
    assert (await client.get("/admin/state", headers={"Authorization": "Bearer admin-secret"})).status == 200
    assert life.events == []


async def test_model_allowlist_and_invalid_images_do_not_load(aiohttp_client, tmp_path):
    life = FakeLifecycle()
    client = await aiohttp_client(create_app(configured(tmp_path, "http://localhost:1"), life))
    r = await client.post(
        "/v1/images/generations",
        headers={"Authorization": "Bearer test-key-bob-long"},
        json={"model": "image", "prompt": "test"},
    )
    assert r.status == 403
    for change in ({"size": "1024x1024"}, {"negative_prompt": 12}, {"n": 2}, {"response_format": "b64_json"}):
        r = await client.post(
            "/v1/images/generations", headers=AUTH, json={"model": "image", "prompt": "test"} | change
        )
        assert r.status == 400
    assert not life.events


async def test_stream_and_context_passthrough(aiohttp_client, aiohttp_server, tmp_path):
    observed = {}

    async def worker(request):
        observed.update(await request.json())
        observed["trace"] = request.headers.get("traceparent")
        observed["auth"] = request.headers.get("Authorization")
        r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await r.prepare(request)
        await r.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
        await asyncio.sleep(0.01)
        await r.write(b"data: [DONE]\n\n")
        return r

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", worker)
    server = await aiohttp_server(upstream)
    client = await aiohttp_client(create_app(configured(tmp_path, server.make_url("/")), FakeLifecycle()))
    payload = {"model": "text", "messages": [], "stream": True, "temperature": 0.2, "top_k": 64}
    r = await client.post("/v1/chat/completions", json=payload, headers=AUTH | {"traceparent": "test-trace"})
    text = await r.text()
    assert "hello" in text and "[DONE]" in text
    assert observed == payload | {"trace": "test-trace", "auth": None}


async def test_image_settings_and_artifact_survive_release(aiohttp_client, aiohttp_server, tmp_path):
    observed = {}
    png = b"\x89PNG\r\n\x1a\n" + b"test"

    async def worker(request):
        observed.update(await request.json())
        return web.json_response({"images": [base64.b64encode(png).decode()]})

    upstream = web.Application()
    upstream.router.add_post("/sdapi/v1/txt2img", worker)
    server = await aiohttp_server(upstream)
    life = FakeLifecycle()
    client = await aiohttp_client(create_app(configured(tmp_path, server.make_url("/")), life))
    r = await client.post(
        "/v1/images/generations",
        headers=AUTH,
        json={"model": "image", "prompt": "test", "negative_prompt": "text", "size": "512x768"},
    )
    assert r.status == 200
    data = await r.json()
    name = data["data"][0]["url"].rsplit("/", 1)[1]
    await asyncio.sleep(0.15)
    assert life.owner is None
    assert await (await client.get("/generated/" + name, headers=AUTH)).read() == png
    assert (
        await client.get("/generated/" + name, headers={"Authorization": "Bearer test-key-bob-long"})
    ).status == 404
    assert observed == {
        "prompt": "test",
        "negative_prompt": "text",
        "width": 512,
        "height": 768,
        "steps": 28,
        "cfg_scale": 7,
        "sampler_name": "dpm++2m",
        "scheduler": "karras",
        "batch_size": 1,
    }


async def test_stream_deadline_emits_error_not_success(aiohttp_client, aiohttp_server, tmp_path):
    async def worker(request):
        r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await r.prepare(request)
        await r.write(b'data: {"hello":1}\n\n')
        await asyncio.sleep(0.2)
        return r

    upstream = web.Application()
    upstream.router.add_post("/v1/chat/completions", worker)
    server = await aiohttp_server(upstream)
    c = configured(tmp_path, server.make_url("/"))
    c.models["text"].max_runtime = 0.05
    life = FakeLifecycle()
    client = await aiohttp_client(create_app(c, life))
    r = await client.post("/v1/chat/completions", json={"model": "text", "stream": True}, headers=AUTH)
    text = await r.text()
    assert '"error"' in text
    assert "[DONE]" not in text
