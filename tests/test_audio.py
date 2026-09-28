import asyncio
import io
import json
import shutil
import struct
import wave
from array import array

import aiohttp
import pytest
from aiohttp import web
from test_http import AUTH, configured
from test_scheduler import FakeLifecycle, until

from avifors.audio import DEFAULTS, plan_chunks, stitch
from avifors.server import create_app


def wav(seconds=1, silent=False):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((b"\0\0" if silent else struct.pack("<h", 1234)) * round(16000 * seconds))
    return out.getvalue()


def form(data=None, **fields):
    f = aiohttp.FormData()
    f.add_field("file", wav() if data is None else data, filename="input.wav", content_type="audio/wav")
    for key, value in ({"model": "speech"} | fields).items():
        f.add_field(key, value)
    return f


def speech_config(tmp_path, url):
    cfg = configured(tmp_path / "images", url)
    model = cfg.models.pop("image")
    model.id, model.kind = "speech", "stt"
    model.max_hold, model.max_runtime = 2, 1
    model.paths = ["/v1/audio/transcriptions"]
    cfg.models["speech"] = model
    cfg.audio = {
        "store": str(tmp_path / "audio"),
        "max_upload_bytes": 5 * 1024**2,
        "max_duration_seconds": 600,
        "min_free_bytes": 1,
        "min_execution_budget": 0.05,
    }
    return cfg


@pytest.fixture(autouse=True)
def ffmpeg_path(monkeypatch):
    import os

    if not shutil.which("ffmpeg") and shutil.which("/opt/homebrew/bin/ffmpeg"):
        monkeypatch.setenv("PATH", "/opt/homebrew/bin:" + os.environ["PATH"])


async def service(aiohttp_client, aiohttp_server, tmp_path, worker=None):
    async def default_worker(request):
        assert request.content_type == "application/octet-stream"
        assert len(await request.read()) > 0
        return web.json_response({"text": "hello world"})

    app = web.Application()
    app.router.add_post("/transcribe", worker or default_worker)
    upstream = await aiohttp_server(app)
    cfg = speech_config(tmp_path, upstream.make_url("/"))
    life = FakeLifecycle()
    client = await aiohttp_client(create_app(cfg, life))
    return client, life, cfg


async def wait_job(client, jid, status="completed"):
    async with asyncio.timeout(10):
        while True:
            r = await client.get("/v1/audio/transcriptions/jobs/" + jid, headers=AUTH)
            row = await r.json()
            if row["status"] == status:
                return row
            assert row["status"] not in {"failed", "cancelled"}, row
            await asyncio.sleep(0.02)


async def test_sync_and_async_result_formats(aiohttp_client, aiohttp_server, tmp_path):
    client, life, cfg = await service(aiohttp_client, aiohttp_server, tmp_path)
    r = await client.post("/v1/audio/transcriptions", data=form(language="en"), headers=AUTH)
    assert r.status == 200, await r.text()
    assert await r.json() == {"text": "hello world"}
    jid = r.headers["X-Transcription-Job-ID"]
    for fmt, expected in [("text", "hello world"), ("srt", "00:00:00,000"), ("vtt", "WEBVTT")]:
        r = await client.get(
            f"/v1/audio/transcriptions/jobs/{jid}/result?response_format={fmt}", headers=AUTH
        )
        assert expected in await r.text()
    r = await client.get(
        f"/v1/audio/transcriptions/jobs/{jid}/result?response_format=verbose_json", headers=AUTH
    )
    assert (await r.json())["timestamp_granularity"] == "chunk"
    await until(lambda: not (tmp_path / "audio" / jid).exists())


async def test_ownership_idempotency_and_no_catalog_activation(aiohttp_client, aiohttp_server, tmp_path):
    client, life, cfg = await service(aiohttp_client, aiohttp_server, tmp_path)
    await client.get("/v1/models", headers=AUTH)
    assert life.events == []
    r = await client.post(
        "/v1/audio/transcriptions/jobs", data=form(), headers=AUTH | {"Idempotency-Key": "one"}
    )
    assert r.status == 202
    jid = (await r.json())["id"]
    repeat = await client.post(
        "/v1/audio/transcriptions/jobs", data=form(), headers=AUTH | {"Idempotency-Key": "one"}
    )
    assert (await repeat.json())["id"] == jid
    for method, suffix in [("get", ""), ("delete", ""), ("get", "/result")]:
        r = await getattr(client, method)(
            f"/v1/audio/transcriptions/jobs/{jid}{suffix}",
            headers={"Authorization": "Bearer test-key-bob-long"},
        )
        assert r.status == 404
    await wait_job(client, jid)


async def test_upload_validation_and_limits(aiohttp_client, aiohttp_server, tmp_path):
    client, life, cfg = await service(aiohttp_client, aiohttp_server, tmp_path)
    for fields in [
        {"model": "text"},
        {"prompt": "unsupported"},
        {"language": "../../bad"},
        {"response_format": "bad"},
    ]:
        r = await client.post("/v1/audio/transcriptions/jobs", data=form(**fields), headers=AUTH)
        assert r.status in {400, 404}
    r = await client.post("/v1/audio/transcriptions/jobs", data=form(), headers={})
    assert r.status == 401
    client.app["audio"].cfg["max_upload_bytes"] = 5
    r = await client.post("/v1/audio/transcriptions/jobs", data=form(), headers=AUTH)
    assert r.status == 413
    assert life.events == []


async def test_invalid_audio_fails_without_gpu(aiohttp_client, aiohttp_server, tmp_path):
    client, life, cfg = await service(aiohttp_client, aiohttp_server, tmp_path)
    r = await client.post("/v1/audio/transcriptions/jobs", data=form(b"not audio"), headers=AUTH)
    jid = (await r.json())["id"]
    async with asyncio.timeout(5):
        while client.app["audio"].row(jid)["state"] != "failed":
            await asyncio.sleep(0.03)
    assert not life.events


async def test_active_cancel_does_not_reset_worker(aiohttp_client, aiohttp_server, tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    async def worker(request):
        entered.set()
        await release.wait()
        return web.json_response({"text": "discarded"})

    client, life, cfg = await service(aiohttp_client, aiohttp_server, tmp_path, worker)
    r = await client.post("/v1/audio/transcriptions/jobs", data=form(), headers=AUTH)
    jid = (await r.json())["id"]
    await asyncio.wait_for(entered.wait(), 5)
    r = await client.delete("/v1/audio/transcriptions/jobs/" + jid, headers=AUTH)
    assert (await r.json())["status"] == "cancelled"
    release.set()
    await until(lambda: not client.app["scheduler"].active)
    assert client.app["audio"].row(jid)["state"] == "cancelled"
    assert not any(e == ("stop", "speech", True) for e in life.events)


async def test_recovery_resumes_committed_frontier(aiohttp_client, aiohttp_server, tmp_path):
    calls = []

    async def worker(request):
        calls.append(len(await request.read()))
        return web.json_response({"text": "new chunk"})

    client, life, cfg = await service(aiohttp_client, aiohttp_server, tmp_path, worker)
    manager = client.app["audio"]
    row, _ = manager.reserve("alice", "speech", None, None)
    jid = row["id"]
    (manager.root / jid / "audio.pcm").write_bytes(struct.pack("<h", 1234) * 32000)
    plan = [
        {"start": 0, "end": 1, "silent": False, "overlap": 0},
        {"start": 1, "end": 2, "silent": False, "overlap": 0},
    ]
    manager.db.execute("INSERT INTO chunks VALUES (?,?,?,?,?)", (jid, 0, 0, 1, "committed"))
    manager.update(jid, plan=json.dumps(plan), duration=2, next_chunk=1, options="{}")
    await client.close()
    # Recreate interrupted state on disk before starting a fresh broker.
    import sqlite3

    with sqlite3.connect(tmp_path / "audio/jobs.sqlite3") as db:
        db.execute("UPDATE jobs SET state='running' WHERE id=?", (jid,))
    client2 = await aiohttp_client(create_app(cfg, FakeLifecycle()))
    row = await wait_job(client2, jid)
    assert row["chunks_completed"] == 2
    assert len(calls) == 1
    r = await client2.get(f"/v1/audio/transcriptions/jobs/{jid}/result", headers=AUTH)
    assert (await r.json())["text"] == "committed new chunk"


def test_two_hour_chunk_plan_is_bounded_and_complete(tmp_path):
    path = tmp_path / "long.pcm"
    # Sparse two-hour PCM file with speech markers at both ends.
    with path.open("wb") as out:
        out.write(array("h", [1000] * 16000).tobytes())
        out.seek(7200 * 32000 - 32000)
        out.write(array("h", [1000] * 16000).tobytes())
    chunks = plan_chunks(path, DEFAULTS)
    assert chunks[0]["start"] == 0 and chunks[-1]["end"] == 7200
    assert not chunks[0]["silent"] and not chunks[-1]["silent"]
    assert all(0 < c["end"] - c["start"] <= 30 + 1e-9 for c in chunks)
    assert all(b["start"] <= a["end"] for a, b in zip(chunks, chunks[1:]))
    assert sum(c["silent"] for c in chunks) > 200


def test_exact_overlap_stitch_does_not_rewrite():
    assert stitch("hello beautiful world", "beautiful world again", 0.8) == "again"
    assert stitch("yes", "yes indeed", 0.8) == "yes indeed"
    assert stitch("hello world", "world unrelated", 0) == "world unrelated"


async def test_failed_job_can_retry_from_checkpoint(aiohttp_client, aiohttp_server, tmp_path):
    client, life, cfg = await service(aiohttp_client, aiohttp_server, tmp_path)
    manager = client.app["audio"]
    row, _ = manager.reserve("alice", "speech", None, None)
    jid = row["id"]
    (manager.root / jid / "audio.pcm").write_bytes(struct.pack("<h", 1234) * 16000)
    manager.update(
        jid,
        state="failed",
        error="test interruption",
        duration=1,
        plan=json.dumps([{"start": 0, "end": 1, "silent": False, "overlap": 0}]),
    )
    r = await client.post(f"/v1/audio/transcriptions/jobs/{jid}/retry", headers=AUTH)
    assert r.status == 202
    await wait_job(client, jid)
    r = await client.get("/v1/audio/transcriptions/jobs", headers=AUTH)
    assert [j["id"] for j in (await r.json())["data"]] == [jid]


async def test_decode_duration_limit_before_gpu(aiohttp_client, aiohttp_server, tmp_path):
    client, life, cfg = await service(aiohttp_client, aiohttp_server, tmp_path)
    client.app["audio"].cfg["max_duration_seconds"] = 0.5
    r = await client.post("/v1/audio/transcriptions/jobs", data=form(wav(2)), headers=AUTH)
    jid = (await r.json())["id"]
    await until(lambda: client.app["audio"].row(jid)["state"] == "failed")
    assert not life.events


async def test_fragmented_worker_json(aiohttp_client, aiohttp_server, tmp_path):
    async def worker(request):
        response = web.StreamResponse(headers={"Content-Type": "application/json"})
        await response.prepare(request)
        await response.write(b'{"text":')
        await asyncio.sleep(0.01)
        await response.write(b'"complete transcript"}')
        return response

    client, life, cfg = await service(aiohttp_client, aiohttp_server, tmp_path, worker)
    r = await client.post("/v1/audio/transcriptions", data=form(), headers=AUTH)
    assert r.status == 200
    assert (await r.json())["text"] == "complete transcript"
