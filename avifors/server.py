from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import tempfile
import time
from email.parser import BytesParser

import aiohttp
from aiohttp import web

from .audio import AudioJobs, inspect_job, list_jobs, submit
from .config import load
from .image_profiles import LEGACY_SIZES, dimensions, resize_png
from .lifecycle import Lifecycle
from .scheduler import Rejected, Scheduler
from .telemetry import Telemetry

LOG = logging.getLogger(__name__)
NAME = re.compile(r"[A-Za-z0-9_-]{32}\.png\Z")
FORWARD = {"content-type", "x-request-id", "x-session-id", "x-workflow-id", "traceparent", "tracestate"}


def error(status, message):
    return web.json_response(
        {"error": {"message": message, "type": "avifors_error"}},
        status=status,
        headers={"Retry-After": "5"} if status in {429, 503} else {},
    )


def identity(request):
    cfg = request.app["config"]
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""
    for user in cfg.users:
        if hmac.compare_digest(token, user.key):
            return user.name, user.models, user.max_pending
    try:
        address = ipaddress.ip_address(request.remote)
        if any(address in net for net in cfg.trusted_networks):
            return "trusted-network", ["*"], cfg.trusted_max_pending
    except ValueError:
        pass
    raise Rejected(401, "valid API key required")


def admin(request):
    cfg = request.app["config"]
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not cfg.admin_key or not hmac.compare_digest(token, cfg.admin_key):
        raise Rejected(401, "administrator API key required")


@web.middleware
async def errors(request, handler):
    try:
        return await handler(request)
    except Rejected as exc:
        return error(exc.status, exc.message)
    except (ValueError, json.JSONDecodeError):
        return error(400, "invalid request")
    except web.HTTPException:
        raise
    except Exception:
        LOG.exception("request failed path=%s", request.path)
        return error(502, "inference worker failed")


async def models(request):
    _, allowed, _ = identity(request)
    cfg, scheduler = request.app["config"], request.app["scheduler"]
    entries = list(cfg.models.values())
    backend = request.match_info.get("backend")
    if backend:
        entries = [m for m in entries if m.parameters.get("route", m.id) == backend]
        if not entries:
            raise Rejected(404, "unknown backend")
        if scheduler.fault or any(m.id in scheduler.model_faults for m in entries):
            raise Rejected(503, "model worker faulted")
    return web.json_response(
        {
            "object": "list",
            "data": [
                m.metadata | {"id": m.id, "object": "model", "owned_by": "avifors"}
                for m in entries
                if "*" in allowed or m.id in allowed
            ],
        }
    )


async def health(request):
    identity(request)
    scheduler = request.app["scheduler"]
    return web.json_response(
        {"status": "fault" if scheduler.fault else "ok"}, status=503 if scheduler.fault else 200
    )


async def state(request):
    admin(request)
    return web.json_response(request.app["scheduler"].snapshot())


async def metrics(request):
    identity(request)
    s = request.app["scheduler"].snapshot()
    values = {
        "active": s["active"],
        "queued": s["queued"],
        "oldest_wait_seconds": s["oldest_wait_seconds"],
        "lease_remaining_seconds": s["lease_remaining_seconds"],
        "up": int(not s["fault"]),
    }
    output = []
    for key, value in values.items():
        output += [f"# TYPE avifors_{key} gauge", f"avifors_{key} {value}"]
    for key, value in s["counters"].items():
        output += [f"# TYPE avifors_{key}_total counter", f"avifors_{key}_total {value}"]
    image_models = [m.id for m in request.app["config"].models.values() if m.kind == "sdapi"]
    if image_models:
        totals = {
            k: sum(s["model_stats"].get(m, {}).get(k, 0) for m in image_models)
            for k in ("requests", "failures", "busy", "last_duration_seconds")
        }
        output += [
            "imagegen_api_up 1",
            f"imagegen_generation_active {int(s['resident'] in image_models and s['active'] > 0)}",
        ]
        for k, value in totals.items():
            suffix = "" if k.endswith("seconds") else "_total"
            output.append(f"imagegen_{k}{suffix} {value}")
    for m in request.app["config"].models.values():
        if m.kind == "openai":
            try:
                async with request.app["http"].get(
                    m.upstream + "/metrics", timeout=aiohttp.ClientTimeout(total=2)
                ) as r:
                    if r.status == 200:
                        output.append(await r.text())
            except (aiohttp.ClientError, TimeoutError):
                pass
    output.append(f"avifors_trace_dropped_total {request.app['telemetry'].dropped}")
    if "audio" in request.app:
        audio = request.app["audio"]
        for row in audio.db.execute("SELECT state, COUNT(*) FROM jobs GROUP BY state"):
            output.append(f'avifors_audio_jobs{{state="{row[0]}"}} {row[1]}')
        reserved = audio.db.execute("SELECT COALESCE(SUM(reserved),0) FROM jobs").fetchone()[0]
        output.append(f"avifors_audio_reserved_bytes {reserved}")
    return web.Response(text="\n".join(output) + "\n", content_type="text/plain")


def sd_payload(body, model):
    if len(body) > 8192:
        raise Rejected(400, "request body must be 1–8192 bytes")
    r = json.loads(body)
    prompt = r.get("prompt")
    if not isinstance(prompt, str) or not 0 < len(prompt.strip()) <= 2000:
        raise Rejected(400, "prompt must contain 1–2000 characters")
    if "negative_prompt" in r and (
        not isinstance(r["negative_prompt"], str) or len(r["negative_prompt"]) > 2000
    ):
        raise Rejected(400, "negative_prompt must be a string of at most 2000 characters")
    if r.get("n", 1) != 1 or r.get("response_format", "url") != "url":
        raise Rejected(400, "only n=1 and URL responses are supported")
    size = r.get("size", model.parameters.get("default_size", "640x640"))
    if size == "auto":
        size = model.parameters.get("default_size", "640x640")
    if not isinstance(size, str) or size not in model.parameters.get("sizes", LEGACY_SIZES):
        raise Rejected(400, "unsupported image size")
    output_size = dimensions(size)
    render_size = model.parameters.get("render_sizes", {}).get(size, size)
    width, height = dimensions(render_size)
    p = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": 28,
        "cfg_scale": 7,
        "sampler_name": "dpm++2m",
        "scheduler": "karras",
        "batch_size": 1,
    }
    for k in ("steps", "cfg_scale", "sampler_name", "scheduler"):
        if k in model.parameters:
            p[k] = model.parameters[k]
    if r.get("negative_prompt") and model.parameters.get("negative_prompt_mode") == "instruction":
        p["prompt"] += "\n\nExclude the following from the image: " + r["negative_prompt"].strip()
    elif "negative_prompt" in r:
        p["negative_prompt"] = r["negative_prompt"]
    if (width, height) != output_size:
        p["_output_size"] = output_size
    return p


def atomic(path, data):
    handle, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o640)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


async def generate(request, model, payload, user):
    cfg = request.app["config"]
    worker_payload = {k: v for k, v in payload.items() if k != "_output_size"}
    async with request.app["http"].post(
        model.upstream + "/sdapi/v1/txt2img", json=worker_payload
    ) as response:
        response.raise_for_status()
        # Bound encoded image responses too; don't accept an unbounded upstream payload.
        chunks, size = [], 0
        async for chunk in response.content.iter_chunked(65536):
            size += len(chunk)
            if size > 32 * 1024 * 1024:
                raise ValueError("oversized image result")
            chunks.append(chunk)
        body = b"".join(chunks)
        result = json.loads(body)
    encoded = result["images"][0]
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    data = base64.b64decode(encoded, validate=True)
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) > 20 * 1024 * 1024:
        raise ValueError("invalid PNG result")
    if "_output_size" in payload:
        data = await asyncio.to_thread(
            resize_png, data, payload["_output_size"], (payload["width"], payload["height"])
        )
    name = secrets.token_urlsafe(24) + ".png"
    # Keep ownership across restarts without storing user prompts in the broker.
    atomic(cfg.store / (name + ".json"), json.dumps({"user": user}).encode())
    atomic(cfg.store / name, data)
    return web.json_response({"created": int(time.time()), "data": [{"url": f"{cfg.public_base}/{name}"}]})


async def artifact(request):
    user, _, _ = identity(request)
    cfg = request.app["config"]
    name = request.match_info["name"]
    if not NAME.fullmatch(name):
        raise web.HTTPNotFound()
    path = cfg.store / name
    if not path.exists() or time.time() - path.stat().st_mtime > cfg.artifact_ttl:
        raise web.HTTPNotFound()
    owner = json.loads((cfg.store / (name + ".json")).read_text())["user"]
    if user != "trusted-network" and user != owner:
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Cache-Control": "private, max-age=300"})


async def forward(request, model, body, path, holder):
    headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD}
    async with request.app["http"].post(model.upstream + path, data=body, headers=headers) as upstream:
        # Never forward internal lifecycle endpoints, worker cookies, or credentials.
        response_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() in {"content-type", "x-request-id", "retry-after"}
        }
        if "text/event-stream" not in upstream.headers.get("Content-Type", ""):
            data = await upstream.read()
            return web.Response(body=data, status=upstream.status, headers=response_headers)
        response = web.StreamResponse(status=upstream.status, headers=response_headers)
        disconnected = request.transport is None or request.transport.is_closing()
        if not disconnected:
            try:
                await response.prepare(request)
                holder["response"] = response
            except ConnectionError:
                disconnected = True
        async for data in upstream.content.iter_any():
            if not disconnected:
                try:
                    await response.write(data)
                except ConnectionError:
                    disconnected = True
        if not disconnected:
            with contextlib.suppress(ConnectionError):
                await response.write_eof()
        return response if response.prepared else web.Response(status=upstream.status)


async def infer(request):
    user, allowed, limit = identity(request)
    cfg = request.app["config"]
    body = await request.read()
    path = "/v1/" + request.match_info["tail"]
    if request.content_type == "multipart/form-data":
        msg = BytesParser().parsebytes(
            b"Content-Type: " + request.headers["Content-Type"].encode() + b"\r\n\r\n" + body
        )
        if not msg.is_multipart():
            raise Rejected(400, "invalid multipart body")
        fields = {
            p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
            for p in msg.get_payload()
        }
        model_id = fields.get("model", b"").decode()
    else:
        data = json.loads(body)
        if not isinstance(data, dict):
            raise Rejected(400, "request must be a JSON object")
        model_id = data.get("model")
    if not isinstance(model_id, str) or model_id not in cfg.models:
        raise Rejected(404, "unknown model")
    if "*" not in allowed and model_id not in allowed:
        raise Rejected(403, "model not allowed for this user")
    model = cfg.models[model_id]
    backend = request.match_info.get("backend")
    if backend and backend != model.parameters.get("route", model.id):
        raise Rejected(400, "model does not match backend")
    if path not in model.paths:
        raise Rejected(400, "endpoint not supported by model")
    payload = sd_payload(body, model) if model.kind == "sdapi" else None
    holder = {}
    call = (
        (lambda: generate(request, model, payload, user))
        if payload is not None
        else (lambda: forward(request, model, body, path, holder))
    )
    scheduler = request.app["scheduler"]
    job = scheduler.enqueue(model, user, limit, call)
    started_ns, status = time.time_ns(), "error"
    try:
        while not job.future.done():
            if request.transport is None or request.transport.is_closing():
                scheduler.cancel(job)
                raise ConnectionResetError("client disconnected")
            await asyncio.wait({job.future}, timeout=0.1)
        result = job.future.result()
        status = "ok" if result.status < 400 else "error"
        return result
    except asyncio.CancelledError:
        scheduler.cancel(job)
        raise
    except Exception as exc:
        scheduler.cancel(job)
        response = holder.get("response")
        if response is not None and response.prepared:
            message = exc.message if isinstance(exc, Rejected) else "inference stream interrupted"
            with contextlib.suppress(Exception):
                await response.write(
                    (
                        "data: "
                        + json.dumps({"error": {"message": message, "type": "avifors_error"}})
                        + "\n\n"
                    ).encode()
                )
                await response.write_eof()
            return response
        raise
    finally:
        # Retrieve failures even after the downstream connection disappeared.
        def retrieve(f):
            if not f.cancelled():
                f.exception()

        job.future.add_done_callback(retrieve)
        request.app["telemetry"].record(
            request.headers.get("traceparent"),
            model.id,
            started_ns,
            (job.started or time.monotonic()) - job.queued,
            status,
        )


async def prune(app):
    while True:
        cfg = app["config"]
        for path in cfg.store.glob("*.png"):
            if time.time() - path.stat().st_mtime > cfg.artifact_ttl:
                path.unlink(missing_ok=True)
                path.with_suffix(".png.json").unlink(missing_ok=True)
        await asyncio.sleep(3600)


async def context(app):
    cfg = app["config"]
    cfg.store.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=5), trust_env=False, auto_decompress=False
    ) as http:
        app["http"] = http
        app["telemetry"] = Telemetry(http, cfg.otlp_endpoint)
        lifecycle = app.get("lifecycle") or Lifecycle(cfg, http)
        app["scheduler"] = Scheduler(cfg, lifecycle)
        await app["scheduler"].start()
        if any(m.kind == "stt" for m in cfg.models.values()):
            app["audio"] = AudioJobs(app)
            await app["audio"].start()
        app["telemetry"].start()
        pruner = asyncio.create_task(prune(app))
        try:
            yield
        finally:
            pruner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pruner
            if "audio" in app:
                await app["audio"].close()
            await app["scheduler"].close()
            await app["telemetry"].close()


def create_app(cfg, lifecycle=None):
    app = web.Application(middlewares=[errors], client_max_size=cfg.max_body)
    app["config"] = cfg
    if lifecycle:
        app["lifecycle"] = lifecycle
    app.cleanup_ctx.append(context)
    app.add_routes(
        [
            web.get("/health", health),
            web.get("/metrics", metrics),
            web.get("/admin/state", state),
            web.get("/v1/models", models),
            web.get("/backend/{backend}/v1/models", models),
            web.get("/generated/{name}", artifact),
            web.post("/v1/audio/transcriptions", submit),
            web.post("/v1/audio/transcriptions/jobs", submit),
            web.get("/v1/audio/transcriptions/jobs", list_jobs),
            web.get("/v1/audio/transcriptions/jobs/{job}", inspect_job),
            web.delete("/v1/audio/transcriptions/jobs/{job}", inspect_job),
            web.get("/v1/audio/transcriptions/jobs/{job}/result", inspect_job),
            web.post("/v1/audio/transcriptions/jobs/{job}/retry", inspect_job),
            web.post("/backend/{backend}/v1/audio/transcriptions", submit),
            web.post("/v1/{tail:.*}", infer),
            web.post("/backend/{backend}/v1/{tail:.*}", infer),
        ]
    )
    return app


def main():
    parser = argparse.ArgumentParser(description="Demand-loaded inference broker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    cfg = load(args.config)
    if args.check:
        print(f"Valid configuration: {len(cfg.models)} models, policy={cfg.policy}")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    web.run_app(
        create_app(cfg),
        host=cfg.host,
        port=cfg.port,
        access_log=None,
        shutdown_timeout=20,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
