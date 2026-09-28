"""Durable, bounded transcription jobs. GPU work always passes through Scheduler."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import secrets
import shutil
import signal
import sqlite3
import time
from array import array
from pathlib import Path

import aiohttp
from aiohttp import web

from .scheduler import Rejected

LOG = logging.getLogger(__name__)
RATE = 16000
TERMINAL = {"completed", "failed", "cancelled"}
DEFAULTS = {
    "max_upload_bytes": 2 * 1024**3,
    "max_duration_seconds": 8 * 3600,
    "max_storage_bytes": 20 * 1024**3,
    "min_free_bytes": 2 * 1024**3,
    "max_jobs": 16,
    "max_jobs_per_user": 4,
    "upload_timeout": 3600,
    "decode_timeout": 3600,
    "job_timeout": 7 * 86400,
    "retention_seconds": 2 * 86400,
    "sync_timeout": 600,
    "chunk_seconds": 30,
    "min_chunk_seconds": 20,
    "overlap_seconds": 0.8,
    "min_execution_budget": 60,
    "chunk_retries": 3,
}


def stitch(previous, current, overlap):
    """Remove only an exact overlap; never heuristically rewrite the transcript."""
    if not overlap or not previous:
        return current.strip()
    left, right = previous.split(), current.split()
    norm = lambda s: re.sub(r"[^\w]", "", s).casefold()  # noqa: E731
    for n in range(min(24, len(left), len(right)), 1, -1):
        if [norm(s) for s in left[-n:]] == [norm(s) for s in right[:n]]:
            return " ".join(right[n:])
    # Languages without word separators need a character overlap instead.
    if len(right) <= 1:
        for n in range(min(48, len(previous), len(current)), 3, -1):
            if previous[-n:] == current[:n]:
                return current[n:]
    return current.strip()


def plan_chunks(path, settings):
    """Scan bounded PCM windows; prefer quiet boundaries and overlap forced cuts.

    Only digital silence is skipped. Quiet speech is retained. Every source sample
    belongs to a chunk or a recorded silent interval; memory is O(chunk length).
    """
    total = path.stat().st_size // 2
    maximum = int(settings["chunk_seconds"] * RATE)
    minimum = int(settings["min_chunk_seconds"] * RATE)
    overlap = int(settings["overlap_seconds"] * RATE)
    chunks, start, previous_end = [], 0, 0
    with path.open("rb") as stream:
        while start < total:
            end = min(start + maximum, total)
            stream.seek(start * 2)
            samples = array("h", stream.read((end - start) * 2))
            if os.sys.byteorder != "little":
                samples.byteswap()
            silent = not any(samples)
            quiet = False
            if end < total and not silent:
                # 100ms RMS windows, within the final 10 seconds of a chunk.
                window = RATE // 10
                candidates = [
                    (sum(v * v for v in samples[i : i + window]), i)
                    for i in range(minimum, len(samples) - window + 1, window)
                ]
                if candidates:
                    energy, offset = min(candidates)
                    quiet = energy / window < 32**2
                    end = start + offset + window // 2
            chunks.append(
                {
                    "start": start / RATE,
                    "end": end / RATE,
                    "silent": silent,
                    "overlap": max(0, previous_end - start) / RATE,
                }
            )
            previous_end = end
            start = end if quiet or silent or end == total else end - overlap
    return chunks


async def decode_audio(source, dest, cfg):
    """Decode with local-file-only protocols; cap output even for forged durations."""
    limit = int((cfg["max_duration_seconds"] + 1) * RATE * 2)
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "2",
        "-protocol_whitelist",
        "file,pipe",
        "-format_whitelist",
        "wav,mp3,flac,ogg,mov,matroska,webm,aac,aiff,au",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(RATE),
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    count = 0
    try:
        async with asyncio.timeout(cfg["decode_timeout"]):
            with dest.open("wb") as target:
                while data := await process.stdout.read(65536):
                    count += len(data)
                    if count > limit:
                        raise Rejected(413, "audio duration exceeds configured limit")
                    if shutil.disk_usage(dest.parent).free < cfg["min_free_bytes"]:
                        raise Rejected(507, "insufficient audio storage")
                    target.write(data)
                target.flush()
                os.fsync(target.fileno())
            code = await process.wait()
        if code or not count or count / (RATE * 2) > cfg["max_duration_seconds"]:
            raise Rejected(400 if code or not count else 413, "invalid audio or duration limit exceeded")
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
    return count / (RATE * 2)


class AudioJobs:
    def __init__(self, app):
        self.app = app
        raw = dict(app["config"].audio)
        self.root = Path(raw.pop("store", app["config"].store.parent / "audio"))
        unknown = set(raw) - DEFAULTS.keys()
        if unknown:
            raise ValueError(f"unknown audio settings: {sorted(unknown)}")
        self.cfg = DEFAULTS | raw
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in self.cfg.values()):
            raise ValueError("audio limits must be finite positive numbers")
        if not self.cfg["overlap_seconds"] < self.cfg["min_chunk_seconds"] < self.cfg["chunk_seconds"] < 40:
            raise ValueError("audio chunks must satisfy overlap < minimum < maximum < 40 seconds")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.root / "jobs.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, owner TEXT NOT NULL, model TEXT NOT NULL,
            state TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
            options TEXT NOT NULL, duration REAL DEFAULT 0, next_chunk INTEGER DEFAULT 0,
            plan TEXT DEFAULT '[]', error TEXT, trace TEXT, idem TEXT, reserved INTEGER DEFAULT 0,
            UNIQUE(owner, idem));
          CREATE TABLE IF NOT EXISTS chunks (
            job TEXT NOT NULL, idx INTEGER NOT NULL, start REAL, end REAL, text TEXT NOT NULL,
            PRIMARY KEY(job, idx));
        """)
        # An interrupted upload has no complete source; committed jobs resume.
        self.db.execute("UPDATE jobs SET state='failed',error='upload interrupted' WHERE state='uploading'")
        self.db.execute("UPDATE jobs SET state='queued' WHERE state='running'")
        self.db.commit()
        self.tasks = {}
        self.inflight = {}
        self.preparation = asyncio.Semaphore(1)
        self.closed = False
        self.runner = None

    def row(self, job_id):
        return self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def update(self, job_id, **values):
        values["updated"] = time.time()
        self.db.execute(
            "UPDATE jobs SET " + ",".join(k + "=?" for k in values) + " WHERE id=?",
            (*values.values(), job_id),
        )
        self.db.commit()

    def view(self, row):
        chunks = json.loads(row["plan"])
        done = row["next_chunk"]
        processed = chunks[done - 1]["end"] if done else 0
        return {
            "id": row["id"],
            "object": "audio.transcription.job",
            "model": row["model"],
            "status": row["state"],
            "created_at": row["created"],
            "duration": row["duration"],
            "processed_seconds": processed,
            "progress": processed / row["duration"] if row["duration"] else 0,
            "chunks_completed": done,
            "chunks_total": len(chunks),
            "error": row["error"],
            "expires_at": row["updated"] + self.cfg["retention_seconds"]
            if row["state"] in TERMINAL
            else None,
            "result_url": f"/v1/audio/transcriptions/jobs/{row['id']}/result",
        }

    def reserve(self, user, model, idem, trace):
        if idem:
            existing = self.db.execute("SELECT * FROM jobs WHERE owner=? AND idem=?", (user, idem)).fetchone()
            if existing:
                return existing, False
        active = self.db.execute(
            "SELECT owner FROM jobs WHERE state NOT IN ('completed','failed','cancelled')"
        ).fetchall()
        if (
            len(active) >= self.cfg["max_jobs"]
            or sum(r[0] == user for r in active) >= self.cfg["max_jobs_per_user"]
        ):
            raise Rejected(429, "audio job quota exceeded")
        # Reserve worst-case input + PCM before consuming any request bytes.
        reservation = int(self.cfg["max_upload_bytes"] + (self.cfg["max_duration_seconds"] + 1) * RATE * 2)
        used = self.db.execute("SELECT COALESCE(SUM(reserved),0) FROM jobs").fetchone()[0]
        if (
            used + reservation > self.cfg["max_storage_bytes"]
            or shutil.disk_usage(self.root).free < reservation + self.cfg["min_free_bytes"]
        ):
            raise Rejected(507, "audio storage quota exceeded")
        job_id, now = secrets.token_hex(16), time.time()
        self.db.execute(
            "INSERT INTO jobs (id,owner,model,state,created,updated,options,trace,idem,reserved) VALUES (?,?,?,'uploading',?,?,'{}',?,?,?)",
            (job_id, user, model, now, now, trace, idem, reservation),
        )
        self.db.commit()
        (self.root / job_id).mkdir(mode=0o700)
        return self.row(job_id), True

    def authorized(self, job_id, user, allowed):
        row = self.row(job_id)
        if not row or row["owner"] != user or ("*" not in allowed and row["model"] not in allowed):
            raise Rejected(404, "unknown transcription job")
        return row

    async def start(self):
        self.runner = asyncio.create_task(self.loop())

    async def close(self):
        self.closed = True
        if self.runner:
            self.runner.cancel()
            await asyncio.gather(self.runner, return_exceptions=True)
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.db.close()

    async def loop(self):
        while True:
            now = time.time()
            for row in self.db.execute("SELECT * FROM jobs").fetchall():
                jid = row["id"]
                if jid in self.tasks:
                    continue
                if row["state"] in TERMINAL:
                    # Failed work retains its checkpoint/audio for explicit retry.
                    if row["state"] != "failed":
                        shutil.rmtree(self.root / jid, ignore_errors=True)
                        if row["reserved"]:
                            self.update(jid, reserved=0)
                    if now - row["updated"] > self.cfg["retention_seconds"]:
                        shutil.rmtree(self.root / jid, ignore_errors=True)
                        self.db.execute("DELETE FROM chunks WHERE job=?", (jid,))
                        self.db.execute("DELETE FROM jobs WHERE id=?", (jid,))
                        self.db.commit()
                elif row["state"] != "uploading":
                    if now - row["created"] > self.cfg["job_timeout"]:
                        self.update(jid, state="failed", error="job deadline exceeded")
                    else:
                        task = asyncio.create_task(self.run(jid))
                        self.tasks[jid] = task
                        task.add_done_callback(lambda t, j=jid: self.tasks.pop(j, None))
            await asyncio.sleep(0.25)

    async def prepare(self, jid):
        async with self.preparation:
            row = self.row(jid)
            if json.loads(row["plan"]):
                return
            self.update(jid, state="preparing")
            directory = self.root / jid
            duration = await decode_audio(directory / "source", directory / "audio.pcm", self.cfg)
            plan = await asyncio.to_thread(plan_chunks, directory / "audio.pcm", self.cfg)
            self.update(jid, duration=duration, plan=json.dumps(plan), state="queued")
            (directory / "source").unlink(missing_ok=True)
            self.update(jid, reserved=(directory / "audio.pcm").stat().st_size)

    async def call_chunk(self, jid, chunk, model, options):
        path = self.root / jid / "audio.pcm"
        begin, end = round(chunk["start"] * RATE), round(chunk["end"] * RATE)
        with path.open("rb") as stream:
            stream.seek(begin * 2)
            data = stream.read((end - begin) * 2)
        if len(data) != (end - begin) * 2:
            raise RuntimeError("audio checkpoint is incomplete")
        headers = {"Content-Type": "application/octet-stream"}
        if self.row(jid)["trace"]:
            headers["traceparent"] = self.row(jid)["trace"]
        params = {"language": options.get("language", "")}
        async with self.app["http"].post(
            model.upstream + "/transcribe", data=data, params=params, headers=headers
        ) as r:
            if r.status == 400:
                raise Rejected(400, "unsupported language or invalid audio chunk")
            r.raise_for_status()
            body = bytearray()
            async for part in r.content.iter_chunked(65536):
                body.extend(part)
                if len(body) > 1024 * 1024:
                    raise RuntimeError("oversized transcript")
            result = json.loads(body)
            if not isinstance(result.get("text"), str):
                raise RuntimeError("invalid transcript")
            return result["text"]

    async def run(self, jid):
        scheduler = self.app["scheduler"]
        job = None
        try:
            await self.prepare(jid)
            row = self.row(jid)
            model = self.app["config"].models.get(row["model"])
            if not model or model.kind != "stt":
                raise Rejected(404, "transcription model is unavailable")
            options, plan = json.loads(row["options"]), json.loads(row["plan"])
            failures = 0
            while (row := self.row(jid))["next_chunk"] < len(plan):
                if row["state"] == "cancelled":
                    return
                if time.time() - row["created"] > self.cfg["job_timeout"]:
                    raise Rejected(504, "job deadline exceeded")
                idx = row["next_chunk"]
                chunk = plan[idx]
                if chunk["silent"]:
                    text = ""
                else:
                    self.update(jid, state="queued")
                    try:

                        async def call(c=chunk):
                            self.update(jid, state="running")
                            return await self.call_chunk(jid, c, model, options)

                        budget = min(self.cfg["min_execution_budget"], model.max_hold * 0.8)
                        job = scheduler.enqueue(
                            model, row["owner"], self.user_limit(row["owner"]), call, min_budget=budget
                        )
                        self.inflight[jid] = job
                        started = time.time_ns()
                        try:
                            text = await asyncio.shield(job.future)
                        finally:
                            self.app["telemetry"].record(
                                row["trace"],
                                model.id,
                                started,
                                (job.started or time.monotonic()) - job.queued,
                                "ok"
                                if job.future.done()
                                and not job.future.cancelled()
                                and job.future.exception() is None
                                else "error",
                            )
                            self.inflight.pop(jid, None)
                    except (Rejected, aiohttp.ClientError, TimeoutError) as exc:
                        if self.row(jid)["state"] == "cancelled":
                            return
                        status = getattr(exc, "status", 502)
                        if status in {400, 401, 403, 404}:
                            raise
                        # Admission/queue expiry does not consume an inference retry.
                        if status not in {429, 503} and not (
                            isinstance(exc, Rejected) and exc.message == "queue deadline exceeded"
                        ):
                            failures += 1
                        if failures >= self.cfg["chunk_retries"]:
                            raise Rejected(502, "audio chunk failed repeatedly") from exc
                        self.update(jid, state="queued")
                        await asyncio.sleep(5)
                        continue
                if self.row(jid)["state"] == "cancelled":
                    return
                # Commit text and frontier in one transaction, so recovery never duplicates it.
                previous = self.db.execute(
                    "SELECT text FROM chunks WHERE job=? ORDER BY idx DESC LIMIT 1", (jid,)
                ).fetchone()
                text = stitch(previous[0] if previous else "", text, chunk["overlap"])
                self.db.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?)",
                    (jid, idx, chunk["start"], chunk["end"], text),
                )
                self.update(jid, next_chunk=idx + 1, state="queued")
                failures = 0
                await asyncio.sleep(0)
            self.update(jid, state="completed", error=None)
        except asyncio.CancelledError:
            self.inflight.pop(jid, None)
            if job:
                scheduler.cancel(job)
                job.future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            raise
        except Exception as exc:
            if self.row(jid)["state"] == "cancelled":
                return
            LOG.warning("transcription job failed id=%s type=%s", jid, type(exc).__name__)
            self.update(
                jid,
                state="failed",
                error=exc.message if isinstance(exc, Rejected) else "audio processing failed",
            )

    def user_limit(self, owner):
        cfg = self.app["config"]
        return next((u.max_pending for u in cfg.users if u.name == owner), cfg.trusted_max_pending)

    def result(self, row, fmt):
        segments = [
            dict(r)
            for r in self.db.execute(
                "SELECT idx AS id,start,end,text FROM chunks WHERE job=? ORDER BY idx", (row["id"],)
            )
            if r["text"]
        ]
        text = " ".join(s["text"] for s in segments)
        headers = {"Cache-Control": "no-store", "X-Transcription-Job-ID": row["id"]}
        if fmt == "text":
            return web.Response(text=text, headers=headers)
        if fmt in {"srt", "vtt"}:

            def stamp(value):
                ms = round(value * 1000)
                h, ms = divmod(ms, 3600000)
                m, ms = divmod(ms, 60000)
                s, ms = divmod(ms, 1000)
                return f"{h:02}:{m:02}:{s:02}{'.' if fmt == 'vtt' else ','}{ms:03}"

            lines = ["WEBVTT\n"] if fmt == "vtt" else []
            for n, seg in enumerate(segments, 1):
                lines.append(f"{n}\n{stamp(seg['start'])} --> {stamp(seg['end'])}\n{seg['text']}\n")
            return web.Response(
                text="\n".join(lines),
                content_type="text/vtt" if fmt == "vtt" else "text/plain",
                headers=headers,
            )
        result = {"text": text}
        if fmt == "verbose_json":
            result |= {
                "duration": row["duration"],
                "segments": segments,
                "timestamp_granularity": "chunk",
                "language": json.loads(row["options"]).get("language"),
            }
        return web.json_response(result, headers=headers)


FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}


async def list_jobs(request):
    from .server import identity

    user, allowed, _ = identity(request)
    manager = request.app.get("audio")
    if not manager:
        raise Rejected(404, "transcription is not configured")
    rows = manager.db.execute("SELECT * FROM jobs WHERE owner=? ORDER BY created DESC LIMIT 100", (user,))
    return web.json_response(
        {
            "object": "list",
            "data": [manager.view(r) for r in rows if "*" in allowed or r["model"] in allowed],
        },
        headers={"Cache-Control": "no-store"},
    )


async def submit(request):
    from .server import identity

    user, allowed, _ = identity(request)
    manager = request.app.get("audio")
    if not manager:
        raise Rejected(404, "transcription is not configured")
    if request.content_type != "multipart/form-data":
        raise Rejected(400, "multipart audio upload required")
    idem = request.headers.get("Idempotency-Key")
    if idem and len(idem) > 256:
        raise Rejected(400, "idempotency key is too long")
    # Store a digest, never caller-provided identifying key material.
    idem = hashlib.sha256(idem.encode()).hexdigest() if idem else None
    row, fresh = manager.reserve(user, "", idem, request.headers.get("traceparent"))
    if not fresh:
        manager.authorized(row["id"], user, allowed)
        return web.json_response(manager.view(row), status=200, headers={"Cache-Control": "no-store"})
    jid = row["id"]
    size, fields, has_file = 0, {}, False
    try:
        async with asyncio.timeout(manager.cfg["upload_timeout"]):
            reader = await request.multipart()
            async for part in reader:
                if part.name == "file":
                    if (
                        has_file
                        or part.headers.get("Content-Encoding")
                        or part.headers.get("Content-Transfer-Encoding")
                    ):
                        raise Rejected(400, "one unencoded audio file is required")
                    has_file = True
                    with (manager.root / jid / "source").open("wb") as out:
                        while chunk := await part.read_chunk(65536):
                            size += len(chunk)
                            if size > manager.cfg["max_upload_bytes"]:
                                raise Rejected(413, "audio upload limit exceeded")
                            out.write(chunk)
                        out.flush()
                        os.fsync(out.fileno())
                else:
                    if part.name not in {"model", "language", "response_format"} or part.name in fields:
                        raise Rejected(400, "unsupported or duplicate transcription field")
                    value = bytearray()
                    while chunk := await part.read_chunk(4096):
                        value.extend(chunk)
                        if len(value) > 256:
                            raise Rejected(400, "transcription field is too long")
                    fields[part.name] = value.decode("utf-8")
            model = request.app["config"].models.get(fields.get("model"))
            if not model or model.kind != "stt":
                raise Rejected(404, "unknown transcription model")
            if "*" not in allowed and model.id not in allowed:
                raise Rejected(403, "model not allowed for this user")
            backend = request.match_info.get("backend")
            if backend and backend != model.parameters.get("route", model.id):
                raise Rejected(400, "model does not match backend")
            if not has_file or not size:
                raise Rejected(400, "audio file is required")
            if fields.get("response_format", "json") not in FORMATS:
                raise Rejected(400, "unsupported response format")
            language = fields.get("language", "")
            if language and not re.fullmatch(r"[A-Za-z]{2,3}(?:_[A-Za-z]{4})?", language):
                raise Rejected(400, "language must be an ISO code or ISO-script identifier")
            manager.update(jid, model=model.id, options=json.dumps(fields), state="uploaded")
    except BaseException:
        manager.update(jid, state="failed", error="upload rejected or interrupted")
        shutil.rmtree(manager.root / jid, ignore_errors=True)
        manager.update(jid, reserved=0)
        raise
    location = f"/v1/audio/transcriptions/jobs/{jid}"
    headers = {"Location": location, "Cache-Control": "no-store", "X-Transcription-Job-ID": jid}
    if request.path.endswith("/jobs"):
        return web.json_response(manager.view(manager.row(jid)), status=202, headers=headers)
    # Disconnect leaves the durable job alive. Long clients should use /jobs.
    until = time.monotonic() + manager.cfg["sync_timeout"]
    while time.monotonic() < until:
        row = manager.row(jid)
        if row["state"] == "completed":
            return manager.result(row, fields.get("response_format", "json"))
        if row["state"] in TERMINAL:
            return web.json_response(manager.view(row), status=422, headers=headers)
        if request.transport is None or request.transport.is_closing():
            break
        await asyncio.sleep(0.25)
    return web.json_response(manager.view(manager.row(jid)), status=202, headers=headers)


async def inspect_job(request):
    from .server import identity

    user, allowed, _ = identity(request)
    manager = request.app.get("audio")
    if not manager:
        raise Rejected(404, "transcription is not configured")
    row = manager.authorized(request.match_info["job"], user, allowed)
    if request.method == "POST":
        directory = manager.root / row["id"]
        if row["state"] != "failed" or not any((directory / n).exists() for n in ("source", "audio.pcm")):
            raise Rejected(409, "job has no retryable audio checkpoint")
        active = manager.db.execute(
            "SELECT owner FROM jobs WHERE state NOT IN ('completed','failed','cancelled')"
        ).fetchall()
        if (
            len(active) >= manager.cfg["max_jobs"]
            or sum(r[0] == user for r in active) >= manager.cfg["max_jobs_per_user"]
        ):
            raise Rejected(429, "audio job quota exceeded")
        manager.update(row["id"], state="queued", error=None, created=time.time())
        return web.json_response(
            manager.view(manager.row(row["id"])), status=202, headers={"Cache-Control": "no-store"}
        )
    if request.method == "DELETE":
        if row["state"] not in {"completed", "cancelled"}:
            manager.update(row["id"], state="cancelled", error=None)
            # Running work drains under the original deadline; do not reset peers.
            job = manager.inflight.get(row["id"])
            if job and job in request.app["scheduler"].pending:
                request.app["scheduler"].cancel(job)
            task = manager.tasks.get(row["id"])
            if task and not job:
                task.cancel()
        return web.json_response(manager.view(manager.row(row["id"])), headers={"Cache-Control": "no-store"})
    if request.path.endswith("/result"):
        if row["state"] != "completed":
            return web.json_response(manager.view(row), status=409, headers={"Cache-Control": "no-store"})
        fmt = request.query.get("response_format", json.loads(row["options"]).get("response_format", "json"))
        if fmt not in FORMATS:
            raise Rejected(400, "unsupported response format")
        return manager.result(row, fmt)
    return web.json_response(manager.view(row), headers={"Cache-Control": "no-store"})
