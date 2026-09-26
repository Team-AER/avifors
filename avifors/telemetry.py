"""Bounded, best-effort OTLP/HTTP export; no prompts, keys or response bodies."""

import asyncio
import logging
import re
import secrets
import time

import aiohttp

LOG = logging.getLogger(__name__)
CONTEXT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})\Z", re.I)


class Telemetry:
    def __init__(self, http, endpoint):
        self.http, self.endpoint = http, endpoint
        self.queue = asyncio.Queue(maxsize=256)
        self.dropped = 0
        self.task = None

    def start(self):
        if self.endpoint:
            self.task = asyncio.create_task(self.run())

    def record(self, context, model, started_ns, queue_seconds, status):
        if not self.endpoint:
            return
        match = CONTEXT.fullmatch(context or "")
        span = {
            "traceId": match[1] if match else secrets.token_hex(16),
            "spanId": secrets.token_hex(8),
            "name": "avifors.inference",
            "kind": 2,
            "startTimeUnixNano": str(started_ns),
            "endTimeUnixNano": str(time.time_ns()),
            "attributes": [
                {"key": "gen_ai.request.model", "value": {"stringValue": model}},
                {"key": "avifors.queue_seconds", "value": {"doubleValue": queue_seconds}},
            ],
            "status": {"code": 1 if status == "ok" else 2},
        }
        if match:
            span["parentSpanId"] = match[2]
        try:
            self.queue.put_nowait(span)
        except asyncio.QueueFull:
            self.dropped += 1

    async def run(self):
        while True:
            span = await self.queue.get()
            body = {
                "resourceSpans": [
                    {
                        "resource": {
                            "attributes": [{"key": "service.name", "value": {"stringValue": "avifors"}}]
                        },
                        "scopeSpans": [{"scope": {"name": "avifors"}, "spans": [span]}],
                    }
                ]
            }
            try:
                async with self.http.post(
                    self.endpoint, json=body, timeout=aiohttp.ClientTimeout(total=3)
                ) as r:
                    r.raise_for_status()
            except (aiohttp.ClientError, TimeoutError):
                self.dropped += 1
                LOG.warning("trace export failed")
            finally:
                self.queue.task_done()

    async def close(self):
        if self.task:
            try:
                async with asyncio.timeout(3):
                    await self.queue.join()
            except TimeoutError:
                pass
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
