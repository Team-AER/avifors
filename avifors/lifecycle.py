from __future__ import annotations

import asyncio
import logging
import os
import signal

import aiohttp

LOG = logging.getLogger(__name__)


async def execute(argv, timeout):
    """Commands are administrator-owned argv, never request-controlled shell input."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        async with asyncio.timeout(timeout):
            code = await process.wait()
        if code:
            raise RuntimeError(f"worker command failed (exit {code})")
    except BaseException:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
        raise


class Lifecycle:
    def __init__(self, config, http):
        self.config, self.http = config, http
        self.sleeping = set()

    async def recover(self):
        # Fail closed if any owned worker cannot be confirmed stopped.
        for model in self.config.models.values():
            await self.release(model, hard=True, check=False)
        await self.check()

    async def check(self):
        if self.config.release_check:
            await execute(self.config.release_check, 10)

    async def activate(self, model):
        async with asyncio.timeout(min(model.activation_timeout, model.max_hold)):
            if model.id in self.sleeping:
                async with self.http.post(model.upstream + "/wake_up") as r:
                    r.raise_for_status()
                self.sleeping.remove(model.id)
            else:
                await execute(model.start, model.activation_timeout)
            while True:
                try:
                    async with self.http.get(
                        model.upstream + model.health_path, timeout=aiohttp.ClientTimeout(total=2)
                    ) as r:
                        if r.status == 200:
                            return
                except (aiohttp.ClientError, TimeoutError):
                    pass
                await asyncio.sleep(0.25)

    async def release(self, model, hard=False, check=True):
        if model.sleep and not hard:
            try:
                async with self.http.post(
                    model.upstream + "/sleep?level=1",
                    timeout=aiohttp.ClientTimeout(total=model.release_timeout),
                ) as r:
                    r.raise_for_status()
                self.sleeping.add(model.id)
                await self.check()
                return
            except (aiohttp.ClientError, TimeoutError, RuntimeError):
                LOG.warning("sleep failed; stopping worker model=%s", model.id)
        await execute(model.stop, model.release_timeout)
        if model.verify_stopped:
            await execute(model.verify_stopped, model.release_timeout)
        self.sleeping.discard(model.id)
        if check:
            await self.check()
