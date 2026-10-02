"""Root-side worker controller: the only Avifors process that can reach the Docker Engine.

The broker runs unprivileged and asks this sidecar, through ``avifors-workerctl`` over a Unix
socket, to start, stop or verify a worker or to check GPU memory. Requests name a *role*, never a
container: the root-owned allowlist maps each role to one container, and the controller refuses
any container that does not carry the Avifors compose labels for that role.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import yaml
from aiohttp import web

LOG = logging.getLogger("avifors.controller")

# Docker Engine 25 introduced API 1.44 and Engine 29 made it the oldest supported version, so this
# pin works unchanged on every engine from 26 through 29.
DOCKER_API = "v1.44"
PROJECT_LABEL = "com.docker.compose.project"
ROLE_LABEL = "org.team-aer.avifors.role"
ROLE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
CONTAINER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")
# Running, restarting, paused or being removed: the worker may still hold memory or the GPU.
ACTIVE_STATES = {"running", "restarting", "paused", "removing"}


class Failure(Exception):
    """An operation that did not succeed; ``status`` is the controller's HTTP status."""

    def __init__(self, status, message, docker_status=None):
        super().__init__(message)
        self.status, self.message, self.docker_status = status, message, docker_status


@dataclass(frozen=True)
class Worker:
    role: str
    container: str
    stop_timeout: int = 10


@dataclass(frozen=True)
class Settings:
    workers: dict[str, Worker]
    project: str = "avifors"
    socket: Path = Path("/run/avifors/control.sock")
    socket_group: int = 990
    docker_socket: str = "/var/run/docker.sock"
    gpu_free_mib: int = 12000
    nvidia_smi: list[str] = field(default_factory=lambda: ["nvidia-smi"])


def load(path) -> Settings:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("the allowlist must be a mapping")
    known = {"project", "socket", "socket_group", "docker_socket", "gpu_free_mib", "nvidia_smi", "workers"}
    if unknown := set(raw) - known:
        raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
    workers = {}
    for role, spec in (raw.get("workers") or {}).items():
        if not isinstance(role, str) or not ROLE.fullmatch(role):
            raise ValueError(f"invalid role {role!r}")
        if not isinstance(spec, dict) or set(spec) - {"container", "stop_timeout"}:
            raise ValueError(f"role {role}: only container and stop_timeout are allowed")
        name = spec.get("container")
        if not isinstance(name, str) or not CONTAINER.fullmatch(name):
            raise ValueError(f"role {role}: invalid container name")
        timeout = spec.get("stop_timeout", 10)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 0 <= timeout <= 300:
            raise ValueError(f"role {role}: stop_timeout must be an integer between 0 and 300")
        workers[role] = Worker(role, name, timeout)
    if not workers:
        raise ValueError("the allowlist names no workers")
    if len({w.container for w in workers.values()}) != len(workers):
        raise ValueError("two roles share one container")
    nvidia_smi = raw.get("nvidia_smi", ["nvidia-smi"])
    if (
        not isinstance(nvidia_smi, list)
        or not nvidia_smi
        or not all(isinstance(x, str) and x for x in nvidia_smi)
    ):
        raise ValueError("nvidia_smi must be a nonempty argv array")
    gpu_free = raw.get("gpu_free_mib", 12000)
    group = raw.get("socket_group", 990)
    for name, value in (("gpu_free_mib", gpu_free), ("socket_group", group)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    project = raw.get("project", "avifors")
    if not isinstance(project, str) or not CONTAINER.fullmatch(project):
        raise ValueError("invalid compose project name")
    return Settings(
        workers=workers,
        project=project,
        socket=Path(raw.get("socket", "/run/avifors/control.sock")),
        socket_group=group,
        docker_socket=str(raw.get("docker_socket", "/var/run/docker.sock")),
        gpu_free_mib=gpu_free,
        nvidia_smi=nvidia_smi,
    )


class Docker:
    """The few Docker Engine API calls the controller needs, over the Engine's Unix socket."""

    def __init__(self, path):
        self.session = aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=path), trust_env=False)

    async def close(self):
        await self.session.close()

    async def call(self, method, path, *, timeout=30, params=None, ok=(200,)):
        url = f"http://docker/{DOCKER_API}{path}"
        try:
            async with self.session.request(
                method, url, params=params, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as r:
                status, body = r.status, await r.read()
        except TimeoutError as exc:  # before ClientError: aiohttp's timeouts are both
            raise Failure(504, f"docker {method} {path} timed out") from exc
        except (aiohttp.ClientError, OSError) as exc:
            raise Failure(502, f"docker engine unreachable: {exc}") from exc
        if status not in ok:
            message = body.decode(errors="replace").strip()
            with contextlib.suppress(ValueError, AttributeError):
                message = json.loads(body).get("message", message)
            raise Failure(502, f"docker {method} {path}: {status} {message}", docker_status=status)
        return body

    async def ping(self):
        await self.call("GET", "/_ping", timeout=5)

    async def inspect(self, name):
        try:
            body = await self.call("GET", f"/containers/{name}/json")
        except Failure as exc:
            if exc.docker_status == 404:
                return None
            raise
        return json.loads(body)

    async def start(self, cid):
        await self.call("POST", f"/containers/{cid}/start", ok=(204, 304))

    async def stop(self, cid, grace):
        await self.call(
            "POST", f"/containers/{cid}/stop", params={"t": str(grace)}, timeout=grace + 30, ok=(204, 304)
        )
        # Stop normally returns after the exit; waiting for not-running also covers an engine that
        # gave up waiting after SIGKILL, so "stopped" is only reported once the process is gone.
        await self.call("POST", f"/containers/{cid}/wait", params={"condition": "not-running"}, timeout=30)


class Controller:
    def __init__(self, settings: Settings, docker):
        self.settings, self.docker = settings, docker
        self.locks = {role: asyncio.Lock() for role in settings.workers}

    def worker(self, role) -> Worker:
        if not isinstance(role, str) or not ROLE.fullmatch(role) or role not in self.settings.workers:
            raise Failure(404, f"unknown role {role!r}")
        return self.settings.workers[role]

    async def checked(self, worker: Worker):
        """Inspect the role's container and refuse it unless it is the labelled Avifors worker."""
        info = await self.docker.inspect(worker.container)
        if info is None:
            raise Failure(
                424,
                f"container {worker.container} does not exist; run docker compose --profile workers create",
            )
        labels = (info.get("Config") or {}).get("Labels") or {}
        if (
            info.get("Name", "").lstrip("/") != worker.container
            or labels.get(PROJECT_LABEL) != self.settings.project
            or labels.get(ROLE_LABEL) != worker.role
        ):
            raise Failure(
                403,
                f"container {worker.container} is not labelled {PROJECT_LABEL}={self.settings.project}, "
                f"{ROLE_LABEL}={worker.role}",
            )
        if not info.get("Id"):
            raise Failure(502, f"docker returned no ID for {worker.container}")
        return info

    async def start(self, role):
        worker = self.worker(role)
        async with self.locks[role]:
            info = await self.checked(worker)
            await self.docker.start(info["Id"])
        return {"role": role, "container": worker.container, "state": "started"}

    async def stop(self, role):
        worker = self.worker(role)
        async with self.locks[role]:
            info = await self.checked(worker)
            await self.docker.stop(info["Id"], worker.stop_timeout)
        return {"role": role, "container": worker.container, "state": "stopped"}

    async def verify(self, role):
        worker = self.worker(role)
        async with self.locks[role]:
            info = await self.checked(worker)
        state = info.get("State") or {}
        status = state.get("Status", "")
        if state.get("Running") or state.get("Restarting") or status in ACTIVE_STATES:
            raise Failure(409, f"{worker.container} is still {status or 'running'}")
        return {"role": role, "container": worker.container, "state": status or "stopped"}

    async def gpu_free(self):
        argv = [*self.settings.nvidia_smi, "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
        try:
            process = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
            )
        except OSError as exc:
            raise Failure(502, f"cannot run nvidia-smi: {exc}") from exc
        try:
            async with asyncio.timeout(8):
                out, _ = await process.communicate()
        except TimeoutError as exc:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise Failure(504, "nvidia-smi timed out") from exc
        if process.returncode:
            raise Failure(502, f"nvidia-smi failed (exit {process.returncode})")
        try:
            free = [int(line.strip()) for line in out.decode().splitlines() if line.strip()]
        except ValueError as exc:
            raise Failure(502, "unparseable nvidia-smi output") from exc
        if not free:
            raise Failure(502, "nvidia-smi reported no GPUs")
        result = {"free_mib": free, "required_mib": self.settings.gpu_free_mib}
        if min(free) < self.settings.gpu_free_mib:
            raise Failure(409, f"only {min(free)} MiB GPU memory free, {self.settings.gpu_free_mib} required")
        return result

    async def ping(self):
        await self.docker.ping()
        return {"docker": "ok"}


def create_app(controller: Controller):
    async def run(operation, label):
        started = time.monotonic()
        try:
            result = await operation
        except Failure as exc:
            LOG.warning("%s failed status=%s: %s", label, exc.status, exc.message)
            return web.json_response({"ok": False, "error": exc.message}, status=exc.status)
        LOG.info("%s ok elapsed=%.2f", label, time.monotonic() - started)
        return web.json_response({"ok": True} | result)

    async def act(request):
        role, action = request.match_info["role"], request.match_info["action"]
        method = {"start": "POST", "stop": "POST", "verify": "GET"}.get(action)
        if method is None:
            return web.json_response({"ok": False, "error": "unknown action"}, status=404)
        if request.method != method:
            return web.json_response({"ok": False, "error": "method not allowed"}, status=405)
        return await run(getattr(controller, action)(role), f"{action} {role!r}")

    async def gpu_free(request):
        return await run(controller.gpu_free(), "gpu-free")

    async def ping(request):
        return await run(controller.ping(), "ping")

    app = web.Application(client_max_size=1024)
    app.router.add_route("*", "/workers/{role}/{action}", act)
    app.router.add_get("/gpu-free", gpu_free)
    app.router.add_get("/ping", ping)
    return app


def prepare_socket(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(mode):
        raise SystemExit(f"{path} exists and is not a socket; refusing to replace it")
    path.unlink()


def secure_socket(path: Path, group: int):
    # Only root and the broker's group may connect; anything else is refused by the kernel.
    os.chown(path, -1, group)
    os.chmod(path, 0o660)


async def serve(settings: Settings, docker=None):
    prepare_socket(settings.socket)
    docker = docker or Docker(settings.docker_socket)
    runner = web.AppRunner(create_app(Controller(settings, docker)), access_log=None, handle_signals=False)
    await runner.setup()
    previous = os.umask(0o117)
    try:
        site = web.UnixSite(runner, str(settings.socket))
        await site.start()
    finally:
        os.umask(previous)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopping.set)
    try:
        secure_socket(settings.socket, settings.socket_group)
        LOG.info("listening on %s for %d workers", settings.socket, len(settings.workers))
        await stopping.wait()
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(sig)
        await runner.cleanup()
        await docker.close()
        with contextlib.suppress(FileNotFoundError):
            settings.socket.unlink()


def main():
    parser = argparse.ArgumentParser(description="Avifors worker controller (runs as root beside Docker)")
    parser.add_argument("--config", default="/etc/avifors/workers.yaml", help="root-owned worker allowlist")
    parser.add_argument("--check", action="store_true", help="validate the allowlist and exit")
    args = parser.parse_args()
    settings = load(args.config)
    if args.check:
        print(f"Valid allowlist: {', '.join(sorted(settings.workers))}")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(serve(settings))


if __name__ == "__main__":
    main()
