import ast
import asyncio
import os
import shutil
import socket
import stat
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from aiohttp import web

from avifors import controller as ctl
from avifors import workerctl
from avifors.config import Config, Model, command
from avifors.config import load as load_config
from avifors.lifecycle import Lifecycle, execute

DEPLOY = Path(__file__).resolve().parent.parent / "deploy" / "docker"
ROLES = {"text", "image", "stt", "stt-qwen", "decision", "decision-hedwig", "decision-guard"}


@pytest.fixture
def sockdir():
    # AF_UNIX paths are limited to ~104 bytes on macOS; pytest's tmp_path can be longer.
    path = Path(tempfile.mkdtemp(prefix="avf"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


class FakeDocker:
    """Docker Engine API v1.44 subset on a Unix socket, with per-container state."""

    def __init__(self):
        self.containers = {}
        self.calls = []
        self.fail = {}  # (action, name) -> (status, message)
        self.delay = 0.0

    def add(self, name, role, project="avifors", status="exited"):
        self.containers[name] = {
            "Id": f"id-{name}",
            "Name": f"/{name}",
            "Config": {"Labels": {ctl.PROJECT_LABEL: project, ctl.ROLE_LABEL: role}},
            "State": {"Status": status, "Running": status in {"running", "paused"}, "Restarting": False},
        }

    def find(self, key):
        for name, c in self.containers.items():
            if key in (name, c["Id"]):
                return name, c
        raise web.HTTPNotFound(text='{"message": "No such container"}', content_type="application/json")

    def app(self):
        async def ping(request):
            self.calls.append(("ping",))
            return web.Response(text="OK")

        async def inspect(request):
            name, c = self.find(request.match_info["id"])
            self.calls.append(("inspect", name))
            await asyncio.sleep(self.delay)
            return web.json_response(c)

        async def act(request):
            action = request.match_info["action"]
            name, c = self.find(request.match_info["id"])
            self.calls.append((action, request.match_info["id"], dict(request.query)))
            if (action, name) in self.fail:
                status, message = self.fail[action, name]
                return web.json_response({"message": message}, status=status)
            state = c["State"]
            if action == "start":
                if state["Running"]:
                    return web.Response(status=304)
                state.update(Status="running", Running=True)
                return web.Response(status=204)
            if action == "stop":
                if not state["Running"]:
                    return web.Response(status=304)
                state.update(Status="exited", Running=False)
                return web.Response(status=204)
            if action == "wait":
                return web.json_response({"StatusCode": 0})
            return web.json_response({"message": "unsupported"}, status=400)

        app = web.Application()
        app.router.add_get("/v1.44/_ping", ping)
        app.router.add_get("/v1.44/containers/{id}/json", inspect)
        app.router.add_post("/v1.44/containers/{id}/{action}", act)
        return app


async def serve_unix(app, path):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.UnixSite(runner, str(path)).start()
    return runner


def fake_smi(directory, body, code=0):
    script = directory / "nvidia-smi"
    script.write_text(f"#!/bin/sh\nprintf '{body}'\nexit {code}\n")
    script.chmod(0o755)
    return [str(script)]


@pytest.fixture
async def stack(sockdir):
    """A FakeDocker engine, a controller wired to it, and a workerctl runner."""
    docker = FakeDocker()
    for role in ("text", "image", "decision"):
        docker.add(f"avifors-w-{role}", role)
    engine = await serve_unix(docker.app(), sockdir / "docker.sock")
    settings = ctl.Settings(
        workers={
            "text": ctl.Worker("text", "avifors-w-text", 5),
            "image": ctl.Worker("image", "avifors-w-image", 8),
            "decision": ctl.Worker("decision", "avifors-w-decision", 10),
            "missing": ctl.Worker("missing", "avifors-w-missing", 10),
        },
        socket=sockdir / "control.sock",
        docker_socket=str(sockdir / "docker.sock"),
        gpu_free_mib=12000,
        nvidia_smi=fake_smi(sockdir, "13000\\n"),
    )
    client = ctl.Docker(settings.docker_socket)
    control = await serve_unix(ctl.create_app(ctl.Controller(settings, client)), settings.socket)

    async def run(*argv, wait=0):
        return await workerctl.execute(
            workerctl.parse(["--socket", str(settings.socket), "--wait", str(wait), *argv])
        )

    yield docker, settings, run
    await control.cleanup()
    await client.close()
    await engine.cleanup()


async def test_start_stop_verify_cycle(stack):
    docker, _, run = stack
    assert await run("verify", "text") == 0
    assert await run("start", "text") == 0
    assert docker.containers["avifors-w-text"]["State"]["Running"]
    assert ("start", "id-avifors-w-text", {}) in docker.calls  # acts on the inspected ID, not the name
    assert await run("verify", "text") == 1  # still running
    assert await run("start", "text") == 0  # already running (304) is success
    assert await run("stop", "text") == 0
    assert ("stop", "id-avifors-w-text", {"t": "5"}) in docker.calls
    assert ("wait", "id-avifors-w-text", {"condition": "not-running"}) in docker.calls
    assert await run("verify", "text") == 0
    assert await run("stop", "text") == 0  # already stopped (304) is success


async def test_stop_uses_each_roles_grace(stack):
    docker, _, run = stack
    assert await run("start", "image") == 0
    assert await run("stop", "image") == 0
    assert ("stop", "id-avifors-w-image", {"t": "8"}) in docker.calls


@pytest.mark.parametrize("status", ["running", "restarting", "paused", "removing"])
async def test_verify_fails_while_container_is_active(stack, status):
    docker, _, run = stack
    docker.containers["avifors-w-decision"]["State"].update(Status=status, Running=status != "removing")
    assert await run("verify", "decision") == 1


@pytest.mark.parametrize("status", ["created", "exited", "dead"])
async def test_verify_accepts_stopped_states(stack, status):
    docker, _, run = stack
    docker.containers["avifors-w-decision"]["State"].update(Status=status, Running=False)
    assert await run("verify", "decision") == 0


async def test_unknown_role_is_a_usage_error_without_docker_calls(stack):
    docker, _, run = stack
    assert await run("start", "shell") == 2
    assert await run("stop", "avifors-w-text") == 2  # container names are not roles
    assert docker.calls == []


@pytest.mark.parametrize(
    "labels",
    [
        {ctl.PROJECT_LABEL: "other", ctl.ROLE_LABEL: "text"},
        {ctl.PROJECT_LABEL: "avifors", ctl.ROLE_LABEL: "image"},
        {ctl.PROJECT_LABEL: "avifors"},
        {},
    ],
)
async def test_refuses_containers_without_matching_labels(stack, labels):
    docker, _, run = stack
    docker.containers["avifors-w-text"]["Config"]["Labels"] = labels
    for action in ("start", "stop", "verify"):
        assert await run(action, "text") == 1
    assert [c for c in docker.calls if c[0] != "inspect"] == []


async def test_missing_container_fails_closed(stack):
    docker, _, run = stack
    assert await run("stop", "missing") == 1
    assert await run("verify", "missing") == 1
    assert await run("start", "missing") == 1


async def test_docker_errors_fail(stack):
    docker, _, run = stack
    docker.fail["start", "avifors-w-text"] = (500, "could not select device driver")
    assert await run("start", "text") == 1
    docker.containers["avifors-w-image"]["State"].update(Status="running", Running=True)
    docker.fail["stop", "avifors-w-image"] = (
        500,
        "tried to kill container, but did not receive an exit event",
    )
    assert await run("stop", "image") == 1


@pytest.mark.parametrize(
    ("body", "code", "expected"),
    [
        ("13000\\n", 0, 0),
        ("12000\\n", 0, 0),
        ("11999\\n", 0, 1),
        ("15000\\n9000\\n", 0, 1),  # every GPU must have room
        ("", 0, 1),
        ("[N/A]\\n", 0, 1),
        ("13000\\n", 9, 1),
    ],
)
async def test_gpu_free(stack, sockdir, body, code, expected):
    _, settings, run = stack
    settings.nvidia_smi[:] = fake_smi(sockdir, body, code)
    assert await run("gpu-free") == expected


async def test_gpu_free_without_nvidia_smi(stack, sockdir):
    _, settings, run = stack
    settings.nvidia_smi[:] = [str(sockdir / "absent")]
    assert await run("gpu-free") == 1


async def test_ping(stack, sockdir):
    docker, _, run = stack
    assert await run("ping") == 0
    assert ("ping",) in docker.calls


async def test_ping_fails_without_docker(sockdir):
    settings = ctl.Settings(
        workers={"text": ctl.Worker("text", "avifors-w-text")},
        socket=sockdir / "control.sock",
        docker_socket=str(sockdir / "absent.sock"),
    )
    client = ctl.Docker(settings.docker_socket)
    control = await serve_unix(ctl.create_app(ctl.Controller(settings, client)), settings.socket)
    try:
        args = workerctl.parse(["--socket", str(settings.socket), "--wait", "0", "ping"])
        assert await workerctl.execute(args) == 1
        args = workerctl.parse(["--socket", str(settings.socket), "--wait", "0", "verify", "text"])
        assert await workerctl.execute(args) == 1
    finally:
        await control.cleanup()
        await client.close()


async def test_concurrent_operations_on_one_role_are_serialised(stack):
    docker, _, run = stack
    docker.delay = 0.05  # without the role lock both inspections would land before either action
    assert await asyncio.gather(run("start", "text"), run("stop", "text")) == [0, 0]
    actions = [c[0] for c in docker.calls]
    assert actions in (
        ["inspect", "start", "inspect", "stop", "wait"],
        ["inspect", "stop", "wait", "inspect", "start"],
    )


async def test_serve_secures_and_removes_its_socket(sockdir):
    class NoDocker:
        async def ping(self):
            pass

        async def close(self):
            pass

    path = sockdir / "run" / "control.sock"
    path.parent.mkdir()
    with socket.socket(socket.AF_UNIX) as stale:
        stale.bind(str(path))  # closing without unlinking leaves the file, as after a crash
    assert path.exists()
    settings = ctl.Settings(workers={"text": ctl.Worker("text", "c")}, socket=path, socket_group=os.getgid())
    task = asyncio.create_task(ctl.serve(settings, NoDocker()))
    try:
        args = workerctl.parse(["--socket", str(path), "--wait", "5", "ping"])
        assert await workerctl.execute(args) == 0
        info = path.stat()
        assert stat.S_ISSOCK(info.st_mode)
        assert stat.S_IMODE(info.st_mode) == 0o660
        assert info.st_gid == os.getgid()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not path.exists()


async def test_serve_refuses_to_replace_a_regular_file(sockdir):
    path = sockdir / "control.sock"
    path.write_text("not a socket")
    settings = ctl.Settings(workers={"text": ctl.Worker("text", "c")}, socket=path, socket_group=os.getgid())
    with pytest.raises(SystemExit):
        await ctl.serve(settings, object())
    assert path.read_text() == "not a socket"


async def test_broker_lifecycle_drives_workers_through_workerctl(stack):
    """The unchanged broker lifecycle runs workerctl as a subprocess; only the argv differs."""
    docker, settings, _ = stack
    ctl_argv = [sys.executable, "-m", "avifors.workerctl", "--socket", str(settings.socket), "--wait", "5"]
    await execute([*ctl_argv, "start", "text"], 30)
    assert docker.containers["avifors-w-text"]["State"]["Running"]
    with pytest.raises(RuntimeError, match="exit 1"):
        await execute([*ctl_argv, "verify", "text"], 30)
    lifecycle = Lifecycle(
        Config(models={}, release_check=[*ctl_argv, "gpu-free"]),
        http=None,
    )
    model = Model(
        id="text",
        upstream="http://127.0.0.1:18000",
        start=[*ctl_argv, "start", "text"],
        stop=[*ctl_argv, "stop", "text"],
        verify_stopped=[*ctl_argv, "verify", "text"],
    )
    await lifecycle.release(model, hard=True)  # stop, verify, then the GPU release check
    assert not docker.containers["avifors-w-text"]["State"]["Running"]
    with pytest.raises(RuntimeError, match="exit 2"):
        await execute([*ctl_argv, "start", "shell"], 30)


# ── allowlist parsing ──────────────────────────────────────────────────────────────────────────


def write(tmp_path, data):
    path = tmp_path / "workers.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_defaults(tmp_path):
    s = ctl.load(write(tmp_path, {"workers": {"text": {"container": "avifors-w-text"}}}))
    assert s.workers["text"] == ctl.Worker("text", "avifors-w-text", 10)
    assert (s.project, s.gpu_free_mib, s.socket_group) == ("avifors", 12000, 990)
    assert s.socket == Path("/run/avifors/control.sock")


@pytest.mark.parametrize(
    "data",
    [
        {},
        [],
        {"workers": {}},
        {"workers": {"Text": {"container": "a"}}},
        {"workers": {"text": {"container": "a b"}}},
        {"workers": {"text": {"container": "a", "image": "evil"}}},
        {"workers": {"text": {"container": "a", "stop_timeout": -1}}},
        {"workers": {"text": {"container": "a", "stop_timeout": "5"}}},
        {"workers": {"text": {"container": "a"}, "image": {"container": "a"}}},
        {"workers": {"text": {"container": "a"}}, "gpu_free_mib": "lots"},
        {"workers": {"text": {"container": "a"}}, "nvidia_smi": "nvidia-smi"},
        {"workers": {"text": {"container": "a"}}, "commands": ["rm"]},
    ],
)
def test_load_rejects_invalid_allowlists(tmp_path, data):
    with pytest.raises(ValueError):
        ctl.load(write(tmp_path, data))


# ── the shipped deployment files must agree with each other ───────────────────────────────────


def test_shipped_allowlist_matches_compose_and_config():
    settings = ctl.load(DEPLOY / "workers.yaml")
    assert set(settings.workers) == ROLES
    assert (settings.project, settings.gpu_free_mib, settings.socket_group) == ("avifors", 12000, 990)

    compose = yaml.safe_load((DEPLOY / "compose.yaml").read_text())
    assert compose["name"] == settings.project
    services = compose["services"]
    for role, worker in settings.workers.items():
        service = services[role]
        assert service["container_name"] == worker.container
        assert service["labels"][ctl.ROLE_LABEL] == role
        assert service["profiles"] == ["workers"]
        assert service["restart"] == "no"
    assert services["broker"]["network_mode"] == "host"
    assert services["broker"]["user"] == "996:990"
    assert services["controller"]["network_mode"] == "none"

    config = yaml.safe_load((DEPLOY / "config.docker.example.yaml").read_text())
    assert command(config["server"]["release_check"]) == ["avifors-workerctl", "gpu-free"]
    used = set()
    for model in config["models"]:
        role = model["start"][2]
        used.add(role)
        assert model["start"] == ["avifors-workerctl", "start", role]
        assert model["stop"] == ["avifors-workerctl", "stop", role]
        assert model["verify_stopped"] == ["avifors-workerctl", "verify", role]
    assert used == ROLES


def test_shipped_docker_config_is_valid(monkeypatch):
    monkeypatch.setenv("AVIFORS_USER_KEY", "example-key-long-enough")
    cfg = load_config(DEPLOY / "config.docker.example.yaml")
    assert {m.lane for m in cfg.models.values()} == {"gpu", "cpu"}


def test_stt_worker_stays_python_312_compatible():
    # The Omnilingual image is the Python 3.12 exception (avifors-stt-omni.Dockerfile).
    for name in ("__init__.py", "stt_worker.py"):
        source = (Path(ctl.__file__).parent / name).read_text()
        ast.parse(source, filename=name, feature_version=(3, 12))


def test_workerctl_console_script_exit_code(sockdir):
    # The broker runs the installed script; its exit status is the whole contract.
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "avifors.workerctl",
            "--socket",
            str(sockdir / "absent"),
            "--wait",
            "0",
            "ping",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "controller unreachable" in result.stderr
