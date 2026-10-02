import asyncio
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from aiohttp import web

from avifors import workerctl


@pytest.fixture
def sockdir():
    path = Path(tempfile.mkdtemp(prefix="avf"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


class FakeController:
    def __init__(self):
        self.status = 200
        self.body = {"ok": True}
        self.requests = []

    def app(self):
        async def handle(request):
            self.requests.append((request.method, request.path))
            if isinstance(self.body, str):
                return web.Response(status=self.status, text=self.body)
            return web.json_response(self.body, status=self.status)

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handle)
        return app


async def listen(app, path):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.UnixSite(runner, str(path)).start()
    return runner


@pytest.fixture
async def controller(sockdir):
    fake = FakeController()
    runner = await listen(fake.app(), sockdir / "control.sock")
    yield fake, sockdir / "control.sock"
    await runner.cleanup()


def args(path, *argv, wait=0):
    return workerctl.parse(["--socket", str(path), "--wait", str(wait), *argv])


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["start"],
        ["verify"],
        ["gpu-free", "text"],
        ["ping", "text"],
        ["restart", "text"],
        ["start", "text", "extra"],
        ["start", "../text"],
        ["start", "Text"],
        ["--wait", "-1", "ping"],
        ["--wait", "soon", "ping"],
    ],
)
def test_usage_errors_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        workerctl.parse(argv)
    assert exc.value.code == 2


def test_wait_from_environment(monkeypatch):
    monkeypatch.setenv("AVIFORS_WORKERCTL_WAIT", "3")
    assert workerctl.parse(["ping"]).wait == 3
    assert workerctl.parse(["--wait", "1", "ping"]).wait == 1
    monkeypatch.setenv("AVIFORS_WORKERCTL_WAIT", "never")
    with pytest.raises(SystemExit):
        workerctl.parse(["ping"])


def test_socket_from_environment(monkeypatch):
    assert workerctl.parse(["ping"]).socket == "/run/avifors/control.sock"
    monkeypatch.setenv("AVIFORS_CONTROL_SOCKET", "/tmp/x.sock")
    assert workerctl.parse(["ping"]).socket == "/tmp/x.sock"


@pytest.mark.parametrize(
    ("argv", "request_line"),
    [
        (["start", "decision-guard"], ("POST", "/workers/decision-guard/start")),
        (["stop", "stt-qwen"], ("POST", "/workers/stt-qwen/stop")),
        (["verify", "text"], ("GET", "/workers/text/verify")),
        (["gpu-free"], ("GET", "/gpu-free")),
        (["ping"], ("GET", "/ping")),
    ],
)
async def test_routes(controller, argv, request_line):
    fake, path = controller
    assert await workerctl.execute(args(path, *argv)) == 0
    assert fake.requests == [request_line]


@pytest.mark.parametrize(
    ("status", "code"),
    [(200, 0), (409, 1), (403, 1), (424, 1), (500, 1), (502, 1), (504, 1), (404, 2), (400, 2)],
)
async def test_exit_codes(controller, status, code, capsys):
    fake, path = controller
    fake.status, fake.body = status, {"ok": status == 200, "error": "reason"}
    assert await workerctl.execute(args(path, "stop", "text")) == code
    if code:
        assert "reason" in capsys.readouterr().err


async def test_non_json_reply(controller):
    fake, path = controller
    fake.status, fake.body = 502, "<html>bad gateway</html>"
    assert await workerctl.execute(args(path, "verify", "text")) == 1


async def test_missing_socket_fails_after_wait(sockdir, capsys):
    started = time.monotonic()
    assert await workerctl.execute(args(sockdir / "absent.sock", "verify", "text", wait=0)) == 1
    assert time.monotonic() - started < 2
    assert "controller unreachable" in capsys.readouterr().err


async def test_waits_for_a_controller_that_is_still_booting(sockdir):
    path = sockdir / "control.sock"
    fake = FakeController()

    async def boot_late():
        await asyncio.sleep(1)
        return await listen(fake.app(), path)

    booting = asyncio.create_task(boot_late())
    assert await workerctl.execute(args(path, "stop", "text", wait=10)) == 0
    await (await booting).cleanup()
    assert fake.requests == [("POST", "/workers/text/stop")]


def test_usage_exit_code_of_the_module():
    result = subprocess.run(
        [sys.executable, "-m", "avifors.workerctl", "start"], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 2
