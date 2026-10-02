"""Unprivileged client for avifors-controller, keeping the old sudo helper's command contract.

    avifors-workerctl start|stop|verify ROLE
    avifors-workerctl gpu-free
    avifors-workerctl ping

Exit 0: done, or confirmed stopped / enough GPU memory free. Exit 1: failed, still running, not
enough GPU memory, or the controller is unreachable. Exit 2: usage error or a role the controller's
allowlist does not name. The broker's ``start``/``stop``/``verify_stopped``/``release_check`` argv
can therefore point here instead of ``sudo avifors-worker`` with no other change.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time

import aiohttp

SOCKET = "/run/avifors/control.sock"
WAIT = 20.0  # seconds to retry a missing or refusing socket, e.g. while the controller boots
TIMEOUT = 150.0  # the broker kills the command earlier, at its own activation/release timeout
ROLE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
ROUTES = {
    "start": ("POST", "/workers/{role}/start"),
    "stop": ("POST", "/workers/{role}/stop"),
    "verify": ("GET", "/workers/{role}/verify"),
    "gpu-free": ("GET", "/gpu-free"),
    "ping": ("GET", "/ping"),
}
ROLE_COMMANDS = {"start", "stop", "verify"}


def parse(argv=None):
    parser = argparse.ArgumentParser(prog="avifors-workerctl", description="Ask avifors-controller to act")
    parser.add_argument("--socket", default=os.environ.get("AVIFORS_CONTROL_SOCKET", SOCKET))
    parser.add_argument(
        "--wait",
        type=float,
        default=None,
        help=f"seconds to retry while the controller socket is unavailable (default {WAIT:g}, "
        "or AVIFORS_WORKERCTL_WAIT)",
    )
    parser.add_argument("command", choices=list(ROUTES))
    parser.add_argument("role", nargs="?")
    args = parser.parse_args(argv)
    if (args.command in ROLE_COMMANDS) != (args.role is not None):
        parser.error(f"{args.command} {'needs' if args.command in ROLE_COMMANDS else 'takes no'} ROLE")
    if args.role is not None and not ROLE.fullmatch(args.role):
        parser.error("invalid role")
    if args.wait is None:
        try:
            args.wait = float(os.environ.get("AVIFORS_WORKERCTL_WAIT", WAIT))
        except ValueError:
            parser.error("AVIFORS_WORKERCTL_WAIT must be a number")
    if not 0 <= args.wait <= 600:
        parser.error("--wait must be between 0 and 600 seconds")
    return args


def exit_code(status):
    if status == 200:
        return 0
    # Unknown role or malformed request: a usage error, exactly like the old fixed allowlist.
    return 2 if status in {400, 404} else 1


async def request(socket, method, path, wait):
    deadline = time.monotonic() + wait
    async with aiohttp.ClientSession(
        connector=aiohttp.UnixConnector(path=socket), trust_env=False
    ) as session:
        while True:
            try:
                async with session.request(
                    method, "http://avifors-controller" + path, timeout=aiohttp.ClientTimeout(total=TIMEOUT)
                ) as r:
                    try:
                        body = await r.json(content_type=None)
                    except ValueError:
                        body = None
                    return r.status, body if isinstance(body, dict) else {}
            except aiohttp.ClientConnectorError:
                # Only a connection that was never made is retried; a request the controller
                # accepted is never repeated.
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.5)


async def execute(args) -> int:
    method, path = ROUTES[args.command]
    try:
        status, body = await request(args.socket, method, path.format(role=args.role), args.wait)
    except (aiohttp.ClientError, OSError, TimeoutError) as exc:
        print(f"avifors-workerctl: controller unreachable at {args.socket}: {exc}", file=sys.stderr)
        return 1
    code = exit_code(status)
    if code:
        print(
            f"avifors-workerctl: {args.command} failed ({status}): {body.get('error', '')}", file=sys.stderr
        )
    elif args.command in {"gpu-free", "ping"}:
        print(" ".join(f"{k}={v}" for k, v in body.items() if k != "ok"))
    return code


def main(argv=None) -> int:
    return asyncio.run(execute(parse(argv)))


if __name__ == "__main__":
    sys.exit(main())
