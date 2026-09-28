from __future__ import annotations

import ipaddress
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import yaml


def positive(value, name, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def command(value):
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise ValueError("worker commands must be nonempty argv arrays (no shell)")
    return value


@dataclass
class Model:
    id: str
    upstream: str
    start: list[str]
    stop: list[str]
    paths: list[str] = field(default_factory=lambda: ["/v1/chat/completions", "/v1/completions"])
    kind: str = "openai"
    health_path: str = "/health"
    concurrency: int = 1
    max_pending: int = 16
    idle_timeout: float = 300
    max_hold: float = 300
    max_runtime: float = 300
    queue_timeout: float = 300
    activation_timeout: float = 180
    release_timeout: float = 15
    weight: float = 1
    sleep: bool = False
    verify_stopped: list[str] | None = None
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        u = urlsplit(self.upstream)
        if u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password or u.query:
            raise ValueError("invalid worker upstream")
        self.upstream = self.upstream.rstrip("/")
        command(self.start)
        command(self.stop)
        if self.verify_stopped:
            command(self.verify_stopped)
        if self.kind not in {"openai", "sdapi", "stt"}:
            raise ValueError("kind must be openai, sdapi or stt")
        if not self.id or self.concurrency < 1 or self.max_pending < 1:
            raise ValueError("invalid model ID or queue/concurrency limit")
        for key in (
            "max_hold",
            "max_runtime",
            "queue_timeout",
            "activation_timeout",
            "release_timeout",
            "weight",
        ):
            setattr(self, key, positive(getattr(self, key), key))
        self.idle_timeout = positive(self.idle_timeout, "idle_timeout", zero=True)
        if not self.paths or any(not p.startswith("/v1/") for p in self.paths):
            raise ValueError("only explicit /v1/ inference paths are allowed")


@dataclass
class User:
    name: str
    key: str
    models: list[str]
    max_pending: int = 16


@dataclass
class Config:
    models: dict[str, Model]
    users: list[User] = field(default_factory=list)
    trusted_networks: list = field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 8000
    policy: str = "latency"
    max_pending: int = 64
    trusted_max_pending: int = 32
    admin_key: str = ""
    store: Path = Path("/var/lib/avifors/images")
    public_base: str = "http://localhost:8000/generated"
    artifact_ttl: float = 86400
    max_body: int = 32 * 1024 * 1024
    drain_margin: float = 15
    release_check: list[str] | None = None
    otlp_endpoint: str = ""
    audio: dict = field(default_factory=dict)


def load(path):
    raw = yaml.safe_load(Path(path).read_text())
    defaults = raw.get("defaults", {})
    models = [Model(**(defaults | x)) for x in raw["models"]]
    if len({m.id for m in models}) != len(models):
        raise ValueError("duplicate model IDs")
    settings = dict(raw.get("server", {}))
    settings["store"] = Path(settings.get("store", "/var/lib/avifors/images"))
    settings["trusted_networks"] = [ipaddress.ip_network(n) for n in settings.get("trusted_networks", [])]
    admin_env = settings.pop("admin_key_env", "AVIFORS_ADMIN_KEY")
    settings["admin_key"] = os.environ.get(admin_env, "")
    users = []
    for u in raw.get("users", []):
        key = os.environ.get(u["key_env"], "")
        if len(key) < 16:
            raise ValueError(f"missing/short API key environment for {u['name']}")
        users.append(User(u["name"], key, u.get("models", ["*"]), u.get("max_pending", 16)))
    if len({u.name for u in users}) != len(users) or len({u.key for u in users}) != len(users):
        raise ValueError("duplicate users or keys")
    cfg = Config(models={m.id: m for m in models}, users=users, **settings)
    if not models or not (users or cfg.trusted_networks):
        raise ValueError("configure users or explicit trusted_networks")
    if cfg.policy not in {"latency", "fifo", "fair", "throughput"}:
        raise ValueError("unknown scheduling policy")
    if cfg.max_pending < 1 or cfg.trusted_max_pending < 1 or any(u.max_pending < 1 for u in users):
        raise ValueError("queue limits must be positive")
    cfg.artifact_ttl = positive(cfg.artifact_ttl, "artifact_ttl")
    cfg.drain_margin = positive(cfg.drain_margin, "drain_margin", zero=True)
    if cfg.release_check:
        command(cfg.release_check)
    return cfg
