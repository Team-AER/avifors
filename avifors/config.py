from __future__ import annotations

import ipaddress
import math
import os
import re
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
    # Models in one lane share one exclusive residency slot. "gpu" is the single-GPU slot; a model
    # in another lane (e.g. a CPU decision worker) gets its own scheduler and never evicts the GPU.
    lane: str = "gpu"
    # Resident memory, used only by lanes with a capacity (pool mode) to decide what fits together.
    memory_mib: float = 0

    def __post_init__(self):
        u = urlsplit(self.upstream)
        if u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password or u.query:
            raise ValueError("invalid worker upstream")
        self.upstream = self.upstream.rstrip("/")
        command(self.start)
        command(self.stop)
        if self.verify_stopped:
            command(self.verify_stopped)
        if self.kind not in {"openai", "sdapi", "stt", "decision"}:
            raise ValueError("kind must be openai, sdapi, stt or decision")
        if not re.fullmatch(r"[a-z0-9_-]{1,32}", self.lane):
            raise ValueError("lane must be a short lowercase name")
        if self.kind == "decision":
            if self.paths == ["/v1/chat/completions", "/v1/completions"]:
                self.paths = ["/v1/systemone"]
            if self.paths != ["/v1/systemone"]:
                raise ValueError("decision models serve only /v1/systemone")
        if self.kind == "sdapi":
            from .image_profiles import validate_profile

            validate_profile(self.parameters)
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
        self.memory_mib = positive(self.memory_mib, "memory_mib", zero=True)
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
    # {lane: {"capacity_mib": N}}: lanes with a capacity keep every model that fits resident together.
    lanes: dict = field(default_factory=dict)


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
    validate_lanes(cfg)
    return cfg


def validate_lanes(cfg):
    used = {m.lane for m in cfg.models.values()}
    for lane, spec in cfg.lanes.items():
        if lane not in used or not isinstance(spec, dict) or set(spec) - {"capacity_mib"}:
            raise ValueError(f"lane {lane!r}: unknown lane or settings (only capacity_mib)")
        capacity = positive(spec.get("capacity_mib", 0), f"lanes.{lane}.capacity_mib")
        for m in cfg.models.values():
            if m.lane == lane and not 0 < m.memory_mib <= capacity:
                raise ValueError(f"model {m.id!r}: memory_mib must be positive and fit lane {lane!r} capacity")
        if lane == "gpu" and cfg.release_check:
            raise ValueError("a pooled gpu lane cannot use release_check (it asserts an empty device)")
