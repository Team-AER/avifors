# Production deployment

The supported deployment is Docker ([below](#docker-deployment)). The systemd units, the sudo
worker helper and the venv instructions that follow describe the previous deployment; they stay
in the repository, unchanged, until the Docker rollback window has closed.

## Legacy systemd deployment

Install the broker into `/opt/avifors/.venv`, create an `avifors` service user,
and create `/var/lib/avifors/images` owned by that user. Keep the application,
configuration and worker-control helper root-owned. Copy `examples/config.yaml`
to `/etc/avifors/config.yaml`; change its store to `/var/lib/avifors/images`.
Store the referenced API keys in `/etc/avifors/secrets.env` with mode `0600`.
The supplied systemd unit reads that file before changing user.

Install your inference engines independently and pin their versions and weights.
Create `avifors-text.service` and `avifors-image.service` with `Restart=no` and no
boot enablement. Bind their HTTP ports to loopback. For a Docker text worker,
use `docker start -a` as `ExecStart`, `docker stop -t 5` as `ExecStop`, and Docker
`--restart=no`. Set a finite systemd stop timeout and verify that stopping the
unit actually terminates its GPU processes. Preserve your existing model flags.

Install `avifors-worker` root-owned at `/usr/local/sbin/avifors-worker`, mode
`0755`. An example narrowly scoped sudoers entry is:

```sudoers
avifors ALL=(root) NOPASSWD: /usr/local/sbin/avifors-worker start text, /usr/local/sbin/avifors-worker stop text, /usr/local/sbin/avifors-worker verify text, /usr/local/sbin/avifors-worker start image, /usr/local/sbin/avifors-worker stop image, /usr/local/sbin/avifors-worker verify image
```

Validate it with `visudo -cf`. Adapt the fixed helper allowlist if your units have
different names. Do not grant the broker unrestricted Docker or systemctl access.
Configure `server.release_check` to a root-owned command that confirms enough GPU
memory is free for the next worker. Sleep can leave a small CUDA context, so set
the threshold from measurements with your engines, rather than assuming zero use.

For vLLM sleep, add `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`, configure
`sleep: true`, and reserve host RAM for offloaded weights as well as the image
worker. Persist Hugging Face and vLLM compilation caches. A complete local model
cache can use `HF_HUB_OFFLINE=1` to avoid network lookups during activation.
Validate both a fresh process start and wake from sleep against the ownership
deadline; first-time kernel compilation may need deployment-time preparation.

On an NVIDIA LXC, use the host driver and matching guest userspace libraries,
map only the required NVIDIA devices, and validate GPU access inside the worker.
Avifors controls its configured workers; it cannot arbitrate unrelated GPU users.
Disable the old workers' guest and engine autostart before enabling the new guest.

Copy `avifors.service` into `/etc/systemd/system`, reload systemd, validate the
configuration with the required key environment, then start the broker. Verify
health, text, streaming, tools, images, mixed-model queueing, idle release,
cancellation and reboot recovery before enabling it for production traffic.
Enable only `avifors.service` and the new guest at boot.

Put TLS and your normal gateway in front of the broker. Use separate user keys
for per-user quotas and artifact isolation; a trusted-network gateway deliberately
shares one identity and quota. Health checks must use `/v1/models` or a scoped
`/backend/NAME/v1/models`, never synthetic inference. Keep the admin key separate.
Optional `server.otlp_endpoint` accepts an OTLP/HTTP `/v1/traces` URL; `/metrics`
reports queue and lifecycle counters without activating a model.

Configuration changes take effect on restart, which drains by cancelling work and
stopping workers. Schedule such changes during a maintenance window. Requests are
not persisted or retried automatically. Keep previous workers and route backups
until acceptance; rollback must stop Avifors before restoring their GPU access.

## Docker deployment

Everything runs in containers from `deploy/docker/compose.yaml` (compose project `avifors`), so
OS upgrades no longer touch the Python runtimes. All images use Python 3.14, except the
Omnilingual speech worker (see [the exception](#python-312-exception-avifors-stt-omni)).

| Service | Image | Runs as | Notes |
|---|---|---|---|
| `controller` | `avifors` | root, `network_mode: none` | The only container with `/var/run/docker.sock`; serves `/run/avifors/control.sock` (root:990, 0660) |
| `broker` | `avifors` | 996:990, `network_mode: host` | Read-only root, no capabilities, starts after the controller is healthy |
| `text` | `vllm/vllm-openai:v0.26.0` (vendor, pin by digest) | image default | `127.0.0.1:18000`, `ipc: host`, `/etc/avifors/vllm.env` |
| `image` | `avifors-image` (sd.cpp `36746936`, CUDA SM 89) | 996:990 | `127.0.0.1:1234`, 28 GiB memory, 512 pids |
| `stt` | `avifors-stt-omni` (Python 3.12) | 996:990 | `127.0.0.1:18002` |
| `stt-qwen` | `avifors-stt-qwen` | 996:990 | `127.0.0.1:18003` |
| `decision`, `decision-hedwig`, `decision-guard` | `avifors-decision` (CPU torch) | 996:990 | `127.0.0.1:18004`–`18006`, 4 GiB memory and no swap, no GPU |

**Worker control.** The broker still runs administrator-owned argv commands with the old exit
codes, so its code and the lifecycle are unchanged; only the commands in `config.yaml` change from
`[sudo, /usr/local/sbin/avifors-worker, ACTION, ROLE]` to `[avifors-workerctl, ACTION, ROLE]`, and
`release_check` from `[/usr/local/bin/avifors-gpu-free]` to `[avifors-workerctl, gpu-free]`.
`avifors-workerctl` asks `avifors-controller` over the Unix socket. The controller acts only on
the container that the root-owned allowlist `/etc/avifors/workers.yaml` names for a role, and only
if Docker reports it with the labels `com.docker.compose.project=avifors` and
`org.team-aer.avifors.role=ROLE`; it talks to the Docker Engine API (pinned to v1.44, supported by
Docker 26 through 29). `start` is `POST /containers/ID/start`, `stop` is `POST .../stop?t=GRACE`
followed by `POST .../wait?condition=not-running`, `verify` succeeds only when the container is not
running, restarting, paused or being removed, and `gpu-free` runs `nvidia-smi` and requires 12000
MiB free on every GPU. A missing container, wrong labels or a Docker error fails closed (exit 1);
an unknown role is a usage error (exit 2), as with the old helper. The broker cannot name a
container, image, mount or command, so it gains no Docker access; it needs no sudo.

**Workers are created, never started, by compose.** `docker compose --profile workers create`
creates the seven `avifors-w-*` containers with `restart: "no"`; the broker starts and stops them
on demand through the controller. Never `up` the workers profile or `docker start` a worker by
hand. Workers listen on 0.0.0.0 inside the `avifors_workers` bridge network and are published on
the host's 127.0.0.1 only, so the broker's upstream URLs are unchanged.

**The broker must use host networking.** That keeps :8000 subject to the host's iptables INPUT
chain (`AVIFORS-IN` in production). A published port would be DNATed through FORWARD and
bypass it. Set `server.host: 0.0.0.0` as today.

Logs go to journald: `journalctl CONTAINER_NAME=avifors-broker-1 -f`, or
`journalctl -t avifors-w-text` for a worker (the tag is the container name).

### Files on the host

| Path | Owner, mode | Purpose |
|---|---|---|
| `/etc/avifors/config.yaml` | root:root 0644 | Broker config with `avifors-workerctl` commands (mounted read-only) |
| `/etc/avifors/secrets.env` | root:root 0600 | `AVIFORS_ADMIN_KEY` and user keys (compose `env_file`) |
| `/etc/avifors/workers.yaml` | root:root 0644 | Controller allowlist, from `deploy/docker/workers.yaml` (controller only) |
| `/etc/avifors/vllm.env` | as today | Text worker environment |
| `/var/lib/avifors/{images,audio}` | avifors | Broker stores (read-write) |
| `/var/lib/avifors/{image-models,stt-models,decision}` | avifors | Weights and checkpoints (read-only) |
| `/var/lib/avifors/fairseq2`, `hf-cache` | avifors | Omnilingual cache (read-write), vLLM Hugging Face cache |
| `/var/lib/avifors/.cache/huggingface` | avifors | Hugging Face home of the old workers, mounted read-only into the Qwen and decision workers (must exist) |
| `deploy/docker/.env` | root 0600 | Compose variables from `.env.example`: vLLM digest, text arguments, volume names (no secrets) |

The broker's `/health` needs an identity. To make the container healthcheck detect scheduler
faults, add `AVIFORS_HEALTHCHECK_KEY=<an existing user key>` to `secrets.env`; without it a 401
still proves the broker is serving.

### Build

Build on an x86-64 Docker host from a checkout or `git archive` export of this repository (for
example `/opt/avifors-src`). The image worker compiles CUDA kernels and needs the ~8 GB CUDA
devel image; images can also be built elsewhere and moved with `docker save | docker load`.
Compose v2.24 or newer is required (`env_file` with `required`).

```sh
cd /opt/avifors-src/deploy/docker
install -m 0600 .env.example .env    # then fill AVIFORS_TEXT_ARGS and the volume names (below)
docker compose build                     # avifors (broker + controller)
docker compose --profile workers build   # avifors-image, -stt-omni, -stt-qwen, -decision
docker compose --profile workers pull text
```

Fill `.env` from the running text container: its image digest
(`docker image inspect vllm/vllm-openai:v0.26.0 --format '{{index .RepoDigests 0}}'`), its
arguments (`docker inspect avifors-text --format '{{json .Config.Cmd}}'`, with the image's default
entrypoint) and its named volumes and mount targets (`docker inspect avifors-text --format
'{{json .Mounts}}'`). The compose `text` worker reuses those volumes, so caches carry over.

### Cutover

Preparation needs no downtime. Run every command in `deploy/docker` of the checkout:

```sh
install -m 0644 -o root -g root workers.yaml /etc/avifors/workers.yaml
cp -p /etc/avifors/config.yaml /etc/avifors/config.systemd.yaml       # rollback copy, mode kept
python3 - <<'EOF'
import re
from pathlib import Path
text = Path("/etc/avifors/config.systemd.yaml").read_text()
# Flow ([sudo, ...]) and block (- sudo) lists both appear in configs; handle both.
text = re.sub(r"\[\s*sudo\s*,\s*/usr/local/sbin/avifors-worker\s*,", "[avifors-workerctl,", text)
text = re.sub(r"(\n(\s*)- )sudo\n\s*- /usr/local/sbin/avifors-worker\n", r"\1avifors-workerctl\n", text)
text = re.sub(r"\[\s*/usr/local/bin/avifors-gpu-free\s*\]", "[avifors-workerctl, gpu-free]", text)
text = re.sub(r"(release_check:)[ \t]*\n[ \t]*- /usr/local/bin/avifors-gpu-free[ \t]*\n",
              r"\1 [avifors-workerctl, gpu-free]\n", text)
Path("/etc/avifors/config.docker.yaml").write_text(text)
EOF
chown root:root /etc/avifors/config.docker.yaml && chmod 0644 /etc/avifors/config.docker.yaml
grep -nE 'sudo|avifors-worker([^c]|$)|avifors-gpu-free' /etc/avifors/config.docker.yaml   # must print nothing
diff /etc/avifors/config.systemd.yaml /etc/avifors/config.docker.yaml                      # only commands differ
[ -d /var/lib/avifors/.cache/huggingface ] || install -d -m 0750 -o avifors -g avifors /var/lib/avifors/.cache/huggingface
docker compose run --rm --no-deps -v /etc/avifors/config.docker.yaml:/etc/avifors/config.docker.yaml:ro \
  broker avifors --config /etc/avifors/config.docker.yaml --check
docker compose --profile workers create      # creates (does not start) controller, broker, workers
```

In a maintenance window, with `/admin/state` idle (a broker outage of more than about a minute
makes llm-proxy quarantine every Avifors model until two healthy probes pass):

```sh
systemctl disable --now avifors              # its shutdown stops and verifies the old workers
systemctl is-active avifors-text avifors-image avifors-stt avifors-stt-qwen \
  avifors-decision avifors-decision-hedwig avifors-decision-guard     # all inactive
docker ps --filter name=avifors-text                                  # old container stopped
install -m 0644 -o root -g root /etc/avifors/config.docker.yaml /etc/avifors/config.yaml
docker compose up -d broker                  # controller first (healthy), then the broker
docker compose ps
docker compose exec broker avifors-workerctl gpu-free
```

The broker's startup stops and verifies every worker and runs the GPU check, so a wrong
allowlist, label or socket shows up immediately as a broker exit (it is restarted by Docker).
Then verify as for any broker change: `/health`, `/admin/state`, a real request for every model
through llm-proxy (text and streaming, image, speech with and without `language=en`, each
decision model), a GPU handoff between text, image and speech, idle release, the llm-proxy
consumer-contract `--live` check, and a reboot of the host (the controller and broker restart
by policy; workers stay stopped until requested). Check that the memory limits are enforced
(`docker info` shows no "No memory limit support" warning, and
`docker inspect -f '{{.HostConfig.Memory}}' avifors-w-decision` is 4294967296).

Do not delete the old `avifors-text` container, the systemd units, `/usr/local/sbin/avifors-worker`,
the sudoers entry or `/opt/avifors*` venvs until the rollback window has closed.

### Rollback

```sh
docker compose stop broker                   # its shutdown stops every worker through the controller
docker compose --profile workers stop        # belt and braces: no Docker worker may hold the GPU
install -m 0644 -o root -g root /etc/avifors/config.systemd.yaml /etc/avifors/config.yaml
systemctl enable --now avifors
```

`docker compose stop` keeps the controller and broker stopped across reboots (`unless-stopped`).

### Upgrades

Restarting the broker drains work, so treat every change as a maintenance-window change. Broker
or controller: rebuild, then `docker compose up -d broker` (or `controller`). A worker image or
worker settings: stop the broker first, then `docker compose --profile workers create
--force-recreate SERVICE`, then start the broker; recreating a worker the broker is using would
cut off its requests. A new worker role needs a compose service with the role label, an entry in
`/etc/avifors/workers.yaml` (restart the controller) and `avifors-workerctl` commands in the
config; no sudoers change.

### Python 3.12 exception (avifors-stt-omni)

All Avifors images run Python 3.14 except `avifors-stt-omni`. omnilingual-asr requires Python
3.12 or older and fairseq2n publishes wheels only up to CPython 3.12, so that one image is a
**time-limited exception** on `python:3.12-slim` with production's `deploy/stt/requirements.lock`
(torch 2.8.0 CUDA 12.8, fairseq2 0.6). Move it to Python 3.14 and remove this exception as soon as
fairseq2/fairseq2n and omnilingual-asr support 3.14.
