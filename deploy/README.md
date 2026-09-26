# Production deployment

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
