# Decision models (System One)

Avifors serves Jev-style decision models through `POST /v1/systemone`, compatible with
[Ollama's System One API](https://docs.ollama.com/api/systemone) as served for
[Nimble](https://ollama.com/library/nimble) (Ollama 0.35+). A client written for
`ollama` + `nimble` works against Avifors by changing the base URL, the model name and adding the
bearer key. The bundled worker runs [Laya](https://github.com/NandhaKishorM/laya) checkpoints
(ModernBERT decision models), including fine-tuned ones.

```sh
curl http://127.0.0.1:8000/v1/systemone -H "Authorization: Bearer $KEY" -d '{
  "model": "aer-laya",
  "state": "Our checkout has returned 500 errors since 9am.",
  "questions": {
    "label": {"type": "choice", "instructions": "Which label fits this ticket?",
              "criteria": {"billing": null, "bug": null, "account": null}}
  }
}'
```

```json
{"model": "aer-laya",
 "answers": {"label": {"type": "choice", "choice": "bug",
                       "probabilities": {"billing": 0.1657, "bug": 0.6718, "account": 0.1625},
                       "confidence": 0.2169}},
 "usage": {"input_tokens": 29, "output_tokens": 1}}
```

## Contract

| | |
|---|---|
| Request | `model`, `state` (nonempty string, JSON object or array), `questions` (1–64 named), optional `keep_alive` |
| `choice` | `criteria`: 2–26 options mapped to a description or `null` → `choice`, `probabilities` (request order), `confidence` |
| `noul` | optional `criteria` describing `"false"`/`"true"` → `noul` = P(true) |
| `score` | `criteria`: 2–26 ordered level descriptions → `score` = Σ i·P(i) (0…N−1, unrounded), `legend`, `probabilities`, `confidence` |
| `confidence` | `1 − H(p)/ln N`: concentration of the distribution, **not** the chance the answer is right |
| `usage` | `input_tokens`: rendered prompt tokens summed over questions; `output_tokens`: one scored decision per question (the encoder generates no text) |
| Limits | 64 KiB body (413), 1–64 questions, 2–26 options/levels, no truncation: a state that does not fit the model context is a 400 |
| Errors | `{"error": "message"}` with 400 / 404 / 413 / 500, and Avifors admission errors 401 / 429 / 503 / 504 in the same shape |

Ties go to the first option in request order. Requests are validated by the broker before any worker
is woken, so malformed input never loads a model.

### Where Avifors follows Ollama's server rather than its documentation

Diffed with `scripts/systemone_conformance.py` against Ollama 0.35.0 serving `nimble:9b-q4_K_M`:

- generation-style and unknown fields (`stream`, `options`, `images`, `format`, `tools`, `think`, …)
  are ignored and the request is answered normally (the docs say they are unsupported);
- numbers are returned unrounded (the docs' examples show four decimals);
- `instructions` may be a nonempty string, object or array;
- validation messages use Ollama's wording (`questions must contain 1–64 fields`,
  `question "x": criteria must contain 2–26 candidates`, …).

Deliberate differences: the unknown-model message is `model "X" not found` (there is no `pull`); a
model outside the caller's allowlist is reported as not found (404), not forbidden; `keep_alive` is
validated like Ollama's but residency follows the model's `idle_timeout`, so one client cannot pin a
worker for everyone. Probabilities come from a different model than Nimble, so answers differ.

## Lanes and keeping models together

Models in one `lane` share one residency domain. The default lane `gpu` is the managed GPU.

Without a capacity, a lane keeps today's exclusive single-owner scheduler (one model resident,
time-sliced with leases). That is unchanged GPU behaviour.

With a capacity, the lane runs in pool mode, configured like this:

```yaml
server:
  lanes:
    cpu: {capacity_mib: 7000}
models:
  - id: aer-laya
    lane: cpu
    memory_mib: 2500
```

In pool mode, every model that fits stays loaded together. Each pooled model declares `memory_mib`.
- A model that does not fit evicts idle residents, least recently used first.
- A busy resident is never cut off mid-job. Once it has held its lease (`max_hold`) while another
  model waits for room, it stops taking new work, drains, and is released.
- Models load concurrently when they fit.
- `idle_timeout` still releases unused models.
- A pooled `gpu` lane cannot use `release_check`, because that check asserts an empty device.


A decision model with `lane: cpu` uses a separate scheduler, so scoring a ticket
does not evict a resident text, image or speech model in the GPU lane. Each lane
has its own queue limits; only an exclusive GPU lane runs `release_check`.
`/admin/state` reports other lanes under `lanes`,
`/metrics` exports `avifors_lane_{active,queued,up}`. A decision model may also stay in the `gpu`
lane when it should be time-sliced on the GPU like any other worker.

## Verified

On 2026-10-01, `scripts/systemone_conformance.py` ran 19 cases against Avifors (this branch, CPU lane,
Laya multitask checkpoint, torch fp32) and against Ollama 0.35.0 serving `nimble:9b-q4_K_M`:

- **19/19 cases conform.** Every case had the same HTTP status and the same response shape as Nimble.
- The cases were:
  - the announcement example;
  - mixed types with an object state;
  - an array state;
  - 26 options;
  - 64 questions;
  - `keep_alive`;
  - ignored fields;
  - object instructions;
  - eight 400 cases;
  - an unknown model (404);
  - a body over 64 KiB (413).
- Error messages were identical apart from the deliberate unknown-model wording.
- Warm latency on 6 CPU threads (i7-12700) was 0.1–0.2 s for one question, 0.9 s for four mixed
  questions and 9 s for 64. These were measured while other CPU jobs were running.

## Worker

```sh
python -m avifors.decision_worker --checkpoint DIR --device cpu --threads 6 --port 18004
python -m avifors.decision_worker --checkpoint DIR --onnx DIR/onnx/laya.int8.onnx --threads 6
```

Torch on CPU or CUDA, or ONNX Runtime on CPU (Laya's `scripts/export_onnx.py`). **Do not use the INT8
export for fine-tuned checkpoints:** it agreed with the torch model on only 78–79% of answers
(probabilities moved by 0.22 on average) on our Hedwig and Pensieve tasks. Use torch fp32 on CPU.
One request is scored at a time; logs carry sizes and timings only, never states or questions.
In production each checkpoint runs in its own container from the `avifors-decision` image
(`deploy/docker/avifors-decision.Dockerfile`: Python 3.14, CPU torch from the PyTorch CPU index,
hash-pinned in `deploy/docker/requirements/decision.txt`). The compose services run as the
`avifors` user (996:990) without the NVIDIA runtime and with `CUDA_VISIBLE_DEVICES=`, 3 threads,
offline Hugging Face, a read-only root and the checkpoint mounted read-only, and a 4 GB memory cap
without swap:

| Model | Role / compose service | Container | Port | Checkpoint |
|---|---|---|---|---|
| `aer-laya` (Hedwig + Pensieve multitask) | `decision` | `avifors-w-decision` | 18004 | `/var/lib/avifors/decision/aer-laya` |
| `aer-laya-hedwig` (Hedwig only) | `decision-hedwig` | `avifors-w-decision-hedwig` | 18005 | `/var/lib/avifors/decision/aer-laya-hedwig` |
| `aer-laya-guard` (mail threats) | `decision-guard` | `avifors-w-decision-guard` | 18006 | `/var/lib/avifors/decision/aer-laya-guard` |

The legacy units in `deploy/decision/` (venv from `pip install '.[decision]'`, same settings) are
kept for the rollback window only.

All three sit in the `cpu` pool at `memory_mib: 3000` each, within the
example pool capacity of `10000` MiB. This is declared residency
accounting, separate from each container's enforced 4 GiB memory limit. Measure
actual memory use and leave host headroom when sizing a pool.

## Deployment checklist

For a new decision model on the Docker deployment ([deploy/README.md](deploy/README.md#docker-deployment)):

1. Copy the checkpoint to `/var/lib/avifors/decision/<id>` (no network fetch at runtime).
2. Add a compose service next to `decision` in `deploy/docker/compose.yaml` (merge `*decision`, a new
   `container_name`, the `org.team-aer.avifors.role` label, port and checkpoint mount) and the role
   to `/etc/avifors/workers.yaml` (root-owned, 0644), then restart the controller. No sudoers change:
   the broker never gains privilege. A role missing from the allowlist makes `avifors-workerctl` exit 2
   and the broker exit during startup recovery.
3. Add the `kind: decision`, `lane: cpu` model with `[avifors-workerctl, start|stop|verify, <role>]`
   commands to `/etc/avifors/config.yaml` (keep root:root mode 644), raise the pool's `capacity_mib`
   if the new model does not fit beside the residents, and run `avifors --config /etc/avifors/config.yaml --check` in the broker image.
4. In a maintenance window (`/admin/state` idle), stop the broker, `docker compose --profile workers
   create <service>`, start the broker; verify with `scripts/systemone_conformance.py` and a real
   request through llm-proxy.
5. llm-proxy (LiteLLM) has no System One route: expose `/v1/systemone` as a pass-through endpoint so
   apps keep calling the proxy, not Avifors directly.
