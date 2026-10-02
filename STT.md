# Durable multilingual transcription

Avifors serves short requests and multi-hour recordings through the same GPU
scheduler used by text and image workers. The MIT broker does not bundle model
weights; the optional speech runtime and Meta models retain their own licenses.

## HTTP API

Use the gateway base URL for every operation. Supply your Avifors API key when
user authentication is configured; a reverse proxy must preserve it. A trusted
network deliberately shares one identity and job namespace. Never treat an
unverified `X-User` header as authentication.

```sh
# Recommended for recordings of any length; returns 202 and a durable job ID.
curl "$BASE/v1/audio/transcriptions/jobs" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Idempotency-Key: recording-unique-id' \
  -F model=aer-stt-v1 -F file=@meeting.m4a -F language=en

curl "$BASE/v1/audio/transcriptions/jobs/$JOB_ID" \
  -H "Authorization: Bearer $API_KEY"
curl "$BASE/v1/audio/transcriptions/jobs/$JOB_ID/result?response_format=text" \
  -H "Authorization: Bearer $API_KEY"
curl -X DELETE "$BASE/v1/audio/transcriptions/jobs/$JOB_ID" \
  -H "Authorization: Bearer $API_KEY"
```

`GET /v1/audio/transcriptions/jobs` lists the caller's latest 100 jobs. Reusing an
idempotency key returns the same job and ignores a replacement upload: use a new
key for different input. The identity-scoped key is retained until job expiry.

`POST /v1/audio/transcriptions` accepts the same multipart fields and waits for
completion. It returns a normal `{"text":"..."}` response for completed jobs.
After the configured synchronous wait it returns **202 with the durable handle**,
not a fake completed transcript. `X-Transcription-Job-ID` and `Location` identify
accepted work; `/jobs` is recommended for SDKs that require HTTP 200 from the
synchronous API. Disconnecting does not cancel a committed job.

Fields: `model`, `file`, optional `language`, optional `response_format` (json,
text, verbose_json, srt, vtt). Unknown fields fail explicitly. Translation,
diarization, vocabulary prompts, word alignment and live microphone streaming
are not implemented. Results retain the original spoken language.

Language can be an unambiguous ISO 639 code such as `en` or `hi`, or an explicit
Meta language/script identifier such as `eng_Latn` or `hin_Deva`. For ambiguous
scripts, supply the explicit identifier. Omit language for unconditioned
multilingual recognition; specifying the known language generally helps accuracy.

Progress reports original audio duration, processed seconds, completed/total
chunks, state, errors, expiry and a result URL. SRT/VTT and verbose JSON contain
**chunk boundaries, not forced word/sentence alignment**. Boundary overlap can
make adjacent cues overlap; these exports are draft subtitles.

## Quality and long recordings

The default deployment uses `omniASR_LLM_3B_v2` in BF16 with beam size five and
batch size one. It is selected for broad language coverage and the
available GPU memory, not claimed to beat every model on every language. The
upstream 7B quality claims must not be attributed to the 3B checkpoint.

CPU decoding produces 16 kHz mono PCM on disk. A bounded-window scan chooses low
energy boundaries between 20 and 30 seconds. Forced cuts retain 0.8 seconds of
overlap; exact matching suffix/prefix text is deduplicated without LLM rewriting.
With the optional CPU Silero ONNX model configured, detected speech is padded
by 250 ms and non-speech is excluded before GPU admission. Thresholds are
configurable for quiet recordings. Without VAD only digital silence is skipped.
Audio context is local to each chunk, so proper-name consistency and uninterrupted speech at
forced boundaries should be evaluated for the target recordings.

Every chunk is an ordinary scheduled GPU job. Audio never bypasses model/user
queue limits or extends an active lease. A minimum remaining budget avoids
starting a chunk at the very end of a lease. Results and the next chunk frontier
commit in one SQLite transaction (WAL, synchronous FULL). After restart only the
uncommitted chunk may rerun; completed chunks are never appended twice.

Each job has at most one queued/active chunk. Latency mode yields at chunk
boundaries when another model waits; FIFO, fair and throughput retain their
existing meanings. The five-minute GPU deadline applies to each ownership
period, including model loading, not to recording duration. When necessary the
worker unloads and the job continues in a later ownership period. Other models'
cold-start costs still affect wall-clock completion time.

Admission and queue expiry retry within the overall job deadline. Inference
failures have a bounded per-chunk retry count. Inference is deterministic in
intent but recovery is **at least once execution, exactly once checkpointing**.
Explicit cancellation removes queued work and drains a running chunk under its
original deadline, without resetting another user's request.

## Limits and retention

Configure under `server.audio` (all numbers are positive):

| Setting | Default |
| --- | ---: |
| max_upload_bytes | 2147483648 (2 GiB) |
| max_duration_seconds | 28800 (8 hours) |
| max_storage_bytes | 21474836480 (20 GiB) |
| min_free_bytes | 2147483648 |
| max_jobs / max_jobs_per_user | 16 / 4 |
| upload_timeout / decode_timeout | 3600 / 3600 seconds |
| job_timeout | 604800 seconds (7 days) |
| retention_seconds | 172800 seconds after terminal state |
| sync_timeout | 600 seconds |
| chunk_seconds / min_chunk_seconds | 30 / 20 |
| overlap_seconds | 0.8 |
| min_execution_budget | 60 seconds |
| chunk_retries | 3 |
| vad_threshold / vad_padding | 0.5 / 0.25 seconds |

`vad_model` optionally names a local Silero ONNX model; CPUExecutionProvider is
explicitly selected so preparation never competes for GPU ownership.

`store` defaults to an `audio` sibling of the image store. Admission reserves
worst-case upload plus decoded PCM space before reading a body. After preparation
this shrinks to actual PCM size. Disk capacity can admit fewer jobs than the job
count limit. Raise duration, upload, storage and proxy limits together when needed.
Eight hours of compressed speech fits the default upload limit; large raw PCM
recordings may require a higher limit.

Source audio is deleted after successful preparation. PCM is deleted after
completion or cancellation. Failed jobs retain audio until expiry so that
`POST /v1/audio/transcriptions/jobs/{id}/retry` can resume the committed frontier.
Explicit cancellation also deletes failed-job audio. Transcripts and job metadata expire
after the configured retention; transcripts are never included in routine logs
or traces. Use encrypted storage if required by your deployment. Back up SQLite
with its backup API or a consistent volume snapshot, not by copying only the
main database while WAL writes are active. LXC backups cover the store on Atlas.

## Worker installation

Production runs both speech workers as containers ([Docker deployment](deploy/README.md#docker-deployment)).
The broker image already contains FFmpeg and `avifors[audio]` for CPU VAD.

| Role | Image | Python | Runtime pins | Port |
|---|---|---|---|---|
| `stt` (Omnilingual) | `avifors-stt-omni` | **3.12 (exception)** | `deploy/stt/requirements.lock`: torch 2.8.0 CUDA 12.8, fairseq2/fairseq2n 0.6 | 18002 |
| `stt-qwen` (Qwen3-ASR) | `avifors-stt-qwen` | 3.14 | `deploy/docker/requirements/stt-qwen.txt`: torch/torchaudio 2.11.0 CUDA 12.8, qwen-asr 0.0.6, transformers 4.57.6 | 18003 |

**Python 3.12 is a time-limited exception for `avifors-stt-omni` only.** omnilingual-asr declares
`requires-python <=3.12` and fairseq2n publishes wheels only up to CPython 3.12, so the
Omnilingual engine cannot run on Python 3.14 yet. Every other Avifors image uses Python 3.14.
Remove the exception (rebase the image on `python:3.14-slim` and refresh the lock) as soon as
fairseq2/fairseq2n and omnilingual-asr ship Python 3.14 support. Qwen3-ASR has no such
constraint; its image moved to the first torch release with both CUDA 12.8 and Python 3.14
wheels for torch and torchaudio.

Both images run as the `avifors` user (996:990) with the NVIDIA runtime, listen on 0.0.0.0
inside the worker network and are published on 127.0.0.1 only. The Omnilingual container mounts
`/etc/avifors/stt-assets` and `/var/lib/avifors/stt-models` read-only and
`/var/lib/avifors/fairseq2` read-write; the Qwen container mounts the models read-only and runs
with Hugging Face offline.

For the legacy systemd units, install the speech runtime in a **separate venv** using
`deploy/stt/requirements.lock` (or the pinned input manifest when developing).
Ubuntu build dependencies include Python development headers, CMake, a C++
compiler, libsndfile, zlib, libbz2, liblzma and Eigen headers.

Create the writable stores before restarting the broker:

```sh
install -d -m 700 -o avifors -g avifors /var/lib/avifors/audio
install -d -m 750 -o avifors -g avifors /var/lib/avifors/fairseq2
```

The supplied runtime lock targets Linux x86-64, Python 3.12 and CUDA 12.8.

On Docker, the `stt` and `stt-qwen` roles are already in `deploy/docker/workers.yaml` and
compose; the model configuration from `examples/stt.yaml` uses
`[avifors-workerctl, start|stop|verify, stt]` commands instead of sudo. With the legacy units,
install `deploy/stt/avifors-stt.service`, extend the root-owned fixed worker helper
and sudo allowlist with `start/stop/verify stt`, and add the model configuration
from `examples/stt.yaml`. Workers must have no boot/restart policy; only the
broker starts at boot. Pre-stage the checkpoint/tokenizer before enabling the
route so network downloads cannot consume a GPU lease. Install the supplied
`local-assets.yaml` in `/etc/avifors/stt-assets` and put the named checkpoint and
tokenizer in `/var/lib/avifors/stt-models`. Keep model revision,
runtime package lock and checksum with deployment records.

Route `/v1/audio/transcriptions` and its `/jobs` subpaths through the existing
proxy to Avifors, with HTTP/1.1 and request buffering disabled. Give this location
an upload limit slightly larger than the file limit for multipart overhead.
Preserve Authorization, Idempotency-Key and trace headers. Register the model
and `audio_transcription` capability in the proxy catalog; health checks target
the broker catalog and never activate the GPU.

## Acceptance gates

Unit/integration tests cover ownership, upload limits, malformed audio,
idempotency, result formats, cancellation isolation, fresh-lease admission,
checkpoint recovery and complete two-hour chunk coverage. Deployment acceptance
must additionally test real model output, GPU memory/load time, mixed workload
handoffs, multi-hour media through the proxy, job recovery after restart and
existing text/image regressions. A synthetic long-file test demonstrates duration
and recovery, not an accuracy benchmark on hours of natural conversation.

## Engine selection

`aer-stt-v1` is the stable broad multilingual entry point. The example deployment
uses Meta Omnilingual 3B v2 for unknown languages and broad coverage, and routes
explicit English hints (`en`, `eng`, `eng_Latn`) to Qwen3-ASR-1.7B for punctuation
and casing. `aer-stt-qwen3` also exposes Qwen directly for its supported languages.
Both run BF16, one GPU owner at a time, with separate on-demand worker processes.

`parameters.language_models` is an administrator-controlled mapping from exact
language hints to configured STT model IDs. It is optional. The resolved engine
is saved at submission and reported as `engine_model`; changing configuration
does not change the engine midway through an existing recording. A user allowed
to use the public alias may use its configured engines through that alias, without
being granted direct access to every engine model ID.

The six-sample deployment smoke set is not a general accuracy benchmark. Both
engines matched the three English references after case/punctuation and “Mr.”
normalization. Hindi results were mixed, so the broad default was retained rather
than claiming a universal quality winner. Proper nouns and difficult recordings
still need review.

Pinned upstream artifacts:

- Meta model/runtime: [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr),
  runtime revision `81f51e224ce9e74b02cc2a3eaf21b2d91d743455` (Apache-2.0).
- Qwen: [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B),
  model revision `7278e1e70fe206f11671096ffdd38061171dd6e5` (Apache-2.0),
  `qwen-asr==0.0.6`, Transformers 4.57.6.
- CPU VAD: [Silero](https://github.com/snakers4/silero-vad),
  `src/silero_vad/data/silero_vad.onnx` at revision
  `5cd7945676eb32225748052e2e6a0580e4686a08` (MIT).

Download model files separately from their upstreams, keep their licenses, and
verify the checksums in `deploy/stt/SHA256SUMS`. Qwen's snapshot belongs at
`/var/lib/avifors/stt-models/qwen3-asr-1.7b`. Install both STT service units and
allowlisted helpers if using the example language routing. Neither unit starts
independently at boot. The text vLLM environment remains separate from all of
these dependencies.

For proxy administration, `server.audio.admission_catalog_url` can point to a
trusted catalog with a `models` array of `{id, status}` records. Only `ready` or
`degraded` model IDs admit work; both the public alias and its resolved engine
must be enabled. Avifors refreshes this control plane every five
seconds: disabled/missing models reject new uploads and pause existing jobs at
chunk boundaries, preserving their checkpoints until re-enabled. Catalog outages
also pause jobs; the overall job deadline still applies. The deployment connects
this to the proxy's existing model enable/disable and quarantine controls.
