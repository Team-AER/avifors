# FLUX.2 Klein image worker

Avifors can run the distilled FLUX.2 Klein 9B FP8 model through
stable-diffusion.cpp. It uses the same exclusive GPU scheduler, per-user quotas,
bounded lease, image endpoint and persistent artifact ownership as other workers.
Keep your existing public model ID when replacing an older image model.

Use `deploy/flux2/model.yaml` and `deploy/flux2/avifors-image.service` as examples.
Install only the broker at boot; the image unit must remain static. The fixed
`image` helper entry already controls this unit. Preserve your other model entries
and timeouts. Back up the old unit, broker package and configuration before rollout.

## Weights and engine

The example uses stable-diffusion.cpp revision
`36746936c054d889e9b3c0e6ee490a9e362a0808` with CUDA. Download these exact files
into the component directories shown in the unit; do this before serving traffic:

| Component | Repository | Revision | File |
| --- | --- | --- | --- |
| Diffusion | black-forest-labs/FLUX.2-klein-9b-fp8 | 902d9d510b51533e07729f19211414a3648b77d2 | flux-2-klein-9b-fp8.safetensors |
| Text encoder | unsloth/Qwen3-8B-GGUF | a6adef130ffb23ddaf1a62fec9dced968c9bc482 | Qwen3-8B-Q8_0.gguf |
| Shared FLUX.2 VAE | black-forest-labs/FLUX.2-klein-4B | e7b7dc27f91deacad38e78976d1f2b499d76a294 | vae/diffusion_pytorch_model.safetensors |

Verified SHA-256 checksums:

```text
865ba09f5b4c3cbd3468a4bd3acb9fcb2f8740c54317482f0bcd4ed1d3655cee  flux-2-klein-9b-fp8.safetensors
0cfbf745760f07a76ddeb358dd025a27f2e11d1ca9c9a4169a373d52990fe86e  Qwen3-8B-Q8_0.gguf
ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04  vae/diffusion_pytorch_model.safetensors
```

The model host requires access approval for the 9B weights. Use your own authorized
Hugging Face credential for downloads, never publish it or bake it into the worker
unit. The code's MIT license does not relicense model weights; review their licenses
for your use. The 9B model has separate noncommercial terms.

CPU offload and memory mapping allow model stages to share GPU memory. The example
assumes a 16 GiB NVIDIA GPU and 32 GiB host RAM; qualify your hardware with a cold
request and a model handoff. Keep the five-minute ownership cap; do not extend it
to hide an incompatible model. Worker processes fully exit on release.

An RTX 4060 Ti 16 GiB qualification run produced a 1024x1024 image in 25.02
seconds inside the worker, with 10,922 MiB sampled peak device memory. Loading
the worker's HTTP service took about one second; weight transfers and inference
are included in the generation time. Queueing behind another model is additional.
These are single-run measurements, not a throughput or quality benchmark.

## Image behavior

The distilled model uses four steps, CFG 1, Euler sampling and the FLUX.2 scheduler.
It is not the separate undistilled `base` model. Legacy 512/640 square requests render
at 1024 square; legacy portrait/landscape requests render at 768x1152/1152x768.
Avifors uses Lanczos resizing to return the exact requested PNG dimensions.
Native rendering sizes are also available. The default remains 640x640.

FLUX's distilled CFG-1 path does not use a separate negative conditioning branch.
`negative_prompt_mode: instruction` translates a supplied negative prompt to an
exclusion instruction appended to the prompt. This preserves request acceptance,
but has different semantics from Stable Diffusion CFG negatives. Other profiles
default to forwarding native negative prompts unchanged.

`sizes`, `default_size` and `render_sizes` are administrator-controlled per-model
parameters. Sizes must be multiples of 32, between 256 and 2048 on each axis, and
at most 2 megapixels. Render mappings must preserve aspect ratio and cannot render
below the output resolution. Clients cannot override steps, samplers or render size.

Validate a cold image, native and legacy dimensions, authenticated artifact access,
gateway archiving, GPU release and text/speech handoff before declaring rollout
complete. To roll back, drain requests, stop the broker, restore its previous package,
image unit and config, reload systemd and restart the broker.
