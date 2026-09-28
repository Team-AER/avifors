"""Optional isolated multilingual ASR worker; never imported by the broker."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from aiohttp import web

QWEN_LANGUAGES = dict(
    zip(
        "zh en yue ar de fr es pt id it ko ru th vi ja tr hi ms nl sv da fi pl cs fil fa el ro hu mk".split(),
        "Chinese English Cantonese Arabic German French Spanish Portuguese Indonesian Italian Korean Russian Thai Vietnamese Japanese Turkish Hindi Malay Dutch Swedish Danish Finnish Polish Czech Filipino Persian Greek Romanian Hungarian Macedonian".split(),
    )
)
QWEN_LANGUAGES.update(
    {
        "eng_Latn": "English",
        "hin_Deva": "Hindi",
        "cmn_Hans": "Chinese",
        "arb_Arab": "Arabic",
        "pes_Arab": "Persian",
        "zsm_Latn": "Malay",
    }
)


class QwenPipeline:
    def __init__(self, path):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=1,
            max_new_tokens=1024,
            local_files_only=True,
        )

    def transcribe(self, inputs, lang, batch_size):
        language = QWEN_LANGUAGES[lang[0]] if lang[0] else None
        results = self.model.transcribe(
            audio=[(x["waveform"], x["sample_rate"]) for x in inputs], language=language
        )
        return [r.text for r in results]


def language_id(code, supported):
    if not code:
        return None
    if code in supported:
        return code
    import pycountry

    # ISO macrolanguages whose common ASR form has a specific language ID.
    aliases = {
        "zh": "cmn_Hans",
        "zho": "cmn_Hans",
        "yue": "yue_Hant",
        "ar": "arb_Arab",
        "ara": "arb_Arab",
        "fa": "pes_Arab",
        "fas": "pes_Arab",
        "ms": "zsm_Latn",
        "msa": "zsm_Latn",
        "no": "nob_Latn",
        "nor": "nob_Latn",
    }
    if code in aliases and aliases[code] in supported:
        return aliases[code]
    language = pycountry.languages.get(**({"alpha_2": code} if len(code) == 2 else {"alpha_3": code}))
    alpha2 = getattr(language, "alpha_2", None)
    if alpha2 in supported:
        return alpha2
    prefix = getattr(language, "alpha_3", code) + "_"
    candidates = sorted(x for x in supported if x.startswith(prefix))
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("unsupported or ambiguous language; supply explicit ISO-script code")


def create_worker(pipeline, supported, card):
    import numpy as np

    lock = asyncio.Lock()
    app = web.Application(client_max_size=40 * 16000 * 2)

    async def health(request):
        return web.json_response({"status": "ok", "model": card})

    async def transcribe(request):
        try:
            language = language_id(request.query.get("language", ""), supported)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        data = await request.read()
        if not data or len(data) % 2 or len(data) >= 40 * 16000 * 2:
            return web.json_response({"error": "expected less than 40s of 16kHz mono PCM16"}, status=400)
        waveform = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        started = time.monotonic()
        async with lock:
            result = await asyncio.to_thread(
                pipeline.transcribe,
                [{"waveform": waveform, "sample_rate": 16000}],
                lang=[language],
                batch_size=1,
            )
        # No transcripts/audio in logs, metrics or traces.
        logging.info(
            "transcribed duration=%.3f elapsed=%.3f", len(waveform) / 16000, time.monotonic() - started
        )
        return web.json_response({"text": result[0], "language": language})

    app.router.add_get("/health", health)
    app.router.add_post("/transcribe", transcribe)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="omniASR_LLM_3B_v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18002)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--engine", choices=["omni", "qwen"], default="omni")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    import torch

    torch.set_num_threads(args.threads)
    if not 1 <= args.beam_size <= 10:
        parser.error("beam size must be between 1 and 10")
    if args.engine == "qwen":
        pipeline, supported_langs = QwenPipeline(args.model), set(QWEN_LANGUAGES)
    else:
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaBeamSearchConfig
        from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

        pipeline = ASRInferencePipeline(
            model_card=args.model,
            device="cuda",
            dtype=torch.bfloat16,
            beam_search_config=Wav2Vec2LlamaBeamSearchConfig(nbest=args.beam_size, length_norm=False),
        )
    web.run_app(
        create_worker(pipeline, set(supported_langs), args.model),
        host=args.host,
        port=args.port,
        access_log=None,
        handler_cancellation=False,
    )


if __name__ == "__main__":
    main()
