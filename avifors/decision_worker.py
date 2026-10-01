"""Optional isolated System One decision worker (Laya checkpoints); never imported by the broker.

Serves `POST /v1/systemone` with the Ollama/Nimble contract (see avifors.systemone) and `GET /health`.
One request is scored at a time. Inputs are never truncated: a state that does not fit the
checkpoint's context is rejected with 400, as Ollama does, instead of being silently clipped.

    python -m avifors.decision_worker --checkpoint /var/lib/avifors/laya/aer-multitask --device cpu --threads 6
    python -m avifors.decision_worker --checkpoint DIR --onnx DIR/onnx/laya.int8.onnx --threads 6
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from aiohttp import web

from .systemone import BODY_TOO_LARGE, MAX_BODY, Invalid, answer, error_body, parse

LOG = logging.getLogger(__name__)


class TooLong(Exception):
    pass


class LayaEngine:
    """Laya torch (CPU/CUDA) or ONNX Runtime (CPU) scoring behind one interface."""

    def __init__(self, checkpoint, device="cpu", onnx=None, threads=6):
        import torch

        torch.set_num_threads(threads)
        if onnx:
            import onnxruntime as ort

            make = ort.SessionOptions

            def options():  # ONNXAgent builds its own session; pin its thread count
                so = make()
                so.intra_op_num_threads = threads
                so.inter_op_num_threads = 1
                return so

            ort.SessionOptions = options
            from laya.onnx_agent import ONNXAgent

            self.agent = ONNXAgent(str(checkpoint), onnx_path=str(onnx))
        else:
            import laya

            self.agent = laya.load(str(checkpoint), device=device)
        self.max_len = int(self.agent.cfg.get("max_len", 512))
        self.head_max_len = int(self.agent.cfg.get("head_max_len", 192))

    def tokens(self, state, questions) -> int:
        """Rendered prompt tokens summed over questions; TooLong if any state would be clipped."""
        from laya.common import build_sequence

        total = 0
        for name, q in questions.items():
            internal = self.agent._to_internal(q)
            ids, _, _, clip = build_sequence(
                self.agent.tok, state, internal, self.max_len, self.head_max_len,
                return_stats=True, return_truncation_stats=True,
            )
            if clip["truncated"]:
                raise TooLong(
                    f"question {name!r}: prompt needs {clip['state_tokens'] - clip['state_tokens_used']} more "
                    f"tokens than the model context ({self.max_len}); shorten the state"
                )
            total += len(ids)
        return total

    def score(self, state, questions) -> dict:
        """{question name: {answer key: probability}} in avifors.systemone.keys() terms."""
        result = self.agent.predict(state, questions)["answers"]
        out = {}
        for name, a in result.items():
            if questions[name]["type"] == "noul":
                p = float(a["noul"])
                out[name] = {"false": 1 - p, "true": p}
            else:
                out[name] = {str(k): float(v) for k, v in a["probabilities"].items()}
        return out


def create_worker(engine, card):
    lock = asyncio.Lock()
    app = web.Application(client_max_size=MAX_BODY + 1024)

    async def health(request):
        return web.json_response({"status": "ok", "model": card})

    async def systemone(request):
        if request.content_length is not None and request.content_length > MAX_BODY:
            return web.json_response(error_body(BODY_TOO_LARGE), status=413)
        try:
            body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return web.json_response(error_body(BODY_TOO_LARGE), status=413)
        if len(body) > MAX_BODY:
            return web.json_response(error_body(BODY_TOO_LARGE), status=413)
        try:
            req = parse(body)
        except Invalid as exc:
            return web.json_response(error_body(str(exc)), status=400)
        state, questions = req["state"], req["questions"]
        started = time.monotonic()
        try:
            async with lock:
                n_tokens = await asyncio.to_thread(engine.tokens, state, questions)
                probs = await asyncio.to_thread(engine.score, state, questions)
            answers = {name: answer(q, probs[name]) for name, q in questions.items()}
        except TooLong as exc:
            return web.json_response(error_body(str(exc)), status=400)
        except Exception:
            LOG.exception("decision scoring failed")
            return web.json_response(error_body("decision scoring failed"), status=500)
        # No states or questions in logs: only sizes and timings.
        LOG.info("decided questions=%d tokens=%d elapsed=%.3f", len(questions), n_tokens, time.monotonic() - started)
        return web.json_response(
            {
                "model": req["model"],
                "answers": answers,
                # The encoder scores options in one pass and generates no text; report one scored
                # decision per question, matching Nimble's single scored token per answer.
                "usage": {"input_tokens": n_tokens, "output_tokens": len(questions)},
            }
        )

    app.router.add_get("/health", health)
    app.router.add_post("/v1/systemone", systemone)
    return app


def main():
    parser = argparse.ArgumentParser(description="Laya System One decision worker")
    parser.add_argument("--checkpoint", required=True, help="Laya checkpoint directory (or hub id)")
    parser.add_argument("--onnx", help="score with this ONNX export on CPU instead of torch")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--card", default="", help="name reported by /health")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18004)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.onnx and args.device != "cpu":
        parser.error("the ONNX path runs on CPU only")
    engine = LayaEngine(args.checkpoint, device=args.device, onnx=args.onnx, threads=args.threads)
    web.run_app(
        create_worker(engine, args.card or str(args.checkpoint)),
        host=args.host,
        port=args.port,
        access_log=None,
        handler_cancellation=False,
    )


if __name__ == "__main__":
    main()
