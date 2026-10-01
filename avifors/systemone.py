"""System One decision API, compatible with Ollama's `POST /v1/systemone` as served for Nimble.

Request:  {"model", "state", "questions": {name: {"type", "instructions", "criteria"}}, "keep_alive"?}
Response: {"model", "answers": {name: answer}, "usage": {"input_tokens", "output_tokens"}}
Errors:   {"error": "message"} with 400 / 404 / 413 / 500 (plus Avifors' 429 / 503 / 504 admission errors)

Pure functions only (no aiohttp, no model code): the broker validates with them before waking a
worker, and the decision worker formats answers with them, so both sides agree on the contract.
Reference: https://docs.ollama.com/api/systemone. Where the documentation and Ollama 0.35.0 differ,
this follows the server (diffed with scripts/systemone_conformance.py against nimble):
- unknown and generation fields (stream, options, images, format, tools, think, ...) are ignored;
- numbers are returned unrounded;
- instructions may be a nonempty string, object or array;
- validation messages use Ollama's wording.
"""

from __future__ import annotations

import json
import math
import re

MAX_BODY = 64 * 1024
MAX_QUESTIONS = 64
MIN_OPTIONS, MAX_OPTIONS = 2, 26
TYPES = ("choice", "noul", "score")
BODY_TOO_LARGE = "request body must not exceed 64 KiB"
# Ollama accepts Go durations ("5m", "1h30m", "-1s") or a number of seconds.
DURATION = re.compile(r"-?(\d+(\.\d+)?(ns|us|µs|ms|s|m|h))+\Z")


class Invalid(ValueError):
    """A request the API rejects with 400 and this message."""


def error_body(message: str) -> dict:
    return {"error": message}


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _description(value) -> bool:
    return value is None or isinstance(value, str)


def parse(body: bytes) -> dict:
    """Validate a raw request body; return the decoded request. Raises Invalid (400) on bad input.
    The 64 KiB cap is the caller's job (413 must be decided before reading the whole body)."""
    try:
        request = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise Invalid("invalid JSON request body") from None
    if not isinstance(request, dict):
        raise Invalid("request must be a JSON object")
    if not _text(request.get("model")):
        raise Invalid("model is required")
    state = request.get("state")
    if isinstance(state, str):
        if not state.strip():
            raise Invalid("state must not be empty")
    elif isinstance(state, (dict, list)):
        if not state:
            raise Invalid("state must not be empty")
    else:
        raise Invalid("state must be a string, JSON object or JSON array")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise Invalid(f"questions must contain 1–{MAX_QUESTIONS} fields")
    for name, question in questions.items():
        _question(name, question)
    if "keep_alive" in request:
        _keep_alive(request["keep_alive"])
    return request


def _question(name, q) -> None:
    if not _text(name):
        raise Invalid("question names must not be blank")
    label = f"question {json.dumps(name, ensure_ascii=False)}"
    if not isinstance(q, dict):
        raise Invalid(f"{label}: must be an object")
    kind = q.get("type")
    if kind not in TYPES:
        raise Invalid(f"{label}: type must be choice, noul, or score")
    instructions = q.get("instructions")
    if not (_text(instructions) or (isinstance(instructions, (dict, list)) and instructions)):
        raise Invalid(f"{label}: instructions must be a nonempty string, object, or array")
    criteria = q.get("criteria")
    count = f"{label}: criteria must contain {MIN_OPTIONS}–{MAX_OPTIONS} candidates"
    if kind == "choice":
        if not isinstance(criteria, dict) or not all(_text(k) and _description(v) for k, v in criteria.items()):
            raise Invalid(f"{label}: choice criteria must map option keys to descriptions or null")
        if not MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS:
            raise Invalid(count)
    elif kind == "noul":
        if criteria is not None and (
            not isinstance(criteria, dict)
            or not set(criteria) <= {"true", "false"}
            or not all(_description(v) for v in criteria.values())
        ):
            raise Invalid(f"{label}: noul criteria may only describe \"false\" and \"true\"")
    else:
        if not isinstance(criteria, list) or not all(_text(c) for c in criteria):
            raise Invalid(f"{label}: score criteria must be a list of level descriptions")
        if not MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS:
            raise Invalid(count)


def _keep_alive(value) -> None:
    if isinstance(value, bool) or not (
        (isinstance(value, (int, float)) and math.isfinite(value))
        or (isinstance(value, str) and (DURATION.match(value) or _number(value)))
    ):
        raise Invalid("keep_alive must be a duration such as \"5m\" or a number of seconds")


def _number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def keys(question: dict) -> list[str]:
    """Answer keys in request order: option names (choice), "false"/"true" (noul), level indices (score)."""
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(question["criteria"]))]


def confidence(probabilities: list[float]) -> float:
    """1 - H(p) / ln(N): 0 for a uniform distribution, near 1 when one candidate dominates.
    It measures concentration, not the chance the answer is right."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, min(1.0, 1 - entropy / math.log(n)))


def _normalise(question: dict, probs: dict) -> list[float]:
    values = [float(probs.get(k, 0.0)) for k in keys(question)]
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("model returned an invalid probability")
    total = sum(values)
    if total <= 0:
        raise ValueError("model returned no probability mass")
    return [v / total for v in values]


def answer(question: dict, probs: dict, digits: int | None = None) -> dict:
    """Format one answer from the model's probabilities over `keys(question)`. Like Ollama, numbers
    are unrounded unless `digits` is given."""
    k, p = keys(question), _normalise(question, probs)

    def r(v):
        return v if digits is None else round(v, digits)

    if question["type"] == "noul":
        return {"type": "noul", "noul": r(p[1])}
    probabilities = {key: r(v) for key, v in zip(k, p)}
    if question["type"] == "choice":
        best = max(range(len(p)), key=lambda i: (p[i], -i))  # ties go to the first option in request order
        return {
            "type": "choice",
            "choice": k[best],
            "probabilities": probabilities,
            "confidence": r(confidence(p)),
        }
    return {
        "type": "score",
        "score": r(sum(i * v for i, v in enumerate(p))),  # probability-weighted level, 0..N-1
        "legend": {str(i): text for i, text in enumerate(question["criteria"])},
        "probabilities": probabilities,
        "confidence": r(confidence(p)),
    }
