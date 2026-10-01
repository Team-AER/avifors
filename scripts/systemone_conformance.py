#!/usr/bin/env python3
"""Black-box conformance check for an Ollama/Nimble-compatible `POST /v1/systemone` endpoint.

Runs one battery of requests against a target and checks every answer against the contract
(keys, option order, probabilities, confidence = 1 - H/ln N, argmax/ties, score formula, usage, and
Ollama-shaped {"error": "..."} bodies with 400/404/413). With --compare it runs the same battery
against a reference endpoint (e.g. Ollama 0.35 serving nimble) and diffs status codes and shapes.

    python scripts/systemone_conformance.py --base http://127.0.0.1:8000 --model aer-laya --key $KEY
    python scripts/systemone_conformance.py --base http://127.0.0.1:8000 --model aer-laya \
        --compare http://127.0.0.1:11434 --compare-model nimble

Stdlib only. Request states are synthetic; nothing private is sent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

TICKET = "Our checkout has returned 500 errors since 9am."
LABEL = {
    "type": "choice",
    "instructions": "Which label fits this ticket?",
    "criteria": {"billing": None, "bug": None, "account": None},
}


def cases(model):
    letters = [chr(97 + i) for i in range(26)]
    yield "nimble_example", 200, {"model": model, "state": TICKET, "questions": {"label": LABEL}}
    yield "mixed_types_object_state", 200, {
        "model": model,
        "state": {"subject": "Charged twice", "body": "I see two charges for order #4411. Please refund one."},
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems", "access": None},
            },
            "angry": {"type": "noul", "instructions": "Is the customer angry?"},
            "refund": {
                "type": "noul",
                "instructions": "Does the customer ask for a refund?",
                "criteria": {"true": "They ask for money back", "false": "They do not"},
            },
            "priority": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "normal", "high"]},
        },
    }
    yield "array_state", 200, {
        "model": model,
        "state": [{"role": "user", "text": "My password reset link never arrives."}],
        "questions": {"label": LABEL},
    }
    yield "26_options", 200, {
        "model": model,
        "state": "The letter I mean is q.",
        "questions": {"letter": {"type": "choice", "instructions": "Which letter is meant?",
                                 "criteria": {x: None for x in letters}}},
    }
    yield "64_questions", 200, {
        "model": model,
        "state": TICKET,
        "questions": {f"q{i}": {"type": "noul", "instructions": f"Is statement {i} about payments?"} for i in range(64)},
    }
    yield "keep_alive", 200, {"model": model, "state": TICKET, "questions": {"label": LABEL}, "keep_alive": "5m"}
    yield "no_questions", 400, {"model": model, "state": TICKET, "questions": {}}
    yield "65_questions", 400, {
        "model": model,
        "state": TICKET,
        "questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(65)},
    }
    yield "27_options", 400, {
        "model": model,
        "state": TICKET,
        "questions": {"x": {"type": "choice", "instructions": "x", "criteria": {f"o{i}": None for i in range(27)}}},
    }
    yield "one_option", 400, {
        "model": model, "state": TICKET,
        "questions": {"x": {"type": "choice", "instructions": "x", "criteria": {"only": None}}},
    }
    yield "bad_type", 400, {"model": model, "state": TICKET, "questions": {"x": {"type": "rank", "instructions": "x"}}}
    yield "empty_state", 400, {"model": model, "state": "", "questions": {"label": LABEL}}
    # Ollama 0.35 ignores generation-style and unknown fields and answers normally.
    yield "stream_ignored", 200, {"model": model, "state": TICKET, "questions": {"label": LABEL}, "stream": True}
    yield "extra_fields_ignored", 200, {"model": model, "state": TICKET, "questions": {"label": LABEL},
                                        "options": {"temperature": 0}, "think": True, "foo": 1}
    yield "instructions_object", 200, {
        "model": model, "state": TICKET,
        "questions": {"bug": {"type": "noul", "instructions": {"ask": "Is this a bug report?"}}},
    }
    yield "missing_instructions", 400, {"model": model, "state": TICKET, "questions": {"x": {"type": "noul"}}}
    yield "choice_criteria_list", 400, {
        "model": model, "state": TICKET,
        "questions": {"x": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}},
    }
    yield "unknown_model", 404, {"model": "no-such-model-xyz", "state": TICKET, "questions": {"label": LABEL}}
    yield "body_over_64k", 413, {"model": model, "state": "x" * (64 * 1024), "questions": {"label": LABEL}}


def post(base, payload, key=None, timeout=600):
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} | ({"Authorization": f"Bearer {key}"} if key else {})
    req = urllib.request.Request(base.rstrip("/") + "/v1/systemone", data=data, headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except (TimeoutError, urllib.error.URLError) as e:
        return 0, {"_error": f"{type(e).__name__}: {e}"}, 1000 * (time.perf_counter() - started)
    ms = 1000 * (time.perf_counter() - started)
    try:
        return status, json.loads(raw), ms
    except json.JSONDecodeError:
        return status, {"_raw": raw[:200].decode(errors="replace")}, ms


def check_success(payload, body):
    problems = []
    if set(body) != {"model", "answers", "usage"}:
        problems.append(f"top-level keys {sorted(body)}")
        return problems
    if body["model"] != payload["model"]:
        problems.append("model not echoed")
    usage = body["usage"]
    if set(usage) != {"input_tokens", "output_tokens"} or not all(isinstance(v, int) and v >= 0 for v in usage.values()):
        problems.append(f"usage {usage}")
    if set(body["answers"]) != set(payload["questions"]):
        problems.append("answers do not match question names")
        return problems
    for name, q in payload["questions"].items():
        a = body["answers"][name]
        if a.get("type") != q["type"]:
            problems.append(f"{name}: type {a.get('type')}")
            continue
        if q["type"] == "noul":
            if set(a) != {"type", "noul"} or not isinstance(a["noul"], (int, float)) or not 0 <= a["noul"] <= 1:
                problems.append(f"{name}: noul answer {a}")
            continue
        keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
        expected = {"type", "choice", "probabilities", "confidence"} if q["type"] == "choice" else {
            "type", "score", "legend", "probabilities", "confidence"}
        if set(a) != expected:
            problems.append(f"{name}: keys {sorted(a)}")
            continue
        probs = a["probabilities"]
        if list(probs) != keys:
            problems.append(f"{name}: probability keys/order {list(probs)[:5]}")
            continue
        p = [float(probs[k]) for k in keys]
        if abs(sum(p) - 1) > 0.01 or any(v < 0 for v in p):
            problems.append(f"{name}: probabilities sum {sum(p):.4f}")
        h = -sum(v * math.log(v) for v in p if v > 0)
        if abs((1 - h / math.log(len(p))) - a["confidence"]) > 0.01:
            problems.append(f"{name}: confidence {a['confidence']} != 1-H/lnN {1 - h / math.log(len(p)):.4f}")
        if q["type"] == "choice":
            top = max(p)
            if a["choice"] not in keys or float(probs[a["choice"]]) < top - 1e-3:
                problems.append(f"{name}: choice {a['choice']} is not the argmax")
        else:
            if abs(sum(i * v for i, v in enumerate(p)) - a["score"]) > 0.01:
                problems.append(f"{name}: score {a['score']} != sum(i*p_i)")
            if a["legend"] != {str(i): c for i, c in enumerate(q["criteria"])}:
                problems.append(f"{name}: legend {a['legend']}")
    return problems


def shape(body):
    """Structural fingerprint used to diff two implementations (values ignored)."""
    if isinstance(body, dict):
        return {k: shape(v) for k, v in sorted(body.items()) if not k.startswith("_")}
    if isinstance(body, list):
        return [shape(body[0])] if body else []
    return type(body).__name__.replace("int", "number").replace("float", "number")


def run(base, model, key):
    results = {}
    for name, expect, payload in cases(model):
        status, body, ms = post(base, payload, key)
        problems = []
        if status != expect:
            problems.append(f"status {status} != {expect}")
        if expect == 200 and status == 200:
            problems += check_success(payload, body)
        elif status != 200 and not (set(body) == {"error"} and isinstance(body["error"], str)):
            problems.append(f"error body {body}")
        if name == "body_over_64k" and body.get("error") != "request body must not exceed 64 KiB":
            problems.append(f"413 message {body.get('error')!r}")
        results[name] = {"status": status, "ms": round(ms), "problems": problems, "body": body}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key")
    ap.add_argument("--compare", help="reference endpoint base URL, e.g. Ollama")
    ap.add_argument("--compare-model")
    ap.add_argument("--compare-key")
    ap.add_argument("--json-out")
    a = ap.parse_args()

    target = run(a.base, a.model, a.key)
    reference = run(a.compare, a.compare_model or a.model, a.compare_key) if a.compare else None
    failed = 0
    print(f"{'case':<26} {'target':>7} {'ms':>7}  problems" + ("   | reference  shape-diff" if reference else ""))
    for name, r in target.items():
        failed += bool(r["problems"])
        line = f"{name:<26} {r['status']:>7} {r['ms']:>7}  {'; '.join(r['problems']) or 'ok'}"
        if reference:
            ref = reference[name]
            same_shape = shape(r["body"]) == shape(ref["body"])
            line += f"   | {ref['status']:>4} {'ok' if not ref['problems'] else 'REF:' + '; '.join(ref['problems'])}"
            line += f"  {'same' if same_shape and ref['status'] == r['status'] else 'DIFF'}"
            mine, theirs = r["body"].get("error"), ref["body"].get("error")
            if mine != theirs and (mine or theirs):
                line += f"  msg: ours={mine!r} ref={theirs!r}"
        print(line)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"target": target, "reference": reference}, f, indent=2)
    print(f"\n{len(target) - failed}/{len(target)} cases conform" + (f"; {failed} failed" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
