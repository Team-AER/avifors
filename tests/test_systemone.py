import json
import math

import pytest

from avifors.systemone import Invalid, answer, confidence, keys, parse

# The request from Ollama's announcement for nimble.
NIMBLE = {
    "model": "nimble",
    "state": "Our checkout has returned 500 errors since 9am.",
    "questions": {
        "label": {
            "type": "choice",
            "instructions": "Which label fits this ticket?",
            "criteria": {"billing": None, "bug": None, "account": None},
        }
    },
}


def body(**change):
    return json.dumps(NIMBLE | change).encode()


def test_confidence_matches_ollama_examples():
    # API reference, 4-decimal probabilities: {0.9781, 0.0125, 0.0093} -> 0.8906
    assert confidence([0.9781, 0.0125, 0.0093]) == pytest.approx(0.8906, abs=0.0005)
    # announcement shows 3-decimal probabilities {0.011, 0.979, 0.010} -> 0.892; the rounding
    # of the displayed probabilities alone moves the entropy by ~0.002
    assert confidence([0.011, 0.979, 0.010]) == pytest.approx(0.892, abs=0.003)
    assert confidence([1 / 3] * 3) == pytest.approx(0)
    assert confidence([1.0, 0.0]) == pytest.approx(1)


def test_choice_answer_shape_and_first_option_wins_ties():
    q = NIMBLE["questions"]["label"]
    a = answer(q, {"billing": 0.011, "bug": 0.979, "account": 0.010})
    assert a == {
        "type": "choice",
        "choice": "bug",
        "probabilities": pytest.approx({"billing": 0.011, "bug": 0.979, "account": 0.01}),
        "confidence": pytest.approx(0.892, abs=0.003),
    }
    assert answer(q, {"billing": 1, "bug": 2, "account": 0}, digits=4)["probabilities"]["bug"] == 0.6667
    assert list(a["probabilities"]) == ["billing", "bug", "account"]  # request order
    tie = answer(q, {"billing": 0.4, "bug": 0.4, "account": 0.2})
    assert tie["choice"] == "billing"


def test_noul_and_score_answers():
    noul = {"type": "noul", "instructions": "Is it urgent?"}
    assert answer(noul, {"false": 0.25, "true": 0.75}) == {"type": "noul", "noul": 0.75}
    score = {"type": "score", "instructions": "How urgent?", "criteria": ["Routine", "Soon", "Urgent"]}
    a = answer(score, {"0": 0.482, "1": 0.342, "2": 0.176})
    assert a["type"] == "score"
    assert a["score"] == pytest.approx(0.342 + 2 * 0.176, abs=1e-4)  # weighted level, not rounded to int
    assert a["legend"] == {"0": "Routine", "1": "Soon", "2": "Urgent"}
    assert a["probabilities"] == pytest.approx({"0": 0.482, "1": 0.342, "2": 0.176})
    assert 0 <= a["confidence"] < 0.1
    assert keys(score) == ["0", "1", "2"]


def test_answer_normalises_and_rejects_bad_probabilities():
    q = {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}}
    assert answer(q, {"a": 2, "b": 2})["probabilities"] == {"a": 0.5, "b": 0.5}
    for bad in ({"a": math.nan, "b": 1}, {"a": -1, "b": 1}, {"a": 0, "b": 0}):
        with pytest.raises(ValueError):
            answer(q, bad)


def test_parse_accepts_documented_shapes():
    assert parse(body())["model"] == "nimble"
    parse(body(state={"subject": "Refund", "body": "Charged twice"}))
    parse(body(state=[{"role": "user", "text": "hi"}]))
    parse(body(questions={"urgent": {"type": "noul", "instructions": "Urgent?"}}))
    parse(body(questions={"urgent": {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "Yes", "false": None}}}))
    parse(body(questions={"how": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]}}))
    for keep_alive in ("5m", "1h30m", "-1", "0", 300, -1, 0, "250ms"):
        parse(body(keep_alive=keep_alive))
    # Ollama 0.35 ignores generation-style and unknown fields rather than rejecting them.
    for extra in ({"stream": True}, {"options": {"temperature": 0}}, {"images": ["aGk="]}, {"think": True}, {"x": 1}):
        parse(body(**extra))
    parse(body(questions={"q": {"type": "noul", "instructions": {"ask": "urgent?"}}}))
    parse(body(questions={"q": {"type": "noul", "instructions": ["urgent?"]}}))


@pytest.mark.parametrize(
    "change",
    [
        {"model": ""},
        {"model": None},
        {"state": ""},
        {"state": "   "},
        {"state": {}},
        {"state": []},
        {"state": 12},
        {"questions": {}},
        {"questions": {f"q{i}": {"type": "noul", "instructions": "x"} for i in range(65)}},
        {"questions": {"": {"type": "noul", "instructions": "x"}}},
        {"questions": {"q": {"type": "rank", "instructions": "x"}}},
        {"questions": {"q": {"type": "noul"}}},
        {"questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"only": None}}}},
        {"questions": {"q": {"type": "choice", "instructions": "x", "criteria": {chr(97 + i): None for i in range(27)}}}},
        {"questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}},
        {"questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": 1, "b": None}}}},
        {"questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"maybe": "?"}}}},
        {"questions": {"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}},
        {"questions": {"q": {"type": "score", "instructions": "x", "criteria": {"a": None, "b": None}}}},
        {"questions": {"q": {"type": "noul", "instructions": {}}}},
        {"keep_alive": "soon"},
        {"keep_alive": True},
    ],
)
def test_parse_rejects_invalid_requests(change):
    with pytest.raises(Invalid):
        parse(body(**change))


def test_parse_rejects_non_objects_and_bad_json():
    for raw in (b"[]", b"not json", b"\xff\xfe"):
        with pytest.raises(Invalid):
            parse(raw)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"questions": {}}, "questions must contain 1–64 fields"),
        ({"questions": {"x": {"type": "rank", "instructions": "x"}}}, 'question "x": type must be choice, noul, or score'),
        ({"questions": {"x": {"type": "noul"}}}, 'question "x": instructions must be a nonempty string, object, or array'),
        (
            {"questions": {"x": {"type": "choice", "instructions": "x", "criteria": {"only": None}}}},
            'question "x": criteria must contain 2–26 candidates',
        ),
        (
            {"questions": {"x": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}},
            'question "x": choice criteria must map option keys to descriptions or null',
        ),
        ({"state": ""}, "state must not be empty"),
    ],
)
def test_validation_messages_match_ollama(change, message):
    """Messages observed from Ollama 0.35.0 serving nimble."""
    with pytest.raises(Invalid) as exc:
        parse(body(**change))
    assert str(exc.value) == message
