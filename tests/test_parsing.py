import json

import pytest

from triage.parsing import parse_llm_output

GOOD = {"category": "bug", "priority": "high", "summary": "App crashes", "suggested_reply": "Sorry!", "confidence": 0.8}


def test_clean_json():
    out = parse_llm_output(json.dumps(GOOD))
    assert out.triage and out.triage.category == "bug" and not out.repaired


@pytest.mark.parametrize("raw", [
    "```json\n" + json.dumps(GOOD) + "\n```",
    "Sure! Here is the triage:\n" + json.dumps(GOOD) + "\nLet me know if you need anything else.",
    json.dumps(GOOD)[:-1] + ",}",  # trailing comma
    str(GOOD),  # python-style single quotes
    json.dumps({"triage": GOOD}),  # wrapped
])
def test_recoverable_formats(raw):
    out = parse_llm_output(raw)
    assert out.triage is not None, out.error
    assert out.triage.category == "bug"
    assert out.repaired


@pytest.mark.parametrize("field,value,expected", [
    ("category", "Feature Request", "feature_request"),
    ("category", "feature-request", "feature_request"),
    ("category", "Billing / Account", "billing"),
    ("category", "technical issue", "bug"),
    ("priority", "Critical", "urgent"),
    ("priority", "P2", "medium"),
    ("confidence", "85%", 0.85),
    ("confidence", 85, 0.85),
    ("confidence", "0.7", 0.7),
])
def test_normalizes_near_misses(field, value, expected):
    out = parse_llm_output(json.dumps({**GOOD, field: value}))
    assert out.triage is not None, out.error
    assert getattr(out.triage, field) == expected


def test_reply_alias_and_multiline_summary():
    data = {**GOOD, "summary": "line one\nline two"}
    data["reply"] = data.pop("suggested_reply")
    out = parse_llm_output(json.dumps(data))
    assert out.triage.suggested_reply == "Sorry!"
    assert "\n" not in out.triage.summary


@pytest.mark.parametrize("raw,err", [
    ("I cannot help with that.", "did not contain a JSON object"),
    ("", "did not contain a JSON object"),
    (json.dumps({**GOOD, "category": "complaint"}), "category"),
    (json.dumps({**GOOD, "priority": "whenever"}), "priority"),
    (json.dumps({**GOOD, "confidence": 4000}), "confidence"),
    (json.dumps({**GOOD, "suggested_reply": ""}), "suggested_reply"),
    ("[1, 2, 3]", "did not contain a JSON object"),
])
def test_unrecoverable_reports_error(raw, err):
    out = parse_llm_output(raw)
    assert out.triage is None
    assert err in out.error


def test_partial_kept_for_salvage():
    out = parse_llm_output(json.dumps({**GOOD, "confidence": "very"}))
    assert out.triage is None
    assert out.partial["category"] == "bug"


def test_impact_accepts_strings_lists_and_unknown_keys():
    out = parse_llm_output(json.dumps({**GOOD, "impact": {"customer_blocked": "yes", "Financial Error": "false",
                                                          "made_up_fact": True}}))
    assert out.triage.impact.customer_blocked is True
    assert out.triage.impact.financial_error is False
    out = parse_llm_output(json.dumps({**GOOD, "impact": ["churn_risk", "legal_or_compliance"]}))
    assert out.triage.impact.churn_risk and out.triage.impact.legal_or_compliance


def test_missing_or_garbage_impact_is_not_fatal():
    assert parse_llm_output(json.dumps(GOOD)).triage.impact is None
    assert parse_llm_output(json.dumps({**GOOD, "impact": "high"})).triage.impact is None
