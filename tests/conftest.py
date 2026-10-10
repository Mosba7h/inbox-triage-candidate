import json
from dataclasses import replace

import pytest

from triage.config import load_settings
from triage.llm import LLMError


class FakeModel:
    """Returns scripted responses in order; an Exception instance is raised instead."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.seen = []

    def complete(self, messages):
        self.calls += 1
        self.seen.append(messages)
        r = self.responses.pop(0) if self.responses else self.default
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps(r)

    default = json.dumps({"category": "other", "priority": "low", "summary": "s",
                          "suggested_reply": "Thanks!", "confidence": 0.8})


GOOD = {"category": "billing", "priority": "high", "summary": "Duplicate charge, wants refund.",
        "suggested_reply": "Hi, sorry about that - we'll look into the duplicate charge.", "confidence": 0.9}


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    return replace(s, state_path=tmp_path / "state.json", results_path=tmp_path / "eval" / "results.json",
                   llm_max_retries=2, concurrency=2)


@pytest.fixture
def fake():
    return FakeModel


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("triage.pipeline.time.sleep", lambda s: None)


__all__ = ["FakeModel", "GOOD", "LLMError"]
