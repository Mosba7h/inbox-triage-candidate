"""Runtime configuration.

Only LLM_BASE_URL, LLM_MODEL and LLM_API_KEY are meant to be set by whoever runs
this. Everything else has a working default so the app boots with just those three.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- the three required knobs (with local-Ollama defaults for convenience) ---
    llm_base_url: str
    llm_model: str
    # Ollama ignores the key, but the OpenAI SDK refuses an empty one.
    llm_api_key: str

    # --- optional tuning, all defaulted ---
    llm_timeout_s: float
    llm_max_retries: int  # extra attempts after the first one
    llm_temperature: float
    concurrency: int
    escalate_below: float  # confidence threshold for escalation
    tickets_path: Path
    labels_path: Path
    results_path: Path
    state_path: Path


def load_settings() -> Settings:
    return Settings(
        llm_base_url=os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1",
        llm_model=os.environ.get("LLM_MODEL") or "llama3.2:3b",
        llm_api_key=os.environ.get("LLM_API_KEY") or "ollama",
        llm_timeout_s=_float("TRIAGE_LLM_TIMEOUT_S", 120.0),
        llm_max_retries=_int("TRIAGE_LLM_MAX_RETRIES", 2),
        llm_temperature=_float("TRIAGE_LLM_TEMPERATURE", 0.0),
        concurrency=max(1, _int("TRIAGE_CONCURRENCY", 4)),
        escalate_below=_float("TRIAGE_ESCALATE_BELOW", 0.6),
        tickets_path=Path(os.environ.get("TRIAGE_TICKETS_PATH", ROOT / "data" / "tickets.json")),
        labels_path=Path(os.environ.get("TRIAGE_LABELS_PATH", ROOT / "data" / "labels.json")),
        results_path=Path(os.environ.get("TRIAGE_RESULTS_PATH", ROOT / "eval" / "results.json")),
        state_path=Path(os.environ.get("TRIAGE_STATE_PATH", ROOT / "data" / "review_state.json")),
    )
