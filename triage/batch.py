from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Settings
from .llm import ChatModel
from .pipeline import triage_ticket
from .schema import Ticket, TriageResult


def load_tickets(path: Path) -> list[Ticket]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Ticket.model_validate(item) for item in raw]


def run_batch(tickets: list[Ticket], model: ChatModel, settings: Settings, on_done=None) -> list[TriageResult]:
    """Triage all tickets concurrently; results come back in input order."""

    def one(t: Ticket) -> TriageResult:
        r = triage_ticket(t, model, settings)
        if on_done:
            on_done(r)
        return r

    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        return list(pool.map(one, tickets))
