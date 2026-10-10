"""Review-queue state: tickets + latest triage + the human's decision.

A JSON file guarded by a lock. Fine for a single-process demo; a real deployment
would use a database (see README, "Next steps").
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .schema import Ticket, TriageResult


class ReviewStore:
    def __init__(self, path: Path, tickets: list[Ticket], seed_details: Path | None = None):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.items: dict[str, dict] = {
            t.id: {"ticket": t.model_dump(by_alias=True), "triage": None, "review": {"status": "pending"}}
            for t in tickets
        }
        if self.path.exists():
            self._merge(json.loads(self.path.read_text(encoding="utf-8")))
        elif seed_details and seed_details.exists():
            # Reuse the last eval run so the UI isn't empty on first boot.
            for p in json.loads(seed_details.read_text(encoding="utf-8")).get("predictions", []):
                if p.get("id") in self.items:
                    self.items[p["id"]]["triage"] = TriageResult.model_validate(p).model_dump()

    def _merge(self, saved: dict) -> None:
        for tid, item in saved.items():
            if tid in self.items:
                self.items[tid]["triage"] = item.get("triage")
                self.items[tid]["review"] = item.get("review") or {"status": "pending"}
            else:  # ad-hoc tickets created through the API
                self.items[tid] = item

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def list(self) -> list[dict]:
        with self._lock:
            return [{"id": k, **v} for k, v in self.items.items()]

    def get(self, tid: str) -> dict | None:
        with self._lock:
            item = self.items.get(tid)
            return {"id": tid, **item} if item else None

    def add_ticket(self, t: Ticket) -> None:
        with self._lock:
            self.items[t.id] = {"ticket": t.model_dump(by_alias=True), "triage": None, "review": {"status": "pending"}}
            self._save()

    def set_triage(self, r: TriageResult) -> None:
        with self._lock:
            self.items[r.id]["triage"] = r.model_dump()
            # new triage invalidates an earlier decision
            self.items[r.id]["review"] = {"status": "pending"}
            self._save()

    def set_review(self, tid: str, *, status: str, reply: str, category: str, priority: str, note: str = "") -> dict:
        with self._lock:
            item = self.items[tid]
            triage = item["triage"] or {}
            item["review"] = {
                "status": status,
                "final_reply": reply,
                "final_category": category,
                "final_priority": priority,
                "note": note,
                # Corrections are the most valuable data this tool produces - keep them explicit.
                "edited_reply": reply != triage.get("suggested_reply"),
                "corrected_category": category != triage.get("category"),
                "corrected_priority": priority != triage.get("priority"),
                "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._save()
            return item["review"]
