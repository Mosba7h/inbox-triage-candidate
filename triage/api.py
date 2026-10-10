"""FastAPI backend + static review-queue UI.

    uvicorn triage.api:app --port 8000      (or: python -m triage.api)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import rules
from .batch import load_tickets
from .config import Settings, load_settings
from .llm import ChatModel, OpenAICompatibleModel
from .pipeline import triage_ticket
from .schema import Category, Priority, Ticket, TriageResult
from .store import ReviewStore

STATIC = Path(__file__).parent / "static"


class NewTicket(BaseModel):
    subject: str = Field(default="", max_length=500)
    body: str = Field(default="", max_length=20_000)
    channel: str = "webform"
    sender: str | None = None


class ReviewIn(BaseModel):
    action: Literal["approve", "reject"]
    reply: str = Field(default="", max_length=10_000)
    category: Category
    priority: Priority
    note: str = Field(default="", max_length=2_000)


def create_app(settings: Settings | None = None, model: ChatModel | None = None) -> FastAPI:
    settings = settings or load_settings()
    model = model or OpenAICompatibleModel(settings)
    store = ReviewStore(settings.state_path, load_tickets(settings.tickets_path),
                        seed_details=settings.results_path.parent / "details.json")
    app = FastAPI(title="Support Inbox Assistant")

    def _get(tid: str) -> dict:
        item = store.get(tid)
        if not item:
            raise HTTPException(404, f"Unknown ticket {tid}")
        return item

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "model": settings.llm_model, "base_url": settings.llm_base_url,
                "llm_calls": model.calls}

    @app.get("/api/tickets")
    def list_tickets():
        return store.list()

    @app.get("/api/tickets/{tid}")
    def get_ticket(tid: str):
        return _get(tid)

    @app.post("/api/tickets", status_code=201)
    def create_ticket(body: NewTicket):
        """Paste in an arbitrary message and triage it immediately."""
        t = Ticket(id=f"NEW-{uuid.uuid4().hex[:6].upper()}", subject=body.subject, body=body.body,
                   channel=body.channel, sender=body.sender)
        store.add_ticket(t)
        store.set_triage(triage_ticket(t, model, settings))
        return _get(t.id)

    # sync handlers run in FastAPI's threadpool, so a slow model call doesn't block the loop
    @app.post("/api/tickets/{tid}/triage", response_model=TriageResult)
    def run_triage(tid: str):
        item = _get(tid)
        result = triage_ticket(Ticket.model_validate(item["ticket"]), model, settings)
        store.set_triage(result)
        return result

    @app.post("/api/tickets/{tid}/review")
    def review(tid: str, body: ReviewIn):
        item = _get(tid)
        if not item["triage"]:
            raise HTTPException(409, "Ticket has not been triaged yet")
        if body.action == "approve" and not body.reply.strip():
            raise HTTPException(422, "Cannot approve an empty reply - reject it or write one")
        if body.action == "approve" and rules.reply_has_placeholders(body.reply):
            raise HTTPException(422, "Reply still contains [CONFIRM: ...] placeholders - fill them in first")
        status = "approved" if body.action == "approve" else "rejected"
        return store.set_review(tid, status=status, reply=body.reply, category=body.category,
                                priority=body.priority, note=body.note)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
