"""Data contracts. The LLM output is validated against `LLMTriage` before anything
downstream sees it; `TriageResult` is what the API / eval / UI consume."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["billing", "bug", "feature_request", "account", "security", "other"]
Priority = Literal["low", "medium", "high", "urgent"]

CATEGORIES: tuple[str, ...] = ("billing", "bug", "feature_request", "account", "security", "other")
PRIORITIES: tuple[str, ...] = ("low", "medium", "high", "urgent")
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}

# Where a triage came from - surfaced to the reviewer so they know how much to trust it.
Source = Literal["llm", "llm_repaired", "fallback_rules", "rule_shortcut"]


class Ticket(BaseModel):
    id: str
    received_at: str | None = None
    channel: str | None = None
    # "from" is a Python keyword
    sender: str | None = Field(default=None, alias="from")
    subject: str = ""
    body: str = ""

    model_config = {"populate_by_name": True}

    @field_validator("subject", "body", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return "" if v is None else v


class Impact(BaseModel):
    """Observable facts the model writes down before it picks a priority. Shown to the
    reviewer as context; the final priority is still the model's own judgement
    (a rule table over these facts measured worse - see README)."""

    widespread_outage: bool = False  # production down / many users or a whole team affected
    security_incident: bool = False  # vulnerability, compromise, data exposure
    customer_blocked: bool = False  # can't log in, can't use a core feature, data lost, no workaround
    financial_error: bool = False  # charged wrongly, refund dispute, payment failing
    legal_or_compliance: bool = False  # legal/regulatory request or obligation (GDPR, CCPA, contracts)
    churn_risk: bool = False  # repeat contact, threatening to cancel or leave
    sales_opportunity: bool = False  # prospect evaluating us, upgrade or expansion question
    minor_or_no_action: bool = False  # how-to, cosmetic, nice-to-have, thanks, spam, noise


IMPACT_FACTS: tuple[str, ...] = tuple(Impact.model_fields)


class LLMTriage(BaseModel):
    """Strict schema the model's (normalized) output must satisfy."""

    category: Category
    # optional so a model that forgets it doesn't cost a retry; we fall back to `priority`
    impact: Impact | None = None
    priority: Priority
    summary: str = Field(min_length=1, max_length=300)
    suggested_reply: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)


class TriageResult(BaseModel):
    id: str
    category: Category
    priority: Priority
    summary: str
    suggested_reply: str
    confidence: float
    escalate: bool
    # --- diagnostics for the reviewer / error analysis (not part of the required output) ---
    source: Source
    flags: list[str] = []
    model_confidence: float | None = None  # what the model claimed, before calibration
    model_priority: Priority | None = None  # the model's own priority, before any guard adjusted it
    impact: dict[str, bool] | None = None  # facts the model noticed (context for the reviewer)
    attempts: int = 0  # LLM calls spent on this ticket
    latency_ms: int = 0

    def public(self) -> dict:
        """The exact prediction shape requested for eval/results.json."""
        return {
            "id": self.id,
            "category": self.category,
            "priority": self.priority,
            "summary": self.summary,
            "suggested_reply": self.suggested_reply,
            "confidence": round(self.confidence, 3),
            "escalate": self.escalate,
        }
