"""Ticket -> TriageResult.

Order of operations:
  1. Short-circuit inputs that need no model (empty body).
  2. Ask the model; parse + normalize + validate. On failure, re-ask with the
     validation error appended (up to N retries).
  3. If the model never produced a valid object: salvage valid fields from the
     best partial answer, fill the rest from the keyword fallback.
  4. Apply guards (injection, security floor, phishing, unsafe reply).
  5. Calibrate confidence and decide escalation.
Every path returns a TriageResult - the pipeline never raises for a single ticket.
"""

from __future__ import annotations

import logging
import time

from . import rules
from .config import Settings
from .llm import ChatModel, LLMError
from .parsing import ParseOutcome, parse_llm_output
from .prompts import build_messages, repair_message
from .schema import CATEGORIES, PRIORITIES, Ticket, TriageResult

log = logging.getLogger(__name__)


def _summary_fallback(t: Ticket) -> str:
    text = t.subject.strip() or t.body.strip()
    text = " ".join(text.split())
    return (text[:117] + "...") if len(text) > 120 else (text or "(no content)")


def _ask_model(model: ChatModel, t: Ticket, max_retries: int) -> tuple[ParseOutcome | None, int, list[str]]:
    """Returns (last outcome, attempts used, flags)."""
    messages = build_messages(t)
    outcome: ParseOutcome | None = None
    best_partial: ParseOutcome | None = None
    flags: list[str] = []
    attempts = 0
    for attempt in range(max_retries + 1):
        attempts += 1
        try:
            raw = model.complete(messages)
        except LLMError as e:
            log.warning("%s: LLM call failed (attempt %d): %s", t.id, attempts, e)
            flags.append("llm_transport_error")
            time.sleep(min(2**attempt, 8) * 0.5)
            continue
        outcome = parse_llm_output(raw)
        if outcome.triage is not None:
            return outcome, attempts, flags
        log.info("%s: invalid model output (attempt %d): %s", t.id, attempts, outcome.error)
        flags.append("llm_invalid_output")
        if outcome.partial and (best_partial is None or _partial_score(outcome) > _partial_score(best_partial)):
            best_partial = outcome
        # conversational repair: show the model its own answer and what was wrong
        messages = messages + [{"role": "assistant", "content": raw[:2000]}, repair_message(outcome.error or "")]
    return best_partial, attempts, flags


def _partial_score(o: ParseOutcome) -> int:
    p = o.partial or {}
    return sum(1 for k in ("category", "priority", "summary", "suggested_reply") if p.get(k))


def triage_ticket(t: Ticket, model: ChatModel, settings: Settings) -> TriageResult:
    start = time.perf_counter()
    flags: list[str] = []

    # 1. nothing to classify -------------------------------------------------
    if rules.is_empty(t):
        return TriageResult(
            id=t.id, category="other", priority="low",
            summary=f"Empty message (subject: {t.subject.strip() or 'none'}); needs details from customer.",
            suggested_reply=rules.SAFE_REPLIES["empty"],
            confidence=0.4, model_confidence=None, escalate=True,
            source="rule_shortcut", flags=["empty_body"], attempts=0,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    # 2. ask the model --------------------------------------------------------
    outcome, attempts, call_flags = _ask_model(model, t, settings.llm_max_retries)
    flags += sorted(set(call_flags))
    retries = attempts - 1

    kw_cat, kw_hits = rules.keyword_classify(t)
    if outcome is not None and outcome.triage is not None:
        tr = outcome.triage
        category, priority = tr.category, tr.priority
        summary, reply, model_conf = tr.summary, tr.suggested_reply, tr.confidence
        source = "llm_repaired" if (outcome.repaired or retries) else "llm"
    else:
        # 3. salvage what we can, fill the rest from keyword rules -------------
        p = (outcome.partial if outcome else None) or {}
        category = p.get("category") if p.get("category") in CATEGORIES else (kw_cat or "other")
        priority = p.get("priority") if p.get("priority") in PRIORITIES else rules.keyword_priority(t)
        summary = p.get("summary") or _summary_fallback(t)
        reply = p.get("suggested_reply") or rules.SAFE_REPLIES["generic"]
        model_conf = 0.2
        source = "fallback_rules"
        flags.append("fallback_used")

    # 4. guards -----------------------------------------------------------------
    if rules.looks_like_injection(t):
        # Don't let the ticket pick its own priority/status, and never ship a
        # reply the model wrote while being manipulated.
        flags.append("prompt_injection_suspected")
        category, priority = "security", "medium"
        reply = rules.SAFE_REPLIES["injection"]
        summary = "Message attempts to manipulate the assistant (prompt injection); no genuine support request found."
    elif rules.looks_like_phishing(t):
        flags += ["suspected_phishing", "no_reply_recommended"]
        category, priority = "other", "low"
        reply = rules.SAFE_REPLIES["phishing"]
    elif rules.looks_like_security(t) or category == "security":
        if category != "security":
            flags.append("security_keywords_override")
            category = "security"
        priority = rules.raise_priority(priority, "high")
        reply = rules.SAFE_REPLIES["security"] if source == "fallback_rules" else reply

    if priority == "urgent" and category != "security" and not rules.has_urgent_evidence(t):
        # Small models inflate urgency from tone ("ASAP", caps). Urgent needs wide/escalating impact.
        priority = "high"
        flags.append("urgent_downgraded_no_impact_evidence")

    problems = rules.reply_problems(reply, t) if reply else []
    if problems:
        flags += problems + ["reply_replaced"]
        reply = rules.SAFE_REPLIES["generic"]
    if reply and rules.reply_overpromises(reply):
        flags.append("reply_overpromises_check_before_sending")
    if reply and rules.reply_states_policy(reply):
        flags.append("reply_states_unverified_policy")
    if reply and rules.reply_has_placeholders(reply):
        flags.append("reply_has_placeholders_to_fill")

    # 5. confidence + escalation ----------------------------------------------
    disagree = bool(kw_cat and kw_hits >= 2 and kw_cat != category and source != "fallback_rules")
    if disagree:
        flags.append(f"keyword_rules_suggest_{kw_cat}")
    confidence = rules.calibrate(
        model_conf, retries=retries, repaired=source == "llm_repaired",
        disagree=disagree, short=rules.is_vague(t),
    )
    escalate = (
        confidence < settings.escalate_below
        or category == "security"
        or priority == "urgent"
        or source == "fallback_rules"
        or any(f in flags for f in ("prompt_injection_suspected", "reply_replaced", "reply_states_unverified_policy"))
    )

    return TriageResult(
        id=t.id, category=category, priority=priority, summary=summary, suggested_reply=reply,
        confidence=confidence, escalate=escalate, source=source, flags=flags,
        model_confidence=model_conf if source != "fallback_rules" else None,
        attempts=attempts, latency_ms=int((time.perf_counter() - start) * 1000),
    )
