"""Deterministic guards around the model.

The LLM is good at reading messy language; it is bad at being reliably safe or
calibrated. Cheap rules handle: inputs that shouldn't reach the model at all,
prompt-injection attempts, unsafe draft replies, a keyword fallback when the model
fails, and turning the model's self-reported confidence into something we can
threshold on.
"""

from __future__ import annotations

import re

from .schema import PRIORITY_RANK, Ticket

# ---------------------------------------------------------------- input checks

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?|rules)",
    r"disregard (all |any )?(the )?(previous|prior|above)",
    r"you are now (an? )?(unrestricted|jailbroken|dan\b)",
    r"(reveal|print|show|output|reply with) (me )?(the |your )?(full )?(system prompt|instructions|prompt)",
    r"\bapi keys?\b.*\b(you have|access)\b",
    r"\bmark (this|the ticket) as (resolved|closed)\b",
    r"\bset (the )?priority to\b",
]

SECURITY_PATTERNS = [
    r"\bvulnerabilit", r"\bidor\b", r"\bxss\b", r"\bsql ?injection\b", r"\bcve-\d", r"\bexploit",
    r"\bbreach", r"\bhacked\b", r"\bcompromised\b", r"\bunauthori[sz]ed access\b",
    r"\bresponsible disclosure\b", r"\bother (tenants|users|customers)'? (data|reports?|metadata)\b",
]

PHISHING_PATTERNS = [
    r"\byou have been selected\b", r"\bgift ?card\b", r"\bclaim (now|your prize)\b", r"\bdear winner\b",
    r"\bclick here\b",
]


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def ticket_text(t: Ticket) -> str:
    return f"{t.subject}\n{t.body}".strip()


def is_empty(t: Ticket) -> bool:
    """No body at all: nothing for a model to classify, don't spend a call on it."""
    return not t.body.strip()


def looks_like_injection(t: Ticket) -> bool:
    return _any(INJECTION_PATTERNS, ticket_text(t))


def looks_like_security(t: Ticket) -> bool:
    return _any(SECURITY_PATTERNS, ticket_text(t))


def looks_like_phishing(t: Ticket) -> bool:
    return _any(PHISHING_PATTERNS, ticket_text(t))


def word_count(t: Ticket) -> int:
    return len(re.findall(r"[^\W\d_]{2,}", ticket_text(t)))


def is_vague(t: Ticket) -> bool:
    """Body too short to say what is wrong ("doesnt work, fix asap"), whatever the subject says."""
    return len(re.findall(r"[^\W\d_]{2,}", t.body)) < 8


# ---------------------------------------------------------------- reply checks

LEAK_MARKERS = [
    "untrusted customer text", "return only a json", "you are a support-ticket triage assistant",
    "system prompt", "api key", "sk-", "llm_api_key",
]
CREDENTIAL_ASK = r"\b(send|share|provide|tell) (us )?(me )?(your )?(password|full card number|cvv|cvc)\b"
OVERPROMISE = (
    r"\b(we have|i have|we've|i've|we|i) (already )?(refunded|processed|fixed|deleted|cancell?ed|removed)\b"
    r"|\b(we|i)('ll| will) (process|issue|send) (a |the |your )?(full )?(refund|invoice)"
    r"|\byou will not be (billed|charged)\b"
    r"|\b\d+\s*(-|to)\s*\d+ (business |working )?days\b|\bwithin (the next )?\d+ (hours|days)\b"
)
# The model doesn't know our policies or product; statements like these are invented.
POLICY_CLAIM = (
    r"\b(our|the) (models?|data|servers?)( are| is)? (trained|stored|hosted|located)\b"
    r"|\bwe (store|host|keep|process) (your |all |customer )?data\b|\bin (full )?compliance with\b"
    r"|\bwe (do not|don't|never) (train|share|sell|store)\b|\bon all (of our )?plans\b"
    r"|\b(we|our product) (fully |currently )?supports?\b|\b(is|are) (only )?available on (the )?\w+ plan"
)
PLACEHOLDER = r"\[(confirm|insert|link|date)[^\]]*\]"


def reply_problems(reply: str, t: Ticket) -> list[str]:
    problems = []
    low = reply.lower()
    if any(m in low for m in LEAK_MARKERS):
        problems.append("reply_possible_leak")
    if re.search(CREDENTIAL_ASK, low):
        problems.append("reply_asks_for_credentials")
    for url in re.findall(r"https?://\S+", ticket_text(t)):
        if url.rstrip(".,)") in reply:
            problems.append("reply_echoes_customer_link")
            break
    return problems


def reply_overpromises(reply: str) -> bool:
    return bool(re.search(OVERPROMISE, reply, flags=re.I))


def reply_states_policy(reply: str) -> bool:
    return bool(re.search(POLICY_CLAIM, reply, flags=re.I))


def reply_has_placeholders(reply: str) -> bool:
    return bool(re.search(PLACEHOLDER, reply, flags=re.I))


def has_urgent_evidence(t: Ticket) -> bool:
    return _any(URGENT_HINTS, ticket_text(t))


# ---------------------------------------------------------------- templates

SAFE_REPLIES = {
    "empty": "Hi, thanks for reaching out. It looks like your message came through without any details. Could you tell us a bit more about what you need help with so we can assist?",
    "injection": "Hi, thanks for contacting support. We weren't able to identify a specific issue in your message. If you need help with your account or our product, please describe the problem and we'll be happy to assist.",
    "security": "Hi, thank you for reporting this responsibly. I've escalated it to our security team, who will contact you through a private channel to collect the details. Please don't share further specifics in this thread.",
    "phishing": "",  # no reply recommended - reviewer should mark as spam
    "generic": "Hi, thanks for contacting us. We've received your message and a member of our team will review it and get back to you shortly.",
}

# ---------------------------------------------------------------- keyword fallback

KEYWORDS = {
    "security": [r"vulnerab", r"\bidor\b", r"exploit", r"breach", r"hacked", r"compromis", r"\bxss\b"],
    "billing": [r"\bcharg", r"refund", r"invoice", r"\bbill", r"subscription", r"\bplan\b", r"pricing",
                r"\bpaid\b", r"payment", r"\bcard\b", r"tarjeta", r"\$\d", r"cancel"],
    "account": [r"log ?in", r"locked out", r"password", r"\binvite\b", r"teammate", r"workspace",
                r"delete (all )?(my |personal )?data", r"\bgdpr\b.*delet", r"\bsso\b"],
    "bug": [r"crash", r"\berror\b", r"\bbroken\b", r"doesn'?t work", r"not working", r"stopped working",
            r"\b5\d\d\b", r"\bslow\b", r"freez", r"fails?\b", r"\bbug\b", r"outage"],
    "feature_request": [r"feature", r"could you add", r"would be (great|nice|cool)", r"is there (any )?way",
                        r"do you support", r"\bimport\b", r"\bexport\b"],
}
# Objective evidence of wide or escalating impact. "urgent" pages someone, so the
# model's word alone isn't enough (pipeline downgrades urgent -> high without it).
URGENT_HINTS = [r"\bproduction\b", r"\bprod\b", r"\bsla\b", r"\boutage\b", r"\bdown for\b", r"\bis down\b",
                r"\b(all|every|entire|whole) (of )?(our )?(users|customers|team|ops|org|company)\b",
                r"\b50[0-4]\b", r"\bdata (leak|exposure|breach)\b"]
HIGH_HINTS = [r"locked out", r"charged twice", r"deadline", r"third time", r"cancel+ing", r"blocking",
              r"\bgdpr\b", r"statutory", r"vulnerab", r"refund"]


def keyword_classify(t: Ticket) -> tuple[str | None, int]:
    """Return (best category, number of keyword hits). None if nothing matched."""
    text = ticket_text(t).lower()
    scores = {cat: sum(bool(re.search(p, text)) for p in pats) for cat, pats in KEYWORDS.items()}
    # security outranks everything when it matches at all
    if scores["security"]:
        return "security", scores["security"]
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] else (None, 0)


def keyword_priority(t: Ticket) -> str:
    text = ticket_text(t).lower()
    if _any(URGENT_HINTS, text):
        return "urgent"
    if _any(HIGH_HINTS, text):
        return "high"
    return "medium" if word_count(t) >= 8 else "low"


def raise_priority(current: str, floor: str) -> str:
    return current if PRIORITY_RANK[current] >= PRIORITY_RANK[floor] else floor


# ---------------------------------------------------------------- confidence

def calibrate(model_conf: float, *, retries: int, repaired: bool, disagree: bool, short: bool) -> float:
    """Model self-confidence from a 3B model is near-constant and over-optimistic,
    so we cap it and discount it using signals we can actually observe."""
    c = min(max(model_conf, 0.05), 0.9)
    if repaired:
        c *= 0.9
    if retries:
        c *= 0.8**retries
    if disagree:
        c *= 0.7
    if short:
        c = min(c, 0.5)
    return round(c, 3)
