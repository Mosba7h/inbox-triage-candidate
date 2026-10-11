"""Turn whatever the model said into a validated `LLMTriage`, or explain why not.

Small local models routinely: wrap JSON in ``` fences, add prose before/after,
emit trailing commas or single quotes, use near-miss labels ("Feature Request",
"critical"), give confidence as "85%" or 85, or nest the object under a key.
We repair the cheap, unambiguous cases here; anything else is reported as an
error so the pipeline can re-ask the model with the error message.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .schema import CATEGORIES, IMPACT_FACTS, PRIORITIES, LLMTriage

CATEGORY_SYNONYMS = {
    "payment": "billing", "payments": "billing", "invoice": "billing", "refund": "billing",
    "subscription": "billing", "pricing": "billing",
    "technical": "bug", "technical_issue": "bug", "defect": "bug", "error": "bug", "outage": "bug",
    "incident": "bug", "performance": "bug",
    "feature": "feature_request", "feature_requests": "feature_request", "enhancement": "feature_request",
    "request": "feature_request", "suggestion": "feature_request",
    "login": "account", "access": "account", "account_access": "account", "privacy": "account",
    "vulnerability": "security", "abuse": "security",
    "spam": "other", "general": "other", "question": "other", "inquiry": "other", "unknown": "other",
    "none": "other", "n/a": "other",
}
PRIORITY_SYNONYMS = {
    "critical": "urgent", "p0": "urgent", "emergency": "urgent", "highest": "urgent", "blocker": "urgent",
    "p1": "high", "important": "high",
    "normal": "medium", "moderate": "medium", "p2": "medium", "med": "medium",
    "p3": "low", "minor": "low", "lowest": "low", "trivial": "low",
}


@dataclass
class ParseOutcome:
    triage: LLMTriage | None
    error: str | None = None
    repaired: bool = False  # True if we had to do more than json.loads(raw)
    partial: dict | None = None  # best-effort fields even when validation failed


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    return m.group(1) if m else text


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} span, string-aware."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc, quote = 0, False, False, ""
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
            elif ch in "\"'":
                in_str, quote = True, ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _loads_lenient(blob: str) -> dict | None:
    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # single-quoted python-ish dicts: {'category': 'bug', ...}
    try:
        py = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", blob)))
        obj = ast.literal_eval(py)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass
    return None


def extract_json(raw: str) -> tuple[dict | None, bool]:
    """Returns (object, repaired)."""
    if raw is None:
        return None, False
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, False
    except (json.JSONDecodeError, TypeError):
        pass
    text = _strip_fences(raw)
    blob = _first_json_object(text)
    if blob is None:
        return None, True
    return _loads_lenient(blob), True


def _norm_label(value, allowed: tuple[str, ...], synonyms: dict[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    v = raw.replace("-", "_").replace(" ", "_")
    if v in allowed:
        return v
    if v in synonyms:
        return synonyms[v]
    if v.replace("_", " ") in synonyms:
        return synonyms[v.replace("_", " ")]
    # "bug|billing" or "billing / account" -> first recognisable part wins
    parts = [p for p in re.split(r"\s*[/|,;]\s*|\s+or\s+", raw) if p]
    if len(parts) > 1:
        for part in parts:
            hit = _norm_label(part, allowed, synonyms)
            if hit:
                return hit
    return None


def _norm_confidence(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        m = re.search(r"-?\d+(\.\d+)?", value)
        if not m:
            return {"high": 0.8, "medium": 0.6, "low": 0.3}.get(value.strip().lower())
        num = float(m.group(0))
        if "%" in value:
            num /= 100
        value = num
    if isinstance(value, (int, float)):
        v = float(value)
        if 1 < v <= 100:  # model answered on a 0-100 scale
            v /= 100
        if 0 <= v <= 1:
            return v
    return None


_TRUE = {"true", "yes", "y", "1"}


def _norm_impact(value) -> dict | None:
    """Accepts {"fact": true/"yes"/1, ...} or a list of the facts that are true.
    Unknown keys are dropped, missing ones default to False. None if unusable."""
    if isinstance(value, list):
        value = {str(v): True for v in value}
    if not isinstance(value, dict):
        return None
    norm = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in value.items()}
    out = {}
    for fact in IMPACT_FACTS:
        v = norm.get(fact, False)
        out[fact] = v if isinstance(v, bool) else str(v).strip().lower() in _TRUE
    return out


def _one_line(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()[:300] if s is not None else ""


def normalize(obj: dict) -> tuple[dict, bool]:
    """Coerce near-miss values into schema values. Returns (normalized, changed)."""
    unwrapped = False
    # unwrap {"triage": {...}} / {"result": {...}}
    if "category" not in obj and len(obj) == 1:
        inner = next(iter(obj.values()))
        if isinstance(inner, dict):
            obj, unwrapped = inner, True
    lower = {str(k).strip().lower().replace(" ", "_"): v for k, v in obj.items()}
    aliases = {"reply": "suggested_reply", "response": "suggested_reply", "draft_reply": "suggested_reply",
               "suggestedreply": "suggested_reply", "confidence_score": "confidence"}
    for a, k in aliases.items():
        if a in lower and k not in lower:
            lower[k] = lower[a]

    out = {
        "category": _norm_label(lower.get("category"), CATEGORIES, CATEGORY_SYNONYMS),
        "impact": _norm_impact(lower.get("impact")),
        "priority": _norm_label(lower.get("priority"), PRIORITIES, PRIORITY_SYNONYMS),
        "summary": _one_line(lower.get("summary")),
        "suggested_reply": str(lower.get("suggested_reply") or "").strip(),
        "confidence": _norm_confidence(lower.get("confidence")),
    }
    changed = unwrapped or (
        out["category"] != lower.get("category")
        or out["priority"] != lower.get("priority")
        or out["confidence"] != lower.get("confidence")
    )
    return out, changed


def parse_llm_output(raw: str) -> ParseOutcome:
    obj, repaired = extract_json(raw)
    if obj is None:
        return ParseOutcome(None, "Output did not contain a JSON object.", repaired=True)
    norm, changed = normalize(obj)
    try:
        return ParseOutcome(LLMTriage(**norm), repaired=repaired or changed, partial=norm)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        return ParseOutcome(None, f"Schema validation failed: {problems}", repaired=True, partial=norm)
