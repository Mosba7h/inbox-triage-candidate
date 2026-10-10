# Support Inbox Assistant

First-pass triage for a support inbox. For each inbound message it produces a
**category**, **priority**, one-line **summary**, an editable **suggested reply**, a
**confidence** score and an **escalate** flag. A human reviews everything in a small
review-queue UI; nothing is ever sent automatically.

Model: `llama3.2:3b` via any OpenAI-compatible endpoint (Ollama by default).

```
tickets.json ─► pipeline ─► guards ─► calibrated confidence ─► review queue (approve / edit / reject)
                  │  ▲
                  ▼  │ retry with validation error
               LLM (unreliable)
```

---

## Quick start

Prereqs: Python 3.10+, `make`, and an OpenAI-compatible endpoint, e.g.

```bash
ollama pull llama3.2:3b          # one-time, ~2 GB
```

```bash
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.2:3b
export LLM_API_KEY=ollama        # Ollama ignores it; any non-empty string works

make eval    # triage all 30 tickets with the live model -> eval/results.json
make run     # API + review UI at http://127.0.0.1:8000
make test    # unit tests, no model needed
```

The first `make` target creates `.venv/` and installs `requirements.txt` into it, so
each of these works on a fresh clone. (Without `make`: `pip install -r requirements.txt`,
then `python -m triage.evaluate` / `python -m triage.api` / `pytest -q`.)

### Configuration

Only the three `LLM_*` variables are meant to be set. Everything else has a default in
[`triage/config.py`](triage/config.py).

| variable | default | purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `llama3.2:3b` | model name |
| `LLM_API_KEY` | `ollama` | API key (never committed; see `.env.example`) |
| `TRIAGE_CONCURRENCY` | `4` | parallel tickets during the eval |
| `TRIAGE_LLM_TIMEOUT_S` | `120` | per-request timeout |
| `TRIAGE_LLM_MAX_RETRIES` | `2` | re-asks after an invalid answer or a transport error |
| `TRIAGE_LLM_TEMPERATURE` | `0` | deterministic-ish output for a reproducible eval |
| `TRIAGE_ESCALATE_BELOW` | `0.6` | confidence threshold for escalation |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | where `make run` listens |

### Where the results are

`make eval` writes three files:

- `eval/results.json`: the requested shape (`metrics` plus 30 `predictions`). `metrics` has a few extra keys beyond the two required ones.
- `eval/report.md`: confusion matrices, every labeled ticket with its misses marked, and the unlabeled tickets for spot-checking.
- `eval/details.json`: each prediction plus diagnostics (`source`, `flags`, `attempts`, `latency_ms`, raw model confidence).

The eval exits non-zero if the model never answered: rule-based fallbacks alone don't count as an eval.

---

## How it works

### 1. Reliable AI layer ([`triage/pipeline.py`](triage/pipeline.py), [`triage/parsing.py`](triage/parsing.py))

The model is treated as an untrusted, unreliable component:

1. **Strict schema.** The output must validate against a Pydantic model (`LLMTriage`) with `Literal` enums, a bounded confidence and non-empty strings. Nothing downstream ever sees unvalidated model output.
2. **Cheap, unambiguous repairs first.** The parser strips ```` ``` ```` fences and surrounding prose, extracts the first balanced `{...}`, and fixes trailing commas and single-quoted dicts. It unwraps `{"triage": {...}}` and maps near-miss labels (`"Feature Request"`, `"critical"`, `"P2"`, `"85%"`, `85`) onto the schema. Anything it can't fix unambiguously is rejected, not guessed.
3. **Retry with the error.** On a validation failure the model is shown its own answer plus the exact validation error and asked again (up to 2 retries). Transport errors (timeouts, connection refused, 5xx) are retried with backoff. JSON mode (`response_format=json_object`) is requested, and dropped automatically if the server doesn't support it.
4. **Salvage, then fall back.** If the model never produces a valid object, valid fields from its best partial answer are kept and the rest come from a keyword classifier. A fallback is always escalated and gets low confidence. The pipeline never raises for a single ticket, so the eval always produces 30 predictions.
5. **Bad input.** A ticket with an empty body is answered by a rule (ask the customer for details) without spending a model call. Ticket text is JSON-encoded inside `<ticket>` tags with `<` escaped, so a message can't close the tag or impersonate a new turn.

### 2. Safety guards ([`triage/rules.py`](triage/rules.py))

- **Prompt injection** ("ignore previous instructions…"): the ticket can't choose its own priority or status. It's classified as `security/medium` and escalated, and the model's reply is discarded and replaced with a neutral template, because a reply written while being manipulated isn't trustworthy.
- **Security reports**: security keywords (IDOR, vulnerability, breach…) force `security` with a priority floor of `high` and always escalate. The reply must route to a private channel, not ask for details in the thread.
- **Phishing/spam**: forced to `other/low` with *no reply recommended*. The model's draft is dropped so we never echo a malicious link.
- **Reply checks**: a draft that may leak the prompt or keys, asks for a password or card number, or repeats a link from the ticket is replaced with a safe template and flagged. Drafts that claim an action already happened ("we've refunded…") are flagged for the reviewer.

### 3. Confidence and escalation

A 3B model's self-reported confidence is close to constant (it says 0.8–0.95 for almost
everything), so it can't be used as a threshold on its own. The final score is the model's
confidence **capped at 0.9**, then discounted by signals we can actually observe:
repaired output (×0.9), each retry (×0.8), disagreement with an independent keyword
classifier (×0.7), and very short or vague messages (cap 0.5).

`escalate = confidence < 0.6 OR category == security OR priority == urgent OR fallback used OR a guard fired`

### 4. Review queue ([`triage/api.py`](triage/api.py), [`triage/static/index.html`](triage/static/index.html))

FastAPI plus a single static HTML page with no build step. Escalated tickets sort to the top,
then by priority. The reviewer sees the original message, the triage, confidence (calibrated
and raw), and every guard flag. They can change the category or priority, edit the reply,
then approve or reject. Approving an empty reply is refused. Each decision records whether
the reply, category or priority was corrected, since that's the data you'd want for improving
the system. You can also paste a new message to triage it live.

| method | path | |
|---|---|---|
| GET | `/api/tickets` | queue with triage and review state |
| POST | `/api/tickets/{id}/triage` | (re)run triage for one ticket |
| POST | `/api/tickets/{id}/review` | `{action: approve\|reject, reply, category, priority, note}` |
| POST | `/api/tickets` | triage an ad-hoc message `{subject, body}` |
| GET | `/api/health` | model, endpoint, number of calls made |

State is a JSON file (`data/review_state.json`, gitignored). On first boot the queue is
seeded from the last `make eval` run, so the UI isn't empty.

---

## Results

All numbers are `llama3.2:3b` on Ollama, temperature 0, scored on the **16 labeled tickets**.
With n=16, one ticket is about 6 percentage points, so read these as directional.

| | category accuracy | priority (exact) | priority (within 1 level) | escalation rate |
|---|---|---|---|---|
| **v1**: first prompt, no urgency guard | 0.750 | 0.562 | 0.938 | 0.53 |
| **final** (3 runs) | **0.938** (0.938 / 0.938 / 0.938) | **0.562** (0.562 / 0.562 / 0.562) | 0.938 | 0.37–0.47 |

- Category and priority labels were **identical across the 3 final runs**. Only confidence varied, because one run hit 5 malformed answers that needed a retry, which lowers confidence by design.
- **Reliability in practice:** across runs, 1–5 of 29 model answers per run failed validation (invalid JSON or out-of-schema values). All were fixed by the retry-with-error loop; the keyword fallback was never needed. The model is called 30–38 times per eval; T-030 has an empty body and is answered by a rule.
- Of the labeled tickets **not** escalated (13/16), category accuracy was 100%. In this run the one category error (T-019) was escalated.
- Wall time is about 3–4 minutes on a laptop GPU with 4 concurrent requests.

Final confusion matrices (from `eval/report.md`):

| category: gold \ pred | billing | bug | feature_request | account | security | other |
|---|---|---|---|---|---|---|
| billing | 3 | | | | | |
| bug | | 5 | | | **1** | |
| feature_request | | | 3 | | | |
| account | | | | 3 | | |
| other | | | | | | 1 |

| priority: gold \ pred | low | medium | high | urgent |
|---|---|---|---|---|
| low | | **3** | | |
| medium | | 3 | **3** | |
| high | **1** | | 4 | |
| urgent | | | | 2 |

### What changed between v1 and the final version, and why

v1 failed in three systematic ways. None of them was a parsing problem:

1. **"security" over-applied** (T-006 outage, T-019 webhook HMAC, T-007 data-policy question): anything that mentioned APIs, auth or data turned into security. *Fix:* explicit boundary sentences in the category definitions (outages and failing integrations → bug; privacy-policy questions → other). Category accuracy went from 0.75 to 0.94.
2. **Urgency inflation** from tone (T-002 "blocking my deadline", T-018 "fix asap", T-025 "sort this out today" all became urgent). *Fix:* the prompt says urgent is rare, and a deterministic guard downgrades urgent → high unless the text contains objective impact evidence (production, outage, SLA, 5xx, "all our users"…). Urgent pages someone, so it should need evidence beyond the model's opinion.
3. **Replies invented facts:** "SAML is on all plans, SCIM is enterprise-only" (T-024), "our models are trained on anonymized data" (T-007), "I've cancelled your subscription" (T-017). For a support tool this is the most dangerous failure, because a confident wrong policy statement can become a legal commitment. *Fix:* the prompt forbids stating product, policy, date or account facts and asks for `[CONFIRM: …]` placeholders instead. The UI and API refuse to approve a reply that still contains a placeholder. Rule-based flags catch overpromises ("we'll process a refund… in 3-5 business days") and invented policy statements, and the policy flag forces escalation.

### Error analysis (final version)

**Category (1 miss out of 16).** T-019 (webhook signature verification failing since this morning) → `security`, labeled `bug`. This one is genuinely debatable: an unannounced signing-secret rotation *is* security-adjacent. It's escalated either way, so the routing outcome is acceptable. I'd rather over-flag security than miss a real compromise.

**Priority (7 misses, all exactly one level off). This is the weak spot.**
- *Systematic +1 on low/medium tickets* (T-003 dark mode, T-009 invoice copy, T-013 how-to → medium; T-017, T-022, T-024 → high). The 3B model treats almost any request as worth at least medium, and the rubric text alone doesn't move it.
- *One real under-prioritisation: T-021 (GDPR deletion request) → `low`, labeled `high`.* This is the miss I'd fix first, because it's a compliance risk: statutory deadlines don't care that the message was polite. The "urgent is rare" instruction made the model more conservative across the board, and this ticket paid for it (v1 had it as urgent). **I deliberately did not add a "GDPR → high" rule after seeing this labeled ticket fail.** That would raise the score by tuning to the test set. In production I'd add a legal/regulatory-request priority floor and validate it on fresh tickets.
- The labels themselves are debatable on a few of these (T-022: is a broken iOS push after an update medium or high?). `priority_within_one_level` is 0.94, and that's the more meaningful number for a review queue where a human adjusts anyway.

**Confidence is weak.** The model reports 0.8–0.95 for almost everything, and on this run the mean confidence of category errors was *higher* than for correct ones (n=1, so not meaningful either way). What carries the escalation decision is the guard layer: injection, security, urgent, phishing, empty or vague input, rule disagreement, invented-policy replies. Treat `confidence` as "how clean was this run" more than as a calibrated probability.

**Unlabeled tickets, hand-checked:**
- T-008 (prompt injection) → neutralised: `security/medium`, safe template reply, escalated. The model's own answer to it was also malformed.
- T-014 (IDOR report) → `security/urgent`, escalated. The reply routes to a private channel.
- T-015 (gift-card phishing) → `other/low`, no reply recommended, the link is never echoed.
- T-004 (gibberish) / T-023 ("nvm, thanks") / T-030 (empty) → `other/low` with low confidence. T-030 doesn't spend a model call.
- T-010 (Spanish, card update error) → `billing/high`. The reply came back **in English**, despite the instruction to reply in the customer's language. That's a known small-model weakness, and nothing flags it yet.
- T-018 ("doesnt work, fix asap") → `bug/high`. Urgent was downgraded, confidence is capped at 0.5 because the body is too vague, it's escalated, and the reply asks what's broken.
- T-027 (export all my data) → `account`. Reasonable, but `feature_request` is also arguable. The keyword rules disagree, so confidence is lowered.

### Next steps (in order)

1. **Priority:** use an ordinal decision procedure instead of one free choice. Ask the model for observable facts (*is the customer blocked? how many users? money taken? legal deadline? security exposure?*) as booleans in the schema, then compute priority with a transparent rule table. It's more debuggable, and it removes the model's "everything is medium+" bias. Add a legal/regulatory floor and validate on fresh data.
2. **Confidence:** self-consistency. Sample 3–5 answers at temperature ~0.7 and use agreement on category/priority as the confidence. It costs 3–5× the calls, but it's the cheapest calibrated signal for a model this size.
3. **Bigger eval set:** 16 labels can't separate a 5-point improvement from noise. Use reviewer corrections (already recorded on every approve/reject: `edited_reply`, `corrected_category`, `corrected_priority`) as a growing labeled set, and keep a held-out split so prompt changes aren't tuned on the test set.
4. **Reply language check:** detect the ticket language and flag or regenerate when the reply language differs.
5. **Retrieval for replies:** ground drafts in real help-centre and policy docs, so the model can answer "where is data stored?" with a citation instead of a placeholder.
6. **Production basics:** a database instead of the JSON file, auth, a background job queue for triage, multi-reviewer locking, and metrics on reviewer edit rate per category.

---

## Trade-offs and corners cut

- **JSON file instead of a database** for review state: single process, no auth, no multi-reviewer locking. Fine for a demo, not for production.
- **No authentication** on the API or UI.
- **Keyword rules are hand-written** and English-centric (the Spanish ticket only gets the LLM's judgment). They are a safety net and a disagreement signal, not a classifier anyone should rely on.
- **The UI triages tickets one by one** (sequential requests from the browser), not with a background job queue. That's simpler, but the tab has to stay open.
- **One model call per ticket.** Self-consistency (sample 3×, use agreement as confidence) would give a much better confidence signal but triples cost and latency. See next steps.
- **Rubric informed by reading the data.** I wrote the category and priority definitions in the system prompt after reading all 30 tickets and the labels, so the reported accuracy is somewhat optimistic. To limit this, the few-shot examples are synthetic (not taken from the dataset), no ticket IDs or ticket-specific rules appear in the prompt, and guards key on general patterns. With only 16 labeled tickets, one ticket is about 6 percentage points; treat the numbers as directional.
