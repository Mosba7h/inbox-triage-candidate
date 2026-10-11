# Support Inbox Assistant

First-pass triage for a support inbox. For each inbound message it produces a
**category**, **priority**, one-line **summary**, an editable **suggested reply**, a
**confidence** score and an **escalate** flag. A human reviews everything in a small
review-queue UI; nothing is ever sent automatically.

Model: `llama3.2:3b` via any OpenAI-compatible endpoint (Ollama by default).

```
ticket ─► empty? ─► LLM (validate · repair · retry · fallback) ─► guards ─► confidence ─► review queue
                     unreliable component, never trusted directly   │                       approve / edit / reject
                                                                     └ injection · security · phishing · legal
                                                                       reply checks · reply language
```

---

## Quick start

Prereqs: Python 3.10+, a POSIX `sh`, and an OpenAI-compatible endpoint, e.g.

```bash
ollama pull llama3.2:3b          # one-time, ~2 GB
```

```bash
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.2:3b
export LLM_API_KEY=ollama        # Ollama ignores it; any string works

sh run.sh eval       # triage all 30 tickets with the live model -> eval/results.json
sh run.sh run        # API + review UI at http://127.0.0.1:8000
sh run.sh test       # unit tests, no model needed
sh run.sh dev-eval   # same eval on the synthetic dev set -> eval/dev/
```

`run.sh` creates `.venv/` and installs `requirements.txt` on first use, so every command works
on a fresh clone. It needs nothing but `sh` and Python. If the machine can't create a venv
(e.g. Debian without `python3-venv`), it installs into the interpreter's user site instead.
Both paths are tested in clean containers. `make run|eval|test` are thin wrappers around the same script.

### Configuration

Only the three `LLM_*` variables need to be set. Everything else has a default in
[`triage/config.py`](triage/config.py).

| variable | default | purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `llama3.2:3b` | model name |
| `LLM_API_KEY` | `ollama` if empty | API key; the OpenAI SDK rejects an empty key. Never committed; see `.env.example` |
| `TRIAGE_CONCURRENCY` | `4` | parallel tickets during the eval (kept low for a shared endpoint) |
| `TRIAGE_LLM_TIMEOUT_S` | `120` | per-request timeout |
| `TRIAGE_LLM_MAX_RETRIES` | `2` | re-asks after an invalid answer or a transport error |
| `TRIAGE_LLM_TEMPERATURE` | `0` | deterministic output for a reproducible eval |
| `TRIAGE_ESCALATE_BELOW` | `0.6` | confidence threshold for escalation |
| `TRIAGE_TICKETS_PATH` / `TRIAGE_LABELS_PATH` / `TRIAGE_RESULTS_PATH` / `TRIAGE_STATE_PATH` | `data/…`, `eval/results.json` | file locations |
| `PORT` / `HOST` | `8000` / `127.0.0.1` | where the app listens |

### Where the results are

`sh run.sh eval` makes the model calls and writes three files in one invocation:

- `eval/results.json`: the requested shape (`metrics` plus one prediction per ticket, all 30). `metrics` has extra keys beyond the two required ones.
- `eval/report.md`: confusion matrices, a confidence-threshold sweep, every labeled ticket with its misses marked, and the unlabeled tickets for spot-checking.
- `eval/details.json`: each prediction plus diagnostics (`source`, `flags`, `attempts`, `latency_ms`, the model's raw confidence and priority, impact signals).

The eval exits non-zero if the model never answered, because rule-based fallbacks alone aren't an eval.
Earlier runs, including the experiments below, are archived in `eval/runs/`.

---

## How it works

### 1. Reliable AI layer ([`pipeline.py`](triage/pipeline.py), [`parsing.py`](triage/parsing.py))

The model is treated as an untrusted, unreliable component:

1. **Strict schema.** The output must validate against a Pydantic model (`LLMTriage`) with `Literal` enums, a bounded confidence and non-empty strings. Nothing downstream sees unvalidated model output.
2. **Cheap, unambiguous repairs first.** The parser strips code fences and surrounding prose, extracts the first balanced `{...}`, and fixes trailing commas and single-quoted dicts. It unwraps `{"triage": {...}}` and maps near-miss values (`"Feature Request"`, `"critical"`, `"P2"`, `"85%"`, `85`) onto the schema. Anything it can't fix unambiguously is rejected, not guessed.
3. **Retry with the error.** On a validation failure the model sees its own answer plus the exact validation error and is asked again (up to 2 retries). Transport errors (timeouts, connection refused, 5xx) are retried with backoff. JSON mode (`response_format=json_object`) is requested, and dropped automatically if the server rejects it.
4. **Salvage, then fall back.** If the model never produces a valid object, valid fields from its best partial answer are kept and the rest come from a keyword classifier. A fallback always escalates with low confidence. The pipeline never raises for a single ticket, so the eval always produces 30 predictions.
5. **Bad input.** A ticket with an empty body gets a rule-based answer (ask the customer for details) without a model call. Ticket text is JSON-encoded inside `<ticket>` tags with `<` escaped, so a message can't close the tag or impersonate a new turn.

In practice: 1–5 of 29 model answers per run failed validation. All were fixed by the retry loop; the keyword fallback has never been needed against the live model.

**Impact signals.** Besides the five required fields, the model fills an `impact` object
(`customer_blocked`, `financial_error`, `legal_or_compliance`, `churn_risk`, …) *before* it
writes the priority. The UI shows these signals to the reviewer next to the priority. I also
tried computing priority from them with a rule table. It measured worse, so the final priority
is the model's own judgement (see [What I tried that didn't work](#what-i-tried-that-didnt-work)).

### 2. Safety guards ([`rules.py`](triage/rules.py))

Deterministic rules around the model, for the things a 3B model can't be trusted with:

- **Prompt injection** ("ignore previous instructions…"): the ticket can't choose its own priority or status. It's set to `security/medium` and escalated, and the model's reply is discarded and replaced with a neutral template, because a reply written while being manipulated isn't trustworthy.
- **Security reports** (IDOR, vulnerability, breach…): forced to `security` with a priority floor of `high`, always escalated, and the reply routes to a private channel.
- **Phishing/spam**: forced to `other/low` with *no reply recommended*. The draft is dropped so a malicious link is never echoed.
- **Legal/compliance requests** (GDPR/CCPA, "statutory", "delete all my data", lawyers…): always escalated. This guard changes **routing only**, never category or priority, so it fixes the product risk without moving the metric.
- **Urgency needs evidence**: the model inflates urgency from tone ("ASAP", caps). `urgent` is downgraded to `high` unless the text shows wide impact (production, outage, SLA, 5xx, "all our users"…).
- **Reply checks** on the final draft:
  - **Replaced** with a safe template: a possible prompt leak or secret-shaped string, a password/CVV request, or a link copied from the ticket.
  - **Flagged and escalated**: unverified policy statements ("we don't train on customer data"), unverified status claims ("the outage has been resolved"), and requests for card or payment details (in English, Spanish, French, German or Portuguese).
  - **Flagged only**: commitments ("we've cancelled…", "within 3-5 business days").
  - **Placeholders** (`[CONFIRM: …]`, `[REPLACE WITH …]`) are flagged, and the API refuses to approve a reply that still contains one. The prompt tells the model to use placeholders instead of inventing facts about our product, policies or the customer's account.
- **Reply language**: a stopword-based detector compares the ticket's language with the reply's. On a mismatch the model gets **one** targeted follow-up. The new reply is used only if it really is in that language; otherwise the ticket is flagged and escalated. The repaired reply then goes through the same reply checks. This never runs on safety-template replies.

### 3. Confidence and escalation

The model's self-reported confidence is near-constant (0.8–0.95 for almost everything). The
score is that value **capped at 0.9**, then discounted by observable signals: repaired output
(×0.9), each retry (×0.8), disagreement with the keyword classifier (×0.7), and a vague body of
fewer than 8 words (capped at 0.5).

```
escalate = confidence < 0.6  OR  category == security  OR  priority == urgent  OR  fallback used
           OR a guard fired (injection, replaced reply, unverified policy/status, payment-details ask,
                             legal request, reply in the wrong language)
```

**Honest caveat:** measured on both eval sets, confidence **does not separate right from wrong
answers** (details in Results). Escalation is driven by the guards, not by the number.

### 4. Review queue ([`api.py`](triage/api.py), [`static/index.html`](triage/static/index.html))

FastAPI plus one static HTML page with no build step. The list has status tabs (pending,
escalated, approved, rejected, untriaged), category/priority filters, search (subject, body,
summary, flags) and sorting (escalated-first, priority, confidence, date). The detail view shows
the original message, the triage, the impact signals and the model's own priority if a guard
changed it, confidence (calibrated and raw) and every flag. The reviewer can change the
category or priority, edit the reply, then approve or reject. Approving an empty reply, or one
with unfilled placeholders, is refused. Each decision records whether the reply, category or
priority was corrected; that's the data for improving the system. You can also paste in a new
message and triage it live.

| method | path | |
|---|---|---|
| GET | `/api/tickets` | queue with triage and review state |
| POST | `/api/tickets/{id}/triage` | (re)run triage for one ticket |
| POST | `/api/tickets/{id}/review` | `{action: approve\|reject, reply, category, priority, note}` |
| POST | `/api/tickets` | triage an ad-hoc message `{subject, body}` |
| GET | `/api/health` | model, endpoint, number of calls made |

State is a JSON file (`data/review_state.json`, gitignored). On first boot the queue is seeded
from the last eval run, so the UI isn't empty.

---

## Evaluation method (and where it isn't clean)

- **Test set:** the 16 labeled tickets in `data/labels.json`, scored on all metrics. With n=16, one ticket is about 6 percentage points.
- **Dev set:** 32 synthetic tickets in `data/dev/`, written for tuning and **not taken from `tickets.json`**. They're labeled against a priority policy written in `data/dev/labels.json` *before* any model run. They cover the same kinds of mess: multilingual, compliance, spam, vague, churn.

How the two rounds of changes were made:

1. **Round 1 (v1 → round 1): tuned on the test set.** I read the 16 labeled failures and changed the prompt and guards in response (category boundaries, the urgency guard, the no-invented-facts rule). That contaminates the test set. Round-1 numbers are optimistic and shown only as history.
2. **Round 2 (round 1 → final): tuned on the dev set only.** Every alternative was compared on dev, the configuration was frozen, and then the test set was run (3 times; see below for the one ticket that varies). It still isn't pristine: the *ideas* for round 2 came partly from round-1 test errors (priority bias, T-010, T-021), and I wrote the dev labels after having seen the test labels.
3. Three reply-safety detectors (status claims, payment-details requests, `[REPLACE …]` placeholders) were added after reading the final test **replies**. They only add flags and escalation; they don't change any scored category or priority.

---

## Results

`llama3.2:3b`, temperature 0. Test = the 16 labeled tickets.

| version (test set) | category accuracy | priority (exact) | priority (±1 level) | both correct |
|---|---|---|---|---|
| v1: first prompt | 0.750 | 0.562 | 0.938 | – |
| round 1: tuned on test (3 runs, identical) | 0.938 | 0.562 | 0.938 | 0.500 |
| **final**: tuned on dev (3 runs) | **0.938** | **0.750 / 0.750 / 0.688** | **1.000** | **0.688 / 0.688 / 0.625** |

**Run-to-run variance:** temperature 0 isn't bit-exact on Ollama when requests are batched
concurrently. In the final config, one ticket (T-022, iOS push notifications) flipped between
`medium` and `high` in 1 of 3 runs. That third run was the fresh-clone dry run in a clean container
(`eval/runs/round2_freshclone_dryrun_*`). Category labels were identical in all runs.
`eval/results.json` is from a 0.750 run; expect 0.69–0.75 for priority on a re-run.

| dev set (32 tickets) | category | priority | ±1 | both correct |
|---|---|---|---|---|
| round-1 code | 0.812 | 0.750 | 0.938 | 20 / 32 |
| + impact signals, priority **from a rule table** | 0.750 | 0.719 | 1.000 | 18 / 32 |
| + impact signals, model's own priority | 0.750 | 0.875 | 1.000 | 21 / 32 |
| **final** (+ category boundary sentences) | **0.812** | **0.812** | 1.000 | **22 / 32** |

Final run on the test set: 32 model calls for 30 tickets (T-030 is answered by a rule; there were
2 validation retries and 1 language follow-up). It took 7 minutes on a 4 GB laptop GPU running on
battery, and about 3–4 minutes when plugged in.

**Is confidence useful?** Confidence alone as the escalation rule (from `eval/report.md`):

| threshold | test: queue escalated | test: errors caught | dev: queue escalated | dev: errors caught |
|---|---|---|---|---|
| < 0.6 (default) | 0% | 0 of 5 | 16% | 1 of 10 |
| < 0.8 | 19% | 0 of 5 | 34% | 4 of 10 |

No. The full escalation rule (guards included) escalates 43% of the test queue and catches 1
of the 5 wrong test tickets (T-019, via the security guard).

### Error analysis (final, test set)

**Category, 1 miss.** T-019 (webhook signature verification failing in production) → `security`,
labeled `bug`. This is debatable: an unannounced signing-secret rotation is security-adjacent.
It's escalated either way, and I'd rather over-flag security than miss a compromise.

**Priority, 4 misses (5 in the run where T-022 went to `high`), all one level off:**
- T-009 (invoice copy with VAT number) and T-013 (how to invite a teammate): `low` → `medium`. The model treats any request that needs action as at least medium.
- T-011 (bulk CSV import, "would save me hours"): `medium` → `low`. The model read it as a nice-to-have; the label counts the stated business impact.
- T-017 (cancel subscription): `medium` → `high`. The model said `urgent` ("effective immediately"); the urgency guard brought it down one level, but not to medium.

The pattern is that **low vs medium is where the model and the labels disagree**, in both
directions. These are judgement calls a human labeler would also debate, which is why ±1 is
100%.

**Fixed since round 1:**
- T-021 (GDPR deletion request): `low` and not escalated → `high` and escalated (legal guard).
- T-010 (Spanish): the reply now comes back in Spanish (language follow-up).

**Unlabeled tickets, hand-checked:**
- T-008 (prompt injection): neutralised, templated reply, escalated.
- T-014 (IDOR report): `security/high`, private-channel reply, escalated.
- T-015 (gift-card phishing): `other/low`, no reply recommended.
- T-004 / T-023 / T-030 (gibberish, "nvm", empty): `other/low`, low confidence. T-030 spends no model call.
- T-010: the Spanish reply asks the customer to "confirm your card details". That's caught by the payment-details flag and escalated; without it this would be a PCI problem in a one-click approve.
- T-029 ("any update on the outage?"): **over-prioritised as `urgent`**, because the urgency guard is keyword-based and this follow-up mentions the 503s. Its reply also claimed "the outage has been resolved", which the model can't know; that's caught by the status-claim flag.
- T-018 ("doesnt work, fix asap"): `bug/high`, confidence capped (vague), escalated, and the reply asks what's broken.

### What I tried that didn't work

- **Priority from a rule table over extracted facts** (dev: priority 0.719 vs 0.875). The model is worse at the individual yes/no facts than at the overall judgement. It left `customer_blocked` false for "I've lost two days of notes" and "can't upload any files", and marked real bugs `minor_or_no_action`. On the 8 dev tickets where the rule table and the model disagreed, the model was right 7 times and the table once. I removed the code path but kept the signals: writing them out before the priority coincided with the best dev priority score, and they're useful context in the UI.
- **Self-consistency for confidence** (3 samples, temperature 0.8, agreement as confidence; 3× the calls). On dev it caught 1 of 11 errors at the 0.6 threshold (baseline: 0) and 3 of 11 at 0.8 (baseline: 4). The model's errors are **systematic, not random**: the samples agree on the same wrong answer. It was removed rather than kept as an unused option.

Both experiments' raw outputs are in `eval/runs/dev/`.

---

## Trade-offs and corners cut

- **Confidence is not calibrated**, and I've said so rather than tuning the number to look useful. The guards carry escalation.
- **The guards are hand-written regexes.** Injection and urgency detection are English-centric and bypassable; they're a last line of defence, not the only one. The payment-details check knows 5 languages. The urgency guard over-fires on follow-ups that mention an outage (T-029).
- **Small eval sets** (16 test, 32 dev), and both were labeled by people who had seen the data. Directional only.
- **JSON file instead of a database**, no auth, no multi-reviewer locking.
- **The UI triages one ticket per request from the browser**, with no background job queue.
- **The 4 GB laptop GPU** made experiments slow (7–22 minutes per run), which capped how many alternatives I could measure in the time-box.

## Next steps

1. **A real labeled set from reviewer corrections.** Every approve/reject already records `corrected_category`, `corrected_priority` and `edited_reply`. Use these as fresh labels with a proper held-out split, so tuning stops leaking into the test set.
2. **Confidence that predicts errors:** a second, stronger model as a verifier on a sample, or a small classifier trained on reviewer corrections (where do humans edit?), instead of the model's self-report.
3. **Low vs medium priority:** few-shot examples per priority level drawn from reviewer-corrected tickets, and evaluate with a bigger model through the same harness (`LLM_MODEL=… sh run.sh eval`).
4. **Retrieval for replies:** ground drafts in real help-centre and policy docs, so "where is data stored?" gets a cited answer instead of a placeholder.
5. **Production basics:** Postgres, auth, a job queue, prompt versioning on every prediction, and dashboards for retry/fallback/escalation rates and reviewer edit rate per category.

## Repo layout

```
run.sh, meta.yaml, Makefile    entry points (meta.yaml -> run.sh)
triage/config.py               settings; only LLM_* need to be set
triage/schema.py               Ticket, LLMTriage (strict model output), TriageResult
triage/prompts.py              system prompt, synthetic few-shots, repair messages
triage/llm.py                  OpenAI-compatible client, counts calls
triage/parsing.py              repair + normalize + validate model output
triage/rules.py                guards, reply checks, language detection, keyword fallback, calibration
triage/pipeline.py             triage_ticket(): the flow described above
triage/evaluate.py             the one-command eval
triage/api.py, store.py        FastAPI + review state
triage/static/index.html       review UI
data/                          tickets.json, labels.json (test), dev/ (synthetic dev set)
eval/                          results.json, report.md, details.json, dev/, runs/ (history)
tests/                         68 tests, no model needed (FakeModel)
```
