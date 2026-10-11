import json

from conftest import GOOD, FakeModel
from fastapi.testclient import TestClient

from triage import evaluate
from triage.api import create_app
from triage.batch import load_tickets, run_batch
from triage.config import load_settings


def test_review_flow(settings):
    client = TestClient(create_app(settings, FakeModel([GOOD] * 50)))
    tickets = client.get("/api/tickets").json()
    assert len(tickets) == 30
    tid = tickets[0]["id"]

    assert client.post(f"/api/tickets/{tid}/review", json={"action": "approve", "reply": "x", "category": "bug",
                                                           "priority": "low"}).status_code == 409
    tr = client.post(f"/api/tickets/{tid}/triage").json()
    assert tr["category"] == "billing"

    # can't approve an empty reply
    r = client.post(f"/api/tickets/{tid}/review", json={"action": "approve", "reply": " ", "category": "billing",
                                                        "priority": "high"})
    assert r.status_code == 422

    r = client.post(f"/api/tickets/{tid}/review", json={"action": "approve", "reply": "Edited reply",
                                                        "category": "billing", "priority": "medium"}).json()
    assert r["status"] == "approved" and r["edited_reply"] and r["corrected_priority"]
    assert not r["corrected_category"]
    # persisted
    assert json.loads(settings.state_path.read_text())[tid]["review"]["status"] == "approved"


def test_rejects_out_of_schema_review(settings):
    client = TestClient(create_app(settings, FakeModel([GOOD])))
    client.post("/api/tickets/T-001/triage")
    r = client.post("/api/tickets/T-001/review", json={"action": "approve", "reply": "x", "category": "spam",
                                                       "priority": "high"})
    assert r.status_code == 422


def test_new_ticket(settings):
    client = TestClient(create_app(settings, FakeModel([GOOD])))
    item = client.post("/api/tickets", json={"subject": "hi", "body": "charged twice"}).json()
    assert item["id"].startswith("NEW-") and item["triage"]["category"] == "billing"


def test_batch_and_metrics_produce_30_predictions(settings):
    tickets = load_tickets(settings.tickets_path)
    results = run_batch(tickets, FakeModel([]), settings)  # default answer: other/low
    assert [r.id for r in results] == [t.id for t in tickets]
    labels = evaluate.load_labels(load_settings().labels_path)
    m = evaluate.compute_metrics(results, labels)
    assert m["n_predictions"] == 30 and m["n_labeled"] == len(labels)
    assert 0 <= m["category_accuracy"] <= 1
    pub = results[0].public()
    assert set(pub) == {"id", "category", "priority", "summary", "suggested_reply", "confidence", "escalate"}


def test_cannot_approve_reply_with_placeholders(settings):
    client = TestClient(create_app(settings, FakeModel([GOOD])))
    client.post("/api/tickets/T-001/triage")
    r = client.post("/api/tickets/T-001/review", json={"action": "approve", "reply": "Renews on [CONFIRM: date]",
                                                       "category": "billing", "priority": "high"})
    assert r.status_code == 422


def test_threshold_sweep_counts_caught_errors():
    from triage.schema import TriageResult
    labels = {"A": {"category": "bug", "priority": "high"}, "B": {"category": "bug", "priority": "high"}}
    base = dict(summary="s", suggested_reply="r", escalate=False, source="llm")
    results = [TriageResult(id="A", category="bug", priority="high", confidence=0.9, **base),     # right, confident
               TriageResult(id="B", category="billing", priority="high", confidence=0.3, **base)]  # wrong, unsure
    row = next(r for r in evaluate.threshold_sweep(results, labels) if r["threshold"] == 0.6)
    assert row == {"threshold": 0.6, "escalated_share": 0.5, "errors_caught": 1.0}
