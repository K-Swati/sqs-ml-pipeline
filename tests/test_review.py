import dataclasses
import sqlite3

import pytest
from fastapi.testclient import TestClient

from pipeline.api import create_app
from pipeline.classifier.worker import ClassifierWorker
from pipeline.queues import queue_depth
from pipeline.store import Store
from tests.conftest import StubModel


@pytest.fixture
def client(settings, sqs, store):
    with TestClient(create_app(settings, sqs=sqs, store=store)) as c:
        yield c


def _low_confidence_ticket(client, settings, sqs, urls, store, metrics, text="hgkj"):
    mid = client.post("/messages", json={"customer_id": "c1", "text": text}).json()["message_id"]
    ClassifierWorker(settings, sqs, urls.incoming, urls.incoming_dlq, urls.classified, urls.human_review,
                     StubModel("technical_issue", 0.27), store, metrics).run_once(0)
    assert store.get_message(mid)["status"] == "needs_human_review"
    return mid


def test_claim_complete_with_corrected_intent(client, settings, sqs, urls, store, metrics):
    mid = _low_confidence_ticket(client, settings, sqs, urls, store, metrics)

    c = client.post("/review/claim", json={"reviewer": "asha"}).json()
    assert c["ticket"]["message_id"] == mid and c["drafts"]["billing"]
    assert queue_depth(sqs, urls.human_review) == {"visible": 0, "in_flight": 1, "delayed": 0}  # locked
    assert client.post("/review/claim", json={"reviewer": "ben"}).json() == {"ticket": None}  # not handed out twice

    r = client.post(f"/review/{mid}/complete", json={
        "receipt_handle": c["receipt_handle"], "reviewer": "asha", "intent": "general_inquiry", "response": "Hi! ...",
    }).json()
    assert r["saved"] is True
    m = r["message"]
    assert (m["status"], m["intent"], m["model_intent"], m["generator"], m["reviewed_by"]) == (
        "responded", "general_inquiry", "technical_issue", "human", "anonymous")
    assert queue_depth(sqs, urls.human_review)["in_flight"] + queue_depth(sqs, urls.human_review)["visible"] == 0
    assert client.get("/review/corrections").json()[0]["human_intent"] == "general_inquiry"


def test_release_puts_ticket_back(client, settings, sqs, urls, store, metrics):
    mid = _low_confidence_ticket(client, settings, sqs, urls, store, metrics)
    c = client.post("/review/claim", json={"reviewer": "asha"}).json()
    assert client.post(f"/review/{mid}/release", json={"receipt_handle": c["receipt_handle"]}).json() == {"released": True}
    assert client.post("/review/claim", json={"reviewer": "ben"}).json()["ticket"]["message_id"] == mid


def test_second_completion_is_a_no_op(client, settings, sqs, urls, store, metrics):
    mid = _low_confidence_ticket(client, settings, sqs, urls, store, metrics)
    h = client.post("/review/claim", json={"reviewer": "asha"}).json()["receipt_handle"]
    body = {"receipt_handle": h, "reviewer": "asha", "intent": "billing", "response": "first"}
    assert client.post(f"/review/{mid}/complete", json=body).json()["saved"] is True
    again = client.post(f"/review/{mid}/complete", json={**body, "response": "second"}).json()
    assert again["saved"] is False and again["message"]["response"] == "first"


def test_complete_validates_intent_and_message(client, settings, sqs, urls, store, metrics):
    mid = _low_confidence_ticket(client, settings, sqs, urls, store, metrics)
    body = {"receipt_handle": "x", "reviewer": "a", "intent": "not-a-label", "response": "hi"}
    assert client.post(f"/review/{mid}/complete", json=body).status_code == 422
    assert client.post("/review/nope/complete", json={**body, "intent": "billing"}).status_code == 404


def test_claim_skips_tickets_already_answered(client, settings, sqs, urls, store, metrics):
    mid = _low_confidence_ticket(client, settings, sqs, urls, store, metrics)
    store.save_human_review(mid, "billing", "answered elsewhere", "asha")  # e.g. a stale redelivery
    assert client.post("/review/claim", json={"reviewer": "ben"}).json() == {"ticket": None}
    assert queue_depth(sqs, urls.human_review)["visible"] == 0  # the stale copy was removed


def test_old_database_gets_review_columns(tmp_path, settings):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:  # results table as it was before human review existed
        c.execute("CREATE TABLE results (message_id TEXT PRIMARY KEY, intent TEXT, confidence REAL, model_version TEXT,"
                  " route TEXT, response TEXT, generator TEXT, classified_at TEXT, responded_at TEXT)")
    Store(str(path))
    with sqlite3.connect(path) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(results)")}
    assert {"model_intent", "reviewed_by", "reviewed_at"} <= cols
    assert dataclasses.replace(settings).review_lock_seconds == 300
