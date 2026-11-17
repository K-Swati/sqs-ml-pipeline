import pytest
from fastapi.testclient import TestClient

from pipeline.api import create_app
from pipeline.classifier.worker import ClassifierWorker
from pipeline.demo import DemoControls, OutageAwareModel
from pipeline.queues import queue_depth, send_message


@pytest.fixture
def client(settings, sqs, store):
    with TestClient(create_app(settings, sqs=sqs, store=store)) as c:
        yield c


@pytest.fixture
def demo():
    return DemoControls()


@pytest.fixture
def demo_client(settings, sqs, store, demo):
    with TestClient(create_app(settings, sqs=sqs, store=store, demo=demo)) as c:
        yield c


def test_dashboard_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Support-ticket pipeline" in r.text


def test_list_messages_newest_first_and_filter(client):
    for n in range(3):
        client.post("/messages", json={"customer_id": f"c{n}", "text": f"ticket {n}"})
    rows = client.get("/messages").json()
    assert [r["customer_id"] for r in rows] == ["c2", "c1", "c0"]
    assert len(client.get("/messages?limit=1").json()) == 1
    assert client.get("/messages?status=responded").json() == []
    assert len(client.get("/messages?status=received").json()) == 3


def test_stats_include_config_and_no_demo_by_default(client, settings):
    s = client.get("/stats").json()
    assert s["config"]["confidence_threshold"] == settings.confidence_threshold
    assert s["config"]["queue_names"]["incoming_dlq"] == settings.incoming_dlq
    assert s["demo"] is None


def test_demo_routes_absent_without_controls(client):
    assert client.post("/demo/poison").status_code == 404
    assert client.post("/demo/outage", json={"on": True}).status_code == 404


def test_peek_is_restricted_and_does_not_consume(client, sqs, urls, settings):
    assert client.get(f"/queues/{settings.incoming_queue}/peek").status_code == 400
    send_message(sqs, urls.incoming_dlq, "{ bad", {"error_reason": "invalid_json"})
    for _ in range(2):  # peeking twice still sees the message
        items = client.get(f"/queues/{settings.incoming_dlq}/peek").json()
        assert len(items) == 1 and items[0]["attributes"]["error_reason"] == "invalid_json" and items[0]["body"] is None
    assert queue_depth(sqs, urls.incoming_dlq)["visible"] == 1


def test_demo_burst_and_poison(demo_client, sqs, urls):
    r = demo_client.post("/demo/burst", json={"per_intent": 1}).json()
    assert r["submitted"] == len(r["message_ids"]) > 0
    demo_client.post("/demo/poison")
    assert queue_depth(sqs, urls.incoming)["visible"] == r["submitted"] + 2


def test_demo_outage_makes_classifier_retry(demo_client, demo, settings, sqs, store, urls, metrics, trained_model):
    assert demo_client.post("/demo/outage", json={"on": True}).json() == {"model_outage": True}
    assert demo_client.get("/stats").json()["demo"] == {"model_outage": True}
    demo_client.post("/messages", json={"customer_id": "c", "text": "I was charged twice"})
    worker = ClassifierWorker(settings, sqs, urls.incoming, urls.incoming_dlq, urls.classified, urls.human_review,
                              OutageAwareModel(trained_model, demo), store, metrics)  # type: ignore[arg-type]
    worker.run_once(0)
    assert metrics.snapshot().get("classifier.retried") == 1

    demo_client.post("/demo/outage", json={"on": False})
    worker.run_once(0)
    assert metrics.snapshot().get("classifier.processed") == 1
    assert demo_client.get("/messages").json()[0]["status"] in ("classified", "needs_human_review")
