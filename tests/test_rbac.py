"""Access control with auth on: one signed-in client per persona."""

import dataclasses

import pytest
from fastapi.testclient import TestClient

from pipeline.api import create_app
from pipeline.auth import PERMISSIONS, AuthStore, hash_password, verify_password
from pipeline.classifier.worker import ClassifierWorker
from pipeline.demo import DemoControls
from tests.conftest import StubModel

PASSWORD = "correct-horse-1"
PERSONAS = ("viewer", "requester", "reviewer", "admin")


@pytest.fixture
def auth_settings(settings):
    return dataclasses.replace(settings, auth_enabled=True, admin_username="root", admin_password="root-password-1")


@pytest.fixture
def app(auth_settings, sqs, store):
    a = create_app(auth_settings, sqs=sqs, store=store, demo=DemoControls())
    with TestClient(a):  # run startup (creates the first admin)
        auth = a.state.auth
        for role in PERSONAS:
            auth.create_user(role, PASSWORD, role, actor="root")
        auth.create_user("requester2", PASSWORD, "requester", actor="root")
        yield a


def login(app, username, password=PASSWORD) -> TestClient:
    c = TestClient(app)
    r = c.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    c.headers["authorization"] = "Bearer " + r.json()["token"]
    return c


@pytest.fixture
def as_(app):
    return {name: login(app, name) for name in (*PERSONAS, "requester2")}


def test_password_hashing():
    h = hash_password("s3cret-pass")
    assert h.startswith("scrypt$") and "s3cret" not in h
    assert verify_password("s3cret-pass", h) and not verify_password("wrong", h)
    assert not verify_password("x", "garbage")


def test_everything_but_health_and_login_needs_a_session(app):
    anon = TestClient(app)
    assert anon.get("/health").status_code == 200
    assert anon.get("/").status_code == 200  # the page itself; it shows the sign-in screen
    assert anon.get("/auth/config").json()["auth_enabled"] is True
    for method, path in [("get", "/stats"), ("get", "/messages"), ("post", "/messages"), ("get", "/model"),
                         ("post", "/review/claim"), ("get", "/users"), ("get", "/audit"), ("post", "/demo/poison")]:
        assert getattr(anon, method)(path).status_code == 401, path
    anon.headers["authorization"] = "Bearer not-a-real-token"
    assert anon.get("/auth/me").status_code == 401


def test_bad_login_is_rejected_and_audited(app):
    c = TestClient(app)
    assert c.post("/auth/login", json={"username": "viewer", "password": "nope"}).status_code == 401
    assert c.post("/auth/login", json={"username": "ghost", "password": "nope"}).status_code == 401
    actions = [a["action"] for a in app.state.auth.recent_audit()]
    assert actions.count("login_failed") == 2


def test_first_admin_is_bootstrapped(app):
    assert login(app, "root", "root-password-1").get("/auth/me").json()["role"] == "admin"


# (method, path, json) -> which personas may call it
MATRIX = [
    (("get", "/stats", None), {"viewer", "reviewer", "admin"}),
    (("get", "/model", None), {"viewer", "reviewer", "admin"}),
    (("get", "/queues/incoming-dlq/peek", None), {"viewer", "reviewer", "admin"}),
    (("post", "/messages", {"customer_id": "c", "text": "I was charged twice"}), {"requester", "admin"}),
    (("get", "/my/stats", None), {"requester", "admin"}),
    (("post", "/review/claim", None), {"reviewer", "admin"}),
    (("get", "/review/corrections", None), {"reviewer", "admin"}),
    (("get", "/users", None), {"admin"}),
    (("get", "/audit", None), {"admin"}),
    (("post", "/demo/poison", None), {"admin"}),
    (("post", "/demo/outage", {"on": False}), {"admin"}),
]


@pytest.mark.parametrize(("call", "allowed"), MATRIX, ids=[c[0][1] + ":" + c[0][0] for c in MATRIX])
def test_permission_matrix(as_, call, allowed):
    method, path, body = call
    for persona in PERSONAS:
        kwargs = {"json": body} if body is not None else {}
        code = getattr(as_[persona], method)(path, **kwargs).status_code
        # allowed means "got past the permission check" (e.g. /model may still 404 without a report)
        if persona in allowed:
            assert code not in (401, 403), f"{persona} {method} {path} -> {code}"
        else:
            assert code == 403, f"{persona} {method} {path} -> {code}"


def test_requester_sees_only_own_tickets_without_model_internals(app, as_, auth_settings, sqs, urls, store, metrics):
    mine = as_["requester"].post("/messages", json={"customer_id": "c1", "text": "refund please"}).json()["message_id"]
    other = as_["requester2"].post("/messages", json={"customer_id": "c2", "text": "cannot log in"}).json()["message_id"]
    ClassifierWorker(auth_settings, sqs, urls.incoming, urls.incoming_dlq, urls.classified, urls.human_review,
                     StubModel("billing", 0.9), store, metrics).run_once(0)

    rows = as_["requester"].get("/messages").json()
    assert [r["message_id"] for r in rows] == [mine]
    assert "intent" not in rows[0] and "confidence" not in rows[0] and "correlation_id" not in rows[0]
    assert as_["requester"].get(f"/messages/{other}").status_code == 404  # not 403: no hint it exists
    assert as_["requester"].get(f"/messages/{mine}").json()["status"] == "classified"
    assert as_["requester"].get("/my/stats").json()["statuses"] == {"classified": 1}

    full = as_["viewer"].get("/messages").json()
    assert {r["message_id"] for r in full} == {mine, other}
    assert full[0]["created_by"] in ("requester", "requester2") and full[0]["intent"] == "billing"


def test_requester_cannot_reuse_someone_elses_idempotency_key(as_):
    body = {"customer_id": "c", "text": "charged twice", "idempotency_key": "order-1"}
    assert as_["requester"].post("/messages", json=body).status_code == 202
    assert as_["requester"].post("/messages", json=body).json()["duplicate"] is True
    assert as_["requester2"].post("/messages", json=body).status_code == 409


def test_review_is_recorded_under_the_signed_in_reviewer(app, as_, auth_settings, sqs, urls, store, metrics):
    mid = as_["requester"].post("/messages", json={"customer_id": "c", "text": "hgkj"}).json()["message_id"]
    ClassifierWorker(auth_settings, sqs, urls.incoming, urls.incoming_dlq, urls.classified, urls.human_review,
                     StubModel("technical_issue", 0.2), store, metrics).run_once(0)
    c = as_["reviewer"].post("/review/claim").json()
    r = as_["reviewer"].post(f"/review/{mid}/complete", json={
        "receipt_handle": c["receipt_handle"], "intent": "general_inquiry", "response": "Happy to help",
        "reviewer": "someone-else",  # ignored: identity comes from the session
    }).json()
    assert r["message"]["reviewed_by"] == "reviewer"
    assert as_["requester"].get(f"/messages/{mid}").json()["response"] == "Happy to help"
    assert any(a["action"] == "review_completed" and a["actor"] == "reviewer" for a in as_["admin"].get("/audit").json())


def test_admin_manages_users_and_changes_apply_immediately(app, as_):
    admin = as_["admin"]
    assert admin.post("/users", json={"username": "priya", "password": PASSWORD, "role": "viewer"}).status_code == 201
    assert admin.post("/users", json={"username": "priya", "password": PASSWORD, "role": "viewer"}).status_code == 422
    assert admin.post("/users", json={"username": "x", "password": "short", "role": "god"}).status_code == 422

    priya = login(app, "priya")
    assert priya.post("/review/claim").status_code == 403
    assert admin.patch("/users/priya", json={"role": "reviewer"}).status_code == 200
    assert priya.post("/review/claim").status_code == 200  # same session, new role

    assert admin.patch("/users/priya", json={"active": False}).status_code == 200
    assert priya.get("/auth/me").status_code == 401  # disabled: sessions end
    assert TestClient(app).post("/auth/login", json={"username": "priya", "password": PASSWORD}).status_code == 401

    assert admin.patch("/users/priya", json={"active": True, "password": "new-password-9"}).status_code == 200
    login(app, "priya", "new-password-9")
    assert admin.patch("/users/nobody", json={"role": "viewer"}).status_code == 404
    names = {u["username"]: u for u in admin.get("/users").json()}
    assert names["priya"]["role"] == "reviewer" and "password_hash" not in names["priya"]


def test_last_admin_cannot_be_removed(app, as_):
    admin = as_["admin"]
    assert admin.patch("/users/root", json={"role": "viewer"}).status_code == 200  # another admin remains
    assert admin.patch("/users/admin", json={"role": "viewer"}).status_code == 409
    assert admin.patch("/users/admin", json={"active": False}).status_code == 409


def test_logout_ends_the_session(app):
    c = login(app, "viewer")
    assert c.post("/auth/logout").status_code == 200
    assert c.get("/stats").status_code == 401


def test_old_messages_table_gets_created_by(tmp_path):
    import sqlite3

    from pipeline.store import Store

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, text TEXT NOT NULL,"
                  " correlation_id TEXT NOT NULL, idempotency_key TEXT UNIQUE, status TEXT NOT NULL,"
                  " created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    store = Store(str(path))
    AuthStore(store)
    store.create_message("m1", "c", "t", "corr", created_by="u1")
    assert store.get_message("m1")["created_by"] == "u1"


def test_permission_table_is_what_the_docs_say():
    assert PERMISSIONS["viewer"] == {"dashboard.view"}
    assert PERMISSIONS["requester"] == {"messages.create", "messages.own"}
    assert PERMISSIONS["reviewer"] == {"dashboard.view", "review.work"}
    assert {"users.manage", "demo.control", "review.work", "messages.create"} <= PERMISSIONS["admin"]
