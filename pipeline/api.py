"""Ingest API: accepts messages, enqueues them, and reports their processing status.

Every route except ``/health``, ``/auth/*`` and the dashboard page itself requires a
session and checks one permission (see ``pipeline/auth.py`` for the role matrix).
The dashboard is served at ``/`` (``pipeline/ui/index.html``).
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from pipeline.auth import ANONYMOUS_ADMIN, PERMISSIONS, ROLE_LABELS, ROLES, AuthError, AuthStore, User
from pipeline.config import Settings
from pipeline.demo import DemoControls
from pipeline.metrics import correlation_context, log_event
from pipeline.metrics import metrics as default_metrics
from pipeline.queues import (
    QueueUrls,
    change_visibility,
    create_client,
    delete_message,
    provision_all,
    queue_depth,
    receive_messages,
    resolve_all,
    send_message,
)
from pipeline.responder.generator import TEMPLATES
from pipeline.schemas import INTENTS, IncomingMessage, MessageAccepted, MessageIn, dump_message, new_id
from pipeline.store import Store

logger = logging.getLogger(__name__)

UI_FILE = Path(__file__).parent / "ui" / "index.html"

# Fields a requester may see on their own tickets (no model internals or ids used for log search).
REQUESTER_FIELDS = ("message_id", "customer_id", "text", "status", "created_at", "updated_at", "response", "responded_at")


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class NewUser(BaseModel):
    username: str
    password: str
    role: str


class UserUpdate(BaseModel):
    role: str | None = None
    active: bool | None = None
    password: str | None = None


class OutageToggle(BaseModel):
    on: bool


class ReviewDecision(BaseModel):
    receipt_handle: str = Field(min_length=1)
    intent: str
    response: str = Field(min_length=1, max_length=5000)


class ReleaseRequest(BaseModel):
    receipt_handle: str = Field(min_length=1)


class BurstRequest(BaseModel):
    per_intent: int = Field(default=2, ge=1, le=10)


def create_app(
    settings: Settings | None = None,
    sqs: Any | None = None,
    store: Store | None = None,
    demo: DemoControls | None = None,
    demo_accounts: list[dict[str, str]] | None = None,
) -> FastAPI:
    """``demo_accounts`` (local mode only) are shown on the sign-in screen as one-click logins."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.sqs = sqs or create_client(settings)
        app.state.store = store or Store(settings.db_path)
        app.state.auth = AuthStore(app.state.store, settings.session_hours)
        app.state.urls = (
            provision_all(app.state.sqs, settings) if settings.provision_on_start else resolve_all(app.state.sqs, settings)
        )
        if settings.auth_enabled:
            generated = app.state.auth.bootstrap_admin(settings.admin_username, settings.admin_password)
            if generated:
                logger.warning("created first admin %r with generated password: %s  (change it after signing in)",
                               settings.admin_username, generated)
        else:
            logger.warning("AUTH_ENABLED=false: every caller has full admin access")
        yield

    app = FastAPI(title="SQS ML pipeline - ingest API", version="1.2.0", lifespan=lifespan)

    # ---- authentication and permission checks -------------------------
    def _token(request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        return token.strip() if scheme.lower() == "bearer" and token.strip() else None

    def current_user(request: Request) -> User:
        if not settings.auth_enabled:
            return ANONYMOUS_ADMIN
        token = _token(request)
        user = app.state.auth.user_for_token(token) if token else None
        if user is None:
            raise HTTPException(401, "sign in required", headers={"WWW-Authenticate": "Bearer"})
        return user

    def require(*permissions: str) -> Callable[..., User]:
        """Allow the request if the user has any one of ``permissions``."""

        def check(user: User = Depends(current_user)) -> User:
            if not any(user.can(p) for p in permissions):
                raise HTTPException(403, f"your role ({user.role}) cannot do this; needs {' or '.join(permissions)}")
            return user

        return check

    def visible(record: dict, user: User) -> dict:
        """Full record for dashboard users; a requester's own ticket without model internals."""
        if user.can("dashboard.view"):
            return record
        return {k: record.get(k) for k in REQUESTER_FIELDS}

    @app.get("/auth/config")
    def auth_config() -> dict:
        return {"auth_enabled": settings.auth_enabled, "demo_accounts": demo_accounts or [],
                "roles": [{"role": r, "label": ROLE_LABELS[r], "permissions": sorted(PERMISSIONS[r])} for r in ROLES]}

    @app.post("/auth/login")
    def login(req: LoginRequest) -> dict:
        result = app.state.auth.login(req.username, req.password)
        if result is None:
            raise HTTPException(401, "wrong username or password, or the account is disabled")
        token, user = result
        return {"token": token, "user": user.public(), "expires_in_hours": settings.session_hours}

    @app.post("/auth/logout")
    def logout(request: Request, user: User = Depends(current_user)) -> dict:
        token = _token(request)
        if token:
            app.state.auth.logout(token)
            app.state.auth.audit(user.username, "logout", user.username)
        return {"signed_out": True}

    @app.get("/auth/me")
    def me(user: User = Depends(current_user)) -> dict:
        return {**user.public(), "auth_enabled": settings.auth_enabled}

    # ---- users (admin) -------------------------------------------------
    @app.get("/users")
    def list_users(_: User = Depends(require("users.manage"))) -> list[dict]:
        return app.state.auth.list_users()

    @app.post("/users", status_code=201)
    def create_user(req: NewUser, user: User = Depends(require("users.manage"))) -> dict:
        try:
            return app.state.auth.create_user(req.username, req.password, req.role, actor=user.username).public()
        except AuthError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.patch("/users/{username}")
    def update_user(username: str, req: UserUpdate, user: User = Depends(require("users.manage"))) -> dict:
        try:
            return app.state.auth.update_user(username, user.username, role=req.role, active=req.active,
                                              password=req.password).public()
        except KeyError as exc:
            raise HTTPException(404, "user not found") from exc
        except AuthError as exc:
            raise HTTPException(409 if "last active admin" in str(exc) else 422, str(exc)) from exc

    @app.get("/audit")
    def audit(limit: int = Query(100, ge=1, le=1000), _: User = Depends(require("users.manage"))) -> list[dict]:
        return app.state.auth.recent_audit(limit)

    # ---- messages ------------------------------------------------------
    def accept(body: MessageIn, user: User) -> MessageAccepted:
        st: Store = app.state.store
        urls: QueueUrls = app.state.urls
        message_id, correlation_id = new_id(), new_id()
        record, created = st.create_message(message_id, body.customer_id, body.text, correlation_id,
                                            body.idempotency_key, created_by=user.username)
        if not created:  # same idempotency key: return the original, enqueue nothing
            if record.get("created_by") != user.username and not user.can("dashboard.view"):
                raise HTTPException(409, "this idempotency key was already used by someone else")
            return MessageAccepted(
                message_id=record["message_id"], correlation_id=record["correlation_id"],
                status=record["status"], duplicate=True,
            )
        envelope = IncomingMessage(
            message_id=message_id, correlation_id=correlation_id, customer_id=body.customer_id, text=body.text
        )
        try:
            send_message(app.state.sqs, urls.incoming, dump_message(envelope), {"correlation_id": correlation_id})
        except Exception as exc:
            st.delete_message(message_id)  # do not leave a record that will never be processed
            with correlation_context(correlation_id):
                log_event(logger, logging.ERROR, "enqueue_failed", error=repr(exc))
            raise HTTPException(503, "queue unavailable, please retry") from exc
        default_metrics.inc("api.accepted")
        with correlation_context(correlation_id):
            log_event(logger, logging.INFO, "message_accepted", message_id=message_id, user=user.username)
        return MessageAccepted(message_id=message_id, correlation_id=correlation_id, status=record["status"])

    @app.post("/messages", status_code=202, response_model=MessageAccepted)
    def submit(body: MessageIn, user: User = Depends(require("messages.create"))) -> MessageAccepted:
        return accept(body, user)

    @app.get("/messages")
    def list_messages(
        limit: int = Query(50, ge=1, le=500), status: str | None = None, mine: bool = False,
        user: User = Depends(require("dashboard.view", "messages.own")),
    ) -> list[dict]:
        only = user.username if (mine or not user.can("dashboard.view")) else None
        return [visible(r, user) for r in app.state.store.recent_messages(limit, status, created_by=only)]

    @app.get("/messages/{message_id}")
    def get_message(message_id: str, user: User = Depends(require("dashboard.view", "messages.own"))) -> dict:
        record = app.state.store.get_message(message_id)
        # A requester gets 404, not 403, for other people's tickets: no hint that the id exists.
        if record is None or (not user.can("dashboard.view") and record.get("created_by") != user.username):
            raise HTTPException(404, "message not found")
        return visible(record, user)

    @app.get("/my/stats")
    def my_stats(user: User = Depends(require("messages.own", "messages.create"))) -> dict:
        return {"statuses": app.state.store.status_counts(created_by=user.username)}

    @app.get("/stats")
    def stats(_: User = Depends(require("dashboard.view"))) -> dict:
        urls: QueueUrls = app.state.urls
        return {
            "statuses": app.state.store.status_counts(),
            "queues": {name: queue_depth(app.state.sqs, url) for name, url in urls.names.items()},
            "counters": default_metrics.snapshot(),
            "config": {
                "confidence_threshold": settings.confidence_threshold,
                "max_receive_count": settings.max_receive_count,
                "backoff_base_seconds": settings.backoff_base_seconds,
                "review_lock_seconds": settings.review_lock_seconds,
                "queue_names": {
                    "incoming": settings.incoming_queue, "incoming_dlq": settings.incoming_dlq,
                    "classified": settings.classified_queue, "classified_dlq": settings.classified_dlq,
                    "human_review": settings.human_review_queue,
                },
            },
            "demo": None if demo is None else {"model_outage": demo.model_outage.is_set()},
        }

    @app.get("/queues/{name}/peek")
    def peek(name: str, _: User = Depends(require("dashboard.view"))) -> list[dict]:
        """Look at up to 10 messages on a dead-letter or human-review queue without consuming them.

        Uses a zero visibility timeout, so messages stay available. Only allowed on queues that
        no worker reads from, because a peek counts as a receive.
        """
        urls: QueueUrls = app.state.urls
        allowed = {settings.incoming_dlq, settings.classified_dlq, settings.human_review_queue}
        if name not in allowed:
            raise HTTPException(400, f"peek is only allowed on: {', '.join(sorted(allowed))}")
        out = []
        for m in receive_messages(app.state.sqs, urls.names[name], max_messages=10, visibility_timeout=0):
            try:
                body: Any = json.loads(m.body)
            except ValueError:
                body = None
            out.append({"sqs_message_id": m.message_id, "receive_count": m.receive_count,
                        "attributes": m.attributes, "body": body, "raw": m.body[:2000]})
        return out

    # ---- human review --------------------------------------------------
    # Reviewers are consumers of the human_review queue, like the workers are of theirs:
    # claiming a ticket receives it with a long visibility timeout (a lock), finishing it
    # deletes it, and an abandoned claim simply reappears for someone else.

    @app.post("/review/claim")
    def claim(user: User = Depends(require("review.work"))) -> dict:
        urls: QueueUrls = app.state.urls
        st: Store = app.state.store
        for _ in range(5):  # skip stale deliveries of tickets that are already answered
            got = receive_messages(app.state.sqs, urls.human_review, max_messages=1,
                                   visibility_timeout=settings.review_lock_seconds)
            if not got:
                return {"ticket": None}
            m = got[0]
            try:
                body = json.loads(m.body)
                record = st.get_message(body["message_id"])
            except (ValueError, KeyError, TypeError):
                record = None
            if record is None or record.get("response"):
                delete_message(app.state.sqs, urls.human_review, m.receipt_handle)
                continue
            default_metrics.inc("review.claimed")
            with correlation_context(record["correlation_id"]):
                log_event(logger, logging.INFO, "review_claimed", reviewer=user.username,
                          message_id=record["message_id"])
            return {
                "ticket": record,
                "receipt_handle": m.receipt_handle,
                "lock_seconds": settings.review_lock_seconds,
                "intents": list(INTENTS),
                "drafts": {i: TEMPLATES[i] for i in INTENTS},
            }
        return {"ticket": None}

    @app.post("/review/{message_id}/complete")
    def complete(message_id: str, req: ReviewDecision, user: User = Depends(require("review.work"))) -> dict:
        if req.intent not in INTENTS:
            raise HTTPException(422, f"intent must be one of: {', '.join(INTENTS)}")
        st: Store = app.state.store
        record = st.get_message(message_id)
        if record is None:
            raise HTTPException(404, "message not found")
        saved = st.save_human_review(message_id, req.intent, req.response.strip(), user.username)
        try:
            delete_message(app.state.sqs, app.state.urls.human_review, req.receipt_handle)
        except Exception as exc:  # lock expired: the reply is saved, a later claim will skip it
            log_event(logger, logging.WARNING, "review_delete_failed", error=repr(exc))
        if saved:
            default_metrics.inc("review.completed")
            corrected = req.intent != record["intent"]
            if corrected:
                default_metrics.inc("review.intent_corrected")
            app.state.auth.audit(user.username, "review_completed", message_id,
                                 f"{record['intent']}->{req.intent}" if corrected else f"confirmed {req.intent}")
            with correlation_context(record["correlation_id"]):
                log_event(logger, logging.INFO, "review_completed", reviewer=user.username,
                          model_intent=record["intent"], human_intent=req.intent)
        return {"saved": saved, "message": st.get_message(message_id)}

    @app.post("/review/{message_id}/release")
    def release(message_id: str, req: ReleaseRequest, _: User = Depends(require("review.work"))) -> dict:
        """Give a claimed ticket back so another reviewer can take it now."""
        try:
            change_visibility(app.state.sqs, app.state.urls.human_review, req.receipt_handle, 0)
        except Exception as exc:
            raise HTTPException(409, "claim already expired") from exc
        return {"released": True}

    @app.get("/review/corrections")
    def corrections(_: User = Depends(require("review.work"))) -> list[dict]:
        """Reviewed tickets: the model's guess next to the human label, for retraining."""
        return app.state.store.review_corrections()

    @app.get("/model")
    def model_report(_: User = Depends(require("dashboard.view"))) -> dict:
        path = Path(settings.reports_dir) / "classifier_metrics.json"
        if not path.exists():
            raise HTTPException(404, "no training report; run `python -m pipeline train`")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(UI_FILE, media_type="text/html")

    if demo is not None:
        _add_demo_routes(app, accept, demo, require)

    return app


def _add_demo_routes(app: FastAPI, accept: Any, demo: DemoControls, require: Callable[..., Any]) -> None:
    from pipeline.classifier.data import generate

    admin = Depends(require("demo.control"))

    @app.post("/demo/burst")
    def burst(req: BurstRequest, user: User = admin) -> dict:
        """Submit tickets in phrasings the model never saw in training."""
        tickets = generate(req.per_intent, seed=random.randrange(1_000_000), held_out=True)
        ids = [accept(MessageIn(customer_id=f"demo-{n:02d}", text=t.text), user).message_id
               for n, t in enumerate(tickets)]
        return {"submitted": len(ids), "message_ids": ids}

    @app.post("/demo/poison")
    def poison(_: User = admin) -> dict:
        """Put malformed messages straight on the incoming queue (bypassing API validation)."""
        urls: QueueUrls = app.state.urls
        send_message(app.state.sqs, urls.incoming, "{ this is not json")
        send_message(app.state.sqs, urls.incoming, json.dumps({"schema_version": 99, "message_id": "x"}))
        return {"injected": 2}

    @app.post("/demo/outage")
    def outage(req: OutageToggle, user: User = admin) -> dict:
        """Make the classifier's model raise, so messages retry with backoff and then reach the DLQ."""
        (demo.model_outage.set if req.on else demo.model_outage.clear)()
        app.state.auth.audit(user.username, "demo_outage", None, "on" if req.on else "off")
        return {"model_outage": demo.model_outage.is_set()}
