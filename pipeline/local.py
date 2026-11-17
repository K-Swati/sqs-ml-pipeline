"""Run the whole system on one machine with no Docker and no AWS account.

    python -m pipeline local          # then open http://localhost:8000

Starts an in-process SQS server (moto), trains the model if needed, and runs the
API plus both workers as threads. The demo controls (burst, poison, outage) are
enabled so you can watch failures being handled. Needs ``moto[server]``, which
``requirements-dev.txt`` installs.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import socket
import threading
from pathlib import Path

from pipeline.config import Settings

logger = logging.getLogger(__name__)

LOCAL_DB = "data/local-ui-{port}.db"  # per port, so two local instances never wipe each other

# One account per persona, recreated on every start (the local database is reset).
DEMO_PASSWORD = "demo-pass-123"
DEMO_ACCOUNTS = [
    {"username": "viewer", "role": "viewer", "about": "sees the dashboard, changes nothing"},
    {"username": "requester", "role": "requester", "about": "raises tickets, sees only their own"},
    {"username": "reviewer", "role": "reviewer", "about": "answers tickets in human review"},
    {"username": "admin", "role": "admin", "about": "everything, plus users and failure controls"},
]


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def run_local(port: int = 8000, sqs_port: int = 5055) -> None:
    import uvicorn
    from moto.server import ThreadedMotoServer

    from pipeline.api import create_app
    from pipeline.auth import AuthStore
    from pipeline.classifier.model import IntentModel
    from pipeline.classifier.train import train_and_report
    from pipeline.classifier.worker import ClassifierWorker
    from pipeline.demo import DemoControls, OutageAwareModel
    from pipeline.metrics import metrics
    from pipeline.queues import create_client, provision_all
    from pipeline.responder.generator import get_responder
    from pipeline.responder.worker import ResponderWorker
    from pipeline.store import Store

    for p, what in ((port, "dashboard/API (change with --port)"), (sqs_port, "local SQS server (change with --sqs-port)")):
        if _port_in_use(p):
            raise SystemExit(f"port {p} is already in use, needed for the {what}. "
                             f"Find the owner with: lsof -nP -iTCP:{p} -sTCP:LISTEN")

    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        os.environ.setdefault(k, "test")

    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # the local SQS server logs every request
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=sqs_port, verbose=False)
    server.start()

    base = Settings.from_env()
    settings = dataclasses.replace(
        base,
        aws_endpoint_url=f"http://127.0.0.1:{sqs_port}",
        db_path=os.environ.get("DB_PATH", LOCAL_DB.format(port=port)),
        wait_time_seconds=min(base.wait_time_seconds, 1),  # short polls keep the dashboard lively
        provision_on_start=True,
    )

    # Queues live in memory and start empty, so start the store empty too (no orphaned records).
    for suffix in ("", "-wal", "-shm"):
        Path(settings.db_path + suffix).unlink(missing_ok=True)

    if not Path(settings.model_path).exists():
        print("training intent model (first run only)...")
        train_and_report(settings)

    sqs = create_client(settings)
    urls = provision_all(sqs, settings)
    store = Store(settings.db_path)
    auth = AuthStore(store, settings.session_hours)
    for acct in DEMO_ACCOUNTS:
        auth.create_user(acct["username"], DEMO_PASSWORD, acct["role"], actor="local-mode")
    demo = DemoControls()
    model = OutageAwareModel(IntentModel.load(settings.model_path), demo)

    workers = [
        ClassifierWorker(settings, sqs, urls.incoming, urls.incoming_dlq, urls.classified, urls.human_review,
                         model, store, metrics),  # type: ignore[arg-type]
        ResponderWorker(settings, create_client(settings), urls.classified, urls.classified_dlq,
                        get_responder(settings.responder, settings.llm_model), store, metrics),
    ]
    stop = threading.Event()
    threads = [threading.Thread(target=w.run_forever, args=(stop,), name=w.stage, daemon=True) for w in workers]
    for t in threads:
        t.start()

    print(f"\n  dashboard:  http://localhost:{port}\n  API docs:   http://localhost:{port}/docs\n")
    print(f"  demo accounts (password for all: {DEMO_PASSWORD}):")
    for acct in DEMO_ACCOUNTS:
        print(f"    {acct['username']:<10} {acct['about']}")
    print()
    try:
        uvicorn.run(create_app(settings, sqs=create_client(settings), store=store, demo=demo,
                               demo_accounts=[{**a, "password": DEMO_PASSWORD} for a in DEMO_ACCOUNTS]),
                    host="127.0.0.1", port=port, log_level="warning")
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=settings.wait_time_seconds + 2)
        server.stop()
