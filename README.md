# Event-Driven ML Pipeline on AWS SQS

[![CI](https://github.com/K-Swati/sqs-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/K-Swati/sqs-ml-pipeline/actions/workflows/ci.yml)

A support-ticket pipeline built from decoupled services that talk only through queues. Each stage can fail, retry and scale on its own, and one bad message never blocks the rest.

```
 client ──> FastAPI ingest ──> [incoming] ──> classifier worker ──┬──> [classified] ──> responder worker ──> result store
              │  idempotency       │ DLQ                         │         │ DLQ
              └─ status API        ▼                             │         ▼
                              [incoming-dlq]                     └──> [human_review]   (low confidence)
                                                                        [classified-dlq]
```

**What it demonstrates**

- **Failure isolation**: malformed messages go straight to a dead-letter queue (DLQ) with the reason attached; transient errors retry with exponential backoff, then SQS moves them to the DLQ
- **Idempotency**: duplicate API calls (idempotency key) and duplicate queue deliveries (SQS is at-least-once) are processed once
- **ML in the loop**: a scikit-learn intent classifier, with low-confidence tickets routed to a human review queue instead of guessed
- **Observability**: JSON logs carrying a correlation id from API to final response, counters, optional CloudWatch metrics
- **Local AWS**: runs on LocalStack with Docker Compose, and tests run against `moto` with no credentials
- Versioned message schemas, FastAPI, Docker, GitHub Actions CI

## Quick start (no Docker, no AWS)

Needs Python 3.12+.

```bash
pip install -r requirements-dev.txt
python -m pipeline local  # whole system + dashboard at http://localhost:8000 (--port / --sqs-port to change)
pytest -q                 # 68 tests against mocked SQS
python scripts/demo.py    # scripted end-to-end run, printed to the terminal
```

## Dashboard

`python -m pipeline local` starts an in-process SQS server (moto), the API and both workers, then serves a live dashboard at `http://localhost:8000`. Sign in first. Local mode creates one demo account per role (password `demo-pass-123`) and shows them as one-click buttons on the sign-in screen.

- **Live flow**: depth of every queue (waiting and in flight), worker counters, and DLQ and human-review queues that light up when something lands in them
- **Send a ticket**: submit through the real API, including an idempotency key, so you can watch the duplicate being rejected
- **Messages and trace**: click any message to see each step with timings, the intent and confidence against the threshold, the generated reply, and the correlation id to search the logs for
- **Dead-letter queues**: peek at failed messages with the reason attached (`invalid_json`, `unsupported_schema_version`, retries exhausted)
- **Human review**: reviewers take the next low-confidence ticket, correct the intent if the model was wrong, edit the drafted reply and send it. A taken ticket is locked to that reviewer (SQS visibility timeout, 5 minutes by default), and "Put back in queue" or an expired lock hands it to someone else. Corrections are kept next to the model's guess (`GET /review/corrections`) as training data
- **Try a failure** (local mode only): send unseen tickets, inject malformed messages, or simulate a model outage and watch messages retry with backoff and then reach the DLQ
- **Users and access** (admin): add users, change roles, disable accounts, reset passwords, and read the audit log

The dashboard is also served by the normal `api` service (for example under Docker Compose) at `/`, without the failure buttons. Local mode keeps queues in memory and resets `data/local-ui-<port>.db` on every start.

## Roles and access control

Each route checks a named permission, never a role name, and the dashboard shows only what the signed-in user's permissions allow. The API enforces the same rules either way.

| Permission | Viewer | Requester | Reviewer | Admin |
|---|:-:|:-:|:-:|:-:|
| See the pipeline, all tickets, queues and model (`dashboard.view`) | ✓ | | ✓ | ✓ |
| Raise tickets (`messages.create`) | | ✓ | | ✓ |
| Follow their own tickets and replies (`messages.own`) | | ✓ | | |
| Answer tickets in human review (`review.work`) | | | ✓ | ✓ |
| Failure controls, local mode (`demo.control`) | | | | ✓ |
| Manage users, read the audit log (`users.manage`) | | | | ✓ |

- **Requesters** see only tickets they raised, without model internals (intent, confidence, correlation id). Asking for someone else's ticket returns `404`, so it does not reveal that the id exists.
- **Sessions** are random bearer tokens; only their SHA-256 is stored. Passwords use scrypt. Role changes take effect on the user's next request, and disabling a user or resetting their password ends their sessions.
- The last active admin cannot be demoted or disabled.
- The **audit log** records sign-ins (including failed ones), user changes, completed reviews and outage toggles.
- **First admin**: when there are no users, the API creates `ADMIN_USERNAME` (default `admin`) with `ADMIN_PASSWORD`. If that is unset it generates a password and logs it once.
- `AUTH_ENABLED=false` gives every caller full access. The pipeline tests and `scripts/demo.py` use it, and it should never be used in a deployment.

Demo output (40 tickets in phrasings the model has never seen, plus 2 malformed messages):

```
resubmitting key t0 -> duplicate=True (nothing enqueued)
  tickets submitted : 40  ->  {'responded': 28, 'needs_human_review': 12}
  intent correct    : 34/40 (unseen phrasings, synthetic data)
  queue incoming-dlq: 2 message(s)
  queue human_review: 12 message(s)
```

## Run the full system with LocalStack

```bash
ADMIN_PASSWORD='choose-a-password' docker compose up --build   # or read the generated one from `docker compose logs api`
TOKEN=$(curl -s localhost:8000/auth/login -H 'content-type: application/json' \
  -d '{"username":"admin","password":"choose-a-password"}' | python -c 'import sys,json; print(json.load(sys.stdin)["token"])')
AUTH="authorization: Bearer $TOKEN"
curl -s localhost:8000/messages -H "$AUTH" -H 'content-type: application/json' \
  -d '{"customer_id":"cust-1","text":"I was charged twice for my Pro plan","idempotency_key":"abc-1"}'
curl -s localhost:8000/messages/<message_id> -H "$AUTH"   # received -> classified -> responded
curl -s localhost:8000/stats -H "$AUTH"                    # queue depths, status counts, counters
```

Pointing at real AWS needs only IAM credentials and no `AWS_ENDPOINT_URL`. Every setting is an environment variable (see `pipeline/config.py`).

## Failure handling

| Situation | What happens |
|---|---|
| Invalid JSON, unsupported `schema_version`, missing field | Copied to the DLQ with `error_reason`, `failed_stage` and `correlation_id` attributes, then deleted. **Not retried**, because it can never succeed |
| Handler raises (model backend down, timeout) | Visibility timeout is set to `base x 2^(attempt-1)` seconds. After `maxReceiveCount` receives, SQS moves it to the DLQ |
| One bad message in a batch of 10 | Only that message is retried or dead-lettered; the other nine complete normally |
| Same message delivered twice | Second delivery is acknowledged with no side effects (idempotency ledger) |
| Same API request sent twice | Same `idempotency_key` returns the original `message_id`, nothing enqueued |
| Enqueue fails in the API | The stored record is removed and the client gets `503`, so no orphan is left |
| Confidence below threshold (default 0.6) | Routed to `human_review` instead of auto-responding; a reviewer answers it from the dashboard |
| Reviewer abandons a ticket | Its lock (visibility timeout) expires and it returns to the queue for another reviewer |
| A ticket on the review queue was already answered | Skipped and removed when the next reviewer claims |

All of these are covered by tests in `tests/test_workers.py`, `tests/test_api_e2e.py` and `tests/test_review.py`.

## Classifier results (read this before quoting a number)

The tickets are **synthetic**: generated from phrase templates with random slot values and typo noise (no public dataset is bundled). To avoid flattering numbers, the **last two templates of each intent are held out** and used only for evaluation.

| Evaluation set | Accuracy | Macro F1 |
|---|---|---|
| Fresh noisy tickets from the *training* templates | 1.000 | 1.000 |
| Tickets from **templates never seen in training** | **0.833** | 0.832 |

- The first row is memorisation of templates and means little. The **second row is the honest figure**.
- The biggest confusions are `billing -> general_inquiry` and `technical_issue -> general_inquiry`, the vaguest category.
- Mean confidence drops from 0.99 to 0.72 on unseen phrasing, which is why the human-review route matters.
- These numbers describe synthetic data only and say nothing about real customer text. The value of this project is the pipeline design, and the model is a swappable component.

Reproduce: `python -m pipeline train` writes `reports/classifier_metrics.json`.

## Project layout

```
pipeline/
  api.py              ingest API: messages, stats, peek, model, human review, users, audit; each route checks a permission
  auth.py             roles and permissions, scrypt passwords, sessions, users, audit log
  ui/index.html       dashboard served at / (plain HTML + JS, no build step)
  local.py            `python -m pipeline local`: moto SQS + API + workers in one process
  demo.py             failure-injection switches used by the dashboard in local mode
  worker_base.py      consume loop, retry/backoff, poison handling, idempotency hook
  classifier/         synthetic data, TF-IDF + logistic regression, training report, worker
  responder/          template responder (offline) and optional OpenAI responder, worker
  queues.py           boto3 SQS helpers: provisioning, batch send, long poll, backoff
  schemas.py          versioned pydantic messages, poison-message detection
  store.py            SQLite store with idempotency ledger (swap for DynamoDB/Postgres)
  metrics.py          counters, correlation-id logging, CloudWatch export
scripts/demo.py       one-process end-to-end demo
tests/                68 tests (moto-mocked SQS), including the role matrix in test_rbac.py
docker-compose.yml    LocalStack + API + two workers
```

## Honest limitations

- `docker-compose.yml` and the Docker image were written but **not run** in the environment this was developed in (no Docker daemon); the tests and demo run against `moto`. Run `docker compose up --build` once on your machine before relying on it.
- SQLite is fine for a demo and a single host. Use DynamoDB or Postgres for several hosts.
- Standard SQS queues are at-least-once and unordered; the design assumes that and does not need FIFO queues. Retries use visibility timeouts rather than SQS delay queues.
- The OpenAI responder is optional (`RESPONDER=openai`) and unit-tested with a fake client, not against the live API.
- `/queues/{name}/peek` receives with a zero visibility timeout, so each peek raises the message's receive count. It is limited to the DLQs and the human-review queue, which no worker consumes.
- Sign-in is built in (users in SQLite, bearer tokens). For an organisation you would put SSO (OIDC) in front and map its groups to these roles. There is no login rate limiting or lockout, so put the API behind a proxy that rate limits `/auth/login`.
- The bearer token is kept in the browser's `localStorage`. That is simple, but a cross-site scripting bug could read it. The page escapes all data it renders, and an HttpOnly cookie with CSRF protection would be the stronger choice for production.
- Not included: rate limiting, autoscaling policy, Terraform.

## Ideas for extension

- Replace SQLite with DynamoDB and add Terraform for the queues
- Add a redrive tool that replays DLQ messages after a fix
- Track classifier drift using the stored confidence values

## License

MIT. See [LICENSE](LICENSE).
