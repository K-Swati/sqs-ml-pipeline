"""Role-based access control: users, sessions, permissions and an audit log.

Roles map to named permissions, and every API route checks a permission, never a
role name. The dashboard asks ``/auth/me`` for the permissions and shows only
what the user may do; the API enforces the same rules either way.

| permission       | viewer | requester | reviewer | admin |
|------------------|:------:|:---------:|:--------:|:-----:|
| dashboard.view   |   x    |           |    x     |   x   |  pipeline flow, all messages, stats, DLQs, model
| messages.create  |        |     x     |          |   x   |  raise a ticket
| messages.own     |        |     x     |          |       |  see your own tickets and their replies
| review.work      |        |           |    x     |   x   |  claim and answer human-review tickets
| demo.control     |        |           |          |   x   |  failure injection (local mode)
| users.manage     |        |           |          |   x   |  create users, change roles, disable, audit log

Users, sessions and the audit log live in the same SQLite file as messages.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from pipeline.store import Store

ROLES: tuple[str, ...] = ("viewer", "requester", "reviewer", "admin")

PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"dashboard.view"}),
    "requester": frozenset({"messages.create", "messages.own"}),
    "reviewer": frozenset({"dashboard.view", "review.work"}),
    "admin": frozenset({"dashboard.view", "messages.create", "review.work", "demo.control", "users.manage"}),
}

ROLE_LABELS = {
    "viewer": "Viewer",
    "requester": "Requester",
    "reviewer": "Reviewer",
    "admin": "Admin",
}

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,31}$")
MIN_PASSWORD = 8

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL,
    created_by    TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    username   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    at     TEXT NOT NULL,
    actor  TEXT,
    action TEXT NOT NULL,
    target TEXT,
    detail TEXT
);
"""


class AuthError(ValueError):
    """Invalid user input or a rule violation (for example removing the last admin)."""


@dataclass(frozen=True)
class User:
    username: str
    role: str
    active: bool = True

    @property
    def permissions(self) -> frozenset[str]:
        return PERMISSIONS.get(self.role, frozenset())

    def can(self, permission: str) -> bool:
        return permission in self.permissions

    def public(self) -> dict[str, Any]:
        return {"username": self.username, "role": self.role, "role_label": ROLE_LABELS.get(self.role, self.role),
                "active": self.active, "permissions": sorted(self.permissions)}


# Used when AUTH_ENABLED=false (tests, scripts): full access, clearly named.
ANONYMOUS_ADMIN = User("anonymous", "admin")


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    n, r, p = 2**14, 8, 1
    digest = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, digest = stored.split("$")
        if algo != "scrypt":
            return False
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(digest) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), digest)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# A real hash, so a login for an unknown user costs the same time as a wrong password.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


class AuthStore:
    def __init__(self, store: Store, session_hours: int = 12) -> None:
        self.store = store
        self.session_hours = session_hours
        with store.transaction() as c:
            c.executescript(AUTH_SCHEMA)

    # ---- users ---------------------------------------------------------
    def create_user(self, username: str, password: str, role: str, actor: str | None = None) -> User:
        username = username.strip().lower()
        if not USERNAME_RE.match(username):
            raise AuthError("username: 3-32 characters, lowercase letters, digits, '.', '_' or '-'")
        if len(password) < MIN_PASSWORD:
            raise AuthError(f"password must be at least {MIN_PASSWORD} characters")
        if role not in ROLES:
            raise AuthError(f"role must be one of: {', '.join(ROLES)}")
        with self.store.transaction() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO users (username, password_hash, role, active, created_at, created_by)"
                " VALUES (?,?,?,1,?,?)",
                (username, hash_password(password), role, _iso(_now()), actor),
            )
            if cur.rowcount != 1:
                raise AuthError(f"user {username!r} already exists")
        self.audit(actor, "user_created", username, f"role={role}")
        return User(username, role)

    def get_user(self, username: str) -> User | None:
        with self.store.transaction() as c:
            row = c.execute("SELECT username, role, active FROM users WHERE username = ?", (username,)).fetchone()
        return User(row["username"], row["role"], bool(row["active"])) if row else None

    def list_users(self) -> list[dict[str, Any]]:
        with self.store.transaction() as c:
            rows = c.execute(
                "SELECT username, role, active, created_at, created_by FROM users ORDER BY created_at, username"
            ).fetchall()
        return [{**dict(r), "active": bool(r["active"]), "role_label": ROLE_LABELS.get(r["role"], r["role"])}
                for r in rows]

    def count_users(self) -> int:
        with self.store.transaction() as c:
            return int(c.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def update_user(
        self, username: str, actor: str, *, role: str | None = None, active: bool | None = None,
        password: str | None = None,
    ) -> User:
        user = self.get_user(username)
        if user is None:
            raise KeyError(username)
        if role is not None and role not in ROLES:
            raise AuthError(f"role must be one of: {', '.join(ROLES)}")
        if password is not None and len(password) < MIN_PASSWORD:
            raise AuthError(f"password must be at least {MIN_PASSWORD} characters")
        new_role = role if role is not None else user.role
        new_active = active if active is not None else user.active
        loses_admin = user.role == "admin" and user.active and (new_role != "admin" or not new_active)
        changes = []
        with self.store.transaction() as c:
            if loses_admin:
                admins = c.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0]
                if admins <= 1:
                    raise AuthError("cannot remove the last active admin")
            if role is not None and role != user.role:
                c.execute("UPDATE users SET role=? WHERE username=?", (role, username))
                changes.append(f"role {user.role}->{role}")
            if active is not None and active != user.active:
                c.execute("UPDATE users SET active=? WHERE username=?", (int(active), username))
                changes.append("enabled" if active else "disabled")
            if password is not None:
                c.execute("UPDATE users SET password_hash=? WHERE username=?", (hash_password(password), username))
                changes.append("password reset")
            if active is False or password is not None:  # end existing sessions
                c.execute("DELETE FROM sessions WHERE username=?", (username,))
        if changes:
            self.audit(actor, "user_updated", username, ", ".join(changes))
        return User(username, new_role, new_active)

    # ---- sessions ------------------------------------------------------
    def login(self, username: str, password: str) -> tuple[str, User] | None:
        username = username.strip().lower()
        with self.store.transaction() as c:
            row = c.execute(
                "SELECT username, password_hash, role, active FROM users WHERE username = ?", (username,)
            ).fetchone()
        ok = verify_password(password, row["password_hash"] if row else _DUMMY_HASH)
        if not (row and ok and row["active"]):
            self.audit(username or None, "login_failed", username, "disabled" if row and ok else "bad credentials")
            return None
        token = secrets.token_urlsafe(32)
        now = _now()
        with self.store.transaction() as c:
            c.execute("DELETE FROM sessions WHERE expires_at < ?", (_iso(now),))
            c.execute("INSERT INTO sessions VALUES (?,?,?,?)",
                      (_token_hash(token), username, _iso(now), _iso(now + timedelta(hours=self.session_hours))))
        self.audit(username, "login", username)
        return token, User(row["username"], row["role"], True)

    def user_for_token(self, token: str) -> User | None:
        """The live user behind a session. Role changes and disabling take effect on the next request."""
        with self.store.transaction() as c:
            row = c.execute(
                """SELECT u.username, u.role FROM sessions s JOIN users u USING (username)
                   WHERE s.token_hash = ? AND s.expires_at > ? AND u.active = 1""",
                (_token_hash(token), _iso(_now())),
            ).fetchone()
        return User(row["username"], row["role"]) if row else None

    def logout(self, token: str) -> None:
        with self.store.transaction() as c:
            c.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))

    # ---- audit ---------------------------------------------------------
    def audit(self, actor: str | None, action: str, target: str | None = None, detail: str | None = None) -> None:
        with self.store.transaction() as c:
            c.execute("INSERT INTO audit_log (at, actor, action, target, detail) VALUES (?,?,?,?,?)",
                      (_iso(_now()), actor, action, target, detail))

    def recent_audit(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.store.transaction() as c:
            rows = c.execute("SELECT at, actor, action, target, detail FROM audit_log ORDER BY id DESC LIMIT ?",
                             (limit,)).fetchall()
        return [dict(r) for r in rows]

    def bootstrap_admin(self, username: str, password: str | None) -> str | None:
        """Create the first admin if there are no users. Returns a generated password, if one was made."""
        if self.count_users():
            return None
        generated = None
        if not password:
            password = generated = secrets.token_urlsafe(12)
        self.create_user(username, password, "admin", actor="system")
        return generated
