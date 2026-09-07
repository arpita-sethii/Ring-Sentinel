"""
Login/register/logout for Ring Sentinel — in-memory, resets on server
restart, matching the same "session-only" philosophy already used
everywhere else in this app (retrain, feedback stats, drift acknowledgment
all reset on restart too — this isn't a special exception).

Deliberately stdlib-only (hashlib.pbkdf2_hmac, no bcrypt/passlib) — this
project has a real history of dependency pain (xgboost/sklearn version
pins, shap/numba DLL issues, pickle cross-version breaks), so a new auth
library is a real, avoidable risk for a feature this size. pbkdf2_hmac
with a per-user random salt and 100k iterations is genuinely secure
password storage, not a toy — just without a third-party package.
"""
import hashlib
import os
import secrets
from datetime import datetime, timezone

USERS: dict[str, dict] = {}       # username -> {"salt": bytes, "password_hash": str, "created_at": str}
SESSIONS: dict[str, str] = {}     # session_token -> username


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


def register_user(username: str, password: str) -> dict:
    username = username.strip()
    if not username or not password:
        raise ValueError("Username and password are required.")
    if len(username) < 2:
        raise ValueError("Username must be at least 2 characters.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters.")
    if username in USERS:
        raise ValueError(f"Username '{username}' is already taken.")

    salt = os.urandom(16)
    USERS[username] = {
        "salt": salt,
        "password_hash": _hash_password(password, salt),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"username": username}


def verify_login(username: str, password: str) -> bool:
    user = USERS.get(username)
    if not user:
        return False
    return _hash_password(password, user["salt"]) == user["password_hash"]


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = username
    return token


def get_user_from_token(token: str | None) -> str | None:
    if not token:
        return None
    return SESSIONS.get(token)


def destroy_session(token: str | None):
    if token and token in SESSIONS:
        del SESSIONS[token]
