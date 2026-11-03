"""A site's signed session cookie: who the browser is signed in as, or that it signed out, signed with HMAC-SHA256
under ``WEBSITE_SESSION_SECRET``, or a secret of the process's own when that isn't set. The cookie names no domain,
so it holds under whatever host the site is reached at."""
from __future__ import annotations

import base64
import binascii
import datetime
import hashlib
import hmac
import json
import os
import secrets
import time

SESSION_COOKIE = "agentenv_session"
SECRET_ENV = "WEBSITE_SESSION_SECRET"
SESSION_TTL_S = 7 * 24 * 3600
SECRET_BYTES = 32


class SessionSigner:
    def __init__(self, secret: str | bytes | None = None) -> None:
        secret = secret if secret is not None else os.environ.get(SECRET_ENV)
        self._key = (secret.encode() if isinstance(secret, str) else secret) or secrets.token_bytes(SECRET_BYTES)

    def encode(self, user: str | None, ttl_s: int = SESSION_TTL_S) -> tuple[str, int]:
        """The cookie value for ``user`` signed in, or signed out when None, and when it expires, in epoch
        seconds."""
        expires = int(time.time()) + ttl_s
        payload = base64.urlsafe_b64encode(json.dumps({"u": user, "exp": expires}).encode()).decode().rstrip("=")
        return f"{payload}.{self._sign(payload)}", expires

    def decode(self, value: str | None) -> dict | None:
        """The session a cookie value carries, ``{"user": ...}``; None for one that is absent, forged or expired."""
        payload, _, signature = (value or "").partition(".")
        if not payload or not hmac.compare_digest(signature.encode("utf-8", "surrogateescape"),
                                                  self._sign(payload).encode()):
            return None
        try:
            data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except (ValueError, binascii.Error):
            return None
        if not isinstance(data, dict) or not isinstance(data.get("exp"), int) or data["exp"] < time.time():
            return None
        return {"user": data.get("u")}

    def cookie(self, user: str | None, ttl_s: int = SESSION_TTL_S) -> dict:
        """The cookie a browser sets to be signed in as ``user``, as a browser automation adds one; its domain is
        left to whoever adds it."""
        value, expires = self.encode(user, ttl_s)
        return {"name": SESSION_COOKIE, "value": value, "path": "/", "expires": expires, "httpOnly": True,
                "sameSite": "Lax"}

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode("utf-8", "surrogateescape"), hashlib.sha256).hexdigest()


def expires_at(epoch: int) -> str:
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat().replace("+00:00", "Z")
