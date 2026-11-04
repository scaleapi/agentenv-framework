"""A site's signed session cookie: who the browser is signed in as, or that it signed out, signed with HMAC-SHA256
under ``WEBSITE_SESSION_SECRET``, which a deploy sets for each site so its sessions outlive its process, or a secret
of the process's own when that isn't set. The cookie names no domain, so it holds under whatever host the site is
reached at, and its lifetime is a ``Max-Age``, never an absolute ``Expires``, so the browser's clock and the site's
needn't agree."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time

SESSION_COOKIE = "agentenv_session"
SESSION_SECRET_ENV = "WEBSITE_SESSION_SECRET"
SESSION_TTL_S = 7 * 24 * 3600
SECRET_BYTES = 32


class SessionSigner:
    def __init__(self, secret: str | bytes | None = None) -> None:
        secret = secret if secret is not None else os.environ.get(SESSION_SECRET_ENV)
        self._key = (secret.encode() if isinstance(secret, str) else secret) or secrets.token_bytes(SECRET_BYTES)

    def encode(self, user: str | None, ttl_s: int = SESSION_TTL_S) -> str:
        """The cookie value for ``user`` signed in, or signed out when None, good for ``ttl_s`` seconds."""
        payload = base64.urlsafe_b64encode(json.dumps({"u": user, "exp": int(time.time()) + ttl_s}).encode())
        payload = payload.decode().rstrip("=")
        return f"{payload}.{self._sign(payload)}"

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
        """The cookie a browser adds to be signed in as ``user``, in browser/v1's shape with ``max_age`` for its
        lifetime; its domain is left to whoever adds it."""
        return {"name": SESSION_COOKIE, "value": self.encode(user, ttl_s), "path": "/", "max_age": ttl_s,
                "httpOnly": True, "sameSite": "Lax"}

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode("utf-8", "surrogateescape"), hashlib.sha256).hexdigest()
