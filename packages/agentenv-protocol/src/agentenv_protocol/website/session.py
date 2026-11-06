"""A site's signed session cookie: who the browser is signed in as, or that it signed out, signed with HMAC-SHA256
under ``WEBSITE_SESSION_SECRET``, which a deploy sets for each site so its sessions outlive its process, or a secret
of the process's own when that isn't set. The cookie names no domain, so it holds under whatever host the site is
reached at, and its lifetime is a ``Max-Age``, never an absolute ``Expires``, so the browser's clock and the site's
needn't agree. Its name is the site's own, ``agentenv_session_<name>``, so two sites a browser reaches under one gateway
host (``/website/a/``, ``/website/b/``) keep a session each."""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import time

SESSION_COOKIE = "agentenv_session"
SESSION_SECRET_ENV = "WEBSITE_SESSION_SECRET"
SESSION_TTL_S = 7 * 24 * 3600
SECRET_BYTES = 32
_NOT_TOKEN = re.compile(r"[^A-Za-z0-9_.-]")


def _cookie_suffix(environment_name: str) -> str:
    """The part of a site's cookie names that is its own: the name itself when a cookie name can hold each of its
    characters, else the name with each such character made ``_`` and ``~`` and a hash of the whole name appended. No
    name kept as itself holds a ``~``, so two distinct names never make one suffix."""
    if not _NOT_TOKEN.search(environment_name):
        return environment_name
    digest = hashlib.sha256(environment_name.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return f"{_NOT_TOKEN.sub('_', environment_name)}~{digest}"


def session_cookie_name(environment_name: str | None) -> str:
    """The session cookie of the site named ``environment_name``: ``agentenv_session_<name>``, a name a cookie name
    can't hold spelled as ``_cookie_suffix`` spells it; the shared ``agentenv_session`` for a site with no name."""
    return f"{SESSION_COOKIE}_{_cookie_suffix(environment_name)}" if environment_name else SESSION_COOKIE


class SessionSigner:
    def __init__(self, secret: str | bytes | None = None, cookie_name: str = SESSION_COOKIE) -> None:
        secret = secret if secret is not None else os.environ.get(SESSION_SECRET_ENV)
        self._key = (secret.encode() if isinstance(secret, str) else secret) or secrets.token_bytes(SECRET_BYTES)
        self.cookie_name = cookie_name

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

    def read(self, cookies: dict) -> dict | None:
        """The session of a request's ``cookies``, from the site's own cookie."""
        return self.decode(cookies.get(self.cookie_name))

    def cookie(self, user: str | None, ttl_s: int = SESSION_TTL_S) -> dict:
        """The cookie a browser adds to be signed in as ``user``, in browser/v1's shape with ``max_age`` for its
        lifetime; its domain is left to whoever adds it."""
        return {"name": self.cookie_name, "value": self.encode(user, ttl_s), "path": "/", "max_age": ttl_s,
                "httpOnly": True, "sameSite": "Lax"}

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode("utf-8", "surrogateescape"), hashlib.sha256).hexdigest()
