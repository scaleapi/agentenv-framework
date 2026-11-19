"""A browser's session on the env's pgweb console, in place of the control token, which never reaches a browser.

The deployment's owner mints a short link from the control token (``ConsoleSession.link``); the gateway trades it for a
cookie that opens ``/svc/pgweb/`` only. Both are ``<expiry>.<signature>``: unix seconds, and the HMAC-SHA256 of
``link:<expiry>`` or ``cookie:<expiry>``, base64url without padding, under ``HMAC-SHA256(control token,
"pgweb-console-v1")``. A link or cookie minted for another deployment fails its signature.

Every Modal tunnel is same-site with every other (``modal.host`` is not a public suffix), so SameSite alone would
not keep the cookie to its own deployment's pages. The cookie therefore counts only on a request from a page of the
gateway's own origin, or on a navigation to the console's index.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import time
from collections.abc import Callable, Iterable
from urllib.parse import parse_qsl

from .constants import PGWEB_CONSOLE

LINK_PARAM = "pgweb_link"
COOKIE_NAME = "__Host-agentenv-pgweb"
CONSOLE_PREFIX = f"/svc/{PGWEB_CONSOLE}/"
LINK_MAX_SECONDS = 300
SESSION_SECONDS = 3600
_KEY_LABEL = b"pgweb-console-v1"
_SIGNED = re.compile(r"([0-9]{1,12})\.([A-Za-z0-9_-]{43})")
_LINK_IN_LOG = re.compile(rf"({LINK_PARAM}=)[^&\s\"']+")


class ConsoleSession:
    """Mints and checks a deployment's pgweb links and cookies. ``now`` is the clock, in unix seconds."""

    def __init__(self, control_token: str, now: Callable[[], float] = time.time):
        self._key = hmac.new(control_token.encode(), _KEY_LABEL, hashlib.sha256).digest()
        self._now = now

    def _sign(self, purpose: str, expires_at: int) -> str:
        digest = hmac.new(self._key, f"{purpose}:{expires_at}".encode(), hashlib.sha256).digest()
        return f"{expires_at}.{base64.urlsafe_b64encode(digest).rstrip(b'=').decode()}"

    def _valid(self, purpose: str, value: str, longest: int) -> bool:
        match = _SIGNED.fullmatch(value)
        if not match:
            return False
        expires_at = int(match[1])
        now = int(self._now())
        return now < expires_at <= now + longest and hmac.compare_digest(self._sign(purpose, expires_at), value)

    def link(self, seconds: int = 120) -> str:
        return self._sign("link", int(self._now()) + seconds)

    def opens(self, link: str) -> bool:
        return self._valid("link", link, LINK_MAX_SECONDS)

    def set_cookie(self) -> str:
        value = self._sign("cookie", int(self._now()) + SESSION_SECONDS)
        return (f"{COOKIE_NAME}={value}; Max-Age={SESSION_SECONDS}; Path=/; Secure; HttpOnly; SameSite=None; "
                "Partitioned")

    def admits(self, scope: dict) -> bool:
        """Whether ``scope``, a request to the pgweb console, carries a live cookie and comes from where one counts."""
        headers = scope.get("headers", ())
        if not any(self._valid("cookie", value, SESSION_SECONDS) for value in cookie_values(headers)):
            return False
        fetch = {name: value for name, value in headers if name in (b"sec-fetch-site", b"sec-fetch-mode")}
        if fetch.get(b"sec-fetch-site") == b"same-origin":
            return True
        return (fetch.get(b"sec-fetch-mode") == b"navigate" and scope["method"] == "GET"
                and scope["path"] == CONSOLE_PREFIX)

    def opening_link(self, scope: dict) -> str | None:
        """The link a GET of the console's index carries, or None."""
        if scope["method"] != "GET" or scope["path"] != CONSOLE_PREFIX:
            return None
        return dict(parse_qsl(scope.get("query_string", b"").decode("latin-1"))).get(LINK_PARAM)


def cookie_values(headers: Iterable[tuple[bytes, bytes]]) -> list[str]:
    """Every value the request's Cookie headers give the session cookie."""
    found = []
    for name, value in headers:
        if name != b"cookie":
            continue
        for pair in value.decode("latin-1").split(";"):
            key, _, cookie = pair.strip().partition("=")
            if key == COOKIE_NAME:
                found.append(cookie)
    return found


def without_session_cookie(headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """``headers`` with the session cookie taken out of each Cookie header, and a header left empty dropped."""
    kept = []
    for name, value in headers:
        if name == b"cookie":
            pairs = [pair for pair in value.decode("latin-1").split(";")
                     if pair.strip().partition("=")[0] != COOKIE_NAME]
            if not any(pair.strip() for pair in pairs):
                continue
            value = ";".join(pairs).strip().encode("latin-1")
        kept.append((name, value))
    return kept


class LinkRedaction(logging.Filter):
    """Takes the link out of each log line that names one. It rewrites the argument that holds it, since uvicorn's
    access log formatter reads its request line from the record's arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str) and LINK_PARAM in record.msg:
            record.msg = _redacted(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_redacted(str(arg)) if LINK_PARAM in str(arg) else arg for arg in record.args)
        return True


def _redacted(text: str) -> str:
    return _LINK_IN_LOG.sub(r"\1<redacted>", text)
