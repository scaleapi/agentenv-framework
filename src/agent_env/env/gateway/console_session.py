"""A browser's session on the env's pgweb console in place of the control token, which never reaches a browser: a
short single-use link minted from the token, which the gateway trades for a cookie of this deploy's own."""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import secrets
import time
from collections.abc import Callable, Iterable

from .constants import PGWEB_CONSOLE

LINK_PARAM = "pgweb_link"
COOKIE_PREFIX = "__Host-agentenv-pgweb-"
CONSOLE_PREFIX = f"/svc/{PGWEB_CONSOLE}/"
CONSOLE_INDEX_PATHS = (CONSOLE_PREFIX, CONSOLE_PREFIX.rstrip("/"))
LINK_DEFAULT_SECONDS = 120
LINK_MAX_SECONDS = 300
# The session lasts this long after its last use: each request past half of it carries a fresh cookie.
SESSION_SECONDS = 900
EXPIRY_MAX_DIGITS = 12
SIGNATURE_CHARS = 43  # an unpadded base64url SHA-256 digest
NONCE_BYTES, NONCE_CHARS = 8, 11  # a link's nonce, and its unpadded base64url
COOKIE_NAME_HEX_CHARS = 8
_KEY_LABEL = b"pgweb-console-v1"
_COOKIE = re.compile(rf"([0-9]{{1,{EXPIRY_MAX_DIGITS}}})\.([A-Za-z0-9_-]{{{SIGNATURE_CHARS}}})")
_LINK = re.compile(rf"([0-9]{{1,{EXPIRY_MAX_DIGITS}}})\.([A-Za-z0-9_-]{{{NONCE_CHARS}}})\.([A-Za-z0-9_-]{{{SIGNATURE_CHARS}}})")
_LINK_IN_LOG = re.compile(rf"({LINK_PARAM}=)[^&\s\"']+")
_FRAME_ANCESTOR = re.compile(r"https?://(\*\.)?[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*(:[0-9]{1,5})?")


class ConsoleSession:
    """Mints and checks one deploy's pgweb links and session cookies; ``now`` is the clock, in unix seconds."""

    def __init__(self, control_token: str, now: Callable[[], float] = time.time):
        self._key = hmac.new(control_token.encode(), _KEY_LABEL, hashlib.sha256).digest()
        self._now = now
        self._spent: dict[str, int] = {}
        name = hmac.new(self._key, b"cookie-name", hashlib.sha256).hexdigest()[:COOKIE_NAME_HEX_CHARS]
        self.cookie_name = COOKIE_PREFIX + name

    def _mac(self, text: str) -> str:
        digest = hmac.new(self._key, text.encode(), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    def _live(self, expires_at: int, longest: int) -> bool:
        now = int(self._now())
        return now < expires_at <= now + longest

    def link(self, seconds: int = LINK_DEFAULT_SECONDS) -> str:
        """A link that opens the console once within ``seconds``: ``<expiry>.<nonce>.<signature>``, so two links minted
        in the same second still differ."""
        if not 0 < seconds <= LINK_MAX_SECONDS:
            raise ValueError(f"a pgweb link lasts 1 to {LINK_MAX_SECONDS} seconds, not {seconds}")
        expires_at, nonce = int(self._now()) + seconds, secrets.token_urlsafe(NONCE_BYTES)
        return f"{expires_at}.{nonce}.{self._mac(f'link:{expires_at}:{nonce}')}"

    def opens(self, link: str) -> bool:
        """Whether ``link`` is live and unused; a link that opens is spent."""
        now = int(self._now())
        self._spent = {spent: expires_at for spent, expires_at in self._spent.items() if expires_at > now}
        match = _LINK.fullmatch(link)
        if link in self._spent or not match or not self._live(expires_at := int(match[1]), LINK_MAX_SECONDS) \
                or not hmac.compare_digest(self._mac(f"link:{expires_at}:{match[2]}"), match[3]):
            return False
        self._spent[link] = expires_at
        return True

    def _cookie_value(self, expires_at: int) -> str:
        return f"{expires_at}.{self._mac(f'cookie:{expires_at}')}"

    def set_cookie(self) -> str:
        value = self._cookie_value(int(self._now()) + SESSION_SECONDS)
        return (f"{self.cookie_name}={value}; Max-Age={SESSION_SECONDS}; Path=/; Secure; HttpOnly; SameSite=None; "
                "Partitioned")

    def _session_expiry(self, scope: dict) -> int | None:
        expiries = []
        for value in cookie_values(scope.get("headers", ()), self.cookie_name):
            match = _COOKIE.fullmatch(value)
            if match and self._live(expires_at := int(match[1]), SESSION_SECONDS) \
                    and hmac.compare_digest(self._cookie_value(expires_at), value):
                expiries.append(expires_at)
        return max(expiries, default=None)

    def admits(self, scope: dict) -> bool:
        """Whether ``scope``, a request to the pgweb console, carries a live cookie and comes from where one counts:
        a page of the gateway's own origin, or a navigation to the console's index."""
        if self._session_expiry(scope) is None:
            return False
        fetch = {name: value for name, value in scope.get("headers", ())
                 if name in (b"sec-fetch-site", b"sec-fetch-mode")}
        if fetch.get(b"sec-fetch-site") == b"same-origin":
            return True
        return (fetch.get(b"sec-fetch-mode") == b"navigate" and scope["method"] == "GET"
                and scope["path"] in CONSOLE_INDEX_PATHS)

    def renews(self, scope: dict) -> bool:
        """Whether the session ``scope`` carries is past half its life, so its answer carries a fresh cookie."""
        expires_at = self._session_expiry(scope)
        return expires_at is not None and expires_at - int(self._now()) < SESSION_SECONDS // 2

    def opening_link(self, scope: dict) -> str | None:
        """The link a GET of the console's index carries under its literal name, or None."""
        if scope["method"] != "GET" or scope["path"] not in CONSOLE_INDEX_PATHS:
            return None
        for pair in scope.get("query_string", b"").decode("latin-1").split("&"):
            name, equals, value = pair.partition("=")
            if name == LINK_PARAM and equals:
                return value
        return None


def frame_ancestor_problem(origin: str) -> str | None:
    """Why ``origin`` can't be a frame-ancestors source: it must be ``http(s)://host[:port]``, a leading ``*.`` allowed."""
    if _FRAME_ANCESTOR.fullmatch(origin):
        return None
    return f"{origin!r} is not an origin like https://app.example.com"


def cookie_values(headers: Iterable[tuple[bytes, bytes]], name: str) -> list[str]:
    """Every value the request's Cookie headers give the cookie ``name``."""
    found = []
    for header, value in headers:
        if header != b"cookie":
            continue
        for pair in value.decode("latin-1").split(";"):
            key, _, cookie = pair.strip().partition("=")
            if key == name:
                found.append(cookie)
    return found


def without_session_cookie(headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """``headers`` with every deploy's pgweb session cookie taken out of each Cookie header, a header left empty
    dropped."""
    kept = []
    for header, value in headers:
        if header == b"cookie":
            pairs = [pair for pair in value.decode("latin-1").split(";")
                     if not pair.strip().partition("=")[0].startswith(COOKIE_PREFIX)]
            if not any(pair.strip() for pair in pairs):
                continue
            value = ";".join(pairs).strip().encode("latin-1")
        kept.append((header, value))
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
