"""The secrets an agent's atif/1 reads leave out: every value the SDK holds or is handed (its control token, write-only
agent-config values, MCP and CDP registrations' credentials, the grants and tokens in extension calls, the model
credentials in its environment) and every one the agent registers. Each read scrubs every string it serves, so a
secret learned after a step was written is still left out; native reads are never scrubbed."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, TypeVar
from urllib.parse import parse_qsl, quote, urlsplit

from pydantic import BaseModel, SecretStr

from ..transfers import HttpGetGrant, HttpPostPolicyGrant, HttpPutGrant
from .extensions import CdpAddRequest, ClockSyncTimeRequest, McpAddRequest

logger = logging.getLogger(__name__)

#: A shorter value isn't scrubbed: replacing it everywhere it occurs would mangle ordinary text.
MIN_SECRET_LENGTH = 8
#: The model credentials an agent's environment holds, which a tool can echo.
MODEL_KEY_ENV = "LITELLM_API_KEY"
MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"
#: Header names whose values describe a request rather than authorize it.
_PLAIN_HEADERS = frozenset({"accept", "content-encoding", "content-length", "content-type", "user-agent"})
#: Words in a signed upload form's field names that mark the values that sign it.
_SIGNING_FIELDS = ("policy", "signature", "credential", "token", "accesskey", "accessid")
#: A path segment a URL with no query can carry its credential in, as a grant server's signed token.
_TOKEN_SEGMENT = re.compile(r"[A-Za-z0-9_.=~+-]{16,}")

T = TypeVar("T")


class Secrets:
    """The values an agent's atif/1 reads leave out, each written ``[redacted:<length>]`` wherever it occurs, in its raw,
    JSON-escaped and percent-encoded forms. Safe to add to from the agent's threads while the loop scrubs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._forms: set[str] = set()
        self._pattern: re.Pattern[str] | None = None
        self._short: set[str] = set()

    def add(self, value: str | SecretStr | None, *, source: str, warn: bool = True) -> None:
        """Leave ``value`` out of every atif/1 read from now on; ``source`` names it in the warning a value too short to
        scrub gets, never the value, unless ``warn`` is off, as for the parts of a URL."""
        text = value.get_secret_value() if isinstance(value, SecretStr) else value
        if not isinstance(text, str) or not text:
            return
        if len(text) < MIN_SECRET_LENGTH:
            if warn and source not in self._short:
                self._short.add(source)
                logger.warning("a secret from %s is shorter than %d characters, so atif/1 reads don't scrub it",
                               source, MIN_SECRET_LENGTH)
            return
        forms = {text, json.dumps(text)[1:-1], quote(text, safe="")}
        with self._lock:
            if forms <= self._forms:
                return
            self._forms |= forms
            ordered = sorted(self._forms, key=len, reverse=True)
            self._pattern = re.compile("|".join(re.escape(form) for form in ordered))

    def add_all(self, values: Iterable[str | SecretStr | None], *, source: str, warn: bool = True) -> None:
        for value in values:
            self.add(value, source=source, warn=warn)

    def add_environment(self, environ: Mapping[str, str] | None = None) -> None:
        """The model credentials in the agent's environment: its key, and its extra headers' values."""
        environ = os.environ if environ is None else environ
        self.add(environ.get(MODEL_KEY_ENV), source=MODEL_KEY_ENV)
        try:
            headers = json.loads(environ.get(MODEL_HEADERS_ENV) or "{}")
        except ValueError:
            return
        if isinstance(headers, dict):
            self.add_all(_header_values(headers), source=MODEL_HEADERS_ENV)

    def add_request(self, request: BaseModel | None) -> None:
        """Every secret an extension call hands the agent: its grants' URLs, query values, headers and signing fields,
        an MCP or CDP registration's credentials and a clock follower's read token, wherever the request holds them."""
        for item in _models(request):
            if isinstance(item, (HttpGetGrant, HttpPutGrant, HttpPostPolicyGrant)):
                self.add_all(_url_secrets(str(item.url)), source="a transfer grant", warn=False)
                self.add_all(_grant_secrets(item), source="a transfer grant")
            elif isinstance(item, McpAddRequest):
                self.add_all(_query_values(item.url), source="an MCP registration", warn=False)
                self.add_all(_header_values(item.headers), source="an MCP registration's headers")
            elif isinstance(item, CdpAddRequest):
                self.add_all([*_url_secrets(item.http_url), *_url_secrets(item.ws_url)], source="a CDP registration",
                             warn=False)
                self.add_all(_header_values(item.headers), source="a CDP registration's headers")
            elif isinstance(item, ClockSyncTimeRequest):
                self.add(item.read_token, source="a clock follower's read token")

    def add_values(self, value: Any, *, source: str) -> None:
        """Every string ``value`` holds, in fields, lists and mappings, as a write-only config value may."""
        self.add_all(_strings(value), source=source)

    def scrub(self, value: T) -> T:
        """``value`` with every secret in its strings, its mappings' values and its lists' items redacted."""
        pattern = self._pattern
        return value if pattern is None else _scrub(value, pattern)


def _scrub(value: Any, pattern: re.Pattern[str]) -> Any:
    if isinstance(value, str):
        return pattern.sub(lambda match: f"[redacted:{len(match.group(0))}]", value)
    if isinstance(value, dict):
        return {key: _scrub(item, pattern) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, pattern) for item in value]
    return value


def _models(value: Any) -> Iterator[BaseModel]:
    """``value`` and every model it holds, in fields, lists and mappings."""
    if isinstance(value, BaseModel):
        yield value
        for name in type(value).model_fields:
            yield from _models(getattr(value, name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _models(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _models(item)


def _strings(value: Any) -> Iterator[str | SecretStr]:
    if isinstance(value, (str, SecretStr)):
        yield value
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _strings(getattr(value, name))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)


def _grant_secrets(grant: HttpGetGrant | HttpPutGrant | HttpPostPolicyGrant) -> list[str]:
    """A grant's credential headers, and a signed upload form's signing fields."""
    secrets = _header_values(grant.headers)
    if isinstance(grant, HttpPostPolicyGrant):
        secrets += [value for name, value in grant.fields.items()
                    if name != grant.path_field and any(word in name.lower() for word in _SIGNING_FIELDS)]
    return secrets


def _url_secrets(url: str) -> list[str]:
    """A URL that can carry a credential, in its path, query or user info: whole, and the part that carries it. A
    presigned URL keeps its signature in its query, so its query's values and its path and query as a request line
    names them (``curl -v`` prints that); a URL with no query keeps its token in a long path segment. A bare origin
    carries none."""
    parts = urlsplit(url)
    if not parts.path.strip("/") and not parts.query and parts.password is None:
        return []
    if parts.query:
        carried = [f"{parts.path}?{parts.query}", *_query_values(url)]
    else:
        carried = [segment for segment in parts.path.split("/") if _TOKEN_SEGMENT.fullmatch(segment)]
    return [url, *carried, *([parts.password] if parts.password else [])]


def _query_values(url: str) -> list[str]:
    return [value for _, value in parse_qsl(urlsplit(url).query)]


def _header_values(headers: Mapping[str, Any] | None) -> list[str]:
    return [value for name, value in (headers or {}).items()
            if isinstance(value, str) and name.lower() not in _PLAIN_HEADERS]
