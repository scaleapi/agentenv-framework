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
#: The words of a name (a query parameter, a signed form's field, a key in a write-only mapping) that mark its value as a
#: credential, and the endings that do: ``X-Amz-Signature`` and Azure's ``sig`` are, ``X-Amz-Algorithm``, ``X-Amz-Date``
#: and ``X-Amz-SignedHeaders`` aren't, so ``AWS4-HMAC-SHA256`` is left in an agent's text.
_CREDENTIAL_WORDS = frozenset({
    "auth", "authorization", "bearer", "cookie", "credential", "credentials", "jwt", "key", "passwd", "password",
    "policy", "pwd", "secret", "sig", "signature", "token", "accessid", "googleaccessid", "awsaccesskeyid",
})
_CREDENTIAL_ENDINGS = ("key", "token", "secret", "password", "signature", "credential")
_NAME_WORD = re.compile(r"[a-z0-9]+")
#: How long a path segment of a URL with no query must be to be taken as its credential, as a grant server's signed
#: token is: longer than an ordinary path's words, unlike ``MIN_SECRET_LENGTH``, which bounds what is scrubbed at all.
MIN_TOKEN_SEGMENT_LENGTH = 16
_TOKEN_SEGMENT = re.compile(rf"[A-Za-z0-9_.=~+-]{{{MIN_TOKEN_SEGMENT_LENGTH},}}")

T = TypeVar("T")


class Secrets:
    """The values an agent's atif/1 reads leave out, each written ``[redacted:<length>]`` wherever it occurs, in its raw,
    JSON-escaped (ASCII or not) and percent-encoded forms. Safe to add to from the agent's threads while the loop
    scrubs. The match is compiled as a prefix trie, once per change and when a read needs it, so a call that hands the
    agent a grant per file, or a run that accumulates thousands, costs neither a rebuild per secret nor a slower
    scan."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._forms: set[str] = set()
        self._pattern: re.Pattern[str] | None = None
        self._stale = False
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
        forms = {text, json.dumps(text)[1:-1], json.dumps(text, ensure_ascii=False)[1:-1], quote(text, safe="")}
        with self._lock:
            if not forms <= self._forms:
                self._forms |= forms
                self._stale = True

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
        """The credentials a write-only config value holds: a string or list of strings whole, and in a mapping or a model
        only what a credential-like key names (and everything below it), so a write-only mapping of sampling parameters,
        as pi's ``model_params`` is, doesn't have ``priority`` or a provider's name redacted wherever it occurs."""
        self.add_all(_credentials(value, whole=not isinstance(value, (Mapping, BaseModel))), source=source)

    def scrub(self, value: T) -> T:
        """``value`` with every secret redacted in its strings: its mappings' keys and values and its lists' items."""
        pattern = self._match()
        return value if pattern is None else _scrub(value, pattern)

    def _match(self) -> re.Pattern[str] | None:
        with self._lock:
            if self._stale:
                self._pattern = re.compile(_trie(self._forms))
                self._stale = False
            return self._pattern


def _scrub(value: Any, pattern: re.Pattern[str]) -> Any:
    if isinstance(value, str):
        return pattern.sub(lambda match: f"[redacted:{len(match.group(0))}]", value)
    if isinstance(value, dict):
        return {_scrub(key, pattern): _scrub(item, pattern) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub(item, pattern) for item in value]
    return value


def _trie(forms: Iterable[str]) -> str:
    """One expression matching each of ``forms``, the longest at any position: a prefix trie, whose shared prefixes
    are matched once, with each unbranching run written as one literal, so its nesting is its branching."""
    root: dict[str, Any] = {}
    for form in forms:
        node = root
        for char in form:
            node = node.setdefault(char, {})
        node[""] = {}
    return _branch(root)


def _branch(node: dict[str, Any]) -> str:
    run: list[str] = []
    while len(node) == 1 and "" not in node:
        [(char, node)] = node.items()
        run.append(re.escape(char))
    children = sorted(char for char in node if char)
    if not children:
        return "".join(run)
    group = "(?:" + "|".join(re.escape(char) + _branch(node[char]) for char in children) + ")"
    # An end here as well: the longer match first, the end if it fails.
    return "".join(run) + (f"{group}?" if "" in node else group)


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


def _credentials(value: Any, *, whole: bool) -> Iterator[str | SecretStr]:
    """The strings ``value`` holds that are credentials: all of them when ``whole`` (a write-only string field, or what a
    credential-like key names), else those a credential-like key or field name marks, and everything below it."""
    if isinstance(value, (str, SecretStr)):
        if whole:
            yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _credentials(item, whole=whole)
    elif isinstance(value, BaseModel):
        for name in type(value).model_fields:
            yield from _credentials(getattr(value, name), whole=whole or _names_credential(name))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _credentials(item, whole=whole or isinstance(key, str) and _names_credential(key))


def _names_credential(name: str) -> bool:
    words = _NAME_WORD.findall(name.lower())
    return any(word in _CREDENTIAL_WORDS or word.endswith(_CREDENTIAL_ENDINGS) for word in words)


def _grant_secrets(grant: HttpGetGrant | HttpPutGrant | HttpPostPolicyGrant) -> list[str]:
    """A grant's credential headers, and a signed upload form's signing fields."""
    secrets = _header_values(grant.headers)
    if isinstance(grant, HttpPostPolicyGrant):
        secrets += [value for name, value in grant.fields.items()
                    if name != grant.path_field and _names_credential(name)]
    return secrets


def _url_secrets(url: str) -> list[str]:
    """A URL that can carry a credential, in its path, query or user info: whole, and the part that carries it. A
    presigned URL keeps its signature in its query, so its credential-like query values and its path and query as a
    request line names them (``curl -v`` prints that); a URL with no query keeps its token in a long path segment. A
    bare origin carries none."""
    parts = urlsplit(url)
    if not parts.path.strip("/") and not parts.query and parts.password is None:
        return []
    if parts.query:
        carried = [f"{parts.path}?{parts.query}", *_query_values(url)]
    else:
        carried = [segment for segment in parts.path.split("/") if _TOKEN_SEGMENT.fullmatch(segment)]
    return [url, *carried, *([parts.password] if parts.password else [])]


def _query_values(url: str) -> list[str]:
    """The values of a URL's credential-like query parameters: a signature, a credential, a token, a key."""
    return [value for name, value in parse_qsl(urlsplit(url).query) if _names_credential(name)]


def _header_values(headers: Mapping[str, Any] | None) -> list[str]:
    return [value for name, value in (headers or {}).items()
            if isinstance(value, str) and name.lower() not in _PLAIN_HEADERS]
