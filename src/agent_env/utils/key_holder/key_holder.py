"""The key holder: the one process in an agent box that holds the model key, run as its own user under root's
supervisord, as the egress proxy is. ``seal``, run as root, writes the key, the endpoint and its ``[model.headers]``
from its own environment into a file only this user reads; ``serve`` forwards model calls on loopback to that endpoint
for clients that hold only a per-context token, so no command the model runs holds the key or the headers.

The image copies this file alone and runs it with ``python3 -I``: it imports the standard library, starlette, uvicorn
and httpx, never agent-env or its protocol package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import secrets
import stat
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

PORT = 18431
USER = "agentenv-keyholder"
SEALED_DIR = Path("/run/agentenv-keyholder")
SEALED_FILE = SEALED_DIR / "sealed.json"
SEALED_VERSION = 1
DEFAULT_ROUTE = "default"
TOKEN_PREFIX = "agentenv-ctx-"
#: The model key's names; the first one set is sealed.
KEY_NAMES = ("LITELLM_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
             "GOOGLE_API_KEY")
#: The model endpoint's names; the first one set is sealed.
BASE_URL_NAMES = ("LITELLM_BASE_URL", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "GOOGLE_GEMINI_BASE_URL")
MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"
MAX_BODY_BYTES = 100 * 1024 * 1024
CONNECT_TIMEOUT_S = 60
READ_TIMEOUT_S = 1800
IDLE_CONNECTION_TIMEOUT_S = 3600
#: Shorter sealed values aren't scrubbed from answers: replacing every occurrence of one would hit unrelated text.
MIN_SCRUB_LENGTH = 8
REDACTED = b"[agentenv-redacted]"


class SealError(Exception):
    """The sealed file can't be written or served."""


@dataclass(frozen=True)
class ModelRoute:
    """An endpoint the key holder forwards model calls to, with the key and headers it sets on them."""

    upstream: str
    key: str
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Sealed:
    """The sealed file: model routes by name (``default`` for the run's endpoint) and sealed headers by origin, a slot
    for later credentials (Cloudflare Access fronting, MCP role tokens) that P1 leaves empty."""

    model_routes: dict[str, ModelRoute]
    upstream_headers: dict[str, dict[str, str]] = field(default_factory=dict)

    def to_json(self) -> str:
        routes = {name: {"upstream": route.upstream, "key": route.key, "headers": route.headers}
                  for name, route in self.model_routes.items()}
        return json.dumps({"version": SEALED_VERSION, "model_routes": routes,
                           "upstream_headers": self.upstream_headers})

    @classmethod
    def from_json(cls, raw: str | bytes) -> Sealed:
        try:
            data = json.loads(raw)
        except ValueError as error:
            raise SealError("the sealed file isn't JSON") from error
        if not isinstance(data, dict) or data.get("version") != SEALED_VERSION:
            version = data.get("version") if isinstance(data, dict) else None
            raise SealError(f"the sealed file's version {version!r} isn't {SEALED_VERSION}, the one this key holder "
                            "serves")
        routes = data.get("model_routes")
        upstream_headers = data.get("upstream_headers", {})
        if not isinstance(routes, dict) or not isinstance(upstream_headers, dict):
            raise SealError("the sealed file's model_routes and upstream_headers must be objects")
        parsed = {}
        for name, route in routes.items():
            if not isinstance(route, dict) or not all(isinstance(route.get(part), str) for part in ("upstream", "key")):
                raise SealError(f"the sealed model route {name!r} needs a string upstream and key")
            parsed[name] = ModelRoute(_checked_upstream(route["upstream"]), route["key"],
                                      _string_map(route.get("headers", {}), f"model route {name!r}'s headers"))
        return cls(parsed, {origin: _string_map(headers, f"{origin}'s headers")
                            for origin, headers in upstream_headers.items()})


@dataclass(frozen=True)
class Context:
    """What a per-context token stands for: the run's context, the models it may call (the first is what any other is
    pinned to) and the project and task its calls are attributed to."""

    context_id: str
    models: tuple[str, ...]
    project_id: str | None = None
    task_id: str | None = None


class Tokens:
    """The per-context tokens the key holder accepts, filled by the harness's minting; one lookup per call."""

    def __init__(self) -> None:
        self._contexts: dict[str, Context] = {}

    def add(self, token: str, context: Context) -> None:
        if not token.startswith(TOKEN_PREFIX):
            raise ValueError(f"a key holder token starts with {TOKEN_PREFIX}")
        self._contexts[token] = context

    def drop(self, token: str) -> None:
        self._contexts.pop(token, None)

    def drop_all(self) -> None:
        self._contexts.clear()

    def get(self, token: str) -> Context | None:
        return self._contexts.get(token)


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(24)


def sealed_from_env(environ: Mapping[str, str]) -> Sealed:
    """The run's endpoint as ``environ`` names it, as the ``default`` route; none without both a key and a URL."""
    key = next((environ[name] for name in KEY_NAMES if environ.get(name)), None)
    base = next((environ[name] for name in BASE_URL_NAMES if environ.get(name)), None)
    if not key or not base:
        return Sealed({})
    return Sealed({DEFAULT_ROUTE: ModelRoute(_checked_upstream(_api_root(base)), key,
                                             _model_headers(environ.get(MODEL_HEADERS_ENV)))})


def write_sealed(sealed: Sealed, directory: Path = SEALED_DIR, *, uid: int, gid: int) -> Path:
    """Write ``sealed`` as the file only ``uid`` reads (0400), through a new file renamed into place, into
    ``directory``, which must be ``uid``'s own and closed to anyone else (0700)."""
    info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or stat.S_IMODE(info.st_mode) != 0o700:
        raise SealError(f"{directory} must be a directory owned by uid {uid} with mode 0700")
    target = directory / SEALED_FILE.name
    temporary = directory / f".sealed.{secrets.token_hex(8)}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, 0o400)
        data = sealed.to_json().encode()
        while data:
            data = data[os.write(descriptor, data):]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        os.unlink(temporary)
        raise
    os.close(descriptor)
    os.replace(temporary, target)
    return target


def load_sealed(path: Path = SEALED_FILE) -> Sealed:
    """The sealed file, which must be a regular file this process's user owns with mode 0400."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise SealError(f"can't open the sealed file {path}: {type(error).__name__}") from error
    with os.fdopen(descriptor, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o400:
            raise SealError(f"{path} must be a regular file owned by uid {os.geteuid()} with mode 0400")
        return Sealed.from_json(file.read())


def model_call_kind(method: str, path: str) -> str | None:
    """Which API a request for the decoded ``path`` calls, if it's a model call the key holder forwards; ``None`` for
    anything else. A path with an empty, ``.`` or ``..`` segment, or a character the endpoint would read as more than
    a path, is never one: the endpoint would resolve it to a route the pattern didn't mean."""
    if _UNSAFE_PATH.search(path) or any(segment in ("", ".", "..") for segment in path.split("/")[1:]):
        return None
    for allowed_method, pattern, kind in _ALLOWED:
        if method == allowed_method and pattern.fullmatch(path):
            return kind
    return None


class KeyHolder:
    """Forwards model calls that carry a token in ``tokens`` to the sealed ``default`` route, with its key and
    headers and the context's attribution, pinned to the context's models; ``transport`` replaces the network."""

    def __init__(self, sealed: Sealed, tokens: Tokens | None = None, *, max_body_bytes: int = MAX_BODY_BYTES,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.route = sealed.model_routes.get(DEFAULT_ROUTE)
        self.tokens = tokens if tokens is not None else Tokens()
        self.max_body_bytes = max_body_bytes
        self.stats: Counter[str] = Counter()
        self._sealed_values = _scrub_forms([self.route.key, *self.route.headers.values()] if self.route else [])
        self._client = httpx.AsyncClient(transport=transport, follow_redirects=False,
                                         timeout=httpx.Timeout(CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S))
        self.app = Starlette(routes=[
            Route("/healthz", self._health, methods=["GET"]),
            Route("/{path:path}", self._forward, methods=list(_METHODS)),
        ])

    async def _health(self, _request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def _forward(self, request: Request) -> Response:
        call = _Call(request.method, time.monotonic())
        path = request.scope["path"]
        kind = model_call_kind(request.method, path)
        if kind is None:
            self.stats["refused"] += 1
            return call.answer(403, f"{request.method} {path} is not a model call the key holder forwards")
        call.kind = kind
        context = self._context(request.headers)
        if context is None:
            self.stats["unauthorized"] += 1
            return call.answer(401, "a model call needs the context's key holder token")
        call.context = _digest(context.context_id)
        if self.route is None:
            return call.answer(503, "this box has no model endpoint")
        if kind == "models":
            return call.done(200, JSONResponse(_model_list(path, context.models)))
        if request.headers.get("content-encoding", "identity").strip().lower() != "identity":
            return call.answer(415, "the key holder forwards only uncompressed model calls")
        body = b""
        if request.method == "POST":
            try:
                body = await self._body(request)
            except _TooLarge:
                return call.answer(413, f"a model call's body is limited to {self.max_body_bytes} bytes")
            if not context.models:
                return call.answer(503, "this run names no model")
            try:
                body, call.model, call.pinned_to = self._pinned_body(body, kind, context)
            except ValueError as error:
                self.stats["refused"] += 1
                return call.answer(400, str(error))
            if kind == "gemini":
                path, call.model, call.pinned_to = self._pinned_path(path, context)
            if call.pinned_to:
                self.stats["pinned"] += 1
        self.stats[kind] += 1
        call.bytes_in = len(body)
        url = self._url(path, request.query_params.multi_items())
        upstream = self._client.build_request(request.method, url, headers=self._headers(request, kind, context),
                                              content=body if request.method == "POST" else None)
        try:
            response = await self._client.send(upstream, stream=True)
        except httpx.HTTPError as error:
            self.stats["upstream_errors"] += 1
            call.error = type(error).__name__
            return call.answer(502, f"the model endpoint is unreachable: {call.error}")
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in _DECODABLE:
            await response.aclose()
            call.error = f"content-encoding {encoding}"
            return call.answer(502, "the model endpoint answered in an encoding the key holder can't scrub")
        headers = {name: self._scrubbed_text(value) for name, value in response.headers.items()
                   if name.lower() not in _DROP_RESPONSE}
        call.status = response.status_code
        return StreamingResponse(self._relay(response, call), status_code=response.status_code, headers=headers)

    def _context(self, headers: Mapping[str, str]) -> Context | None:
        """The context of the one token the call presents, as a bearer token or the same value as an API key."""
        presented = {value for value in (_bearer(headers.get("authorization")), headers.get("x-api-key"),
                                         headers.get("x-goog-api-key")) if value}
        contexts = {token: context for token in presented if (context := self.tokens.get(token)) is not None}
        return next(iter(contexts.values())) if len(contexts) == 1 else None

    async def _body(self, request: Request) -> bytes:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self.max_body_bytes:
            raise _TooLarge
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > self.max_body_bytes:
                raise _TooLarge
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _pinned_body(body: bytes, kind: str, context: Context) -> tuple[bytes, str | None, str | None]:
        """``body`` with its model pinned to the context's first one unless the context names it; read as JSON whatever
        its content type says, since it goes upstream as JSON. The requested model and what it was pinned to."""
        if not body and kind == "openai_object":
            return body, None, None
        try:
            data = json.loads(body)
        except ValueError as error:
            raise ValueError("a model call's body must be a JSON object") from error
        if not isinstance(data, dict):
            raise ValueError("a model call's body must be a JSON object")
        if named := sorted(_ENDPOINT_PARAMS.intersection(data)):
            raise ValueError(f"a model call can't name its own endpoint or key ({', '.join(named)})")
        model = data.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError("a model call names its model as a string")
        if model in context.models or (model is None and kind in ("openai_object", "gemini")):
            return body, model, None
        data["model"] = context.models[0]
        return json.dumps(data, ensure_ascii=False).encode(), model, context.models[0]

    @staticmethod
    def _pinned_path(path: str, context: Context) -> tuple[str, str, str | None]:
        """A Gemini call's path, which names its model, pinned to the context's first model unless it names one of the
        context's; the model it named and what it was pinned to."""
        match = _GEMINI_MODEL.search(path)
        if match.group(1) in {model.rsplit("/", 1)[-1] for model in context.models}:
            return path, match.group(1), None
        pinned = context.models[0].rsplit("/", 1)[-1]
        return path[:match.start(1)] + pinned + path[match.end(1):], match.group(1), pinned

    def _url(self, path: str, query: Iterable[tuple[str, str]]) -> httpx.URL:
        base = httpx.URL(self.route.upstream)
        params = [*base.params.multi_items(), *((name, value) for name, value in query if name != "key")]
        return base.copy_with(path=base.path.rstrip("/") + path, query=None).copy_merge_params(params)

    def _headers(self, request: Request, kind: str, context: Context) -> dict[str, str]:
        """The client's headers less its credentials, its attribution and any the route seals, then the route's headers
        with the context's attribution over them and the route's key."""
        sealed = {name.lower(): value for name, value in self.route.headers.items()}
        headers = {name: value for name, value in request.headers.items()
                   if name not in _DROP_REQUEST and name not in sealed}
        headers.update(_with_attribution(sealed, _attribution(context.project_id, context.task_id)))
        headers["authorization"] = f"Bearer {self.route.key}"
        if kind == "anthropic":
            headers["x-api-key"] = self.route.key
        elif kind == "gemini":
            headers["x-goog-api-key"] = self.route.key
        headers["accept-encoding"] = "identity"
        if request.method == "POST":
            headers["content-type"] = "application/json"
        return headers

    async def _relay(self, response: httpx.Response, call: _Call) -> AsyncIterator[bytes]:
        scrubber = _Scrubber(self._sealed_values)
        try:
            async for chunk in response.aiter_bytes():
                if out := scrubber.feed(chunk):
                    call.sent(len(out))
                    yield out
            if tail := scrubber.flush():
                call.sent(len(tail))
                yield tail
        except httpx.HTTPError as error:
            call.error = type(error).__name__
        finally:
            await response.aclose()
            call.log()

    def _scrubbed_text(self, text: str) -> str:
        for value in self._sealed_values:
            text = text.replace(value.decode(errors="replace"), REDACTED.decode())
        return text


@dataclass
class _Call:
    """One call's log line: never the key, a header value, a token or a body."""

    method: str
    started: float
    kind: str = "refused"
    status: int | None = None
    model: str | None = None
    pinned_to: str | None = None
    context: str | None = None
    bytes_in: int = 0
    bytes_out: int = 0
    first_byte_ms: float | None = None
    error: str | None = None

    def sent(self, size: int) -> None:
        if self.first_byte_ms is None:
            self.first_byte_ms = round((time.monotonic() - self.started) * 1000, 1)
        self.bytes_out += size

    def answer(self, status: int, message: str) -> Response:
        return self.done(status, JSONResponse({"error": {"message": message}}, status_code=status))

    def done(self, status: int, response: Response) -> Response:
        self.status = status
        self.log()
        return response

    def log(self) -> None:
        fields = {name: value for name, value in self.__dict__.items() if name != "started" and value is not None}
        log("call", **fields, ms=round((time.monotonic() - self.started) * 1000, 1))


class _Scrubber:
    """Replaces every sealed value in a stream, also one split across chunks, by holding back a chunk's end while it
    could be the start of one."""

    def __init__(self, values: Iterable[bytes]) -> None:
        self._values = sorted(values, key=len, reverse=True)
        self._held = b""

    def feed(self, chunk: bytes) -> bytes:
        data = self._held + chunk
        for value in self._values:
            data = data.replace(value, REDACTED)
        keep = max((size for value in self._values for size in range(1, len(value)) if data.endswith(value[:size])),
                   default=0)
        self._held = data[len(data) - keep:] if keep else b""
        return data[:len(data) - keep] if keep else data

    def flush(self) -> bytes:
        held, self._held = self._held, b""
        return held


class _TooLarge(Exception):
    pass


def log(event: str, **fields: Any) -> None:
    sys.stdout.write(json.dumps({"event": event, "real": time.time(), **fields}) + "\n")
    sys.stdout.flush()


def _api_root(url: str) -> str:
    """``url`` without a trailing ``/`` and one trailing ``/v1``, as clients append their own versioned paths. Core's
    ``api_root``, copied since the key holder can't import core; a unit test keeps the copies equal."""
    parts = urlsplit(url.strip())
    return urlunsplit(parts._replace(path=parts.path.rstrip("/").removesuffix("/v1").rstrip("/")))


def _checked_upstream(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SealError("the model endpoint's URL must be http or https with a host")
    if parts.username is not None or parts.password is not None or parts.fragment:
        raise SealError("the model endpoint's URL can't hold credentials or a fragment")
    return url


def _model_headers(raw: str | None) -> dict[str, str]:
    """``[model.headers]``, from agent-env's JSON object; none when it isn't one, as the bridges read it."""
    if not raw:
        return {}
    try:
        headers = json.loads(raw)
    except ValueError:
        log("ignored", variable=MODEL_HEADERS_ENV, reason="not JSON")
        return {}
    if not isinstance(headers, dict):
        log("ignored", variable=MODEL_HEADERS_ENV, reason="not a JSON object")
        return {}
    return {str(name): str(value) for name, value in headers.items()}


def _string_map(value: Any, what: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise SealError(f"the sealed {what} must map strings to strings")
    return dict(value)


def _attribution(project_id: str | None, task_id: str | None) -> dict[str, str]:
    """LiteLLM's spend attribution headers for a run's project and task, as the bridges set them."""
    if project_id:
        tags = [f"projectId:{project_id}", *([f"taskId:{task_id}"] if task_id else [])]
        return {"x-litellm-customer-id": project_id, "x-litellm-tags": ",".join(tags)}
    return {"x-litellm-tags": f"taskId:{task_id}"} if task_id else {}


def _with_attribution(headers: dict[str, str], attribution: dict[str, str]) -> dict[str, str]:
    """``headers`` with the run's ``attribution`` over them, in any case, except that the run's x-litellm-tags join
    theirs."""
    merged = dict(headers)
    for name, value in attribution.items():
        configured = next((merged.pop(own) for own in list(merged) if own.lower() == name), None)
        merged[name] = _join_tags(configured, value) if name == "x-litellm-tags" and configured else value
    return merged


def _join_tags(configured: str, run: str) -> str:
    """x-litellm-tags as configured, then each of the run's tags whose key (the part before ``:``) they don't name."""
    tags = [tag.strip() for tag in configured.split(",") if tag.strip()]
    keys = {tag.partition(":")[0] for tag in tags}
    return ",".join([*tags, *(tag for tag in run.split(",") if tag and tag.partition(":")[0] not in keys)])


def _scrub_forms(values: Iterable[str]) -> list[bytes]:
    """Each sealed value long enough to scrub, as it would appear raw and inside a JSON string."""
    forms = set()
    for value in values:
        if len(value) >= MIN_SCRUB_LENGTH:
            forms.update({value.encode(), json.dumps(value)[1:-1].encode()})
    return sorted(forms, key=len, reverse=True)


def _bearer(value: str | None) -> str | None:
    scheme, _, token = (value or "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" else None


def _digest(context_id: str) -> str:
    return hashlib.sha256(context_id.encode()).hexdigest()[:12]


def _model_list(path: str, models: Iterable[str]) -> dict[str, Any]:
    """The context's models, in the listing's own shape: Gemini's for a Gemini path, else OpenAI's."""
    if path.startswith("/gemini/") or path.startswith("/v1beta/"):
        return {"models": [{"name": f"models/{model.rsplit('/', 1)[-1]}"} for model in models]}
    return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "agentenv"} for model in models]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="key_holder", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("seal", help="as root: seal the run's model endpoint from this process's environment")
    serve = commands.add_parser("serve", help="as the key holder's user: forward model calls on loopback")
    serve.add_argument("--max-body-bytes", type=int, default=MAX_BODY_BYTES)
    args = parser.parse_args(argv)
    try:
        if args.command == "seal":
            try:
                user = pwd.getpwnam(USER)
            except KeyError as error:
                raise SealError(f"this image has no user {USER}") from error
            sealed = sealed_from_env(os.environ)
            write_sealed(sealed, SEALED_DIR, uid=user.pw_uid, gid=user.pw_gid)
            log("sealed", **_describe(sealed))
            return 0
        if os.geteuid() == 0:
            raise SealError(f"serve runs as {USER}, never as root")
        sealed = load_sealed(SEALED_FILE)
    except (SealError, OSError) as error:
        log("refused", command=args.command, reason=str(error))
        return 1
    holder = KeyHolder(sealed, max_body_bytes=args.max_body_bytes)
    log("serving", port=PORT, **_describe(sealed))
    uvicorn.run(holder.app, host="127.0.0.1", port=PORT, log_level="warning", access_log=False, lifespan="off",
                proxy_headers=False, server_header=False, timeout_keep_alive=IDLE_CONNECTION_TIMEOUT_S)
    return 0


def _describe(sealed: Sealed) -> dict[str, Any]:
    """What the sealed file routes to, by origin and header name: never a key or a header value."""
    return {"routes": {name: {"upstream": _origin(route.upstream), "headers": sorted(route.headers)}
                       for name, route in sealed.model_routes.items()},
            "upstream_headers": {origin: sorted(headers) for origin, headers in sealed.upstream_headers.items()}}


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")
_ALLOWED = (
    ("POST", re.compile(r"/v1/messages(/count_tokens)?"), "anthropic"),
    ("POST", re.compile(r"(/openai)?/v1/(responses|chat/completions|completions)"), "openai"),
    ("POST", re.compile(r"(/openai)?/v1/responses/[A-Za-z0-9_.=-]+(/[a-z_]+)?"), "openai_object"),
    ("GET", re.compile(r"(/openai)?/v1/responses/[A-Za-z0-9_.=-]+(/[a-z_]+)?"), "openai_object"),
    ("GET", re.compile(r"(/openai)?/v1/models"), "models"),
    ("POST", re.compile(r"(/gemini)?/v1(beta)?/models/[A-Za-z0-9_.@-]+:(generateContent|streamGenerateContent|"
                        r"countTokens)"), "gemini"),
    ("GET", re.compile(r"(/gemini)?/v1(beta)?/models"), "models"),
)
#: What a decoded path can't hold: a query, a fragment, an escape left after decoding, a backslash or a control.
_UNSAFE_PATH = re.compile(r"[?#%\\\x00-\x20\x7f]")
_GEMINI_MODEL = re.compile(r"/models/([^/:]+):")
#: Body parameters with which an endpoint (LiteLLM's client-side credentials) would call another endpoint or key.
_ENDPOINT_PARAMS = frozenset({"api_key", "api_base", "base_url"})
_DROP_REQUEST = frozenset({
    "host", "content-length", "content-type", "content-encoding", "connection", "keep-alive", "transfer-encoding",
    "te", "upgrade", "trailer", "accept-encoding", "proxy-authorization", "proxy-connection", "authorization",
    "x-api-key", "x-goog-api-key", "api-key", "x-litellm-api-key", "x-litellm-customer-id", "x-litellm-tags",
    "x-http-method-override",
})
_DROP_RESPONSE = frozenset({"content-length", "content-encoding", "connection", "keep-alive", "transfer-encoding"})
#: What ``aiter_bytes`` decodes with the standard library alone, so the scrub reads the answer as sent.
_DECODABLE = frozenset({"identity", "gzip", "x-gzip", "deflate"})


if __name__ == "__main__":
    sys.exit(main())
