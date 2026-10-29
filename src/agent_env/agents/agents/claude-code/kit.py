"""The pieces of agent plumbing this agent shares in shape with other CLI agents: durable session references,
skill installs, FastAPI-style error bodies, grant reads and FileParts written to disk."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx
from agentenv_protocol.a2a_agent import BundleSkillRequest, FilePart, InlineSkillRequest, SkillAddResponse
from agentenv_protocol.transfers import ReadObject, download
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

_REFS_ATTR = "_durable_session_refs"


class DurableSessionRefs:
    """Mixin, listed before ``AgentEnvAgent``, that keeps every context's session reference for the life of the
    process: the framework's own record is a bounded LRU, so an idle context would otherwise start a fresh CLI
    session."""

    def session_ref_for_context(self, context_id: str) -> str | None:
        return self.__dict__.get(_REFS_ATTR, {}).get(context_id) or super().session_ref_for_context(context_id)

    def set_session_ref_for_context(self, context_id: str, session_ref: str) -> None:
        super().set_session_ref_for_context(context_id, session_ref)
        self.__dict__.setdefault(_REFS_ATTR, {})[context_id] = session_ref

    def bound_session_refs(self) -> list[str]:
        """The session reference of every context this process has bound."""
        return list(self.__dict__.get(_REFS_ATTR, {}).values())


_NO_BODY_STATUSES = frozenset({204, 205, 304})


async def _detail_body(_request: Request, exc: HTTPException) -> Response:
    if exc.status_code < 200 or exc.status_code in _NO_BODY_STATUSES:
        return Response(status_code=exc.status_code, headers=exc.headers)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


def json_error_bodies(app: Starlette) -> Starlette:
    """Answer every ``HTTPException`` with a ``{"detail": ...}`` body, as agent-env reads errors."""
    app.add_exception_handler(HTTPException, _detail_body)
    return app


async def read_object_bytes(source: ReadObject) -> bytes:
    """The bytes behind a read grant."""
    with tempfile.TemporaryDirectory(prefix="agent-read-") as work:
        target = Path(work) / "object"
        await download(source, target)
        return target.read_bytes()


def _validate_skill_name(name: str) -> None:
    if name != Path(name).name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail=f"Invalid skill name '{name}': must not contain path separators or '..'")


async def install_skill(request: InlineSkillRequest | BundleSkillRequest, skills_dir: Path) -> SkillAddResponse:
    """Write ``skills_dir/<name>/``: the inline ``SKILL.md``, or every bundle file through its read grant. A failed
    install leaves no directory behind."""
    _validate_skill_name(request.name)
    skill_dir = skills_dir / request.name
    if skill_dir.exists():
        raise HTTPException(status_code=409, detail=f"Skill '{request.name}' is already registered")
    skill_dir.mkdir(parents=True)
    try:
        if isinstance(request, BundleSkillRequest):
            for item in request.skill_bundle.files:
                await download(item.object, skill_dir / item.path)
        else:
            if not request.skill_md or not request.skill_md.strip():
                raise HTTPException(status_code=400, detail="Skill content must not be empty")
            (skill_dir / "SKILL.md").write_text(request.skill_md)
    except BaseException:
        shutil.rmtree(skill_dir, ignore_errors=True)
        raise
    logger.info(f"Registered skill '{request.name}' at {skill_dir}")
    return SkillAddResponse(name=request.name)


# A FilePart's name, reduced to a safe charset so it can't leave the directory it's written to.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")
# A FilePart URI may be presigned, so its userinfo and query are credentials: errors and logs name it without them.
_URL_USERINFO_RE = re.compile(r"(?i)\b(https?://)[^\s/?#@]+@")
_URL_QUERY_RE = re.compile(r"(?i)\b(https?://[^\s?#]+)\?(?!<redacted>)\S*")
_LOGGED_URL_RE = re.compile(r"(?i)\b(?P<scheme>https?)://(?:[^/?#@\s]*@)?(?P<host>[^/?#\s]+)\S*")
_REDACT_LOGGED_URLS: ContextVar[bool] = ContextVar("claude_code_filepart_download", default=False)
_FILEPART_DOWNLOAD_TIMEOUT_S = 60
# Two parts of one message sharing a name are told apart by this much of a uuid.
_NAME_PREFIX_CHARS = 8


class _RedactDownloadUrls(logging.Filter):
    """httpx logs each request's full URL; while a FilePart downloads, only its origin."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _REDACT_LOGGED_URLS.get():
            record.msg = _LOGGED_URL_RE.sub(r"\g<scheme>://\g<host>/<redacted>", record.getMessage())
            record.args = None
        return True


logging.getLogger("httpx").addFilter(_RedactDownloadUrls())


@contextmanager
def _redacting_logged_urls() -> Iterator[None]:
    token = _REDACT_LOGGED_URLS.set(True)
    try:
        yield
    finally:
        _REDACT_LOGGED_URLS.reset(token)


_T = TypeVar("_T")


async def in_thread_to_its_end(func: Callable[..., _T], /, *args: object) -> _T:
    """``func(*args)`` in a thread, off the event loop, and awaited to its end even when the caller is cancelled, so
    nothing it does outlives a cancelled turn; the cancel is raised once it has ended."""
    work = asyncio.ensure_future(asyncio.to_thread(func, *args))
    cancelled = False
    while not work.done():
        try:
            await asyncio.shield(work)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break
    if cancelled:
        raise asyncio.CancelledError from (None if work.cancelled() else work.exception())
    return work.result()


def _write_bytes(target: Path, data: bytes) -> None:
    target.write_bytes(data)


def _write_base64(target: Path, data: str) -> None:
    _write_bytes(target, base64.b64decode(data, validate=True))


class FilePartMaterializeError(Exception):
    """A FilePart that can't be written to disk, its message free of the URI's credentials."""

    def __init__(self, message: str) -> None:
        message = _URL_USERINFO_RE.sub(r"\1", message)
        super().__init__(_URL_QUERY_RE.sub(r"\1?<redacted>", message))


def _shown(uri: str) -> str:
    """``uri`` without userinfo, query or fragment."""
    try:
        parts = urlsplit(uri)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
    except ValueError:
        return "<unparseable URI>"
    return urlunsplit((parts.scheme, host, parts.path, "<redacted>" if parts.query else "", ""))


def _without_credentials(text: str, uri: str) -> str:
    """``text`` with every copy of ``uri``'s userinfo, fragment and query, raw or as httpx encodes them, removed."""
    try:
        parts, url = urlsplit(uri), httpx.URL(uri)
    except (httpx.InvalidURL, ValueError):
        return text
    userinfos = {parts.netloc.rpartition("@")[0], url.userinfo.decode("ascii", "replace")}
    for userinfo in filter(None, userinfos):
        text = text.replace(f"{userinfo}@", "")
    for fragment in sorted(filter(None, {parts.fragment, url.fragment}), key=len, reverse=True):
        text = text.replace(f"#{fragment}", "")
    for query in sorted(filter(None, {parts.query, url.query.decode("ascii", "replace")}), key=len, reverse=True):
        text = text.replace(query, "<redacted>")
    return text


def _safe_name(candidate: str | None) -> str:
    candidate = _SAFE_NAME_RE.sub("_", Path(candidate).name) if candidate else ""
    return candidate or uuid.uuid4().hex


async def _download_uri_to(uri: str, target: Path) -> None:
    scheme = urlsplit(uri).scheme.lower()
    if scheme not in ("http", "https"):
        raise FilePartMaterializeError(
            f"unsupported URI scheme {scheme!r} in {_shown(uri)}: a file part reaches this agent as bytes or an "
            "http(s) URL, and agent-env sends an object its stores own as an https one")
    try:
        with _redacting_logged_urls():
            async with httpx.AsyncClient(timeout=_FILEPART_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
                response = await client.get(uri)
                response.raise_for_status()
    except httpx.HTTPStatusError as e:  # not chained: httpx's messages can repeat the full URL
        raise FilePartMaterializeError(
            f"http(s) download failed for {_shown(uri)}: HTTP {e.response.status_code}") from None
    except httpx.HTTPError as e:
        raise FilePartMaterializeError(
            f"http(s) download failed for {_shown(uri)}: {_without_credentials(str(e), uri)}") from None
    await in_thread_to_its_end(_write_bytes, target, response.content)


async def _materialize(part: FilePart, target_dir: Path) -> str:
    target_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_name(part.name)
    target = target_dir / name
    if target.exists():
        target = target_dir / f"{uuid.uuid4().hex[:_NAME_PREFIX_CHARS]}-{name}"
    if part.bytes is not None:
        try:
            await in_thread_to_its_end(_write_base64, target, part.bytes)
        except (binascii.Error, ValueError) as e:
            raise FilePartMaterializeError(f"malformed base64 in FilePart '{name}': {e}") from e
        except OSError as e:
            raise FilePartMaterializeError(f"failed to write FilePart '{name}': {e}") from e
    else:
        await _download_uri_to(part.uri, target)
    logger.info(f"Materialized FilePart name='{name}' mime={part.mime_type} size={target.stat().st_size}B to {target}")
    return str(target)


async def materialize_file_part(part: FilePart, target_dir: Path) -> str:
    """Write a FilePart's base64 bytes or http(s) contents to ``target_dir``; the path. Two parts of one message
    sharing a name are kept apart. The download is awaited and the decoding and writing done off the event loop, each
    to its end, so a cancelled turn's ends before the turn does. Raises FilePartMaterializeError."""
    return await _materialize(part, target_dir)
