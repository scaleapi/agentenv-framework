"""Object-store byte seam: stream any object-store blob to the browser."""

from __future__ import annotations

import mimetypes
import os
import tempfile
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from agent_env.config import get_config

# Prefix hardcoded (not imported from app) to avoid a router->app import cycle.
router = APIRouter(prefix="/api/v1/objects", tags=["objects"])

_CONTENT_STREAM_CHUNK = 64 * 1024

# Types a browser runs as a document — served under CSP sandbox so a stored
# HTML/SVG artifact can't execute script on the explorer origin.
_SANDBOX_TYPES = frozenset({
    "text/html", "application/xhtml+xml", "image/svg+xml",
    "application/xml", "text/xml", "application/xslt+xml",
})


def _owned_store(object_url: str):
    """The store serving ``object_url``, which must be one of its objects: the explorer reads
    nothing outside the configured store."""
    store = get_config().get_object_store_at(object_url)
    if not store.owns(object_url):
        raise HTTPException(status_code=400, detail="object_url is not in the configured object store")
    if not store.get_object_key(object_url):
        raise HTTPException(status_code=400, detail="object_url names a bucket, not an object")
    return store


def _content_disposition(filename: str) -> str:
    """``inline`` with the name for the browser to save under. Header values are Latin-1, so a
    name that is not plain ASCII goes in RFC 6266's ``filename*`` beside an ASCII fallback."""
    fallback = "".join(c if " " <= c <= "~" and c not in '"\\' else "_" for c in filename)
    if fallback == filename:
        return f'inline; filename="{filename}"'
    return f"inline; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename, safe='')}"


@router.get("/content")
def object_content(object_url: str = Query(...)):
    """Proxy an object-store blob to the browser, resolved through the configured store."""
    store = _owned_store(object_url)
    meta = store.get_object_metadata_at(object_url)
    if meta is None:
        raise HTTPException(status_code=404, detail="no object at the given object_url")

    # Stream from a temp file to bound memory on large blobs.
    fd, tmp = tempfile.mkstemp(prefix="agentenv-content-")
    os.close(fd)
    try:
        try:
            store.download_to_file(object_url, tmp)
        except Exception:
            # Object existed at the metadata check above, so a read failure here is an
            # upstream store error (permission/timeout/outage), not a missing object.
            raise HTTPException(status_code=502, detail="object store failed to read the object")
        response = _streamed(tmp, object_url, meta)
    except BaseException:
        # The response owns the file once it exists; until then nothing else will remove it.
        os.unlink(tmp)
        raise
    return response


def _filename(object_url: str) -> str:
    return object_url.rstrip("/").rsplit("/", 1)[-1] or "object"


def _content_type(object_url: str, meta) -> str:
    """The type the explorer serves an object as. Local stores don't persist one, so it falls back
    to the filename, keeping the media type and the sandbox decision right on every backend."""
    return meta.content_type or mimetypes.guess_type(_filename(object_url))[0] or "application/octet-stream"


def _streamed(tmp: str, object_url: str, meta) -> StreamingResponse:
    def _stream_and_cleanup():
        try:
            with open(tmp, "rb") as fh:
                while chunk := fh.read(_CONTENT_STREAM_CHUNK):
                    yield chunk
        finally:
            os.unlink(tmp)

    content_type = _content_type(object_url, meta)
    headers = {
        "X-Content-Type-Options": "nosniff",  # untrusted artifact bytes must not sniff-execute
        "Content-Disposition": _content_disposition(_filename(object_url)),
    }
    if content_type.split(";")[0].strip().lower() in _SANDBOX_TYPES:
        headers["Content-Security-Policy"] = "sandbox"
    # Reads return the bytes as stored, so the browser needs the encoding to decode them.
    if meta.content_encoding:
        headers["Content-Encoding"] = meta.content_encoding
    if meta.size is not None:
        headers["Content-Length"] = str(meta.size)
    return StreamingResponse(
        _stream_and_cleanup(),
        media_type=content_type,
        headers=headers,
    )


@router.get("/metadata")
def object_metadata(object_url: str = Query(...)):
    """Size + content-type for an object, without transferring the bytes."""
    store = _owned_store(object_url)
    meta = store.get_object_metadata_at(object_url)
    if meta is None:
        raise HTTPException(status_code=404, detail="no object at the given object_url")
    return {"size_bytes": meta.size, "content_type": _content_type(object_url, meta)}
