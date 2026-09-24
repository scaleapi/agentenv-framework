"""Object-store byte seam: stream any object-store blob to the browser."""

from __future__ import annotations

import mimetypes
import os
import tempfile

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


@router.get("/content")
def object_content(object_url: str = Query(...)):
    """Proxy an object-store blob to the browser, resolved through the configured store."""
    store = get_config().get_object_store()
    # Ownership check: get_object_key raises for a url outside this store (foreign
    # bucket / path traversal), so the route only ever serves this store's objects.
    try:
        store.get_object_key(object_url)
    except Exception:
        raise HTTPException(status_code=400, detail="object_url is not in the configured object store")

    meta = store.get_object_metadata_at(object_url)
    if meta is None:
        raise HTTPException(status_code=404, detail="no object at the given object_url")

    # Stream from a temp file to bound memory on large blobs.
    fd, tmp = tempfile.mkstemp(prefix="agentenv-content-")
    os.close(fd)
    try:
        store.download_to_file(object_url, tmp)
    except Exception:
        os.unlink(tmp)
        # Object existed at the metadata check above, so a read failure here is an
        # upstream store error (permission/timeout/outage), not a missing object.
        raise HTTPException(status_code=502, detail="object store failed to read the object")

    def _stream_and_cleanup():
        try:
            with open(tmp, "rb") as fh:
                while chunk := fh.read(_CONTENT_STREAM_CHUNK):
                    yield chunk
        finally:
            os.unlink(tmp)

    filename = object_url.rstrip("/").rsplit("/", 1)[-1] or "object"
    # Local stores don't persist a content type; fall back to the filename so both the
    # media type and the sandbox decision below are correct on every backend.
    content_type = meta.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    headers = {
        "X-Content-Type-Options": "nosniff",  # untrusted artifact bytes must not sniff-execute
        "Content-Disposition": f'inline; filename="{filename}"',
    }
    if content_type.split(";")[0].strip().lower() in _SANDBOX_TYPES:
        headers["Content-Security-Policy"] = "sandbox"
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
    store = get_config().get_object_store()
    try:
        store.get_object_key(object_url)
    except Exception:
        raise HTTPException(status_code=400, detail="object_url is not in the configured object store")

    meta = store.get_object_metadata_at(object_url)
    if meta is None:
        raise HTTPException(status_code=404, detail="no object at the given object_url")
    return {"size_bytes": meta.size, "content_type": meta.content_type}
