"""Shared pieces for the explorer's read routers, using backend-agnostic store calls."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agent_env.artifact.registry import canonical_type, equivalent_types
from agent_env.config import get_config
from agent_env.explorer.entity_ids import EntityId
from agent_env.store import Filter, Sort


class PaginatedResponse(BaseModel):
    """Pagination envelope shared by every list endpoint."""

    items: list[Any]
    total: int
    limit: int
    offset: int
    has_more: bool = False


def docs():
    return get_config().get_document_store()


def _id_matches(doc: dict, id_field: str, needle: str) -> bool:
    """Case-insensitive substring match on the entity id."""
    value = doc.get(id_field)
    return isinstance(value, str) and needle.lower() in value.lower()


_UNIVERSE_FILE_TYPES = {"file_artifact_universe"}
_UNIVERSE_ENVIRONMENT_TYPES = {"environment_universe"}


def _ref_id_version(ref: Any) -> tuple[Optional[str], Optional[int]]:
    if isinstance(ref, dict):
        return ref.get("id"), ref.get("version")
    return (ref if isinstance(ref, str) else None), None


def _resolve_doc(store, collection: str, ref: Any) -> Optional[dict]:
    """Fetch a referenced artifact doc by (id, version); latest version if unpinned."""
    rid, ver = _ref_id_version(ref)
    if rid is None:
        return None
    if ver is not None:
        return store.find_one(collection, Filter.of(id=rid, version=ver))
    found = store.latest_per_id(collection, Filter.of(id=rid))
    return found[0] if found else None


def _enrich_universe(doc: dict, store, collection: str) -> dict:
    """Resolve a universe's artifact refs into the ``files``/``services`` arrays the UI renders.

    Each file's ``object_url`` is the FileArtifact's seam location, served via /objects/content.
    """
    t = doc.get("type")
    if collection == "artifacts" and isinstance(t, str):
        t = canonical_type(t)
    if t in _UNIVERSE_FILE_TYPES:
        refs = doc.get("file_artifact_refs") or doc.get("file_artifact_ids") or {}
        files = []
        for filename, ref in refs.items():
            fa = _resolve_doc(store, collection, ref)
            rid, ver = _ref_id_version(ref)
            files.append({
                "filename": filename,
                "artifact_id": rid,
                "version": ver,
                "content_type": fa.get("content_type") if fa else None,
                "object_url": (fa.get("object_url") or fa.get("s3_url")) if fa else None,
            })
        doc["files"] = files
    elif t in _UNIVERSE_ENVIRONMENT_TYPES:
        refs = doc.get("service_artifact_refs")
        if not refs:
            refs = [{"id": sid} for sid in (doc.get("service_artifact_ids") or [])]
        services = []
        for ref in refs:
            sa = _resolve_doc(store, collection, ref)
            rid, ver = _ref_id_version(ref)
            services.append({
                "service_name": (sa.get("service_name") if sa else None) or rid,
                "artifact_id": rid,
                "version": ver,
            })
        doc["services"] = services
    return doc


def versioned_router(
    *,
    prefix: str,
    tag: str,
    collection: str,
    noun: str,
    id_field: str = "id",
) -> APIRouter:
    """List / get / versions over one ``(id, version)`` collection (artifacts, envs, tasks,
    agents, evals). ``noun`` names the entity in each route's summary, which the Docs nav
    lists without its path."""
    router = APIRouter(prefix=prefix, tags=[tag])

    @router.get("", response_model=PaginatedResponse, summary=f"List {noun.title()}s")
    def list_items(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        type: Optional[list[str]] = Query(None, description="Filter by type; repeat to union."),
        id: Optional[str] = Query(None, description="Substring match on the entity id."),
        sort_by: str = Query("created_at_utc"),
        descending: bool = Query(True),
    ) -> PaginatedResponse:
        store = docs()
        want_types = set(type) if type else None
        if want_types is not None and collection == "artifacts":
            want_types = {s for t in want_types for s in equivalent_types(t)}
        if id or want_types is not None:
            # Reduce to the latest version per id FIRST, then filter (by id substring
            # and/or type) in Python. Filtering before the reduction could select a
            # superseded version whose type still matches and report it as the live
            # entity; reducing first means the type filter only ever sees live rows.
            reduced = store.latest_per_id(
                collection, Filter({}), id_field=id_field,
                sort=Sort.by(sort_by, descending=descending),
            )
            matched = [
                d for d in reduced
                if (not id or _id_matches(d, id_field, id))
                and (want_types is None or str(d.get("type")) in want_types)
            ]
            total = len(matched)
            items = matched[offset: offset + limit] if limit else matched[offset:]
        else:
            items, total = store.latest_per_id_page(
                collection, Filter({}), id_field=id_field,
                sort=Sort.by(sort_by, descending=descending), limit=limit, offset=offset,
            )
        return PaginatedResponse(
            items=items, total=total, limit=limit, offset=offset,
            has_more=offset + len(items) < total,
        )

    @router.get("/{entity_id}", summary=f"Get {noun.title()}")
    def get_item(entity_id: EntityId, version: Optional[int] = None) -> dict:
        store = docs()
        if version is not None:
            doc = store.find_one(collection, Filter.of(**{id_field: entity_id, "version": version}))
        else:
            found = store.latest_per_id(collection, Filter.of(**{id_field: entity_id}), id_field=id_field)
            doc = found[0] if found else None
        if doc is None:
            raise HTTPException(status_code=404, detail=f"{tag} {entity_id} not found")
        return _enrich_universe(doc, store, collection)

    @router.get("/{entity_id}/versions", summary=f"List {noun.title()} Versions")
    def list_versions(
        entity_id: EntityId,
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[dict]:
        """Every stored version of one entity, newest first, as a bare list; 404 for an unknown id."""
        store = docs()
        filt = Filter.of(**{id_field: entity_id})
        page = store.query(collection, filt, sort=Sort.by("version", descending=True),
                           limit=limit, offset=offset)
        if not page and store.find_one(collection, filt) is None:
            raise HTTPException(status_code=404, detail=f"{tag} {entity_id} not found")
        return page

    return router
