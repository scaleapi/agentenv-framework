"""Snapshots are keyed on the service images that produced their PGDATA, not on the env id.

The env id is the wrong cache key in both directions: re-registering identical images
under a new id orphans a valid snapshot, and rebuilding a server under the same id would
serve a snapshot whose schema no longer matches. These tests pin both directions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent_env.env.snapshot_store import (
    ENV_SNAPSHOTS_COLLECTION,
    EnvSnapshotStore,
    compute_env_fingerprint,
)
from agent_env.store.document_store import AbsentOrNull, Eq
from agent_env.config import set_document_store


def _env(images: dict[str, str], websites: dict[str, tuple[str, str]] | None = None):
    env = MagicMock()
    env.mcp_server_envs = []
    for name, image in images.items():
        m = MagicMock()
        m.environment_name = name
        m.docker_image_artifact.image_name = image
        env.mcp_server_envs.append(m)
    env.website_envs = []
    for name, (backend, frontend) in (websites or {}).items():
        w = MagicMock()
        w.environment_name = name
        w.backend_docker_image_artifact.image_name = backend
        w.frontend_docker_image_artifact.image_name = frontend
        env.website_envs.append(w)
    return env


def test_fingerprint_is_stable_and_order_independent():
    a = _env({"gmail": "img-gmail:v1", "github": "img-github:v3"})
    b = _env({"github": "img-github:v3", "gmail": "img-gmail:v1"})
    assert compute_env_fingerprint(a) == compute_env_fingerprint(b)


def test_fingerprint_changes_when_a_service_image_changes():
    before = _env({"gmail": "img-gmail:v1", "github": "img-github:v3"})
    after = _env({"gmail": "img-gmail:v2", "github": "img-github:v3"})
    assert compute_env_fingerprint(before) != compute_env_fingerprint(after)


def test_fingerprint_covers_website_images():
    a = _env({}, {"shop": ("be:v1", "fe:v1")})
    b = _env({}, {"shop": ("be:v1", "fe:v2")})
    assert compute_env_fingerprint(a) != compute_env_fingerprint(b)


def test_fingerprint_ignores_env_id():
    """The real-world case: same images, different env id -> same key, so the snapshot hits."""
    images = {"gmail": "img-gmail:v1"}
    renamed = _env(images)
    renamed.id = "universe-env-fixed-w2"
    original = _env(images)
    original.id = "universe-env"
    assert compute_env_fingerprint(renamed) == compute_env_fingerprint(original)


def _store_with_docs(docs: list[dict]) -> tuple[EnvSnapshotStore, MagicMock]:
    """A store whose find_one honours Eq/AbsentOrNull, so query construction is exercised."""

    def find_one(collection, query, sort=None):
        assert collection == ENV_SNAPSHOTS_COLLECTION
        matches = []
        for doc in docs:
            ok = True
            for path, predicates in query.conditions.items():
                for predicate in predicates:
                    if isinstance(predicate, Eq):
                        if doc.get(path) != predicate.value:
                            ok = False
                    elif isinstance(predicate, AbsentOrNull):
                        if doc.get(path) is not None:
                            ok = False
                    else:
                        raise AssertionError(f"unexpected predicate {predicate!r}")
            if ok:
                matches.append(doc)
        matches.sort(key=lambda d: d["created_at_utc"], reverse=True)
        return matches[0] if matches else None

    doc_store = MagicMock()
    doc_store.find_one.side_effect = find_one
    store = EnvSnapshotStore()
    set_document_store(doc_store)
    return store, doc_store


def _row(**overrides) -> dict:
    row = {
        "env_id": "env-a",
        "service_universe_id": "uni",
        "service_universe_version": 1,
        "instance_id": "inst-1",
        "is_clean": True,
        "db_image_artifact_id": "snap-img",
        "db_image_artifact_version": 1,
        "created_at_utc": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "env_fingerprint": "fp-abc",
    }
    row.update(overrides)
    return row


def test_fingerprint_match_hits_across_a_renamed_env():
    store, _ = _store_with_docs([_row(env_id="env-old")])
    found = store.get_clean("env-new", "uni", 1, env_fingerprint="fp-abc")
    assert found is not None and found.db_image_artifact_id == "snap-img"


def test_rebuilt_images_do_not_match_even_on_the_same_env_id():
    """The corruption case: same env id, different images -> must MISS, not serve stale PGDATA."""
    store, _ = _store_with_docs([_row(env_id="env-a", env_fingerprint="fp-OLD")])
    assert store.get_clean("env-a", "uni", 1, env_fingerprint="fp-NEW") is None


def test_legacy_row_without_a_fingerprint_still_matches_on_env_id():
    store, _ = _store_with_docs([_row(env_fingerprint=None)])
    found = store.get_clean("env-a", "uni", 1, env_fingerprint="fp-abc")
    assert found is not None


def test_dirty_snapshots_are_never_returned():
    store, _ = _store_with_docs([_row(is_clean=False)])
    assert store.get_clean("env-a", "uni", 1, env_fingerprint="fp-abc") is None


def test_universe_version_is_respected():
    store, _ = _store_with_docs([_row(service_universe_version=38)])
    assert store.get_clean("env-a", "uni", 39, env_fingerprint="fp-abc") is None
    assert store.get_clean("env-a", "uni", 38, env_fingerprint="fp-abc") is not None


def test_newest_clean_snapshot_wins():
    old = _row(instance_id="i-old", db_image_artifact_id="old", created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc))
    new = _row(instance_id="i-new", db_image_artifact_id="new", created_at_utc=datetime(2026, 6, 1, tzinfo=timezone.utc))
    store, _ = _store_with_docs([old, new])
    found = store.get_clean("env-a", "uni", 1, env_fingerprint="fp-abc")
    assert found is not None and found.db_image_artifact_id == "new"


def test_put_persists_the_fingerprint():
    store = EnvSnapshotStore()
    doc_store = MagicMock()
    set_document_store(doc_store)
    snapshot = store.put(
        env_id="env-a",
        environment_universe_id="uni",
        environment_universe_version=1,
        instance_id="inst-1",
        is_clean=True,
        db_image_artifact_id="img",
        db_image_artifact_version=2,
        env_fingerprint="fp-abc",
    )
    assert snapshot.env_fingerprint == "fp-abc"
    assert doc_store.replace.call_args[0][2]["env_fingerprint"] == "fp-abc"
