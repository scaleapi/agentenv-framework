"""Unit tests for config.toml-declared custom Artifact subclasses.

A custom `Artifact` named under `[artifacts]` in `.agentenv/config.toml` is
imported, ABC-guarded, and registered under its own `type` when the registry is
built — with no change to the built-in artifact path. Backend-agnostic; no network.
"""

import textwrap
from typing import Literal

import pytest

from agent_env.artifact.artifact import Artifact
from agent_env.artifact import registry
from agent_env.artifact.store import ArtifactQuery, get_artifact_store, reset_artifact_store
from agent_env.config import ConfigError
from agent_env.store import reset_config, set_document_store
from agent_env.store.document_store import Eq, In
from agent_env.store.query import to_document_query
from tst.unit.store.fakes import FakeDocumentStore


class _CustomArtifact(Artifact):
    """A minimal custom artifact declaring its own `type`."""

    type: Literal["custom_test_artifact"] = "custom_test_artifact"
    payload: str = "default"


class _CollidingArtifact(Artifact):
    """Declares a `type` that a built-in already owns."""

    type: Literal["file"] = "file"


class _CustomArtifactDup(Artifact):
    """A second class claiming the same `type` as _CustomArtifact."""

    type: Literal["custom_test_artifact"] = "custom_test_artifact"


class _NoTypeArtifact(Artifact):
    """Forgets to set its own `type`, so it inherits the base (no default)."""


class _NotAnArtifact:
    pass


class _RenamedArtifact(Artifact):
    """Mid-rename: registers under its new `type` but still validates the old spelling.

    Mirrors the shape a deployment is in while `[artifacts] type_aliases` is carrying
    its stored documents forward — the alias resolves the registry lookup, the widened
    Literal accepts the value still on disk.
    """

    type: Literal["legacy_renamed", "renamed_artifact"] = "renamed_artifact"


class _RenamedArtifactAlias(_RenamedArtifact):
    """Keeps the old spelling alive as its own registered class instead of an alias."""

    type: Literal["legacy_renamed", "renamed_artifact"] = "legacy_renamed"


def _write_config(tmp_path, body: str):
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(textwrap.dedent(body))
    return cfg


_HERE = "tst.unit.artifact.registry_test"


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_config()
    yield
    reset_config()
    reset_artifact_store()
    reset_config()


def test_absent_config_leaves_builtins_only(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    reg = registry.get_artifact_registry()
    assert "custom_test_artifact" not in reg
    assert "file" in reg


def test_no_artifacts_section_leaves_builtins_only(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, '[stores]\ndocument = "local"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reg = registry.get_artifact_registry()
    assert "custom_test_artifact" not in reg
    assert "file" in reg


def test_empty_impls_leaves_builtins_only(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, "[artifacts]\nimpls = []\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reg = registry.get_artifact_registry()
    assert "custom_test_artifact" not in reg
    assert "file" in reg


def test_non_list_impls_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f'[artifacts]\nimpls = "{_HERE}:_CustomArtifact"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="must be a list"):
        registry.get_artifact_registry()


def test_config_toml_artifact_is_registered(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_CustomArtifact"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reg = registry.get_artifact_registry()
    assert reg["custom_test_artifact"] is _CustomArtifact
    assert "file" in reg


def test_collision_with_builtin_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_CollidingArtifact"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="already registered"):
        registry.get_artifact_registry()


def test_two_custom_artifacts_same_type_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_CustomArtifact", "{_HERE}:_CustomArtifactDup"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="already registered"):
        registry.get_artifact_registry()


def test_artifact_without_own_type_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_NoTypeArtifact"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="does not define its own 'type'"):
        registry.get_artifact_registry()


def test_unimportable_impl_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [artifacts]
        impls = ["no.such.module:Thing"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="Cannot import"):
        registry.get_artifact_registry()


def test_non_artifact_impl_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_NotAnArtifact"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="not a subclass"):
        registry.get_artifact_registry()


def test_failed_merge_does_not_memoize_partial_registry(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [artifacts]
        impls = ["no.such.module:Thing"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError):
        registry.get_artifact_registry()
    # A bad manifest must fail loud on EVERY call, not just the first.
    with pytest.raises(ConfigError):
        registry.get_artifact_registry()


def test_custom_artifact_round_trips_through_registry(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_CustomArtifact"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))

    doc = {"id": "a1", "version": 1, "type": "custom_test_artifact", "payload": "hello"}
    cls = registry.get_artifact_registry()[doc["type"]]
    artifact = cls.model_validate(doc)

    assert isinstance(artifact, _CustomArtifact)
    assert artifact.id == "a1"
    assert artifact.version == 1
    assert artifact.payload == "hello"


def test_store_get_routes_custom_type_through_registry(monkeypatch, tmp_path):
    """ArtifactStore.get -> _deserialize resolves a config-registered custom type via
    get_artifact_registry() — the real read path, not just the registry dict."""
    cfg = _write_config(tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_CustomArtifact"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    doc_store = FakeDocumentStore()
    doc_store.docs.append({"id": "cw1", "version": 1, "type": "custom_test_artifact", "payload": "via-store"})
    set_document_store(doc_store)
    reset_artifact_store()

    loaded = Artifact.get("cw1")

    assert type(loaded) is _CustomArtifact
    assert loaded.payload == "via-store"


def test_non_string_impl_element_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, "[artifacts]\nimpls = [123]\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="must be a 'module:Class' string"):
        registry.get_artifact_registry()


def test_store_get_unregistered_type_raises_clear_error(monkeypatch, tmp_path):
    """A stored doc whose type isn't registered fails with an actionable error that
    points at [artifacts].impls — not a bare KeyError deep in deserialization."""
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # no .agentenv -> built-ins only
    doc_store = FakeDocumentStore()
    doc_store.docs.append({"id": "u1", "version": 1, "type": "never_registered_type"})
    set_document_store(doc_store)
    reset_artifact_store()

    with pytest.raises(ValueError, match=r"Unknown artifact type .*\[artifacts\]\.impls"):
        Artifact.get("u1")


class _FilterRecorder(FakeDocumentStore):
    """Captures the filter a store method compiles, so a query's shape can be asserted."""

    last_filter = None

    def query(self, collection, filter, *a, **k):
        self.last_filter = filter
        return []


def _alias_config(tmp_path, monkeypatch, aliases='{ legacy_renamed = "renamed_artifact" }'):
    cfg = _write_config(tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_RenamedArtifact"]
        type_aliases = {aliases}
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    return cfg


def test_absent_type_aliases_is_identity(monkeypatch, tmp_path):
    """The core ships no aliases, so both accessors are the identity on every type."""
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert registry.get_type_aliases() == {}
    assert registry.canonical_type("file") == "file"
    assert registry.equivalent_types("file") == ["file"]


def test_type_aliases_canonicalise_and_expand_both_ways(monkeypatch, tmp_path):
    _alias_config(tmp_path, monkeypatch)
    assert registry.canonical_type("legacy_renamed") == "renamed_artifact"
    assert registry.canonical_type("renamed_artifact") == "renamed_artifact"
    expanded = ["legacy_renamed", "renamed_artifact"]
    assert registry.equivalent_types("legacy_renamed") == expanded
    assert registry.equivalent_types("renamed_artifact") == expanded


def test_equivalent_types_leaves_an_unaliased_type_scalar(monkeypatch, tmp_path):
    """An unaliased type must not become a one-element `$in` — plain equality uses the index."""
    _alias_config(tmp_path, monkeypatch)
    assert registry.equivalent_types("file") == ["file"]


def test_store_get_reads_a_doc_stored_under_the_legacy_spelling(monkeypatch, tmp_path):
    """The real read path: the alias resolves the class, and the model reports canonical.

    The document on disk keeps its own spelling — nothing rewrites it — but a field typed
    `Literal["renamed_artifact"]` cannot hold anything else, so the value the model
    carries is the canonical one. A reader comparing `artifact.type` against the raw
    document will see them differ for every legacy row."""
    _alias_config(tmp_path, monkeypatch)
    doc_store = FakeDocumentStore()
    doc_store.docs.append({"id": "r1", "version": 1, "type": "legacy_renamed"})
    set_document_store(doc_store)
    reset_artifact_store()

    loaded = Artifact.get("r1")

    assert type(loaded) is _RenamedArtifact
    assert loaded.type == "renamed_artifact"
    assert doc_store.docs[0]["type"] == "legacy_renamed"


def test_store_get_still_raises_for_a_type_that_is_neither_spelling(monkeypatch, tmp_path):
    """The control: aliasing one type must not make every unknown type resolve."""
    _alias_config(tmp_path, monkeypatch)
    doc_store = FakeDocumentStore()
    doc_store.docs.append({"id": "r2", "version": 1, "type": "not_a_spelling"})
    set_document_store(doc_store)
    reset_artifact_store()

    with pytest.raises(ValueError, match="Unknown artifact type 'not_a_spelling'"):
        Artifact.get("r2")


def test_query_type_expands_an_aliased_type_to_an_in(monkeypatch, tmp_path):
    """A type filter runs server-side against the stored value, so it has to match both."""
    _alias_config(tmp_path, monkeypatch)
    filt, _ = to_document_query(ArtifactQuery().type("renamed_artifact"))
    assert filt.conditions["type"] == [In(["legacy_renamed", "renamed_artifact"])]


def test_query_type_stays_a_scalar_equality_for_an_unaliased_type(monkeypatch, tmp_path):
    _alias_config(tmp_path, monkeypatch)
    filt, _ = to_document_query(ArtifactQuery().type("file"))
    assert filt.conditions["type"] == [Eq("file")]


def test_latest_by_type_expands_an_aliased_type(monkeypatch, tmp_path):
    _alias_config(tmp_path, monkeypatch)
    doc_store = _FilterRecorder()
    set_document_store(doc_store)
    reset_artifact_store()

    get_artifact_store().latest_by_type("renamed_artifact")

    assert doc_store.last_filter.conditions["type"] == [In(["legacy_renamed", "renamed_artifact"])]


def test_latest_by_type_stays_a_scalar_equality_for_an_unaliased_type(monkeypatch, tmp_path):
    _alias_config(tmp_path, monkeypatch)
    doc_store = _FilterRecorder()
    set_document_store(doc_store)
    reset_artifact_store()

    get_artifact_store().latest_by_type("file")

    assert doc_store.last_filter.conditions["type"] == [Eq("file")]


def test_type_aliases_must_be_a_table(monkeypatch, tmp_path):
    _alias_config(tmp_path, monkeypatch, aliases='["legacy_renamed"]')
    with pytest.raises(ConfigError, match=r"\[artifacts\.type_aliases\] must be a table"):
        registry.get_type_aliases()


def test_type_aliases_value_must_be_a_string(monkeypatch, tmp_path):
    _alias_config(tmp_path, monkeypatch, aliases="{ legacy_renamed = 7 }")
    with pytest.raises(ConfigError, match="must map to a string"):
        registry.get_type_aliases()


def test_type_aliases_rejects_a_self_alias(monkeypatch, tmp_path):
    _alias_config(tmp_path, monkeypatch, aliases='{ legacy_renamed = "legacy_renamed" }')
    with pytest.raises(ConfigError, match="maps to itself"):
        registry.get_type_aliases()


def test_type_aliases_rejects_a_chain(monkeypatch, tmp_path):
    """Chains are rejected rather than resolved, so one lookup is always enough."""
    _alias_config(
        tmp_path, monkeypatch,
        aliases='{ oldest = "legacy_renamed", legacy_renamed = "renamed_artifact" }',
    )
    with pytest.raises(ConfigError, match="chains are not resolved"):
        registry.get_type_aliases()


def test_type_aliases_must_not_redirect_a_type_another_artifact_owns(monkeypatch, tmp_path):
    """Aliasing a live type at a different class silently hijacks every document it owns."""
    _alias_config(tmp_path, monkeypatch, aliases='{ file = "renamed_artifact" }')
    with pytest.raises(ConfigError, match="not both"):
        registry.get_artifact_registry()


def test_type_aliases_and_a_registered_subclass_are_alternatives(monkeypatch, tmp_path):
    """Keeping the old spelling alive as a registered subclass is the other way to do
    this; declaring both makes the alias steal documents from the subclass."""
    cfg = _write_config(tmp_path, f"""
        [artifacts]
        impls = ["{_HERE}:_RenamedArtifact", "{_HERE}:_RenamedArtifactAlias"]
        type_aliases = {{ legacy_renamed = "renamed_artifact" }}
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="not both"):
        registry.get_artifact_registry()


def test_type_aliases_target_must_be_registered(monkeypatch, tmp_path):
    """A typo in the target is caught when the registry is built, not on a later read."""
    _alias_config(tmp_path, monkeypatch, aliases='{ legacy_renamed = "renmaed_artifact" }')
    with pytest.raises(ConfigError, match="which no artifact registers"):
        registry.get_artifact_registry()
