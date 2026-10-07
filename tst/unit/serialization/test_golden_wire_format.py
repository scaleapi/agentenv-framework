"""Golden wire-bytes tests for everything agent-env PERSISTS.

The ``service_*`` to ``environment_*`` rename is additive: readers may learn
new spellings, but the bytes written to MongoDB (and therefore into every
delivered bundle built from them) must not move. That failure mode is silent —
a pydantic ``AliasChoices`` added without a matching ``alias=``, or a ``Literal``
widened with the new value first, re-keys every new document without breaking a
single round-trip test.

Every registered artifact type is dumped with ``model_dump(by_alias=True)`` (what
``ArtifactStore.put_document`` writes) and every registered env type with
``to_dict()`` (what ``EnvStore.put`` writes), then compared against a checked-in
golden JSON file so a regression's diff names the exact key that moved.

Envs are covered in the WRITE direction only: most ``Env.from_dict``
implementations resolve refs through ``Artifact.get`` / ``Env.get``, which need
the network and so cannot run in this (socket-blocked) suite.

Regenerate after an INTENTIONAL wire change, then review the diff key by key:

    python tst/unit/serialization/test_golden_wire_format.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.skill import SkillArtifact
from agent_env.artifact.artifacts.vm_image import VMImageArtifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.artifact.registry import get_artifact_registry
from agent_env.env.envs.gateway_server import GatewayEnv
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.service_db import ServiceDBEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.registry import get_env_registry
from agent_env.task_step.task_steps.apply_server_config import ConfigDirective

GOLDEN_DIR = Path(__file__).parent / "golden"
ARTIFACT_GOLDEN_DIR = GOLDEN_DIR / "artifact"
ENV_GOLDEN_DIR = GOLDEN_DIR / "env"

_BUCKET = "s3://artifact-bucket"


def _docker_image(artifact_id: str, version: int, image_name: str) -> DockerImageArtifact:
    return DockerImageArtifact(
        id=artifact_id,
        version=version,
        description=f"{image_name} image",
        image_name=image_name,
        tar_gz_s3_url=f"{_BUCKET}/artifacts/docker_image/{artifact_id}/{version}/image.tar.gz",
    )


def _mcp_server_env() -> MCPServerEnv:
    return MCPServerEnv(
        id="slack-env",
        version=4,
        docker_image_artifact=_docker_image("slack-mcp-image", 5, "slack-mcp:v5"),
        environment_name="slack",
        metadata={"owner": "example-team"},
    )


def _website_env() -> WebsiteEnv:
    return WebsiteEnv(
        id="shop-site-env",
        version=2,
        backend_docker_image_artifact=_docker_image("shop-backend-image", 3, "shop-backend:v3"),
        frontend_docker_image_artifact=_docker_image("shop-frontend-image", 3, "shop-frontend:v3"),
        environment_name="shop",
        metadata={"owner": "example-team"},
    )


ARTIFACT_FIXTURES = {
    "cli": lambda: CliArtifact(
        id="slack-cli",
        version=3,
        cli_files_id="slack-cli-files",
        cli_s3_url=f"{_BUCKET}/artifacts/cli/slack-cli/3/",
        entrypoint="bin/slack",
        command_name="slack",
        env_id="slack-env",
        env_version=2,
    ),
    "docker_image": lambda: _docker_image("slack-mcp-image", 5, "slack-mcp:v5"),
    "file": lambda: FileArtifact(
        id="slack-universe-data",
        version=2,
        description="Slack universe seed data",
        filename="slack.json",
        content_type="application/json",
        s3_url=f"{_BUCKET}/artifacts/file/slack-universe-data/2/slack.json",
    ),
    "file_artifact_universe": lambda: FileArtifactUniverse(
        id="tutorial-files",
        version=4,
        file_artifact_refs={"README.md": ArtifactRef(id="readme-file", version=1)},
        file_artifact_ids={"README.md": "readme-file"},
        bundle_s3_url=f"{_BUCKET}/artifacts/file_artifact_universe/tutorial-files/4/bundle.tar.gz",
    ),
    # Alias branches of the double registry entries; each pins its alias type value explicitly.
    "environment": lambda: EnvironmentArtifact(
        id="slack-service",
        version=7,
        type="environment",
        service_name="slack",
        service_version=3,
        file_artifact_ref=ArtifactRef(id="slack-universe-data", version=2),
    ),
    "environment_universe": lambda: EnvironmentUniverseArtifact(
        id="acme-universe",
        version=9,
        type="environment_universe",
        environment_artifact_refs=[ArtifactRef(id="slack-service", version=7)],
        metadata_refs={"manifest.json": ArtifactRef(id="acme-manifest", version=1)},
    ),
    "skill": lambda: SkillArtifact(
        id="pdf-filler",
        version=2,
        skill_files_id="pdf-filler-files",
        skill_s3_url=f"{_BUCKET}/artifacts/skill/pdf-filler/2/",
        agent_skills_spec_version="2025-10-01",
        skill_name="pdf-filler",
        description="Fill PDF forms",
        license="MIT",
        compatibility="claude-code",
        allowed_tools="Read,Write",
        skill_metadata={"category": "documents"},
    ),
    "vm_image": lambda: VMImageArtifact(
        id="ubuntu-vm",
        version=6,
        description="Ubuntu VM image",
        ecr_url="123456789012.dkr.ecr.us-west-2.amazonaws.com/vm/ubuntu:v6",
        disk_size_gb=30.0,
        cpu=4.0,
        memory_mb=8192,
    ),
}

ENV_FIXTURES = {
    "gateway_server": lambda: GatewayEnv(
        id="gateway-env",
        version=11,
        docker_image_artifact=_docker_image("gateway-image", 12, "gateway:v12"),
        metadata={"owner": "example-team"},
    ),
    "mcp_server": _mcp_server_env,
    "multi": lambda: MultiEnv(
        id="acme-multi-env",
        version=6,
        mcp_server_envs=[_mcp_server_env()],
        website_envs=[_website_env()],
        metadata={"owner": "example-team"},
    ),
    "service_db": lambda: ServiceDBEnv(
        id="service-db-env",
        db_docker_image_artifact=_docker_image("postgres-image", 1, "postgres:16-alpine"),
        db_web_docker_image_artifact=_docker_image("pgweb-image", 1, "pgweb:v1"),
        db_mcp_docker_image_artifact=_docker_image("db-mcp-image", 1, "db-mcp:v1"),
        version=3,
        metadata={"owner": "example-team"},
    ),
    "website": _website_env,
}

_REGEN = "regenerate with: python tst/unit/serialization/test_golden_wire_format.py"


def _dumps(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"


def _artifact_wire(registry_key: str) -> dict:
    return ARTIFACT_FIXTURES[registry_key]().model_dump(by_alias=True)


def _env_wire(registry_key: str) -> dict:
    return ENV_FIXTURES[registry_key]().to_dict()


@pytest.mark.parametrize("registry_key", sorted(ARTIFACT_FIXTURES))
def test_artifact_wire_bytes_match_golden(registry_key):
    golden = ARTIFACT_GOLDEN_DIR / f"{registry_key}.json"
    assert golden.is_file(), f"missing golden for artifact type {registry_key!r} — {_REGEN}"
    assert _dumps(_artifact_wire(registry_key)) == golden.read_text(), (
        f"artifact type {registry_key!r} changed the keys it writes to MongoDB; "
        f"if that is intentional, {_REGEN}"
    )


@pytest.mark.parametrize("registry_key", sorted(ENV_FIXTURES))
def test_env_wire_bytes_match_golden(registry_key):
    golden = ENV_GOLDEN_DIR / f"{registry_key}.json"
    assert golden.is_file(), f"missing golden for env type {registry_key!r} — {_REGEN}"
    assert _dumps(_env_wire(registry_key)) == golden.read_text(), (
        f"env type {registry_key!r} changed the keys it writes to MongoDB; "
        f"if that is intentional, {_REGEN}"
    )


def test_every_registered_artifact_type_has_a_fixture():
    """A newly registered artifact type must gain a fixture + golden, or the wire
    format of a whole document class ships untested."""
    assert set(ARTIFACT_FIXTURES) == set(get_artifact_registry())


def test_every_registered_env_type_has_a_fixture():
    assert set(ENV_FIXTURES) == set(get_env_registry())


def test_renamed_types_register_every_literal_member_not_just_the_default():
    """Disarms the legacy-key-drop footgun in ``artifact/registry.py``.

    ``_get_type`` derives a registry key from the ``type`` Literal's *default*, so
    deriving half of a renamed pair is a trap: flipping that default when the legacy key is dropped makes
    the derived key equal the hardcoded one, the two entries collapse, and the
    LEGACY spelling silently de-registers — every document ever written under it
    stops resolving (measured on dev: 32,846 + 20,845 docs).

    Set equality, not mere presence, is what makes this fire. Under the derived
    form it passes today (default ``service`` plus a hardcoded ``environment``)
    and fails the instant the default flips — exactly when it matters."""
    from typing import get_args

    for cls in (EnvironmentArtifact, EnvironmentUniverseArtifact):
        members = set(get_args(cls.model_fields["type"].annotation))
        keys = {k for k, v in get_artifact_registry().items() if v is cls}
        assert keys == members, f"{cls.__name__}: registry {keys} != Literal members {members}"


def test_artifact_golden_files_match_the_registry():
    assert {p.stem for p in ARTIFACT_GOLDEN_DIR.glob("*.json")} == set(get_artifact_registry())


def test_env_golden_files_match_the_registry():
    assert {p.stem for p in ENV_GOLDEN_DIR.glob("*.json")} == set(get_env_registry())


def test_service_artifact_writes_both_name_spellings_and_one_version_key():
    """A round-trip stays green when an alias direction flips, so assert the key
    names directly. ``environment_name`` is the attribute, ``service_name`` the
    legacy wire key (the additive wire contract). The dual-write adds the new spelling
    *beside* the old, never in place of it — ``alias=`` alone would re-key the
    document, so the twin is written in the serializer body.

    ``service_version`` was deleted from this model. It was a deprecation, not a
    rename, so nothing replaces it — neither spelling may appear. Stored documents keep
    the key for ever and pydantic extra='ignore' drops it on read; this pins that the
    serializer never writes it back."""
    doc = _artifact_wire("environment")
    assert doc["service_name"] == "slack"
    assert doc["environment_name"] == "slack"
    assert "service_version" not in doc
    assert "environment_version" not in doc
    assert "environment_version" not in doc


def test_service_artifact_legacy_file_artifact_id_key_is_unchanged():
    doc = EnvironmentArtifact(
        id="slack-service", version=1, service_name="slack", service_version=3,
        file_artifact_id="slack-universe-data",
    ).model_dump(by_alias=True)
    assert doc["file_artifact_id"] == "slack-universe-data"
    assert "legacy_file_artifact_id" not in doc


def test_service_universe_artifact_writes_both_ref_spellings():
    doc = _artifact_wire("environment_universe")
    assert doc["service_artifact_refs"] == [{"id": "slack-service", "version": 7}]
    assert doc["environment_artifact_refs"] == doc["service_artifact_refs"]


def test_service_universe_artifact_reads_either_ref_spelling_and_writes_both():
    """validation_alias, not alias: reading the new spelling must not re-key the
    document. The dual-write adds the twin in the serializer body afterwards, so both keys
    appear no matter which spelling came in."""
    ref = [{"id": "slack-service", "version": 7}]
    from_new = EnvironmentUniverseArtifact(id="u", version=1, environment_artifact_refs=ref)
    from_legacy = EnvironmentUniverseArtifact(id="u", version=1, service_artifact_refs=ref)
    assert from_new.environment_artifact_refs == from_legacy.environment_artifact_refs
    doc = from_new.model_dump(by_alias=True)
    assert doc["service_artifact_refs"] == ref
    assert doc["environment_artifact_refs"] == ref


def test_service_universe_artifact_legacy_keys_are_unchanged():
    """The dual-write must not reach unpinned docs at all. The twin is written only where
    pinned refs exist, so an unpinned universe round-trips byte-for-byte as
    before — and crucially gains no ``environment_artifact_refs``, which would
    claim a pinning this document does not have."""
    doc = EnvironmentUniverseArtifact(
        id="acme-universe", version=1,
        service_artifact_ids=["slack-service"],
        metadata={"manifest.json": "acme-manifest"},
    ).model_dump(by_alias=True)
    assert doc["service_artifact_ids"] == ["slack-service"]
    assert doc["metadata"] == {"manifest.json": "acme-manifest"}
    assert "legacy_environment_artifact_ids" not in doc
    assert "legacy_metadata" not in doc
    assert "environment_artifact_refs" not in doc


def test_mcp_server_env_to_dict_writes_both_name_spellings_and_no_version_key():
    """``service_version`` was deleted from both env types, as from the artifact above: neither
    spelling may appear. Stored documents keep the key for ever and ``from_dict`` does not read
    it; this pins that ``to_dict`` never writes it back."""
    doc = _env_wire("mcp_server")
    assert doc["service_name"] == "slack"
    assert doc["environment_name"] == "slack"
    assert "service_version" not in doc
    assert "environment_version" not in doc


def test_website_env_to_dict_writes_both_name_spellings_and_no_version_key():
    doc = _env_wire("website")
    assert doc["service_name"] == "shop"
    assert doc["environment_name"] == "shop"
    assert "service_version" not in doc
    assert "environment_version" not in doc


_AE3_TWINS = {
    "environment_name": "service_name",
    "environment_artifact_refs": "service_artifact_refs",
    "object_url": "s3_url",
    "tar_gz_object_url": "tar_gz_s3_url",
    "build_context_object_url": "build_context_s3_url",
    "bundle_object_url": "bundle_s3_url",
    "skill_object_url": "skill_s3_url",
    "cli_object_url": "cli_s3_url",
}


@pytest.mark.parametrize("wire,keys", [(_artifact_wire, ARTIFACT_FIXTURES), (_env_wire, ENV_FIXTURES)])
def test_ae3_dual_write_only_ever_adds_a_twin_of_an_existing_key(wire, keys):
    """The additive invariant, asserted structurally rather than by eyeballing a
    golden diff, for every registered type at once — including types added after
    this was written.

    Both-or-neither, deliberately. An earlier one-way version of this only
    checked the twin where it was already present, so a key that was never
    dual-written at all asserted nothing and passed green. That is exactly how
    the missing ``ConfigDirective`` twin survived the first round of review."""
    for registry_key in sorted(keys):
        doc = wire(registry_key)
        for new, old in _AE3_TWINS.items():
            assert (new in doc) == (old in doc), (
                f"{registry_key}: {old} and {new} must appear together — a one-sided "
                f"twin is a dual-write decaying into a rename"
            )
            if new in doc:
                assert doc[new] == doc[old], f"{registry_key}: {new} diverged from {old}"


def test_config_directive_to_dict_writes_both_keys():
    """``from_dict`` reads either spelling; ``to_dict`` writes both,
    because this is the ``tasks`` / ``task_steps`` document serializer.

    The frozen surface is the *Harbor applier input*, which
    ``_resolve_task_server_config`` in the plugin-hosted Harbor exporter
    builds straight off the attributes and never routes through here — it
    emits a fourth key this does
    not, so the two provably cannot be the same dict. Without the twin, a task
    authored today would be unloadable the moment the legacy key is dropped,
    which is the task-load KeyError the whole additive program exists to avoid."""
    expected = {
        "service": "slack", "environment": "slack",
        "uri": "urn:agentenv:set-errors/v1", "args": {"error_rate": 0.5},
    }
    for key in ("service", "environment"):
        d = ConfigDirective.from_dict(
            {key: "slack", "uri": "urn:agentenv:set-errors/v1", "args": {"error_rate": 0.5}}
        )
        assert d.to_dict() == expected


def test_artifact_type_literal_defaults_are_the_persisted_values():
    """The invariant is the default ASSIGNMENT, not the Literal member order: ``_get_type``
    reads ``model_fields["type"].default``, so widening the Literal cannot re-key the registry."""
    assert EnvironmentArtifact.model_fields["type"].default == "environment"
    assert EnvironmentUniverseArtifact.model_fields["type"].default == "environment_universe"
    registry = get_artifact_registry()
    assert registry["environment"] is EnvironmentArtifact
    assert registry["environment_universe"] is EnvironmentUniverseArtifact
    assert "service" not in registry
    assert "service_universe" not in registry


def _regenerate() -> None:
    for directory, fixtures, wire in (
        (ARTIFACT_GOLDEN_DIR, ARTIFACT_FIXTURES, _artifact_wire),
        (ENV_GOLDEN_DIR, ENV_FIXTURES, _env_wire),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for registry_key in fixtures:
            (directory / f"{registry_key}.json").write_text(_dumps(wire(registry_key)))


if __name__ == "__main__":
    _regenerate()
