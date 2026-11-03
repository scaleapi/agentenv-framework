"""EnvironmentUniverseArtifact.get_file_artifacts() — the plain-file view that lets a
service universe be staged as a file tree (grading) by the same loaders that
handle a FileArtifactUniverse.

Contract: keys are `<service name>/<wrapped filename>`, or `<service name>/<path>`
for each file of an environment holding several; values are the universe's
EXISTING FileArtifacts (nothing copied or re-registered). A universe holding an
environment of several files can't be written to a universe folder.
"""

from __future__ import annotations

import pytest

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.entity.export import FolderExport, Unwritable


def _file_artifact(id: str, filename: str) -> FileArtifact:
    return FileArtifact(
        id=id,
        version=1,
        description=f"state of {id}",
        filename=filename,
        content_type="application/json",
        s3_url=f"s3://bucket/{id}/{filename}",
    )


def _service_artifact(id: str, environment_name: str, file_artifact: FileArtifact) -> EnvironmentArtifact:
    return EnvironmentArtifact(
        id=id,
        version=1,
        environment_name=environment_name,
        file_artifact_ref=file_artifact.as_ref(),
    )


@pytest.fixture
def registry(monkeypatch):
    """Resolve pinned refs from an in-memory map instead of the artifact store."""
    files: dict[str, FileArtifact] = {}
    services: dict[str, EnvironmentArtifact] = {}

    monkeypatch.setattr(FileArtifact, "get", classmethod(lambda cls, id, version=None: files[id]))
    monkeypatch.setattr(EnvironmentArtifact, "get", classmethod(lambda cls, id, version=None: services[id]))
    return files, services


def _universe(registry, specs: list[tuple[str, str]], universe_id: str = "snapshot-env-abc") -> EnvironmentUniverseArtifact:
    """specs = [(environment_name, wrapped filename)] — one EnvironmentArtifact each."""
    files, services = registry
    refs = []
    for idx, (environment_name, filename) in enumerate(specs):
        fa = _file_artifact(f"fa-{idx}", filename)
        sa = _service_artifact(f"sa-{idx}", environment_name, fa)
        files[fa.id] = fa
        services[sa.id] = sa
        refs.append(ArtifactRef(id=sa.id, version=sa.version))
    return EnvironmentUniverseArtifact(id=universe_id, version=2, service_artifact_refs=refs)


def test_keys_are_service_dir_plus_wrapped_filename(registry):
    universe = _universe(registry, [
        ("slack", "snapshot-env-slack-a1b2c3.json"),
        ("gdrive", "gdrive.zip"),
    ])
    assert sorted(universe.get_file_artifacts()) == [
        "gdrive/gdrive.zip",
        "slack/snapshot-env-slack-a1b2c3.json",
    ]


def test_service_dir_keeps_a_shared_basename_apart(registry):
    """Two services can upload bundles with the same basename; flat keys would
    silently drop one."""
    universe = _universe(registry, [("slack", "data.json"), ("gdrive", "data.json")])
    assert sorted(universe.get_file_artifacts()) == ["gdrive/data.json", "slack/data.json"]


def test_returns_the_universes_existing_file_artifacts(registry):
    """No copy, no re-register: the values ARE the pinned FileArtifacts."""
    universe = _universe(registry, [("slack", "state.json")])
    files, _ = registry
    assert universe.get_file_artifacts()["slack/state.json"] is files["fa-0"]


def test_duplicate_service_names_raise(registry):
    """Two services of the same name would silently collapse to one file —
    the load side guards against exactly this masking, so refuse to stage."""
    universe = _universe(registry, [("slack", "a.json"), ("slack", "a.json")])
    with pytest.raises(ValueError, match="duplicate environment 'slack'"):
        universe.get_file_artifacts()


def test_an_environment_of_several_files_lists_each_at_its_path_and_cant_be_written_to_a_universe_folder(registry, tmp_path):
    files, services = registry
    a, b = _file_artifact("fa-a", "a.json"), _file_artifact("fa-b", "b.json")
    files.update({a.id: a, b.id: b})
    services["mail"] = EnvironmentArtifact(id="mail", version=1, environment_name="mail",
                                           file_artifact_refs={"inbox/a.json": a.as_ref(), "b.json": b.as_ref()})
    universe = EnvironmentUniverseArtifact(id="u", version=1, service_artifact_refs=[ArtifactRef(id="mail", version=1)])

    assert universe.get_file_artifacts() == {"mail/inbox/a.json": a, "mail/b.json": b}
    with pytest.raises(Unwritable, match="its environment 'mail' holds several files, and a universe's folder holds one "
                                         "per environment; write environment artifact mail to a folder of its own"):
        universe.to_toml(FolderExport(tmp_path))
