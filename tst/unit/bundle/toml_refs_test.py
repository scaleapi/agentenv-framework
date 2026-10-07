"""Every type written from a bundle's toml declares exactly the keys that name other entities: each ``toml_refs``
path is a key it takes, ``from_toml`` loads what each declared path names, and what it writes references nothing
else."""

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import Artifact, EnvironmentArtifact, EnvironmentUniverseArtifact, FileArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.ref import ArtifactRef
from agent_env.artifact.store import ArtifactStore
from agent_env.bundle import parse_bundle
from agent_env.bundle.authoring import AuthoringContext
from agent_env.bundle.parse import _EVAL_KEYS, CONFIG_FILES
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.eval.eval import Eval
from tst.unit.bundle._support import layout

WRITTEN_FROM_TOML = {  # each type, the folder that holds one, and the keys it needs beside its references
    MCPServerEnv: ("envs/server", {"environment_name": "server"}),
    WebsiteEnv: ("envs/site", {"environment_name": "site"}),
    MultiEnv: ("envs/suite", {}),
    A2AAgent: ("agents/solver", {}),
    EnvironmentArtifact: ("artifacts/crm-data", {"environment_name": "crm"}),
    EnvironmentUniverseArtifact: ("artifacts/world", {}),
}


def _type(cls):
    return cls.model_fields["type"].default if issubclass(cls, Artifact) else cls.type


@pytest.mark.parametrize("cls", [*WRITTEN_FROM_TOML, Eval], ids=lambda cls: cls.__name__)
def test_each_declared_path_starts_at_a_key_the_type_takes(cls):
    takes = set(_EVAL_KEYS) if cls is Eval else set(cls.toml_keys)
    assert cls.toml_refs and {ref.field for ref in cls.toml_refs} <= takes


def _image(id):
    return DockerImageArtifact(id=id, version=1, description=id, image_name=f"registry.example/{id}:v1",
                               tar_gz_s3_url=f"s3://bucket/{id}.tar.gz")


_STAND_INS = {  # what a load returns for each type a from_toml asks for
    DockerImageArtifact: _image,
    FileArtifact: lambda id: FileArtifact(id=id, version=1, description=id, filename="data.json",
                                          content_type="application/json", s3_url=f"s3://bucket/{id}"),
    EnvironmentArtifact: lambda id: EnvironmentArtifact(id=id, version=1, environment_name=id,
                                                        file_artifact_ref=ArtifactRef(id=f"{id}-file", version=1)),
    MCPServerEnv: lambda id: MCPServerEnv(id, 1, docker_image_artifact=_image(f"{id}-image"), environment_name=id),
    WebsiteEnv: lambda id: WebsiteEnv(id, 1, backend_docker_image_artifact=_image(f"{id}-back"),
                                      frontend_docker_image_artifact=_image(f"{id}-front"), environment_name=id),
}


@pytest.mark.parametrize("cls", WRITTEN_FROM_TOML, ids=lambda cls: cls.__name__)
def test_from_toml_loads_each_declared_ref_and_writes_no_other(tmp_path, monkeypatch, cls):
    folder, needed = WRITTEN_FROM_TOML[cls]
    kind = next(kind for kind in CONFIG_FILES if folder.startswith(f"{kind.value}/"))
    root = layout(tmp_path / "b", {f"{folder}/{CONFIG_FILES[kind]}": f'type = "{_type(cls)}"\n'})
    entry = next(entry for entry in parse_bundle(root).entries if entry.name == folder.split("/")[1])
    sentinels = {ref: f"sentinel-{ref.field}" for ref in cls.toml_refs}
    data = {"type": _type(cls), **needed,
            **{ref.field: [id] if ref.path.endswith("[]") else id for ref, id in sentinels.items()}}
    loads, written = [], []

    def load(self, base, kind, ref, expect):
        loads.append((kind, ref))
        return _STAND_INS[expect](ref)

    def put(**kwargs):
        written.append(cls(version=1, **kwargs))
        return written[-1]

    def put_document(self, artifact):
        written.append(artifact.model_copy(update={"version": 1}))
        return written[-1]

    monkeypatch.setattr(AuthoringContext, "_load", load)
    if issubclass(cls, Artifact):  # the real put, so its refs are built as they're stored
        monkeypatch.setattr(ArtifactStore, "put_document", put_document)
    else:
        monkeypatch.setattr(cls, "put", put)
    cls.from_toml(data, AuthoringContext(parse_bundle(root), entry))

    assert sorted(loads) == sorted((ref.kind, id) for ref, id in sentinels.items())
    (entity,) = written
    document = entity.model_dump(by_alias=True) if isinstance(entity, Artifact) else entity.to_dict()
    assert _referenced(document) and _referenced(document) <= set(sentinels.values())


def _referenced(doc):
    """The ids of the entities a written document names: each nested ``{id, version, ...}`` table."""
    found = set()
    for value in doc.values():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict) and "id" in item and "version" in item:
                found.add(item["id"])
    return found
