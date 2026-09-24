"""The local ObjectStore + local DocumentStore compose end-to-end, no infra.

Exercises the real artifact operations (CRUD, versioning, the url-addressed
``_at`` ops via put_at / put_existing / universe bundling) through
``AGENT_ENV_OBJECT_STORE=local`` + ``AGENT_ENV_DOCUMENT_STORE=local``, with no
S3 or Mongo. This is the fast, network-free proof that the no-infra storage
layer works; the full task-DAG no-infra e2e stays gated on images/secrets/seeding.
"""

import gzip
import io
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts import docker_image
from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.skill import SkillArtifact, download_skill
from agent_env.artifact.store import reset_artifact_store
from agent_env.cli.artifact.file_artifact_universe import file_artifact_universe
from agent_env.store import LocalFilesystemObjectStore
from agent_env.store.ids import fs_safe, key_segment
from agent_env.config import configure, get_config, reset_config, set_image_store
from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore
from tst.unit.store.fakes import FakeImageStore


@pytest.fixture
def local_stores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    configure()
    reset_artifact_store()  # rebind the cached store to this test's fresh local config
    cfg = get_config()
    assert isinstance(cfg.get_object_store(), LocalFilesystemObjectStore)
    assert isinstance(cfg.get_document_store(), LocalSqliteDocumentStore)
    yield cfg
    reset_config()
    reset_artifact_store()


def _write(tmp_path, name, data: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_file_artifact_roundtrip_and_versioning(local_stores, tmp_path):
    src = _write(tmp_path, "payload.json", b'{"hello":"local"}')

    fa1 = FileArtifact.put(id="doc", description="v1", file_path=src)
    assert fa1.version == 1
    assert fa1.object_url.startswith("file://")
    assert FileArtifact.get("doc").load() == b'{"hello":"local"}'

    # second put -> DocumentStore versioning on SQLite
    fa2 = FileArtifact.put(id="doc", description="v2", file_path=src)
    assert fa2.version == 2
    assert FileArtifact.get("doc").version == 2
    assert FileArtifact.get("doc", 1).version == 1


def test_file_artifact_put_at(local_stores, tmp_path):
    """put_at -> ObjectStore.put_file_at (url-addressed write on the local backend)."""
    src = _write(tmp_path, "at.bin", b"at-bytes")
    store = local_stores.get_object_store()
    object_url = store.object_url("artifacts/file/putat/1/at.bin")

    fa = FileArtifact.put_at(id="putat", description="d", file_path=src, object_url=object_url)
    assert fa.object_url.startswith("file://")
    assert FileArtifact.get("putat").load() == b"at-bytes"


def test_file_artifact_put_existing(local_stores, tmp_path):
    """put_existing -> ObjectStore.get_object_metadata_at on an already-written blob."""
    store = local_stores.get_object_store()
    src = _write(tmp_path, "exists.txt", b"already-here")
    written_url = store.put_file("artifacts/file/existing/1/exists.txt", src)

    fa = FileArtifact.put_existing(id="existing", description="d", object_url=written_url)
    assert fa.filename == "exists.txt"
    assert FileArtifact.get("existing").load() == b"already-here"


def test_universe_put_bundled(local_stores, tmp_path):
    store = local_stores.get_object_store()
    files = {
        "a.txt": _write(tmp_path, "a.txt", b"AAA"),
        "sub/b.txt": _write(tmp_path, "b.txt", b"BBB"),
    }
    prefix = store.object_url("artifacts/universe/bundled/1/")

    uni = FileArtifactUniverse.put_bundled(id="bundled", files=files, s3_url=prefix)
    loaded = {rel: fa.load() for rel, fa in FileArtifactUniverse.get(uni.id).get_file_artifacts().items()}
    assert loaded == {"a.txt": b"AAA", "sub/b.txt": b"BBB"}


def test_universe_put_existing_lists_via_list_at(local_stores, tmp_path):
    """put_existing -> ObjectStore.list_at over an existing local prefix."""
    store = local_stores.get_object_store()
    store.put_file("artifacts/universe/existing/1/one.txt", _write(tmp_path, "one.txt", b"1"))
    store.put_file("artifacts/universe/existing/1/nested/two.txt", _write(tmp_path, "two.txt", b"2"))
    prefix = store.object_url("artifacts/universe/existing/1/")

    uni = FileArtifactUniverse.put_existing(id="uni-existing", s3_url=prefix)
    loaded = {rel: fa.load() for rel, fa in FileArtifactUniverse.get(uni.id).get_file_artifacts().items()}
    assert loaded == {"one.txt": b"1", "nested/two.txt": b"2"}


def test_skill_artifact_put_and_download(local_stores, tmp_path):
    """SkillArtifact.put derives its bundle url from the store (not a hardcoded
    s3://), so skill create + download round-trips on the local backend."""
    skill_dir = tmp_path / "skilldir"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: local-skill\ndescription: A local skill for testing.\n---\nHello.\n"
    )
    (skill_dir / "ref.txt").write_bytes(b"aux")

    art = SkillArtifact.put(id="local-skill", skill_dir=skill_dir)
    assert art.skill_object_url.startswith("file://")
    assert local_stores.get_object_store().get_object_key(art.skill_object_url) == "artifacts/skill/local-skill/1"
    assert art.skill_files_id == "local-skill__files"
    with download_skill(art.skill_object_url) as out:
        names = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert names == ["SKILL.md", "ref.txt"]


@pytest.fixture
def local_ids_accepted(monkeypatch):
    """Stands in for @local acceptance, which arrives with namespace-routed stores: lifts the
    '@' reservation so these tests can drive the id sinks end to end on local backends."""
    monkeypatch.setattr("agent_env.store.document_store.document_store.reject_reserved_id", lambda _id: None)
    monkeypatch.setattr("agent_env.artifact.store.reject_reserved_id", lambda _id: None)


HOSTILE = "@local/~/My Work/Tickets V2"
HOSTILE_SEGMENT = "local/my-work-tickets-v2-ee679b8e5d3a"


def test_a_local_id_file_artifact_lands_under_its_encoded_segment(local_stores, local_ids_accepted, tmp_path):
    src = _write(tmp_path, "payload.json", b'{"hostile": true}')
    fa = FileArtifact.put(id=HOSTILE, description="d", file_path=src)
    fb = FileArtifact.put_bytes(id=HOSTILE, description="d", filename="raw.bin", content=b"raw")

    store = local_stores.get_object_store()
    assert store.get_object_key(fa.object_url) == f"artifacts/file/{HOSTILE_SEGMENT}/1/payload.json"
    assert store.get_object_key(fb.object_url) == f"artifacts/file/{HOSTILE_SEGMENT}/2/raw.bin"
    assert FileArtifact.get(HOSTILE, 1).load() == b'{"hostile": true}'
    assert FileArtifact.get(HOSTILE).load() == b"raw"


def test_a_legacy_id_keeps_its_object_key_byte_identical(local_stores, tmp_path):
    fa = FileArtifact.put(id="Legacy/Id v1", description="d", file_path=_write(tmp_path, "p.txt", b"x"))
    assert local_stores.get_object_store().get_object_key(fa.object_url) == "artifacts/file/Legacy/Id v1/1/p.txt"


def _cli_dir(tmp_path, name, files):
    root = tmp_path / name
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def test_a_local_id_cli_bundle_round_trips(local_stores, local_ids_accepted, tmp_path):
    cli_dir = _cli_dir(tmp_path, "cli", {"bin/tool": b"#!/bin/sh\n", "lib/data.txt": b"d"})
    art = CliArtifact.put(id=HOSTILE, command_name="tool", entrypoint="bin/tool", cli_dir=cli_dir)

    assert local_stores.get_object_store().get_object_key(art.cli_object_url) == f"artifacts/cli/{HOSTILE_SEGMENT}/1"
    assert art.cli_files_id == f"{HOSTILE}__files"
    loaded = {rel: fa.load() for rel, fa in CliArtifact.get(HOSTILE).get_cli_files().get_file_artifacts().items()}
    assert loaded == {"bin/tool": b"#!/bin/sh\n", "lib/data.txt": b"d"}


def test_a_local_id_bundle_never_lists_another_ids_files(local_stores, local_ids_accepted, tmp_path):
    parent = CliArtifact.put(
        id="@local/t/A", command_name="a", entrypoint="a", cli_dir=_cli_dir(tmp_path, "a", {"a": b"A"}),
    )
    CliArtifact.put(id="@local/t/A/1", command_name="b", entrypoint="b", cli_dir=_cli_dir(tmp_path, "b", {"b": b"B"}))
    with download_skill(parent.cli_object_url) as out:
        assert sorted(p.name for p in out.rglob("*") if p.is_file()) == ["a"]


class _DockerSave:
    def __init__(self, *args, **kwargs):
        self.stdout = io.BytesIO(b"image-tar-bytes")
        self.stderr = io.BytesIO(b"")
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


@pytest.mark.parametrize("entity_id, repository, tarball", [
    (HOSTILE, HOSTILE_SEGMENT, f"artifacts/docker_image/{HOSTILE_SEGMENT}/1/local-my-work-tickets-v2-ee679b8e5d3a-v1.tar.gz"),
    ("legacy-image", "legacy-image", "artifacts/docker_image/legacy-image/1/legacy-image-v1.tar.gz"),
])
def test_a_docker_image_names_its_repository_and_tarball_from_the_encoded_id(
    local_stores, local_ids_accepted, monkeypatch, entity_id, repository, tarball,
):
    images = FakeImageStore()
    set_image_store(images)
    pushed = []
    monkeypatch.setattr(docker_image, "_push_local_image", lambda src, ref, store: pushed.append(ref))
    monkeypatch.setattr(docker_image.subprocess, "Popen", _DockerSave)

    art = docker_image.DockerImageArtifact.put(id=entity_id, description="d", image_name="src:latest")

    assert images.repositories == [repository]
    assert pushed == [f"fake.registry/{repository}:v1"] and art.image_name == pushed[0]
    assert local_stores.get_object_store().get_object_key(art.tar_gz_object_url) == tarball
    assert gzip.decompress(docker_image.DockerImageArtifact.get(entity_id).load()) == b"image-tar-bytes"



@pytest.mark.parametrize("put", [
    lambda tmp_path: docker_image.DockerImageArtifact.put(id=HOSTILE, description="d", image_name="src:latest"),
    lambda tmp_path: FileArtifact.put(id=HOSTILE, description="d", file_path=_write(tmp_path, "p.txt", b"x")),
    lambda tmp_path: FileArtifact.put_at(
        id=HOSTILE, description="d", file_path=_write(tmp_path, "p.txt", b"x"),
        object_url=get_config().get_object_store().object_url("explicit/p.txt"),
    ),
    lambda tmp_path: FileArtifactUniverse.put_bundled(
        id=HOSTILE, files={"p.txt": Path(_write(tmp_path, "p.txt", b"x"))},
        s3_url=get_config().get_object_store().object_url("bundle/"),
    ),
], ids=["docker_image", "file", "file_put_at", "universe_put_bundled"])
def test_a_reserved_id_is_refused_before_any_image_or_object_is_written(local_stores, monkeypatch, tmp_path, put):
    images = FakeImageStore()
    set_image_store(images)
    pushed = []
    monkeypatch.setattr(docker_image, "_push_local_image", lambda src, ref, store: pushed.append(ref))
    monkeypatch.setattr(docker_image.subprocess, "Popen", _DockerSave)

    with pytest.raises(ValueError, match="reserved"):
        put(tmp_path)

    assert images.repositories == [] and pushed == []
    assert local_stores.get_object_store().list("") == []


@pytest.mark.parametrize("universe_id, segment", [
    (HOSTILE, HOSTILE_SEGMENT),
    ("Legacy/Universe v1", "Legacy/Universe v1"),
])
def test_put_bundled_defaults_its_prefix_to_the_encoded_id(local_stores, monkeypatch, tmp_path, universe_id, segment):
    _write(tmp_path, "a.txt", b"a")
    monkeypatch.setattr(local_stores, "get_s3_bucket", lambda: "bucket")
    calls = []

    def put_bundled(id, files, s3_url):
        calls.append(s3_url)
        return FileArtifactUniverse(
            id=id, version=1, description="d", file_artifact_ids={}, bundle_object_url=s3_url + "bundle.tar.gz",
        )

    monkeypatch.setattr(FileArtifactUniverse, "put_bundled", put_bundled)
    res = CliRunner().invoke(file_artifact_universe, ["put-bundled", "--id", universe_id, "--file-dir", str(tmp_path)])

    assert res.exit_code == 0, res.output
    assert calls == [f"s3://bucket/file_artifact_universe/{segment}/"]

def test_get_many_writes_each_local_id_into_its_own_encoded_directory(local_stores, local_ids_accepted, tmp_path):
    store = local_stores.get_object_store()
    for uid, name in (("@local/t/A", "1"), ("@local/t/A/1", "f")):
        prefix = store.object_url(f"artifacts/file_artifact_universe/{key_segment(uid)}/1/")
        FileArtifactUniverse.put_bundled(id=uid, files={name: _write(tmp_path, f"src-{name}", b"x")}, s3_url=prefix)

    out = tmp_path / "out"
    res = CliRunner().invoke(
        file_artifact_universe,
        ["get-many", "--id", "@local/t/A", "--id", "@local/t/A/1", "--output-dir", str(out)],
    )

    assert res.exit_code == 0, res.output
    assert (out / fs_safe("@local/t/A") / "1").is_file()
    assert (out / fs_safe("@local/t/A/1") / "f").is_file()
    assert not (out / "@local").exists()
