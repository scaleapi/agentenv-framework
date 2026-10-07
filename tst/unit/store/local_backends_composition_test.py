"""The local ObjectStore + local DocumentStore compose end-to-end, no infra.

Exercises the real artifact operations (CRUD, versioning, the url-addressed
``_at`` ops via put_at / put_existing / universe bundling) through
``AGENT_ENV_OBJECT_STORE=local`` + ``AGENT_ENV_DOCUMENT_STORE=local``, with no
S3 or Mongo. This is the fast, network-free proof that the no-infra storage
layer works; the full task-DAG no-infra e2e stays gated on images/secrets/seeding.
"""

import gzip
import io
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts import docker_image
from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.skill import SkillArtifact, download_skill
from agent_env.cli.artifact.file_artifact_universe import file_artifact_universe
from agent_env.store.ids import fs_safe, key_segment
from agent_env.config import get_config, set_image_store, set_object_store
from agent_env.store.image_store import LocalRegistryImageStore
from tst.unit.store.fakes import FakeImageStore, FakeObjectStore


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


@pytest.mark.parametrize("url,refusal", [
    ("fake://home", "does not exist"),
    ("fake://home/", "not a prefix"),
    ("fake://home/dir/", "not a prefix"),
])
def test_put_existing_refuses_a_url_that_names_no_object(local_stores, url, refusal):
    """A prefix is refused by its trailing slash; anything else is left to the store, which holds no object there."""
    store = FakeObjectStore()
    store.put("dir/x.bin", b"v")
    set_object_store(store)

    with pytest.raises(ValueError, match=refusal):
        FileArtifact.put_existing(id="prefix", description="d", object_url=url)


def test_put_existing_refuses_a_local_directory(local_stores):
    """A directory is not an object, with or without the trailing slash."""
    store = local_stores.get_object_store()
    store.put("dir/x.bin", b"v")
    directory = store.object_url("dir")

    with pytest.raises(ValueError, match="not a prefix"):
        FileArtifact.put_existing(id="prefix", description="d", object_url=directory + "/")
    with pytest.raises(ValueError, match="does not exist"):
        FileArtifact.put_existing(id="prefix", description="d", object_url=directory)


def test_put_existing_keeps_a_filename_that_holds_a_hash(local_stores, tmp_path):
    """The filename is the url's last segment, where urlparse would stop at the '#'."""
    store = local_stores.get_object_store()
    written_url = store.put_file("artifacts/file/hashed/1/state#v2.zip", _write(tmp_path, "s.zip", b"PK"))

    fa = FileArtifact.put_existing(id="hashed", description="d", object_url=written_url)
    assert fa.filename == "state#v2.zip"
    assert fa.content_type == "application/zip"


def test_put_existing_registers_an_object_outside_the_stores_own_root(local_stores, tmp_path):
    """Like the url-addressed reads, registration needs the store to read the object, not to
    own it: an object in another root of the same backend (another bucket, say) is accepted."""
    store = FakeObjectStore(root="home")
    elsewhere = store.put_file_at("fake://shared/bundles/w.zip", _write(tmp_path, "w.zip", b"PK"))
    set_object_store(store)
    assert not store.owns(elsewhere)

    fa = FileArtifact.put_existing(id="shared-bundle", description="d", object_url=elsewhere)
    assert fa.filename == "w.zip"
    assert fa.object_url == elsewhere


def test_universe_put_bundled(local_stores, tmp_path):
    store = local_stores.get_object_store()
    files = {
        "a.txt": _write(tmp_path, "a.txt", b"AAA"),
        "sub/b.txt": _write(tmp_path, "b.txt", b"BBB"),
    }
    prefix = store.object_url("artifacts/universe/bundled/1/")

    uni = FileArtifactUniverse.put_bundled(id="bundled", files=files, prefix_url=prefix)
    loaded = {rel: fa.load() for rel, fa in FileArtifactUniverse.get(uni.id).get_file_artifacts().items()}
    assert loaded == {"a.txt": b"AAA", "sub/b.txt": b"BBB"}


def test_universe_put_existing_lists_via_list_at(local_stores, tmp_path):
    """put_existing -> ObjectStore.list_at over an existing local prefix."""
    store = local_stores.get_object_store()
    store.put_file("artifacts/universe/existing/1/one.txt", _write(tmp_path, "one.txt", b"1"))
    store.put_file("artifacts/universe/existing/1/nested/two.txt", _write(tmp_path, "two.txt", b"2"))
    prefix = store.object_url("artifacts/universe/existing/1/")

    uni = FileArtifactUniverse.put_existing(id="uni-existing", prefix_url=prefix)
    loaded = {rel: fa.load() for rel, fa in FileArtifactUniverse.get(uni.id).get_file_artifacts().items()}
    assert loaded == {"one.txt": b"1", "nested/two.txt": b"2"}


def test_a_skill_prefix_without_skill_md_is_refused_through_the_stores_not_found_error():
    """A missing SKILL.md is found through ObjectNotFoundError, not a backend-specific error."""
    store = FakeObjectStore()
    store.put("skills/demo/README.md", b"no skill file here")
    set_object_store(store)

    with pytest.raises(ValueError, match="SKILL.md not found"):
        SkillArtifact.validate(object_url=store.object_url("skills/demo"), expected_name="demo")


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


HOSTILE = "@local/~/My Work/Tickets V2"
HOSTILE_SEGMENT = "local/my-work-tickets-v2-ee679b8e5d3a"


def test_a_local_id_file_artifact_lands_under_its_encoded_segment(local_stores, cli_routing, tmp_path):
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


def test_a_local_id_cli_bundle_round_trips(local_stores, cli_routing, tmp_path):
    cli_dir = _cli_dir(tmp_path, "cli", {"bin/tool": b"#!/bin/sh\n", "lib/data.txt": b"d"})
    art = CliArtifact.put(id=HOSTILE, command_name="tool", entrypoint="bin/tool", cli_dir=cli_dir)

    assert local_stores.get_object_store().get_object_key(art.cli_object_url) == f"artifacts/cli/{HOSTILE_SEGMENT}/1"
    assert art.cli_files_id == f"{HOSTILE}__files"
    loaded = {rel: fa.load() for rel, fa in CliArtifact.get(HOSTILE).get_cli_files().get_file_artifacts().items()}
    assert loaded == {"bin/tool": b"#!/bin/sh\n", "lib/data.txt": b"d"}


def test_a_local_id_bundle_never_lists_another_ids_files(local_stores, cli_routing, tmp_path):
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


@pytest.mark.parametrize("entity_id, registry, repository, tarball", [
    (HOSTILE, "localhost:5000", HOSTILE_SEGMENT,
     (f"artifacts/docker_image/{HOSTILE_SEGMENT}/1-", "/local-my-work-tickets-v2-ee679b8e5d3a-v1.tar.gz")),
    ("legacy-image", "fake.registry", "legacy-image",
     ("artifacts/docker_image/legacy-image/1-", "/legacy-image-v1.tar.gz")),
])
def test_a_docker_image_names_its_repository_and_tarball_from_the_encoded_id(
    local_stores, cli_routing, monkeypatch, entity_id, registry, repository, tarball,
):
    images = FakeImageStore()
    set_image_store(images)
    local_registry = []
    monkeypatch.setattr(LocalRegistryImageStore, "ensure_repository", lambda self, repository: local_registry.append(repository))
    pushed = []
    monkeypatch.setattr(docker_image, "_push_local_image", lambda src, ref, store: pushed.append(ref))
    monkeypatch.setattr(docker_image.subprocess, "Popen", _DockerSave)

    art = docker_image.DockerImageArtifact.put(id=entity_id, description="d", image_name="src:latest")

    assert images.repositories + local_registry == [repository]
    assert pushed == [f"{registry}/{repository}:v1"] and art.image_name == pushed[0]
    before, after = tarball
    key = local_stores.get_object_store().get_object_key(art.tar_gz_object_url)
    assert re.fullmatch(f"{re.escape(before)}[0-9a-f]{{8}}{re.escape(after)}", key), key
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
def test_an_local_id_outside_the_cli_is_refused_before_any_image_or_object_is_written(local_stores, monkeypatch, tmp_path, put):
    images = FakeImageStore()
    set_image_store(images)
    pushed = []
    monkeypatch.setattr(docker_image, "_push_local_image", lambda src, ref, store: pushed.append(ref))
    monkeypatch.setattr(docker_image.subprocess, "Popen", _DockerSave)

    with pytest.raises(ValueError, match="only the @local namespace's store holds"):
        put(tmp_path)

    assert images.repositories == [] and pushed == []
    assert local_stores.get_object_store().list("") == []


@pytest.mark.parametrize("universe_id, segment", [
    (HOSTILE, HOSTILE_SEGMENT),
    ("Legacy/Universe v1", "Legacy/Universe v1"),
])
def test_put_bundled_writes_each_version_under_its_own_prefix(
    local_stores, cli_routing, tmp_path, universe_id, segment,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_bytes(b"a")
    prefix = local_stores.get_object_store().object_url(f"artifacts/file_artifact_universe/{segment}")
    for version in (1, 2):
        res = CliRunner().invoke(file_artifact_universe, ["put-bundled", "--id", universe_id, "--file-dir", str(source)])
        assert res.exit_code == 0, res.output
        url = FileArtifactUniverse.get(universe_id, version).bundle_object_url
        assert re.fullmatch(rf"{re.escape(prefix)}/{version}-[0-9a-f]{{8}}/", url), url


def test_a_put_bundled_that_fails_partway_doesnt_block_the_next(local_stores, monkeypatch, tmp_path):
    files = {"a.txt": _write(tmp_path, "a.txt", b"A"), "b.txt": _write(tmp_path, "b.txt", b"B")}
    put_at, calls = FileArtifact.put_at, []

    def interrupted(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise ConnectionError("interrupted")
        return put_at(**kwargs)

    monkeypatch.setattr(FileArtifact, "put_at", interrupted)
    with pytest.raises(ConnectionError):
        FileArtifactUniverse.put_bundled(id="interrupted", files=files)
    monkeypatch.setattr(FileArtifact, "put_at", put_at)
    universe = FileArtifactUniverse.put_bundled(id="interrupted", files=files)
    assert universe.version == 1
    assert {key: fa.load() for key, fa in universe.get_file_artifacts().items()} == {"a.txt": b"A", "b.txt": b"B"}


def test_a_docker_image_put_that_stops_before_its_document_doesnt_block_the_next(local_stores, monkeypatch):
    set_image_store(FakeImageStore())
    monkeypatch.setattr(docker_image, "_push_local_image", lambda src, ref, store: None)
    monkeypatch.setattr(docker_image.subprocess, "Popen", _DockerSave)
    put_tar = docker_image.DockerImageArtifact.put_tar

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(docker_image.DockerImageArtifact, "put_tar", interrupted)
    with pytest.raises(KeyboardInterrupt):
        docker_image.DockerImageArtifact.put(id="interrupted", description="d", image_name="src:latest")
    monkeypatch.setattr(docker_image.DockerImageArtifact, "put_tar", put_tar)
    art = docker_image.DockerImageArtifact.put(id="interrupted", description="d", image_name="src:latest")

    assert art.version == 1
    assert gzip.decompress(art.load()) == b"image-tar-bytes"


def test_get_many_writes_each_local_id_into_its_own_encoded_directory(local_stores, cli_routing, tmp_path):
    store = local_stores.get_object_store()
    for uid, name in (("@local/t/A", "1"), ("@local/t/A/1", "f")):
        prefix = store.object_url(f"artifacts/file_artifact_universe/{key_segment(uid)}/1/")
        FileArtifactUniverse.put_bundled(id=uid, files={name: _write(tmp_path, f"src-{name}", b"x")}, prefix_url=prefix)

    out = tmp_path / "out"
    res = CliRunner().invoke(
        file_artifact_universe,
        ["get-many", "--id", "@local/t/A", "--id", "@local/t/A/1", "--output-dir", str(out)],
    )

    assert res.exit_code == 0, res.output
    assert (out / fs_safe("@local/t/A") / "1").is_file()
    assert (out / fs_safe("@local/t/A/1") / "f").is_file()
    assert not (out / "@local").exists()


def test_get_many_writes_a_universe_whose_id_is_longer_than_a_filename(local_stores, tmp_path):
    universe_id = "crm-suite__v2-" + "Escalated tickets from the EU desk, " * 7 + "and the US desk, escalated by hand"
    FileArtifactUniverse.put_bundled(id=universe_id, files={"report.md": _write(tmp_path, "report.md", b"# report")})

    out = tmp_path / "out"
    res = CliRunner().invoke(file_artifact_universe, ["get-many", "--id", universe_id, "--output-dir", str(out)])

    assert res.exit_code == 0, res.output
    assert (out / key_segment(universe_id) / "report.md").read_bytes() == b"# report"


def test_get_many_writes_universes_that_differ_by_a_slash_into_different_directories(local_stores, tmp_path):
    for universe_id, data in (("crm/eu", b"eu"), ("crm-eu", b"flat")):
        FileArtifactUniverse.put_bundled(id=universe_id, files={"report.md": _write(tmp_path, f"{data.decode()}.md", data)})

    out = tmp_path / "out"
    res = CliRunner().invoke(file_artifact_universe, ["get-many", "--id", "crm/eu", "--id", "crm-eu", "--output-dir", str(out)])

    assert res.exit_code == 0, res.output
    assert (out / "crm" / "eu" / "report.md").read_bytes() == b"eu"
    assert (out / "crm-eu" / "report.md").read_bytes() == b"flat"
