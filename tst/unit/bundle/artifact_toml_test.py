"""Environment and environment_universe artifacts written from a bundle's artifact folders: an environment over the
file its folder holds or the file artifact it names, and a universe laid out as ``environment-universe get
--output-dir`` writes one, adding the environments it names. Each is checked before any write, rewritten when
what it records changes, and equal, field for field, to one the CLI writes from the same files."""

import json

import pytest
from click.testing import CliRunner

from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact, FileArtifact
from agent_env.artifact.registry import canonical_type
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import check_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.cli import cli
from agent_env.store.routing import namespace_routing
from tst.unit.bundle._support import RefusingStore, layout, local_store, plan_of

ROOT = "@local/~/triage"
TOML = "artifact.toml"


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "triage"


def _bundle(root, files, *loaded):
    """``files`` laid out under ``root``, with a task loading each artifact in ``loaded`` into a local sandbox."""
    steps = [{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm",
              "sandbox_type": "local"}]
    steps += [{"id": f"load-{name}", "type": "load_artifact", "sandbox_name": "box", "artifact_id": name,
               "destination_path": f"/app/{name}", "depends_on": ["box"]} for name in loaded]
    return layout(root, {**files, "tasks/t.json": json.dumps(steps)})


def _environment(name, environment_name, **keys):
    toml = 'type = "environment"\n' + f'environment_name = "{environment_name}"\n'
    toml += "".join(f"{key} = {value}\n" for key, value in keys.items())
    return {f"artifacts/{name}/{TOML}": toml}


def _universe(name, **keys):
    toml = 'type = "environment_universe"\n' + "".join(f"{key} = {value}\n" for key, value in keys.items())
    return {f"artifacts/{name}/{TOML}": toml}


def _run(root):
    return materialize(plan_of(root))


def _summary(materialization):
    return {done.write.id: (done.version, done.reused, done.reasons) for done in materialization.writes}


def _problems(root):
    with pytest.raises(BundleError) as caught:
        plan_of(root)
    return caught.value.problems


def _store_file(id, content, filename="data.json"):
    with namespace_routing():
        return FileArtifact.put_bytes(id, description=id, filename=filename, content=content,
                                      content_type="application/json")


def _canonical(artifact):
    """What an artifact means, less what equivalence ignores (ids, versions, created_at_utc, object URLs and a
    file's description): its type, after canonical_type, its fields, and what its refs name, followed."""
    if isinstance(artifact, FileArtifact):
        return {"type": canonical_type(artifact.type), "filename": artifact.filename,
                "content_type": artifact.content_type, "content": artifact.load()}
    if isinstance(artifact, EnvironmentArtifact):
        return {"type": canonical_type(artifact.type), "environment_name": artifact.environment_name,
                "file": _canonical(artifact.get_file_artifact())}
    return {"type": canonical_type(artifact.type),
            "environments": sorted((_canonical(env) for env in artifact.get_environment_artifacts()),
                                   key=lambda env: env["environment_name"]),
            "metadata": {key: _canonical(file) for key, file in artifact.get_metadata().items()}}


def _fields(artifact):
    """The keys of the document the store holds for ``artifact``."""
    return set(artifact.model_dump(by_alias=True))


# Environments


def test_an_environment_wraps_the_one_file_its_folder_holds_and_is_rewritten_when_it_changes(bundle_dir):
    _bundle(bundle_dir, {**_environment("crm-data", "crm"), "artifacts/crm-data/data.json": '{"rows": 1}\n'},
            "crm-data")

    first = _summary(_run(bundle_dir))
    environment = EnvironmentArtifact.get(f"{ROOT}/crm-data")
    file = environment.get_file_artifact()

    assert first[f"{ROOT}/crm-data"] == (1, False, ("new",))
    assert environment.environment_name == "crm"
    assert (file.id, file.version, file.filename, file.description) == (f"{ROOT}/crm-data__file", 1, "data.json",
                                                                       "data.json")
    assert file.load() == b'{"rows": 1}\n'

    assert all(reused for _, reused, _ in _summary(_run(bundle_dir)).values())
    (bundle_dir / "artifacts/crm-data/data.json").write_text('{"rows": 2}\n')
    assert _summary(_run(bundle_dir))[f"{ROOT}/crm-data"] == (2, False, ("files changed: data.json",))
    assert EnvironmentArtifact.get(f"{ROOT}/crm-data").get_file_artifact().load() == b'{"rows": 2}\n'


def test_a_description_names_the_folders_file(bundle_dir):
    _bundle(bundle_dir, {**_environment("crm-data", "crm", description='"crm rows"'),
                         "artifacts/crm-data/data.json": "{}\n"}, "crm-data")

    _run(bundle_dir)

    assert EnvironmentArtifact.get(f"{ROOT}/crm-data").get_file_artifact().description == "crm rows"


def test_an_environment_over_a_bundle_file_is_rewritten_when_the_file_is(bundle_dir):
    _bundle(bundle_dir, {"artifacts/crm-file/data.json": "{}\n", **_environment("crm-data", "crm", file='"crm-file"')},
            "crm-data")

    first = _summary(_run(bundle_dir))
    file = EnvironmentArtifact.get(f"{ROOT}/crm-data").get_file_artifact()

    assert first[f"{ROOT}/crm-file"][0] == first[f"{ROOT}/crm-data"][0] == 1
    assert (file.id, file.version) == (f"{ROOT}/crm-file", 1)
    (bundle_dir / "artifacts/crm-file/data.json").write_text('{"edited": true}\n')
    again = _summary(_run(bundle_dir))
    assert again[f"{ROOT}/crm-data"] == (2, False, (f"artifact {ROOT}/crm-file is written anew (v1 → v2)",))
    assert EnvironmentArtifact.get(f"{ROOT}/crm-data").get_file_artifact().version == 2


def test_an_environment_over_a_store_file_pins_the_version_the_plan_read(bundle_dir):
    _store_file("crm-rows", b"{}\n")
    _bundle(bundle_dir, {**_environment("crm-data", "crm", file='"crm-rows"'),
                         **_environment("pinned", "crm", file='{ artifact = "crm-rows", version = 1 }')},
            "crm-data", "pinned")

    _run(bundle_dir)
    _store_file("crm-rows", b'{"newer": true}\n')
    again = _summary(_run(bundle_dir))

    assert again[f"{ROOT}/crm-data"] == (2, False, ("artifact crm-rows has a new version in the store (v1 → v2)",))
    assert again[f"{ROOT}/pinned"][1] is True
    assert EnvironmentArtifact.get(f"{ROOT}/crm-data").file_artifact_ref.version == 2
    assert EnvironmentArtifact.get(f"{ROOT}/pinned").file_artifact_ref.version == 1


# Universes


def test_a_universe_writes_the_environments_and_metadata_its_folder_lays_out_then_the_ones_it_names(bundle_dir):
    _bundle(bundle_dir, {
        **_environment("crm-data", "crm"), "artifacts/crm-data/data.json": '{"crm": 1}\n',
        **_universe("world", environment_artifacts='["crm-data"]'),
        "artifacts/world/slack/data.json": '{"slack": 1}\n', "artifacts/world/gmail/mail.zip": "zip\n",
        "artifacts/world/metadata/manifest/manifest.json": '{"m": 1}\n',
    }, "world")

    first = _summary(_run(bundle_dir))
    world = EnvironmentUniverseArtifact.get(f"{ROOT}/world")
    environments = world.get_environment_artifacts()

    assert first[f"{ROOT}/world"] == (1, False, ("new",))
    assert [(env.id, env.environment_name) for env in environments] == [
        (f"{ROOT}/world__gmail", "gmail"), (f"{ROOT}/world__slack", "slack"), (f"{ROOT}/crm-data", "crm")]
    assert environments[0].get_file_artifact().id == f"{ROOT}/world__gmail__file"
    assert world.get_file_artifacts()["slack/data.json"].load() == b'{"slack": 1}\n'
    manifest = world.get_metadata()["manifest"]
    assert (manifest.id, manifest.filename, manifest.load()) == (f"{ROOT}/world__metadata__manifest", "manifest.json",
                                                                b'{"m": 1}\n')
    assert world.legacy_metadata is None and "metadata" not in world.model_dump(by_alias=True)

    assert all(reused for _, reused, _ in _summary(_run(bundle_dir)).values())
    (bundle_dir / "artifacts/crm-data/data.json").write_text('{"crm": 2}\n')
    assert _summary(_run(bundle_dir))[f"{ROOT}/world"] == (
        2, False, (f"artifact {ROOT}/crm-data is written anew (v1 → v2)",))
    (bundle_dir / "artifacts/world/slack/data.json").write_text('{"slack": 2}\n')
    assert _summary(_run(bundle_dir))[f"{ROOT}/world"] == (3, False, ("files changed: slack/data.json",))
    world = EnvironmentUniverseArtifact.get(f"{ROOT}/world")
    assert world.get_file_artifacts()["slack/data.json"].load() == b'{"slack": 2}\n'
    assert EnvironmentArtifact.get(f"{ROOT}/world__slack").version == 3  # the folder's children follow the universe


def test_a_universe_of_store_environments_only_needs_no_files(bundle_dir):
    with namespace_routing():
        mail = EnvironmentArtifact.put("mail-data", environment_name="gmail",
                                       file_artifact=_store_file("mail-rows", b"{}\n"))
    _bundle(bundle_dir, _universe("world", environment_artifacts='["mail-data"]'), "world")

    _run(bundle_dir)

    refs = EnvironmentUniverseArtifact.get(f"{ROOT}/world").environment_artifact_refs
    assert [(ref.id, ref.version) for ref in refs] == [("mail-data", mail.version)]


def test_a_universe_over_store_data_loads_it_through_the_local_namespace(bundle_dir):
    """An @local universe whose environment wraps a store file: its files read through the routed stores."""
    _store_file("crm-rows", b'{"from": "store"}\n')
    _bundle(bundle_dir, {**_environment("crm-data", "crm", file='"crm-rows"'),
                         **_universe("world", environment_artifacts='["crm-data"]')}, "world")

    _run(bundle_dir)

    with namespace_routing():
        files = EnvironmentUniverseArtifact.get(f"{ROOT}/world").get_file_artifacts()
        assert {name: file.load() for name, file in files.items()} == {"crm/data.json": b'{"from": "store"}\n'}


def test_a_universe_the_cli_downloads_is_written_from_a_bundle_equal_to_its_original(bundle_dir, tmp_path):
    for name, content in (("gmail", b'{"inbox": []}\n'), ("slack", b'{"channels": []}\n')):
        (tmp_path / f"{name}.json").write_bytes(content)
        put = CliRunner().invoke(cli, ["artifact", "environment", "put", str(tmp_path / f"{name}.json"),
                                       "--id", f"{name}-data", "--description", name, "--environment-name", name])
        assert put.exit_code == 0, put.output
    (tmp_path / "manifest.json").write_text('{"m": 1}\n')
    put = CliRunner().invoke(cli, ["artifact", "environment-universe", "put", "--id", "original",
                                   "--environment-artifact", "gmail-data", "--environment-artifact", "slack-data",
                                   "--metadata", f"manifest={tmp_path / 'manifest.json'}"])
    assert put.exit_code == 0, put.output
    folder = bundle_dir / "artifacts/copy"
    got = CliRunner().invoke(cli, ["artifact", "environment-universe", "get", "--id", "original", "--output-dir",
                                   str(folder)])
    assert got.exit_code == 0, got.output
    _bundle(bundle_dir, _universe("copy"), "copy")

    _run(bundle_dir)

    with namespace_routing():
        original, copy = EnvironmentUniverseArtifact.get("original"), EnvironmentUniverseArtifact.get(f"{ROOT}/copy")
        assert _canonical(copy) == _canonical(original)
        assert _fields(copy) == _fields(original)
        for written, cli_written in zip(copy.get_environment_artifacts(), original.get_environment_artifacts()):
            assert _fields(written) == _fields(cli_written)
            assert _fields(written.get_file_artifact()) == _fields(cli_written.get_file_artifact())


def test_an_environment_written_either_way_equals_the_one_the_cli_writes(bundle_dir, tmp_path):
    (tmp_path / "data.json").write_text('{"rows": 1}\n')
    put = CliRunner().invoke(cli, ["artifact", "environment", "put", str(tmp_path / "data.json"), "--id", "original",
                                   "--description", "crm", "--environment-name", "crm"])
    assert put.exit_code == 0, put.output
    _bundle(bundle_dir, {**_environment("folder", "crm"), "artifacts/folder/data.json": '{"rows": 1}\n',
                         "artifacts/crm-file/data.json": '{"rows": 1}\n',
                         **_environment("named", "crm", file='"crm-file"')}, "folder", "named")

    _run(bundle_dir)

    with namespace_routing():
        original = EnvironmentArtifact.get("original")
        for name in ("folder", "named"):
            written = EnvironmentArtifact.get(f"{ROOT}/{name}")
            assert _canonical(written) == _canonical(original)
            assert _fields(written) == _fields(original)


# Refused before any write


def _refused(files):
    return {**files, "tasks/t.json": json.dumps([
        {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "local"},
        *({"id": f"load-{name}", "type": "load_artifact", "sandbox_name": "box", "artifact_id": name,
           "destination_path": f"/app/{name}", "depends_on": ["box"]} for name in ("x",)),
    ])}


@pytest.mark.parametrize("files, problem", [
    (_environment("x", "crm") | {"artifacts/x/a.json": "{}", "artifacts/x/b.json": "{}"},
     "artifacts/x: an environment artifact holds one file, and this folder has 2 ('a.json', 'b.json'); name the "
     "artifact to wrap with file, or give each other file an artifact folder of its own"),
    ({f"artifacts/x/{TOML}": 'type = "environment"\n', "artifacts/x/a.json": "{}"},
     "artifacts/x: artifact.toml: environment_name is required: the environment the file seeds"),
    (_environment("x", "") | {"artifacts/x/a.json": "{}"},
     "artifacts/x: artifact.toml: environment_name can't be empty"),
    (_environment("x", "crm", service_name='"crm"') | {"artifacts/x/a.json": "{}"},
     "artifacts/x: artifact.toml: service_name is what the stored document calls it; write environment_name instead"),
    (_environment("x", "crm", file_artifact_id='"f"') | {"artifacts/x/a.json": "{}"},
     "artifacts/x: artifact.toml: file_artifact_id is what the stored document calls it; write file instead"),
    (_environment("x", "crm", file='"f"', description='"d"') | {"artifacts/f/a.json": "{}"},
     "artifacts/x: artifact.toml: description describes the folder's own file, and file names an artifact instead; "
     "drop one"),
    (_environment("x", "crm", file='"f"') | {"artifacts/f/a.json": "{}", "artifacts/x/stray.json": "{}"},
     "artifacts/x: its file is the artifact file names, so the folder holds only artifact.toml, and it also has "
     "'stray.json'; give the file an artifact folder of its own, or leave out file to wrap the folder's one file"),
    (_environment("x", "crm", file='"many"') | {"artifacts/many/a.json": "{}", "artifacts/many/b.json": "{}"},
     "artifacts/x: file: 'many' is this bundle's file_artifact_universe, but this field takes file"),
    (_universe("x"),
     "artifacts/x: a universe needs an environment: a folder of its own, named after it and holding its file, or one "
     "environment_artifacts names"),
    (_universe("x") | {"artifacts/x/stray.json": "{}"},
     "artifacts/x/stray.json: a universe folder holds only artifact.toml and a folder for each environment; move it "
     "into <environment_name>/, or give it an artifact folder of its own"),
    (_universe("x") | {"artifacts/x/gmail/a.json": "{}", "artifacts/x/gmail/b.json": "{}"},
     "artifacts/x/gmail: holds one file, and has 2 ('a.json', 'b.json')"),
    (_universe("x") | {"artifacts/x/gmail/inner/a.json": "{}"},
     "artifacts/x/gmail/inner/a.json: an environment's folder holds its one file directly, gmail/<file>, with no "
     "folders inside"),
    (_universe("x") | {"artifacts/x/g__mail/a.json": "{}"},
     "artifacts/x/g__mail: a name holding __ could clash with the ids derived from it; rename the folder"),
    (_universe("x") | {"artifacts/x/gmail/a.json": "{}", "artifacts/x/metadata/m.json": "{}"},
     "artifacts/x/metadata/m.json: metadata/ holds a folder for each key, with that key's one file in it "
     "(metadata/<key>/<file>); an environment named metadata goes in an artifact folder of its own, named in "
     "environment_artifacts"),
    (_universe("x", environment_artifacts='["gmail-data"]') | {"artifacts/x/gmail/a.json": "{}"}
     | _environment("gmail-data", "gmail") | {"artifacts/gmail-data/a.json": "{}"},
     "artifacts/x: environment_artifacts[0]: environment_name 'gmail' is also the gmail/ folder's; a universe's "
     "environments need names of their own"),
    (_universe("x", service_artifact_refs='["a"]', metadata='{ m = "f" }') | {"artifacts/x/gmail/a.json": "{}"},
     "artifacts/x: artifact.toml: service_artifact_refs is what the stored document calls it; write "
     "environment_artifacts instead"),
], ids=["environment-two-files", "environment-no-name", "environment-empty-name", "environment-service-name",
        "environment-file-artifact-id", "environment-file-and-description", "environment-file-and-a-file",
        "environment-over-a-universe", "universe-empty", "universe-root-file", "universe-two-files",
        "universe-nested-folder", "universe-double-underscore", "universe-flat-metadata", "universe-repeated-name",
        "universe-stored-names"])
def test_an_artifact_toml_that_cant_be_written_is_refused_before_any_write(bundle_dir, files, problem):
    layout(bundle_dir, _refused(files))

    assert problem in _problems(bundle_dir)
    assert not local_store().path.exists()


def test_a_universe_naming_a_store_environment_with_the_name_of_one_of_its_folders_is_refused(bundle_dir):
    with namespace_routing():
        EnvironmentArtifact.put("mail-data", environment_name="gmail", file_artifact=_store_file("mail-rows", b"{}"))
    _bundle(bundle_dir, {**_universe("world", environment_artifacts='["mail-data"]'),
                         "artifacts/world/gmail/a.json": "{}"}, "world")

    assert _problems(bundle_dir) == ("artifacts/world: environment_artifacts[0]: environment_name 'gmail' is also the "
                                     "gmail/ folder's; a universe's environments need names of their own",)


def test_a_check_reports_artifact_toml_problems_without_reading_a_store(bundle_dir, monkeypatch):
    _bundle(bundle_dir, {**_universe("world"), "artifacts/world/stray.json": "{}",
                         f"artifacts/crm/{TOML}": 'type = "environment"\n', "artifacts/crm/a.json": "{}"},
            "world", "crm")
    monkeypatch.setattr("agent_env.config.runtime.Config.get_document_store", lambda self: RefusingStore())

    with pytest.raises(BundleError) as caught:
        check_bundle(resolve_bundle(parse_bundle(bundle_dir)))

    assert sorted(caught.value.problems) == [
        "artifacts/crm: artifact.toml: environment_name is required: the environment the file seeds",
        "artifacts/world/stray.json: a universe folder holds only artifact.toml and a folder for each environment; "
        "move it into <environment_name>/, or give it an artifact folder of its own",
    ]


def test_a_universe_written_by_hand_in_the_store_isnt_touched_by_a_bundle_naming_it(bundle_dir):
    with namespace_routing():
        mail = EnvironmentArtifact.put("mail-data", environment_name="gmail", file_artifact=_store_file("m", b"{}"))
        before = get_artifact_store().get("mail-data").version
    _bundle(bundle_dir, _universe("world", environment_artifacts='["mail-data"]'), "world")

    _run(bundle_dir)

    with namespace_routing():
        assert get_artifact_store().get("mail-data").version == before == mail.version
