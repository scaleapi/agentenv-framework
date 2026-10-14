"""CLI folders: an ``artifacts/<name>/`` folder whose artifact.toml says ``type = "cli"`` is written as a CliArtifact
over every file its walk lists, under an upload prefix of its own, rewritten when its files or toml change, and
refused before any write when it couldn't be installed as written."""

import json

import pytest

from agent_env.artifact import CliArtifact
from agent_env.bundle import BundleError
from agent_env.bundle.materialize import materialize
from agent_env.store.routing import namespace_routing
from tst.unit.bundle._support import layout, local_store, plan_of

ROOT = "@local/~/triage"
CLI = f"{ROOT}/hello"
TOML = 'type = "cli"\ncommand_name = "hello"\nentrypoint = "bin/hello"\n'
FILES = {
    "artifacts/hello/artifact.toml": TOML,
    "artifacts/hello/bin/hello": "#!/bin/sh\ncat \"$(dirname \"$0\")/../lib/greeting.txt\"\n",
    "artifacts/hello/lib/greeting.txt": "hello\n",
    "tasks/t.json": json.dumps([{"id": "skills", "type": "add_skills", "cli_artifact_ids": ["hello"]}]),
}


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "triage"


def _run(root):
    return materialize(plan_of(root))


def _summary(materialization):
    return {done.write.id: (done.version, done.reused, done.reasons) for done in materialization.writes}


def _problems(root):
    with pytest.raises(BundleError) as caught:
        plan_of(root)
    return caught.value.problems


def test_a_cli_folder_is_written_as_a_cli_over_its_files_under_a_prefix_of_its_own(bundle_dir):
    layout(bundle_dir, FILES)

    _run(bundle_dir)

    with namespace_routing():
        cli = CliArtifact.get(CLI)
        files = cli.get_cli_files().get_file_artifacts()
    assert (cli.command_name, cli.entrypoint, cli.cli_files_id) == ("hello", "bin/hello", f"{CLI}__files")
    assert sorted(files) == ["bin/hello", "lib/greeting.txt"]
    prefix, _, attempt = cli.cli_object_url.rstrip("/").rpartition("/")
    assert "/artifacts/cli/" in prefix and attempt.startswith("1-")


def test_a_cli_folder_is_reused_until_its_files_or_toml_change(bundle_dir):
    layout(bundle_dir, FILES)
    _run(bundle_dir)

    assert _summary(_run(bundle_dir))[CLI] == (1, True, ())
    (bundle_dir / "artifacts/hello/lib/greeting.txt").write_text("hi\n")
    assert _summary(_run(bundle_dir))[CLI][:2] == (2, False)
    (bundle_dir / "artifacts/hello/artifact.toml").write_text(TOML.replace('"hello"\n', '"greet"\n', 1))
    assert _summary(_run(bundle_dir))[CLI][:2] == (3, False)
    with namespace_routing():
        assert CliArtifact.get(CLI).command_name == "greet"


def test_a_cli_folder_its_walk_refuses_is_reported_once(bundle_dir, tmp_path):
    layout(bundle_dir, FILES)
    (tmp_path / "outside.txt").write_text("x\n")
    (bundle_dir / "artifacts/hello/bin/link").symlink_to(tmp_path / "outside.txt")

    problems = _problems(bundle_dir)

    assert [problem for problem in problems if "outside the bundle" in problem] == [
        f"artifacts/hello/bin/link: links to {(tmp_path / 'outside.txt').resolve()}, outside the bundle; copy what it "
        "points to into the bundle, or put it in a store and refer to it by id"]


@pytest.mark.parametrize("toml, extra, problem", [
    ('type = "cli"\nentrypoint = "bin/hello"\n', {},
     "artifacts/hello: artifact.toml: command_name is required"),
    ('type = "cli"\ncommand_name = "hello"\n', {},
     "artifacts/hello: artifact.toml: entrypoint is required"),
    (TOML.replace('command_name = "hello"', 'command_name = "../hello"'), {},
     "artifacts/hello: artifact.toml: command_name must be a plain name, of letters, digits, '.', '_' and '-' and "
     "not starting with '.' or '-', since the CLI is installed at /opt/cli/<command_name>; not '../hello'"),
    (TOML.replace('"bin/hello"', '"bin/helo"'), {},
     "artifacts/hello: artifact.toml: entrypoint 'bin/helo' isn't one of this folder's files"),
    (TOML, {"artifacts/hello/bin/.env": "AGENT_ENV_GATEWAY_URL=mine\n"},
     "artifacts/hello: bin/.env: installing the CLI writes the gateway URL to this file, beside its entrypoint; "
     "rename or remove the folder's own"),
    (TOML + 'env_id = "crm"\n', {},
     "artifacts/hello: artifact.toml: unknown key 'env_id'; a cli artifact takes command_name, entrypoint, type "
     "and id"),
], ids=["no command_name", "no entrypoint", "command_name a path", "no such entrypoint", "its own .env",
        "unknown key"])
def test_a_cli_folder_that_couldnt_be_installed_as_written_is_refused_before_any_write(bundle_dir, toml, extra,
                                                                                       problem):
    layout(bundle_dir, {**FILES, "artifacts/hello/artifact.toml": toml, **extra})

    assert problem in _problems(bundle_dir)
    assert not local_store().path.exists()
