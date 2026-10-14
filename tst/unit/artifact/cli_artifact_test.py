"""Writing a CliArtifact: from a folder or a file listing, its entrypoint among the files, retried safely."""

import pytest

from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse


@pytest.fixture
def cli_files(tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/hello").write_text("#!/bin/sh\necho hello\n")
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/util.sh").write_text("true\n")
    return {"bin/hello": tmp_path / "bin/hello", "lib/util.sh": tmp_path / "lib/util.sh"}


def test_a_folder_and_a_listing_write_the_same_files(local_stores, cli_files, tmp_path):
    from_folder = CliArtifact.put("hello-cli", command_name="hello", entrypoint="bin/hello", cli_dir=tmp_path)
    from_listing = CliArtifact.put("hello-cli", command_name="hello", entrypoint="bin/hello", files=cli_files)

    assert sorted(from_folder.get_cli_files().get_file_artifacts()) == ["bin/hello", "lib/util.sh"]
    assert sorted(from_listing.get_cli_files().get_file_artifacts()) == ["bin/hello", "lib/util.sh"]
    assert from_folder.cli_object_url != from_listing.cli_object_url


@pytest.mark.parametrize("sources", [{}, {"cli_dir": "x", "files": {}}], ids=["neither", "both"])
def test_a_put_takes_a_folder_or_a_listing(local_stores, sources):
    with pytest.raises(ValueError, match="exactly one of cli_dir or files"):
        CliArtifact.put("hello-cli", command_name="hello", entrypoint="bin/hello", **sources)


def test_a_folders_entrypoint_is_recorded_as_given(local_stores, cli_files, tmp_path):
    cli = CliArtifact.put("hello-cli", command_name="hello", entrypoint="./bin/hello", cli_dir=tmp_path)

    assert cli.entrypoint == "./bin/hello"


def test_a_listing_without_its_entrypoint_is_refused(local_stores, cli_files):
    with pytest.raises(ValueError, match="entrypoint 'bin/hi' isn't among the CLI's files"):
        CliArtifact.put("hello-cli", command_name="hello", entrypoint="bin/hi", files=cli_files)


def test_a_put_that_fails_after_uploading_its_files_can_be_retried(local_stores, cli_files, monkeypatch):
    real_put = FileArtifactUniverse.put

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(FileArtifactUniverse, "put", real_put)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(FileArtifactUniverse, "put", fail_once)
    with pytest.raises(RuntimeError, match="interrupted"):
        CliArtifact.put("hello-cli", command_name="hello", entrypoint="bin/hello", files=cli_files)

    assert CliArtifact.put("hello-cli", command_name="hello", entrypoint="bin/hello", files=cli_files).version == 1
