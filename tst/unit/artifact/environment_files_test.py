"""An environment artifact holds its environment's files by path. One file whose path is its filename is stored as the one
wrapped file every environment artifact had, document for document; several, or one inside a folder, as
file_artifact_refs. get_file_artifacts reads either shape, get_file_artifact refuses a collection, and a path that isn't
plain file names joined by '/' is refused. One file kept at a path other than its filename can't be written to an
artifact folder, whose one file reads back kept at its filename."""

import pytest

from agent_env.artifact import EnvironmentArtifact, FileArtifact
from agent_env.entity.export import FolderExport, Unwritable


def _file(id: str, filename: str) -> FileArtifact:
    return FileArtifact.put_bytes(id, description=id, filename=filename, content=id.encode(), content_type="text/plain")


def _document(artifact: EnvironmentArtifact) -> dict:
    return {key: value for key, value in artifact.model_dump(mode="json", by_alias=True).items() if key != "id"}


def test_one_file_at_its_filename_is_stored_as_the_one_wrapped_file(local_stores):
    file = _file("seed", "seed.json")

    by_path = EnvironmentArtifact.put("crm", environment_name="crm", file_artifacts={"seed.json": file})
    wrapped = EnvironmentArtifact.put("crm-wrapped", environment_name="crm", file_artifact=file)

    assert _document(by_path) == _document(wrapped)
    assert "file_artifact_refs" not in _document(by_path)
    assert {path: f.id for path, f in by_path.get_file_artifacts().items()} == {"seed.json": "seed"}


@pytest.mark.parametrize("paths", [["a.json", "b.json"], ["inbox/a.json"]], ids=["several", "one-in-a-folder"])
def test_files_are_kept_by_path_when_one_wrapped_file_cant_say_where_each_goes(local_stores, paths):
    files = {path: _file(f"f{index}", path.rsplit("/", 1)[-1]) for index, path in enumerate(paths)}

    stored = EnvironmentArtifact.get(EnvironmentArtifact.put("crm", environment_name="crm", file_artifacts=files).id)

    assert _document(stored) == {"type": "environment", "version": 1, "metadata": {}, "environment_name": "crm",
                                 "service_name": "crm", "file_artifact_refs": {
                                     path: {"id": file.id, "version": 1} for path, file in files.items()}}
    assert {path: file.load() for path, file in stored.get_file_artifacts().items()} == {
        path: file.load() for path, file in files.items()}
    with pytest.raises(ValueError, match=f"holds {len(paths)} files by path, not one; read them with get_file_artifacts"):
        stored.get_file_artifact()


def test_a_legacy_unpinned_document_reads_as_its_one_file(local_stores):
    _file("seed", "seed.json")

    legacy = EnvironmentArtifact(id="crm", version=1, environment_name="crm", file_artifact_id="seed")

    assert {path: f.id for path, f in legacy.get_file_artifacts().items()} == {"seed.json": "seed"}


@pytest.mark.parametrize("path", ["/a.json", "../a.json", "a//b.json", "a/./b.json", "a\\b.json", ""])
def test_a_path_that_isnt_plain_file_names_is_refused(local_stores, path):
    files = {path: _file("f", "a.json"), "b.json": _file("g", "b.json")}

    with pytest.raises(ValueError, match="must be plain file names joined by '/'"):
        EnvironmentArtifact.put("crm", environment_name="crm", file_artifacts=files)


@pytest.mark.parametrize("given, refused", [
    (lambda file: {"file_artifacts": {}}, "file_artifacts can't be empty"),
    (lambda file: {"file_artifact": file, "file_artifacts": {"seed.json": file}}, "holds a file_artifact or file_artifacts"),
], ids=["no-files", "a-file-and-files"])
def test_an_environment_holds_a_file_or_files(local_stores, given, refused):
    with pytest.raises(ValueError, match=refused):
        EnvironmentArtifact.put("crm", environment_name="crm", **given(_file("seed", "seed.json")))


@pytest.mark.parametrize("path", ["inbox/a.json", "notes.json"], ids=["inside-a-folder", "under-another-name"])
def test_one_file_kept_at_a_path_other_than_its_filename_cant_be_written_to_an_artifact_folder(local_stores, tmp_path, path):
    environment = EnvironmentArtifact.put("crm", environment_name="crm", file_artifacts={path: _file("f", "a.json")})

    with pytest.raises(Unwritable, match=f"its one file is kept at '{path}', not at its filename, and a folder's one "
                                         "file reads back kept at its filename"):
        environment.to_toml(FolderExport(tmp_path))
