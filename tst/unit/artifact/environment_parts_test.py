"""An environment artifact of the parts an environment's data/get returned: written and read back in order, stored with
no wrapped file and nothing of the one-file shape, which stores as before; refused where one file is taken, and by an
artifact folder."""

import pytest
from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI

from agent_env.artifact.artifacts.environment import EnvironmentArtifact, EnvironmentPart
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.entity.export import Unwritable


def _file(tmp_path, name, data):
    (tmp_path / name).write_bytes(data)
    return FileArtifact.put(f"file-{name}", description=name, file_path=str(tmp_path / name))


def test_an_environment_of_parts_is_written_and_read_back_in_order_with_nothing_of_one_file(local_stores, tmp_path):
    notes, state = _file(tmp_path, "notes.txt", b"hello"), _file(tmp_path, "state.json", b'{"rows": 1}')
    parts = [EnvironmentPart(kind="file", file_artifact_ref=notes.as_ref(), name="notes.txt", mime_type="text/plain",
                             metadata={DATA_OBJECTS_EXTENSION_URI: {"path": "Documents/notes.txt"}}),
             EnvironmentPart(kind="data", file_artifact_ref=state.as_ref())]

    written = EnvironmentArtifact.put("desk", environment_name="desktop", parts=parts)

    read = EnvironmentArtifact.get("desk", version=written.version)
    assert [(part.kind, part.path, file.load()) for part, file in read.get_parts()] == [
        ("file", "Documents/notes.txt", b"hello"), ("data", None, b'{"rows": 1}')]
    assert read.parts == parts and read.file_artifact_ref is None
    assert "file_artifact_id" not in read.model_dump(by_alias=True)
    with pytest.raises(ValueError, match="holds the 2 parts its environment's data/get returned, not one file"):
        read.get_file_artifact()


def test_one_file_is_stored_as_before_and_read_as_one_file_part(local_stores, tmp_path):
    state = _file(tmp_path, "state.json", b"{}")

    written = EnvironmentArtifact.put("slack", environment_name="slack", file_artifact=state)

    assert "parts" not in written.model_dump(by_alias=True) and written.file_artifact_ref == state.as_ref()
    assert [(part.kind, part.name, file.id) for part, file in EnvironmentArtifact.get("slack").get_parts()] == [
        ("file", "state.json", state.id)]


@pytest.mark.parametrize("given", [{}, {"file_artifact": "both", "parts": []}], ids=["neither", "both"])
def test_an_environment_wraps_one_file_or_holds_parts_but_not_both_or_neither(local_stores, tmp_path, given):
    if given:
        given = {"file_artifact": _file(tmp_path, "state.json", b"{}"), "parts": []}

    with pytest.raises(ValueError, match="wraps a file_artifact or holds parts, one of the two"):
        EnvironmentArtifact.put("x", environment_name="x", **given)


def test_an_artifact_folder_holds_no_environment_of_parts(local_stores, tmp_path):
    desk = EnvironmentArtifact.put("desk", environment_name="desktop", parts=[])
    universe = EnvironmentUniverseArtifact.put("universe", environment_artifacts=[desk])

    with pytest.raises(Unwritable, match="it holds the parts its data/get returned"):
        desk.to_toml(None)
    with pytest.raises(Unwritable, match="its environment 'desktop' holds the parts its data/get returned"):
        universe.to_toml(None)
