"""The universe folder format on its own: ``read_folder`` reads back what ``environment_path`` and ``metadata_path``
lay out, and names its problems by their paths in the folder."""

import hashlib
from pathlib import Path, PurePosixPath

import pytest

from agent_env.artifact.artifacts.environment_universe_folder import (
    UniverseFolderError,
    environment_path,
    metadata_path,
    read_folder,
)

UNIVERSE = "@local/~/b/u"


def _listing(paths):
    """A folder's files keyed by their paths in it, and its folders, as a walk lists them."""
    files = {path.as_posix(): Path("/nowhere") / path for path in paths}
    folders = {PurePosixPath(key).parents[i].parts for key in files for i in range(len(PurePosixPath(key).parts) - 1)}
    return files, folders


def test_reading_a_folder_gives_back_every_name_and_key_its_paths_were_written_for():
    environments = {"gmail": "a.json", "drive/shared": "b.json", "g__mail": "c.json"}
    keys = {"manifest": "m.json", "scenarios": "s.yaml", "scenarios/x.yaml": "x.yaml", "q&a?": "q.json"}
    files, folders = _listing([environment_path(PurePosixPath(), name, filename)
                               for name, filename in environments.items()]
                              + [metadata_path(PurePosixPath(), key, filename) for key, filename in keys.items()])

    read = read_folder(UNIVERSE, files, folders)

    assert {entry.name: entry.filename for entry in read.environments} == environments
    assert {entry.name: entry.filename for entry in read.metadata} == keys
    digest = hashlib.sha256(b"g__mail").hexdigest()[:16]
    assert {entry.name: entry.environment_id for entry in read.environments}["g__mail"] == f"{UNIVERSE}__sha256__{digest}"
    assert {entry.name: entry.file_id for entry in read.metadata}["scenarios/x.yaml"] == (
        f"{UNIVERSE}__metadata__scenarios/x.yaml")


def test_its_problems_name_paths_in_the_folder():
    files, folders = _listing([PurePosixPath("stray.json"), PurePosixPath("gmail/a.json"),
                               PurePosixPath("gmail/b.json")])

    with pytest.raises(UniverseFolderError) as caught:
        read_folder(UNIVERSE, files, folders | {("slack",)})

    assert caught.value.problems == [
        "stray.json: a universe folder holds only artifact.toml and a folder for each environment; move it into "
        "<environment_name>/, or give it an artifact folder of its own",
        "gmail: holds one file, and has 2 ('a.json', 'b.json')",
        "slack: holds one file, and has 0",
    ]
