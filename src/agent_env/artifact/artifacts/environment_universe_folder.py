"""The folder an environment universe is written to and read back from.

``environment-universe get --output-dir`` writes each environment's file to ``<environment_name>/<file>`` and each
metadata file to ``metadata/<key>/<file>`` (``environment_path``, ``metadata_path``), a name or key holding ``/`` in
nested folders. ``read_folder`` is its exact inverse, which a bundle's environment_universe folder is written from,
and ``environment_id`` and ``metadata_id`` name what's written from each folder.
The format's own limits: an environment whose name starts with ``metadata/`` reads back as a metadata key, names are
read in NFC, and names that differ only in case are one folder on a case-insensitive disk.
"""

from __future__ import annotations

import functools
import hashlib
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.store.ids import derive_id, validate_local_id

# What a universe's metadata is called: the folder its files go in, and the segment of the ids they're written under.
METADATA = "metadata"


def environment_id(universe_id: str, environment_name: str) -> str:
    """The id an environment is written under when it's written with its universe: ``<universe>__<name>``."""
    return derive_id(universe_id, environment_name)


def metadata_id(universe_id: str, key: str) -> str:
    """The id a metadata file is written under when it's written with its universe: ``<universe>__metadata__<key>``."""
    return derive_id(derive_id(universe_id, METADATA), key)


def environment_path(folder: Path, environment_name: str, filename: str) -> Path:
    """Where the file of a universe's environment goes in the universe's folder."""
    return folder / environment_name / filename


def metadata_path(folder: Path, key: str, filename: str) -> Path:
    """Where a universe's metadata file goes in the universe's folder."""
    return folder / METADATA / key / filename


@dataclass(frozen=True)
class FolderEntry:
    """One file of a universe's folder, and the ids it's written under."""

    name: str  # its environment's name, or its metadata key
    filename: str
    path: Path
    file_id: str
    environment_id: str | None  # None for a metadata file


@dataclass(frozen=True)
class UniverseFolder:
    """A universe's folder, read: its environments, its metadata files, and every file it holds."""

    environments: tuple[FolderEntry, ...]
    metadata: tuple[FolderEntry, ...]
    files: Mapping[str, Path]


class UniverseFolderError(ValueError):
    """Why a folder can't be read as a universe: ``problems``, each ``<path in the folder>: <why>``."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def read_folder(universe_id: str, files: Mapping[str, Path],
                folders: Iterable[tuple[str, ...]]) -> UniverseFolder:
    """The environments and metadata files a universe's folder holds, from its files (``{path in the folder: path}``)
    and its folders (the names on each path to one): each folder directly holding one file is an environment named by
    its path from the universe's folder, or, under ``metadata/``, a metadata key named by its path from there. An
    environment is written as ``<universe>__<name>`` over ``<universe>__<name>__file`` and a metadata file as
    ``<universe>__metadata__<key>``, with a hash of the name in its place where an id can't hold it. Raises
    UniverseFolderError listing every problem."""
    folders = set(folders)
    # A metadata/ folder spelled in another case is a typo, as for every reserved name in a bundle.
    near = sorted(parts[0] for parts in folders
                  if len(parts) == 1 and parts[0] != METADATA and _fold(parts[0]) == _fold(METADATA))
    problems = [f"{name}: rename to {METADATA}; names are case-sensitive, and an environment named {name} goes in an "
                "artifact folder of its own, named in environment_artifacts" for name in near]
    # Every folder counts, so one left without its file is refused rather than dropped.
    held: dict[tuple[str, ...], list[str]] = {parts: [] for parts in folders
                                              if parts[0] not in near and parts != (METADATA,)}
    for key in files:
        parts = tuple(key.split("/"))
        if parts[0] in near:
            continue
        if len(parts) == 1:
            problems.append(f"{key}: a universe folder holds only artifact.toml and a folder for each environment; "
                            "move it into <environment_name>/, or give it an artifact folder of its own")
        elif parts[:-1] == (METADATA,):
            problems.append(f"{key}: {METADATA}/ holds a folder for each key, with that key's one file in it "
                            f"({METADATA}/<key>/<file>); an environment named {METADATA} goes in an artifact folder "
                            "of its own, named in environment_artifacts")
        else:
            held.setdefault(parts[:-1], []).append(key)
    environments, metadata_files = [], []
    # A folder holding only folders is part of their names.
    holders = {parts[:depth] for parts in held for depth in range(1, len(parts))}
    for parts, keys in sorted(held.items(), key=lambda item: "/".join(item[0])):
        folder = "/".join(parts)
        if not keys and parts in holders:
            continue
        if len(keys) != 1:
            listed = f" ({', '.join(repr(key.rsplit('/', 1)[-1]) for key in keys)})" if keys else ""
            problems.append(f"{folder}: holds one file, and has {len(keys)}{listed}")
            continue
        metadata = parts[0] == METADATA
        name = "/".join(parts[1:] if metadata else parts)
        if metadata:
            env_id, file_id = None, _child_id(functools.partial(metadata_id, universe_id), name, environment=False)
        else:
            env_id = _child_id(functools.partial(environment_id, universe_id), name, environment=True)
            file_id = EnvironmentArtifact.derived_file_id(env_id)
        try:
            for id in (env_id, file_id):
                if id is not None:
                    validate_local_id(id)
        except ValueError as e:
            problems.append(f"{folder}: can't name the artifacts written from it ({e}); give the universe a shorter "
                            "name")
            continue
        entry = FolderEntry(name, keys[0].rsplit("/", 1)[-1], files[keys[0]], file_id, env_id)
        (metadata_files if metadata else environments).append(entry)
    if problems:
        raise UniverseFolderError(problems)
    return UniverseFolder(tuple(environments), tuple(metadata_files), files)


def _child_id(derive: Callable[[str], str], name: str, *, environment: bool) -> str:
    """The id a universe's environment or metadata file named ``name`` is written under, as ``derive``, the
    universe's own naming, derives it from a name: from ``name``, or, for a name holding ``__``, which could clash
    with another's ids, or one an id can't hold (with an environment's file id, ``<id>__file``), from
    ``sha256__<16 hex of its hash>``, which holds ``__``, so no name's own id can be."""
    if "__" not in name:
        id = derive(name)
        try:
            for derived in (id, EnvironmentArtifact.derived_file_id(id)) if environment else (id,):
                validate_local_id(derived)
            return id
        except ValueError:
            pass
    return derive(f"sha256__{hashlib.sha256(name.encode('utf-8')).hexdigest()[:16]}")


def _fold(name: str) -> str:
    """The caseless form names are compared in: casefold output isn't always NFC, so normalize both sides."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())
