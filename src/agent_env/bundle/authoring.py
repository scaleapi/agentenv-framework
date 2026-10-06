"""What an entity's ``from_toml`` builds itself from: its bundle entry, and what its toml names."""

from __future__ import annotations

import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TypeVar, overload

from agent_env.artifact.artifact import Artifact
from agent_env.entity_refs import EntityKind, parse_toml_ref
from agent_env.env.env import Env

from ._fs import os_reason, relative, show, with_article
from .parse import CONFIG_FILES, LEAVINGS, NAMED_BY, Bundle, BundleEntry, BundleError

A = TypeVar("A", bound=Artifact)
E = TypeVar("E", bound=Env)

_BRING_IT_IN = "copy what it points to into the bundle, or put it in a store and refer to it by id"
_TOML_TYPES = {str: "a string", dict: "a table"}


@dataclass(frozen=True)
class AuthoringContext:
    """One bundle entry being authored. ``env`` and ``artifact`` load what a resolved ref names: an
    id, or a table ``{ env = "<id>", version = <n> }`` (``artifact =`` for an artifact), at that
    version or, without one, at its latest."""

    bundle: Bundle
    entry: BundleEntry

    @property
    def id(self) -> str:
        return self.entry.id

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def dir(self) -> Path:
        return self.entry.path

    def files(self) -> dict[str, Path]:
        return entry_files(self.bundle, self.entry)

    def file(self) -> tuple[str, Path]:
        return entry_file(self.bundle, self.entry)

    def accept(self, data: dict, /, **takes: type) -> dict:
        """The keys of ``data`` a type takes, each checked against the type given; any other key but
        ``type`` and ``id`` is refused."""
        fields, problems = self.accepted(data, **takes)
        if problems:
            self.refuse(problems)
        return fields

    def accepted(self, data: dict, /, **takes: type) -> tuple[dict, list[str]]:
        """``accept`` without refusing: the keys of ``data`` a type takes and of the type given, and the
        problems with the rest, for a type that checks more before it refuses."""
        config = CONFIG_FILES[self.entry.kind]
        problems = []
        unknown = sorted(set(data) - {"type", "id", *takes})
        if unknown:
            names = [*takes, "type", "id"]
            problems.append(f"{config}: unknown key{'s' * (len(unknown) > 1)} {', '.join(map(repr, unknown))}; "
                            f"{with_article(f'{self.entry.type} {self.entry.kind.store}')} takes "
                            f"{', '.join(names[:-1])} and {names[-1]}")
        for key, kind in takes.items():
            if key in data and not isinstance(data[key], kind):
                problems.append(f"{config}: {key} must be {_TOML_TYPES.get(kind, kind.__name__)}, not {data[key]!r}")
        fields = {key: data[key] for key, kind in takes.items() if key in data and isinstance(data[key], kind)}
        return fields, problems

    def refuse(self, problems: list[str]) -> NoReturn:
        """Raise BundleError with ``problems``, each named by the entry's folder."""
        where = relative(self.bundle.root, self.entry.path)
        raise BundleError([f"{where}: {problem}" for problem in problems])

    @overload
    def env(self, ref: str | dict[str, Any]) -> Env: ...
    @overload
    def env(self, ref: str | dict[str, Any], expect: type[E]) -> E: ...
    def env(self, ref: str | dict[str, Any], expect: type[Env] = Env) -> Env:
        return self._load(Env, EntityKind.ENV, ref, expect)

    @overload
    def artifact(self, ref: str | dict[str, Any]) -> Artifact: ...
    @overload
    def artifact(self, ref: str | dict[str, Any], expect: type[A]) -> A: ...
    def artifact(self, ref: str | dict[str, Any], expect: type[Artifact] = Artifact) -> Artifact:
        return self._load(Artifact, EntityKind.ARTIFACT, ref, expect)

    def _load(self, base: type, kind: EntityKind, ref: Any, expect: type) -> Any:
        entity_id, version = self._parse(kind, ref)
        named = next((entry for entry in self.bundle.entries
                      if entry.kind in NAMED_BY[kind] and entry.name.casefold() == entity_id.casefold()), None)
        if named is not None:
            self._fail(f"{entity_id!r} matches this bundle's {kind.value} {named.name!r} but wasn't resolved; "
                       "declare its key in toml_refs")
        loaded = base.get(entity_id, version)
        if not isinstance(loaded, expect):
            self._fail(f"{entity_id} is a {loaded.type} {kind.value}, not a {expect.__name__}")
        return loaded

    def _parse(self, kind: EntityKind, ref: Any) -> tuple[str, int | None]:
        try:
            return parse_toml_ref(kind, ref)
        except ValueError as e:
            self._fail(str(e))

    def _fail(self, message: str) -> NoReturn:
        self.refuse([message])


def entry_files(bundle: Bundle, entry: BundleEntry) -> dict[str, Path]:
    """The files in an entry's folder, keyed by POSIX path in NFC and sorted: regular files, dot files
    included, through links that stay inside the bundle, less what the OS or Python leaves behind and
    the entry's own toml. Raises BundleError listing every problem found."""
    return _Walk(bundle, entry).files()


def build_context_files(bundle: Bundle, entry: BundleEntry) -> dict[str, Path]:
    """What an image built from an entry's folder is built from: its ``entry_files`` and its own toml, which
    the Dockerfile can copy too."""
    files = entry_files(bundle, entry)
    toml = entry.path / CONFIG_FILES[entry.kind]
    if toml.is_file():
        files[toml.name] = toml
    return dict(sorted(files.items()))


def entry_file(bundle: Bundle, entry: BundleEntry) -> tuple[str, Path]:
    """The one file in an entry's folder, as ``(name, path)``. Raises BundleError otherwise."""
    files = entry_files(bundle, entry)
    if len(files) != 1:
        listed = ", ".join(repr(name) for name in files)
        raise BundleError([f"{relative(bundle.root, entry.path)}: a {entry.type} {entry.kind.store} holds one file, "
                           f"and this folder has {len(files)} ({listed}); leave out its declared type to write the "
                           "folder as a file_artifact_universe"])
    key, path = next(iter(files.items()))
    return key.rsplit("/", 1)[-1], path


class _Walk:
    """Lists an entry's folder, following a link only when its target is inside the bundle."""

    def __init__(self, bundle: Bundle, entry: BundleEntry):
        self.bundle = bundle
        self.entry = entry
        self.root = _identity(bundle.root)
        self.problems: list[str] = []
        self.found: dict[str, Path] = {}

    def files(self) -> dict[str, Path]:
        top = self.entry.path
        if not self._inside(top):
            how = "links to" if top.is_symlink() else "resolves to"
            self._problem(top, f"{how} {show(os.path.realpath(top))}, outside the bundle; {_BRING_IT_IN}")
        else:
            self._walk(top)
        if not self.found and not self.problems:
            self._problem(top, f"has no files to write ({', '.join(sorted(LEAVINGS))} don't count)")
        if self.problems:
            raise BundleError(self.problems)
        return dict(sorted(self.found.items()))

    def _walk(self, top: Path) -> None:
        config = CONFIG_FILES.get(self.entry.kind)
        # The folders holding the entry's up to the bundle root count too, so a link up to one loops.
        depth = len(top.relative_to(self.bundle.root).parts)
        pending = [(top, (), frozenset(_identity(folder) for folder in (top, *list(top.parents)[:depth])))]
        while pending:
            folder, parts, holders = pending.pop()
            try:
                with os.scandir(folder) as listing:
                    children = sorted(listing, key=lambda child: child.name)
            except OSError as e:
                self._problem(folder, os_reason(e))
                continue
            for child in children:
                if child.name in LEAVINGS or (not parts and child.name == config):
                    continue
                path = folder / child.name
                if child.is_symlink():
                    kind = self._linked(path)
                elif child.is_dir(follow_symlinks=False):
                    kind = stat.S_IFDIR
                elif child.is_file(follow_symlinks=False):
                    kind = stat.S_IFREG
                else:
                    kind = None
                    self._problem(path, "neither a regular file nor a folder")
                if kind == stat.S_IFDIR:
                    identity = _identity(path)
                    if identity in holders:
                        self._problem(path, "links back to a folder that holds it")
                    else:
                        pending.append((path, (*parts, child.name), holders | {identity}))
                elif kind == stat.S_IFREG:
                    if os.access(path, os.R_OK):
                        self._found((*parts, child.name), path)
                    else:
                        self._problem(path, "permission denied")

    def _linked(self, path: Path) -> int | None:
        """The kind a link resolves to, or None after recording why it can't be followed."""
        try:
            mode = os.stat(path).st_mode
        except OSError as e:
            self._problem(path, "a broken symlink" if isinstance(e, FileNotFoundError) else os_reason(e))
            return None
        if not self._inside(path):
            self._problem(path, f"links to {show(os.path.realpath(path))}, outside the bundle; {_BRING_IT_IN}")
            return None
        if stat.S_ISDIR(mode) or stat.S_ISREG(mode):
            return stat.S_IFMT(mode)
        self._problem(path, "links to something that is neither a regular file nor a folder")
        return None

    def _inside(self, path: Path) -> bool:
        """Whether ``path`` resolves to somewhere under the bundle root, compared by device and inode
        so a differently cased or normalized spelling of the root still counts."""
        resolved = Path(os.path.realpath(path))
        return any(_identity(candidate) == self.root for candidate in (resolved, *resolved.parents))

    def _found(self, parts: tuple[str, ...], path: Path) -> None:
        key = "/".join(unicodedata.normalize("NFC", part) for part in parts)
        if key in self.found:
            self._problem(path, f"has the same name as {relative(self.bundle.root, self.found[key])} once normalized "
                          "to NFC; rename one")
        else:
            self.found[key] = path

    def _problem(self, path: Path, message: str) -> None:
        self.problems.append(f"{relative(self.bundle.root, path)}: {message}")


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        status = os.stat(path)
    except OSError:
        return None
    return (status.st_dev, status.st_ino)


