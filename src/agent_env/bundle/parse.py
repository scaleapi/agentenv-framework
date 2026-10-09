"""Read a bundle folder into the entities it defines, without touching a store.

A bundle is a folder holding at least one kind directory (``envs/``, ``agents/``, ``artifacts/``,
``skills/``, ``tasks/``, ``evals/``), where ``<kind>/<name>/`` is an entity of that kind named
``<name>``; a task is ``tasks/<name>.json`` and an eval ``evals/<name>.toml``. Parsing reads the
directory listings and the small config files only. References, registered types and file
contents are checked by the steps that use them.
"""

from __future__ import annotations

import json
import os
import stat
import tomllib
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NoReturn

from agent_env.bundle._fs import Unreadable, fold, home, on_disk, os_reason, read_regular, show
from agent_env.entity_refs import EntityKind
from agent_env.store.ids import LOCAL_PREFIX, MAX_AUTHORED_LOCAL_ID_BYTES, validate_local_id
from agent_env.utils.paths import LEAVINGS

BUNDLE_TOML = "bundle.toml"
DOCKERFILE = "Dockerfile"
SKILL_MD = "SKILL.md"

_DIR, _FILE = "dir", "file"


class BundleKind(Enum):
    """The six kinds; the value is the kind's directory. A plain Enum, so a kind never equals an
    ``EntityKind`` or a string."""

    ENV = "envs"
    AGENT = "agents"
    ARTIFACT = "artifacts"
    SKILL = "skills"
    TASK = "tasks"
    EVAL = "evals"

    @property
    def store(self) -> str:
        """The store the kind is written to; artifacts and skills share one."""
        return "artifact" if self is BundleKind.SKILL else self.value[:-1]


# The kinds whose entries a reference of each entity kind names; artifacts and skills share one store.
NAMED_BY = {
    EntityKind.ENV: (BundleKind.ENV,),
    EntityKind.AGENT: (BundleKind.AGENT,),
    EntityKind.ARTIFACT: (BundleKind.ARTIFACT, BundleKind.SKILL),
    EntityKind.TASK: (BundleKind.TASK,),
}
_ORDER = {kind: index for index, kind in enumerate(BundleKind)}
CONFIG_FILES = {BundleKind.ENV: "env.toml", BundleKind.AGENT: "agent.toml", BundleKind.ARTIFACT: "artifact.toml"}
_DEFAULT_TYPE = {BundleKind.ENV: "mcp_server", BundleKind.AGENT: "a2a_agent"}
_EVAL_KEYS = ("tasks", "type", "id")
_FILE_SUFFIX = {BundleKind.TASK: ".json", BundleKind.EVAL: ".toml"}


@dataclass(frozen=True)
class BundleEntry:
    kind: BundleKind
    name: str  # NFC of the on-disk name: the directory, or the task or eval file's stem
    id: str  # the declared id, else ``<Bundle.id_root>/<name>``
    path: Path  # as the filesystem spells it (possibly NFD), for I/O
    type: str  # declared, else the kind's default; for artifacts, inferred from the directory
    # The toml as written ({} when absent), a task's steps, None for skills.
    config: Any = field(compare=False, repr=False)


@dataclass(frozen=True)
class Bundle:
    root: Path
    name: str
    id_root: str
    description: str | None  # README.md's first line of text
    entries: tuple[BundleEntry, ...]
    ignored: tuple[Path, ...]  # root-relative, as on disk: entries that are neither kinds nor entities


class BundleError(ValueError):
    """A folder that isn't a valid bundle. ``problems`` holds every problem found, each naming its path."""

    def __init__(self, problems: Iterable[str]):
        self.problems = tuple(sorted(problems))
        super().__init__("\n".join(self.problems))

    @property
    def summary(self) -> str:
        """The first problem, and how many more there are."""
        more = len(self.problems) - 1
        return self.problems[0] + (f" (and {more} more)" if more else "")


def is_ignored(name: str) -> bool:
    """Dot entries and ``__pycache__`` (pip compiles the .py files of an installed bundle) are never
    entities."""
    return name.startswith(".") or name == "__pycache__"


def parse_bundle(root: Path, *, id_root: str | None = None, name: str | None = None) -> Bundle:
    """Parse the bundle at ``root``. An entry-point bundle passes its ``@local/<dist>/<name>``
    root and name; otherwise both come from the folder's canonical path. Raises BundleError."""
    return _Parser(_canonical(Path(root)), id_root, name).parse()


class _Skip(Exception):
    """The entity has a problem, already recorded; skip it."""


class _Parser:
    def __init__(self, root: Path, id_root: str | None, name: str | None):
        self.root = root
        self.root_text = str(root)
        self.name = name or unicodedata.normalize("NFC", root.name)
        self.problems: list[str] = []
        self.ignored: list[Path] = []
        self.entry_point = id_root is not None
        self.root_ok = True
        if id_root is None:
            self.id_root = self._derive_id_root()
            return
        self.id_root = id_root
        try:
            self._check_declared(id_root, root, what="the bundle's id root")
            if _size(id_root) > MAX_AUTHORED_LOCAL_ID_BYTES - 2:
                self._skip(root, "the bundle's id root leaves no room for a name; shorten it")
        except _Skip:
            self.root_ok = False

    def parse(self) -> Bundle:
        kinds, description = self._scan_root()
        entries = [entry for kind, path in kinds.items() for entry in self._scan_kind(kind, path)]
        self._check_unique(entries)
        if self.problems:
            raise BundleError(self.problems)
        entries.sort(key=lambda entry: (_ORDER[entry.kind], entry.name))
        return Bundle(
            root=self.root,
            name=self.name,
            id_root=self.id_root,
            description=description,
            entries=tuple(entries),
            ignored=tuple(sorted(self.ignored)),
        )

    def _derive_id_root(self) -> str:
        user_home = home()
        home_relative = user_home is not None and self.root.is_relative_to(user_home)
        if home_relative:
            parts = ["~", *self.root.relative_to(user_home).parts]
            above = user_home
        else:
            parts = list(self.root.parts[1:])
            above = Path(self.root.anchor)
            if parts[0] == "~":
                self._problem(above / "~", "a folder named '~' at the filesystem root would read as home; "
                              "move the bundle")
                self.root_ok = False
        for part in parts[1:] if home_relative else parts:
            above = above / part
            reason = _refused(part)
            if reason:
                self._problem(above, f"move the bundle, or rename the folder: {reason}")
                self.root_ok = False
        return LOCAL_PREFIX + "/".join(unicodedata.normalize("NFC", part) for part in parts)

    # Walking the tree

    def _scan_root(self) -> tuple[dict[BundleKind, Path], str | None]:
        try:
            listing = [entry for entry in self._list(self.root) if not is_ignored(entry.name)]
        except _Skip:
            raise BundleError(self.problems) from None
        names = [entry.name for entry in listing]
        by_dir = {kind.value: kind for kind in BundleKind}
        misspelled = {near for kind_dir in by_dir for near in self._near_miss(self.root, names, kind_dir)}
        if not misspelled and not by_dir.keys() & set(names):
            raise BundleError([f"{show(self.root)}: not a bundle: it has none of "
                               + ", ".join(f"{kind.value}/" for kind in BundleKind)])
        kinds: dict[BundleKind, Path] = {}
        readmes = []
        for entry in listing:
            if entry.name in misspelled:
                continue
            if fold(entry.name) == fold(BUNDLE_TOML):
                self._problem(entry.path, "reserved: a bundle takes its name from the folder and its "
                              "description from README.md")
            elif entry.name in by_dir:
                node = self._node(Path(entry.path), entry)
                if node == _DIR:
                    kinds[by_dir[entry.name]] = Path(entry.path)
                elif node == _FILE:
                    self._problem(entry.path, "a kind directory must be a folder")
            elif fold(entry.name) == "readme.md":
                readmes.append(entry)
            else:
                self.ignored.append(Path(entry.name))
        readme = next((entry for entry in readmes if entry.name == "README.md"), readmes[0] if readmes else None)
        self.ignored.extend(Path(entry.name) for entry in readmes if entry is not readme)
        return kinds, self._description(Path(readme.path)) if readme else None

    def _scan_kind(self, kind: BundleKind, directory: Path) -> list[BundleEntry]:
        try:
            listing = self._list(directory)
        except _Skip:
            return []
        suffix = _FILE_SUFFIX.get(kind)
        names = {listed.name for listed in listing}
        candidates: dict[str, list[tuple[str, Path]]] = defaultdict(list)
        for listed in listing:
            if is_ignored(listed.name):
                continue
            path = Path(listed.path)
            node = self._node(path, listed)
            if node is None:
                continue
            if suffix is None and node == _FILE:
                self.ignored.append(Path(kind.value, listed.name))
            elif suffix is not None and node == _DIR:
                self._problem(path, f"{kind.value[:-1]} folders are not supported yet; "
                              f"write {kind.value}/<name>{suffix}")
            elif suffix is not None and not listed.name.endswith(suffix):
                lowered = listed.name[: -len(suffix)] + suffix
                if fold(listed.name) != fold(lowered):
                    self.ignored.append(Path(kind.value, listed.name))
                elif lowered in names:
                    self._collision([directory / lowered, path])
                else:
                    self._problem(path, f"rename to {show(lowered)}; names are case-sensitive")
            else:
                name = unicodedata.normalize("NFC", listed.name.removesuffix(suffix) if suffix else listed.name)
                candidates[fold(name)].append((name, path))
        entries = []
        for group in candidates.values():
            if len(group) > 1:
                self._collision(path for _, path in group)
                continue
            try:
                entries.append(self._entry(kind, *group[0]))
            except _Skip:
                pass
        return entries

    # One entity per kind

    def _entry(self, kind: BundleKind, name: str, path: Path) -> BundleEntry:
        if kind in (BundleKind.ARTIFACT, BundleKind.SKILL) and "__" in name:
            self._skip(path, f'"__" is reserved for derived ids in artifact and skill names; rename the '
                       f"folder (e.g. {show(name.replace('__', '-'))})")
        if kind in (BundleKind.ENV, BundleKind.AGENT):
            return self._build_context(kind, name, path)
        read = {BundleKind.TASK: self._task, BundleKind.EVAL: self._eval, BundleKind.SKILL: self._skill}
        return read.get(kind, self._artifact)(name, path)

    def _task(self, name: str, path: Path) -> BundleEntry:
        steps = self._read_json(path)
        if not (isinstance(steps, list) and all(isinstance(step, dict) for step in steps)):
            self._skip(path, "a task file must be a JSON array of step objects")
        return self._make(BundleKind.TASK, name, path, "task", steps)

    def _eval(self, name: str, path: Path) -> BundleEntry:
        config = self._read_toml(path)
        problems = []
        unknown = sorted(set(config) - set(_EVAL_KEYS))
        if unknown:
            problems.append(f"unknown key{'s' * (len(unknown) > 1)} {', '.join(map(repr, unknown))}; an eval takes "
                            f"{', '.join(_EVAL_KEYS[:-1])} and {_EVAL_KEYS[-1]}")
        declared = config.get("type", "eval")
        if not (isinstance(declared, str) and declared):
            problems.append("type must be a non-empty string")
        tasks = config.get("tasks")
        if tasks is None or tasks == []:
            problems.append("an eval needs at least one task in tasks = [...]")
        elif not isinstance(tasks, list):
            problems.append(f"tasks must be a list, not {tasks!r}")
        for problem in problems:
            self._problem(path, problem)
        if problems:
            raise _Skip
        return self._make(BundleKind.EVAL, name, path, declared, config, config.get("id"), path)

    def _skill(self, name: str, path: Path) -> BundleEntry:
        children = self._children(path)
        if self._near_miss(path, children, SKILL_MD):
            raise _Skip
        if SKILL_MD not in children:
            self._skip(path, f"a skill needs {SKILL_MD}")
        self._marker(path / SKILL_MD, children[SKILL_MD])
        return self._make(BundleKind.SKILL, name, path, "skill", None)

    def _artifact(self, name: str, path: Path) -> BundleEntry:
        children = self._children(path)
        config, config_path, declared = self._config(BundleKind.ARTIFACT, path, children)
        if declared == "skill" or any(fold(child) == fold(SKILL_MD) for child in children):
            self._skip(path, "this looks like a skill: skills live in skills/<name>/, not artifacts/")
        # A lone Dockerfile spelled another way decides nothing once a type is declared: then it's content.
        if (declared is None or DOCKERFILE in children) and self._near_miss(path, children, DOCKERFILE):
            raise _Skip
        entity_type = declared or self._inferred_type(path, children)
        return self._make(BundleKind.ARTIFACT, name, path, entity_type, config, config.get("id"), config_path)

    def _build_context(self, kind: BundleKind, name: str, path: Path) -> BundleEntry:
        """An env or agent: a Docker build context, so only its Dockerfile and toml are checked."""
        names = {entry.name for entry in self._list(path)}
        config, config_path, declared = self._config(kind, path, names)
        if DOCKERFILE in names or config_path.name not in names:
            if self._near_miss(path, names, DOCKERFILE):
                raise _Skip
        if DOCKERFILE in names:
            self._marker(path / DOCKERFILE)
        elif config_path.name not in names:
            self._skip(path, f"needs a {DOCKERFILE} or an {config_path.name}")
        entity_type = declared or _DEFAULT_TYPE[kind]
        return self._make(kind, name, path, entity_type, config, config.get("id"), config_path)

    def _inferred_type(self, path: Path, children: dict[str, str]) -> str:
        if DOCKERFILE in children:
            self._marker(path / DOCKERFILE, children[DOCKERFILE])
            return "docker_image"
        content = [child for child in children if child != CONFIG_FILES[BundleKind.ARTIFACT]]
        if not content:
            self._skip(path, f"an artifact folder has nothing in it ({', '.join(sorted(LEAVINGS))} don't count)")
        return "file" if len(content) == 1 and children[content[0]] == _FILE else "file_artifact_universe"

    # Parts of an entity

    def _children(self, path: Path) -> dict[str, str]:
        """The folder's entries less what the OS or Python leaves, each checked to be a file or a folder:
        agent-env uploads these files itself (an env or agent folder is left to Docker's rules instead)."""
        children = {entry.name: self._node(Path(entry.path), entry) for entry in self._list(path)
                    if entry.name not in LEAVINGS}
        if None in children.values():
            raise _Skip
        return children

    def _config(self, kind: BundleKind, path: Path, names: Iterable[str]) -> tuple[dict, Path, str | None]:
        """The entity's toml ({} when absent), where it lives, and the type it declares."""
        config_path = path / CONFIG_FILES[kind]
        if self._near_miss(path, names, config_path.name):
            raise _Skip
        if config_path.name not in names:
            return {}, config_path, None
        config = self._read_toml(config_path)
        declared = config.get("type")
        if declared is not None and not (isinstance(declared, str) and declared):
            self._skip(config_path, "type must be a non-empty string")
        return config, config_path, declared

    def _marker(self, path: Path, node: str | None = None) -> None:
        """Skip the entity unless the file that marks it (a Dockerfile, SKILL.md) is a file."""
        node = node or self._node(path)
        if node == _DIR:
            self._problem(path, "must be a file, not a folder")
        if node != _FILE:
            raise _Skip

    def _make(
        self,
        kind: BundleKind,
        name: str,
        path: Path,
        entity_type: str,
        config: Any,
        declared: Any = None,
        declared_in: Path | None = None,
    ) -> BundleEntry:
        entity_id = self._entity_id(kind, name, path, declared, declared_in)
        return BundleEntry(kind=kind, name=name, id=entity_id, path=path, type=entity_type, config=config)

    def _entity_id(self, kind: BundleKind, name: str, path: Path, declared: Any, declared_in: Path | None) -> str:
        if declared is not None:
            self._check_declared(declared, declared_in or path, artifact=kind.store == "artifact")
            return declared
        reason = _refused(name)
        if reason:
            self._skip(path, f"rename it: {reason}")
        # The root and the name are each validated, so only the length is left to check.
        entity_id = f"{self.id_root}/{name}"
        size = _size(entity_id)
        if self.root_ok and size > MAX_AUTHORED_LOCAL_ID_BYTES:
            remedy = "move the bundle to a shorter path, or shorten the name"
            if self.entry_point:
                remedy = "shorten the name"
            self._skip(path, f"id is {size} bytes, over the {MAX_AUTHORED_LOCAL_ID_BYTES}-byte limit; {remedy}")
        return entity_id

    def _check_declared(self, entity_id: Any, where: Path, *, what: str = "id", artifact: bool = False) -> None:
        if not isinstance(entity_id, str) or not entity_id.startswith(LOCAL_PREFIX):
            self._skip(where, f"{what} {_brief(entity_id)}: ids declared in a bundle must start with "
                       f"{LOCAL_PREFIX!r}, since a bare id names an entity in a shared store")
        size = _size(entity_id)
        if size > MAX_AUTHORED_LOCAL_ID_BYTES:
            limit = MAX_AUTHORED_LOCAL_ID_BYTES
            self._skip(where, f"{what} is {size} bytes, over the {limit}-byte limit; shorten it")
        try:
            validate_local_id(entity_id)
        except ValueError as e:
            self._skip(where, f"{what}: {e}")
        if entity_id != unicodedata.normalize("NFC", entity_id):
            self._skip(where, f"{what} {_brief(entity_id)} must be NFC-normalized")
        if artifact and "__" in entity_id.rsplit("/", 1)[1]:
            self._skip(where, f'{what} {_brief(entity_id)}: "__" is reserved for derived artifact ids')

    # Across entities

    def _check_unique(self, entries: list[BundleEntry]) -> None:
        by_name: dict[str, list[BundleEntry]] = defaultdict(list)
        by_id: dict[tuple[str, str], list[BundleEntry]] = defaultdict(list)
        for entry in entries:
            by_id[(entry.kind.store, entry.id)].append(entry)
            if entry.kind.store == "artifact":
                by_name[fold(entry.name)].append(entry)
        # A bare artifact ref looks in artifacts/ and skills/ alike, so a name may live in only one.
        clashing: set[BundleEntry] = set()
        for holders in by_name.values():
            if len(holders) > 1:
                clashing.update(holders)
                self.problems.append(f"{self._paths(h.path for h in holders)} share a name; artifacts and "
                                     "skills share one store, so rename one")
        for (_, entity_id), holders in by_id.items():
            if len(holders) > 1 and not clashing.issuperset(holders):
                self.problems.append(f"{self._paths(h.path for h in holders)} have the same id {_brief(entity_id)}")

    # Files

    def _description(self, readme: Path) -> str | None:
        """README's first line of text, skipping front matter, badges and HTML. The description is
        optional, so an unreadable README has none."""
        try:
            lines = read_regular(readme).decode("utf-8-sig", errors="replace").splitlines()
        except Unreadable:
            return None
        if lines and lines[0].strip() == "---":
            closing = next((index for index, line in enumerate(lines[1:], 1) if line.strip() == "---"), 0)
            lines = lines[closing + 1:]
        for line in lines:
            text = line.strip().lstrip("#").strip()
            if text and not text.startswith(("<", "[![", "![")):
                return show(text)
        return None

    def _read_toml(self, path: Path) -> dict:
        data = self._read(path)
        try:
            return tomllib.loads(data.decode("utf-8-sig"))
        except RecursionError:
            self._skip(path, "not valid TOML: nested too deeply")
        except ValueError as e:
            self._skip(path, f"not valid TOML: {e}")

    def _read_json(self, path: Path) -> Any:
        data = self._read(path)
        try:
            return json.loads(data, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
        except RecursionError:
            self._skip(path, "not valid JSON: nested too deeply")
        except ValueError as e:
            self._skip(path, f"not valid JSON: {e}")

    def _read(self, path: Path) -> bytes:
        try:
            return read_regular(path)
        except Unreadable as e:
            self._skip(path, str(e))

    # Listings and problems

    def _list(self, directory: Path) -> list[os.DirEntry]:
        try:
            with os.scandir(directory) as listing:
                return sorted(listing, key=lambda entry: entry.name)
        except OSError as e:
            self._skip(directory, os_reason(e))

    def _node(self, path: Path, listed: os.DirEntry | None = None) -> str | None:
        """``_DIR`` or ``_FILE``, following links; anything else is recorded as a problem (and None
        returned, so a caller can go on to the next entry). A listed entry that isn't a link is
        classified from the listing, without a stat."""
        try:
            if listed is not None and not listed.is_symlink():
                is_dir, is_file = listed.is_dir(follow_symlinks=False), listed.is_file(follow_symlinks=False)
            else:
                mode = os.stat(path).st_mode
                is_dir, is_file = stat.S_ISDIR(mode), stat.S_ISREG(mode)
        except OSError as e:
            self._problem(path, "a broken symlink" if isinstance(e, FileNotFoundError) else os_reason(e))
            return None
        if is_dir:
            return _DIR
        if is_file:
            return _FILE
        self._problem(path, "neither a regular file nor a folder")
        return None

    def _near_miss(self, parent: Path, names: Iterable[str], spelling: str) -> list[str]:
        """The entries of ``names`` that are ``spelling`` in another case, each recorded as a problem:
        alone, a variant is a typo; beside the exact spelling, the two collide on a case-insensitive
        disk (a clone on macOS keeps only one), so the bundle would mean something else there."""
        names = list(names)
        near = sorted(name for name in names if name != spelling and fold(name) == fold(spelling))
        for name in near:
            if spelling in names:
                self._collision([parent / spelling, parent / name])
            else:
                self._problem(parent / name, f"rename to {spelling}; names are case-sensitive" + _spelled_as(name))
        return near

    def _collision(self, paths: Iterable[Path]) -> None:
        self.problems.append(f"{self._paths(paths)} differ only in case or Unicode form; rename one so the bundle "
                             "means the same on every filesystem")

    def _skip(self, path, message: str) -> NoReturn:
        self._problem(path, message)
        raise _Skip

    def _problem(self, path, message: str) -> None:
        self.problems.append(f"{self._rel(path)}: {message}")

    def _paths(self, paths: Iterable[Path]) -> str:
        return " and ".join(sorted(self._rel(path) for path in paths))

    def _rel(self, path) -> str:
        """``path`` relative to the root for a message; the root itself, and anything outside it,
        stay absolute."""
        text = str(path)
        if text.startswith(self.root_text + os.sep):
            text = text[len(self.root_text) + 1:]
        return show(text)


def _canonical(root: Path) -> Path:
    try:
        resolved = root.expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise BundleError([f"{show(root)}: no such folder"]) from None
    except (OSError, RuntimeError, ValueError) as e:
        raise BundleError([f"{show(root)}: {e}"]) from None
    if not resolved.is_dir():
        raise BundleError([f"{show(root)}: not a folder"])
    if resolved == Path(resolved.anchor):
        raise BundleError(["a bundle cannot be the filesystem root"])
    return on_disk(resolved)


def _refused(segment: str) -> str | None:
    """Why ``segment`` can't be one segment of an id, or None; the reason names the segment."""
    try:
        segment.encode("utf-8")
    except UnicodeEncodeError:
        return "the name is not valid UTF-8"
    as_id = LOCAL_PREFIX + unicodedata.normalize("NFC", segment)
    try:
        validate_local_id(as_id)
    except ValueError as e:
        return str(e).replace(f"local id {as_id!r}", repr(as_id.removeprefix(LOCAL_PREFIX)), 1)
    return None


def _size(entity_id: str) -> int:
    return len(entity_id.encode("utf-8", "surrogatepass"))


def _spelled_as(name: str) -> str:
    """For a near miss that may look right on screen: how it is really spelled, if not plain ASCII."""
    return "" if name.isascii() else f" (it is spelled {ascii(name)})"


def _brief(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = dict(pairs)
    if len(result) < len(pairs):
        duplicates = sorted(key for key, count in Counter(key for key, _ in pairs).items() if count > 1)
        raise ValueError(f"duplicate keys {duplicates}")
    return result


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")
