"""What an entity's ``from_toml`` builds itself from: its bundle entry, and what its toml names."""

from __future__ import annotations

import os
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TypeVar, overload

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.entity_refs import EntityKind, parse_toml_ref
from agent_env.env.env import Env
from agent_env.env.envs._deployment import provider_refusal
from agent_env.providers.env_providers.constants import GATEWAY_SERVICE_NAMES
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_providers.env_provider import _env_provider_class
from agent_env.store.ids import validate_local_id
from agent_env.utils.card_naming import card_names_in_files

from ._fs import fold, os_reason, relative, show, with_article
from .parse import CONFIG_FILES, LEAVINGS, NAMED_BY, Bundle, BundleEntry, BundleError

A = TypeVar("A", bound=Artifact)
E = TypeVar("E", bound=Env)

_BRING_IT_IN = "copy what it points to into the bundle, or put it in a store and refer to it by id"
_TOML_TYPES = {str: "a string", dict: "a table", list: "a list"}


def default_dockerfile(key: str) -> str:
    """The Dockerfile an image key left out builds from the entry's folder: ``Dockerfile`` for ``image``, and
    ``Dockerfile.<role>`` for ``<role>_image``, as a website's ``backend_image`` and ``frontend_image``."""
    return "Dockerfile" if key == "image" else f"Dockerfile.{key.removesuffix('_image')}"


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

    def universe(self) -> UniverseLayout:
        return universe_layout(self.bundle, self.entry)

    def renamed(self, data: dict, names: Mapping[str, str]) -> tuple[dict, list[str]]:
        """``data`` less each key ``names`` maps to the key an author writes instead, and a problem for each:
        the names a stored document uses, which a toml doesn't."""
        problems = [self.config_problem(f"{key} is what the stored document calls it; write {names[key]} instead")
                    for key in data if key in names]
        return {key: value for key, value in data.items() if key not in names}, problems

    def card_names(self, image_key: str) -> list[str] | None:
        """The names ``@environment_card(name=...)`` gives in the source of the image ``image_key`` builds from
        this folder: the ``.py`` files of its build context, those in its Dockerfile's folder first. None when the
        key names an image rather than building one."""
        value = self.entry.config.get(image_key) if isinstance(self.entry.config, dict) else None
        if value is None:
            dockerfile = default_dockerfile(image_key)
        elif isinstance(value, dict) and value.keys() == {"dockerfile"} and isinstance(value["dockerfile"], str):
            dockerfile = value["dockerfile"]
        else:
            return None
        return card_names_in_files(build_context_files(self.bundle, self.entry), dockerfile)

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
        problems = []
        unknown = sorted(set(data) - {"type", "id", *takes})
        if unknown:
            names = [*takes, "type", "id"]
            problems.append(self.config_problem(
                f"unknown key{'s' * (len(unknown) > 1)} {', '.join(map(repr, unknown))}; "
                f"{with_article(f'{self.entry.type} {self.entry.kind.store}')} takes "
                f"{', '.join(names[:-1])} and {names[-1]}"))
        for key, kind in takes.items():
            if key in data and not isinstance(data[key], kind):
                problems.append(self.config_problem(
                    f"{key} must be {_TOML_TYPES.get(kind, kind.__name__)}, not {data[key]!r}"))
        fields = {key: data[key] for key, kind in takes.items() if key in data and isinstance(data[key], kind)}
        return fields, problems

    def accept_env(self, data: dict, env_class: type[Env], *, named_by: str | None = None) -> dict:
        """The keys of ``data``, an env.toml, ``env_class`` takes, checked the way ``accepted_env`` checks
        them; every problem is refused."""
        fields, problems = self.accepted_env(data, env_class, named_by=named_by)
        if problems:
            self.refuse(problems)
        return fields

    def accepted_env(self, data: dict, env_class: type[Env], *,
                     named_by: str | None = None) -> tuple[dict, list[str]]:
        """``accepted`` for an env.toml: the keys ``env_class`` takes, and the problems with them. The fields
        carry its env_provider_type, the gateway's when left out, which must name an installed provider that
        deploys an env of ``env_class``. With ``named_by``, they carry its environment_name too: the one set,
        else the one the ``@environment_card`` in the source of the image that key builds gives, which a
        gateway deploy can't take from one of its own containers."""
        fields, problems = self.accepted(data, **env_class.toml_keys)
        provider_type = fields.setdefault("env_provider_type", EnvironmentGatewayProvider.type)
        try:
            provider = _env_provider_class(provider_type)
        except ValueError as e:
            problems.append(self.config_problem(str(e)))
            provider = None
        if provider is not None and (reason := provider_refusal(provider, env_class)) is not None:
            problems.append(self.config_problem(f"env_provider_type {provider_type!r} {reason}"))
        if named_by is not None:
            name, problem = self._environment_name(data, fields, named_by)
            if problem is not None:
                problems.append(self.config_problem(problem))
            elif name is not None:
                fields["environment_name"] = name
            if (name in GATEWAY_SERVICE_NAMES and provider is not None
                    and issubclass(provider, EnvironmentGatewayProvider)):
                problems.append(self.config_problem(
                    f"environment_name {name!r} is one a gateway deploy names its own containers "
                    f"({', '.join(sorted(GATEWAY_SERVICE_NAMES))}); choose another"))
        return fields, problems

    def _environment_name(self, data: dict, fields: dict, named_by: str) -> tuple[str | None, str | None]:
        """The env's environment_name, and the problem when there's none: the one set, else the one the source
        of the image ``named_by`` builds from this folder declares."""
        if "environment_name" in data:  # one of the wrong type is a problem accepted() reported
            name = fields.get("environment_name")
            return name, "environment_name can't be empty" if name == "" else None
        names = self.card_names(named_by)
        what = "image" if named_by == "image" else named_by.replace("_", " ")
        if names is None:
            return None, (f"environment_name isn't set, and its {what} isn't built from this folder, so there's no "
                          "source to read it from; set it")
        if len(names) != 1:
            found = (f"several environment cards ({', '.join(map(repr, names))})" if names
                     else "no @environment_card(name=...)")
            return None, f"environment_name isn't set, and the source its {what} is built from declares {found}; set it"
        return names[0], None

    def config_problem(self, message: str) -> str:
        """``message`` as a problem with the entry's toml, ready for ``refuse``."""
        return f"{CONFIG_FILES[self.entry.kind]}: {message}"

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


def entry_files(bundle: Bundle, entry: BundleEntry, *, empty_ok: bool = False) -> dict[str, Path]:
    """The files in an entry's folder, keyed by POSIX path in NFC and sorted: regular files, dot files
    included, through links that stay inside the bundle, less what the OS or Python leaves behind and
    the entry's own toml. Raises BundleError listing every problem found, a folder with no files among
    them unless ``empty_ok``."""
    return _Walk(bundle, entry, empty_ok).files()


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
    key, path = next(iter(one_file(bundle, entry).items()))
    return key.rsplit("/", 1)[-1], path


_AS_UNIVERSE = "leave out its declared type to write the folder as a file_artifact_universe"


def one_file(bundle: Bundle, entry: BundleEntry, remedy: str = _AS_UNIVERSE) -> dict[str, Path]:
    """``entry_files`` of a folder that must hold one file. Raises BundleError otherwise, suggesting ``remedy``."""
    return _only(bundle, entry, entry_files(bundle, entry), remedy)


def _only(bundle: Bundle, entry: BundleEntry, files: dict[str, Path], remedy: str) -> dict[str, Path]:
    if len(files) != 1:
        listed = ", ".join(repr(name) for name in files)
        raise BundleError([f"{relative(bundle.root, entry.path)}: {with_article(f'{entry.type} {entry.kind.store}')} "
                           f"holds one file, and this folder has {len(files)} ({listed}); {remedy}"])
    return files


# The keys that name the file an environment artifact wraps: ``file``, and the stored document's names for it, which
# its toml check refuses with a pointer to ``file``.
_NAMES_ITS_FILE = ("file", *(stored for stored, key in EnvironmentArtifact.toml_stored_names.items() if key == "file"))


def environment_files(bundle: Bundle, entry: BundleEntry) -> dict[str, Path]:
    """What an environment artifact's write reads from its folder: its one file, or none when its artifact.toml's
    ``file`` names the artifact to wrap instead. Raises BundleError otherwise."""
    files = entry_files(bundle, entry, empty_ok=True)
    if not (isinstance(entry.config, dict) and any(name in entry.config for name in _NAMES_ITS_FILE)):
        if not files:
            raise BundleError([f"{relative(bundle.root, entry.path)}: has no file for the environment to wrap "
                               f"({', '.join(sorted(LEAVINGS))} don't count); put its one file here, or name the "
                               "artifact to wrap with file"])
        return _only(bundle, entry, files, "name the artifact to wrap with file, or give each other file an artifact "
                                           "folder of its own")
    if files:
        listed = ", ".join(repr(name) for name in files)
        raise BundleError([f"{relative(bundle.root, entry.path)}: its file is the artifact file names, so the folder "
                           f"holds only artifact.toml, and it also has {listed}; give the file an artifact folder of "
                           "its own, or leave out file to wrap the folder's one file"])
    return {}


# The folder of a universe's metadata files, as `environment-universe get --output-dir` writes them.
UNIVERSE_METADATA = EnvironmentUniverseArtifact.metadata_name


@dataclass(frozen=True)
class LaidOut:
    """One file of a universe folder, and the ids it's written under."""

    name: str  # its environment's name, or its metadata key
    filename: str
    path: Path
    file_id: str
    environment_id: str | None  # None for a metadata file


@dataclass(frozen=True)
class UniverseLayout:
    """An environment_universe folder, laid out as ``environment-universe get --output-dir`` writes one: each
    ``<environment_name>/`` folder holds one environment's file, and each ``metadata/<key>/`` folder one metadata
    file. ``files`` lists them all, as ``entry_files`` does."""

    environments: tuple[LaidOut, ...]
    metadata: tuple[LaidOut, ...]
    files: dict[str, Path]


def universe_layout(bundle: Bundle, entry: BundleEntry) -> UniverseLayout:
    """The environments and metadata files an environment_universe's folder holds, each with the ids it's
    written under: ``<universe>__<name>`` over ``<universe>__<name>__file`` for an environment, and
    ``<universe>__metadata__<key>`` for a metadata file. Raises BundleError listing every problem."""
    files = entry_files(bundle, entry, empty_ok=True)
    where = relative(bundle.root, entry.path)
    top = _folders(entry.path)
    # A metadata/ folder spelled in another case is a typo, as for every reserved name in a bundle.
    near = sorted(name for name in top if name != UNIVERSE_METADATA and fold(name) == fold(UNIVERSE_METADATA))
    problems = [f"{where}/{name}: rename to {UNIVERSE_METADATA}; names are case-sensitive, and an environment named "
                f"{name} goes in an artifact folder of its own, named in environment_artifacts" for name in near]
    # Every folder counts, so one left without its file is refused rather than dropped.
    folders: dict[tuple[bool, str], list[str]] = {(False, name): [] for name in top
                                                  if name != UNIVERSE_METADATA and name not in near}
    folders.update({(True, name): [] for name in _folders(entry.path / UNIVERSE_METADATA)})
    for key in files:
        parts = key.split("/")
        if parts[0] in near:
            continue
        metadata = parts[0] == UNIVERSE_METADATA
        if len(parts) == (3 if metadata else 2):
            folders.setdefault((metadata, parts[-2]), []).append(key)
        elif len(parts) == 1:
            problems.append(f"{where}/{key}: a universe folder holds only artifact.toml and a folder for each "
                            "environment; move it into <environment_name>/, or give it an artifact folder of its own")
        elif metadata and len(parts) == 2:
            problems.append(f"{where}/{key}: {UNIVERSE_METADATA}/ holds a folder for each key, with that key's one "
                            f"file in it ({UNIVERSE_METADATA}/<key>/<file>); an environment named "
                            f"{UNIVERSE_METADATA} goes in an artifact folder of its own, named in "
                            "environment_artifacts")
        else:
            folder = "/".join(parts[:2]) if metadata else parts[0]
            what = "a metadata key's" if metadata else "an environment's"
            problems.append(f"{where}/{key}: {what} folder holds its one file directly, {folder}/<file>, with no "
                            "folders inside")
    environments, metadata_files = [], []
    for (metadata, name), keys in sorted(folders.items()):
        folder = f"{where}/{UNIVERSE_METADATA}/{name}" if metadata else f"{where}/{name}"
        if len(keys) != 1:
            listed = f" ({', '.join(repr(key.rsplit('/', 1)[-1]) for key in keys)})" if keys else ""
            problems.append(f"{folder}: holds one file, and has {len(keys)}{listed}")
            continue
        # Renaming an environment's folder renames the environment, so say how to keep its name.
        rename = "rename the folder" if metadata else (
            f"rename the folder, which renames the environment, or keep the name by giving the environment an "
            f'artifact folder of its own, with environment_name = "{name}", named in environment_artifacts')
        if name != name.strip():
            problems.append(f"{folder}: a name that opens or closes with a space can't name the artifacts written "
                            "from it; rename the folder")
            continue
        if "__" in name:
            problems.append(f"{folder}: a name holding __ could clash with the ids derived from it; {rename}")
            continue
        if metadata:
            environment_id, file_id = None, EnvironmentUniverseArtifact.derived_metadata_id(entry.id, name)
        else:
            environment_id = EnvironmentUniverseArtifact.derived_environment_id(entry.id, name)
            file_id = EnvironmentArtifact.derived_file_id(environment_id)
        try:
            for id in (environment_id, file_id):
                if id is not None:
                    validate_local_id(id)
        except ValueError as e:
            problems.append(f"{folder}: can't name the artifacts written from it ({e}); {rename}")
            continue
        laid = LaidOut(name, keys[0].rsplit("/", 1)[-1], files[keys[0]], file_id, environment_id)
        (metadata_files if metadata else environments).append(laid)
    if problems:
        raise BundleError(problems)
    return UniverseLayout(tuple(environments), tuple(metadata_files), files)


def _folders(path: Path) -> list[str]:
    """The names of the folders directly in ``path``, as the walk spells them, or none when it isn't one."""
    try:
        with os.scandir(path) as listing:
            return [unicodedata.normalize("NFC", child.name) for child in listing
                    if child.name not in LEAVINGS and child.is_dir()]
    except OSError:
        return []  # the walk reports why it can't be listed


def universe_files(bundle: Bundle, entry: BundleEntry) -> dict[str, Path]:
    """What an environment_universe's write reads from its folder: every file ``universe_layout`` lays out."""
    return universe_layout(bundle, entry).files


class _Walk:
    """Lists an entry's folder, following a link only when its target is inside the bundle."""

    def __init__(self, bundle: Bundle, entry: BundleEntry, empty_ok: bool = False):
        self.bundle = bundle
        self.entry = entry
        self.empty_ok = empty_ok
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
        if not self.found and not self.problems and not self.empty_ok:
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


