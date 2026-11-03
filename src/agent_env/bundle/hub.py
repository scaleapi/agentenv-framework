"""Bundles run straight from a Hugging Face Hub dataset repo, named ``hf://<org>/<repo>[@<revision>][/<path>]``. Each
commit's bundle is downloaded once into a plain folder, since the Hub's own cache keeps files as links that would leave
the bundle, and its ids are rooted in the ``@hf`` namespace, which only this path produces."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from huggingface_hub.repocard import metadata_load

from agent_env.config.paths import cache_root
from agent_env.plugins._bundles import ALIASES_FILE
from agent_env.plugins._requirements import unmet_problems
from agent_env.store.ids import HF_PREFIX, validate_local_id
from agent_env.store.local_state import holding_locks

from .parse import BUNDLE_TOML, BundleError, BundleKind, read_requires
from .requires import UnmetRequirements

SCHEME = "hf://"
BUNDLES = "bundles"
CARD = "README.md"
SOURCE_KEY = "bundle_source"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA = re.compile(r"[0-9a-f]{40}")
_KINDS = frozenset(kind.value for kind in BundleKind)
_FORM = "hf://<org>/<repo>[@<revision>][/<path>]"


@dataclass(frozen=True)
class HubReference:
    """An ``hf://`` reference's parts: ``<org>/<repo>``, the revision named, and the bundle's folder in the repo."""

    repo: str
    revision: str | None
    path: str | None


@dataclass(frozen=True)
class HubSource:
    """Where a Hub bundle came from: its repo, the revision named (``main`` by default), its commit, and its folder in
    the repo, ``""`` for the root. Each run's context records it under ``SOURCE_KEY``."""

    repo: str
    revision: str
    sha: str
    path: str

    @classmethod
    def from_metadata(cls, data: Mapping[str, str]) -> HubSource:
        return cls(data["repo"], data["revision"], data["sha"], data["path"])

    def to_metadata(self) -> dict[str, str]:
        return {"hub": "hf", "repo": self.repo, "revision": self.revision, "sha": self.sha, "path": self.path}

    def describe(self) -> str:
        """As a run prints it: ``<org>/<repo> at <sha12>, bundle <path>``."""
        return f"{self.repo} at {self.sha[:12]}, bundle {self.path or 'at its root'}"


@dataclass(frozen=True)
class HubBundle:
    """A bundle downloaded from the Hub: its folder, the id root it runs under, where it came from, and what the repo's
    card says it needs: the plugins it requires when it has no bundle.toml, and the command that sets it up."""

    root: Path
    id_root: str
    source: HubSource
    plugins: tuple[str, ...]
    setup: str | None

    @property
    def setup_note(self) -> str | None:
        """What a run refused for something setting the bundle up provides ends with: the card's command for it."""
        return f"{self.source.repo}'s card sets this bundle up with: {self.setup}" if self.setup else None


@dataclass
class _Commit:
    """What this machine holds of a commit, kept in its folder: the commit's file list, and the bundles downloaded."""

    files: list[str]
    downloaded: list[str] = field(default_factory=list)

    @staticmethod
    def path(folder: Path) -> Path:
        return folder / ".cache" / "agentenv" / "commit.json"

    @classmethod
    def read(cls, folder: Path) -> _Commit | None:
        try:
            return cls(**json.loads(cls.path(folder).read_text()))
        except FileNotFoundError:
            return None

    def save(self, folder: Path) -> None:
        path = self.path(folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}")
        partial.write_text(json.dumps({"files": self.files, "downloaded": self.downloaded}))
        os.replace(partial, path)


def is_hub_reference(text: str) -> bool:
    return text.startswith(SCHEME)


def parse_reference(text: str) -> HubReference:
    """``text``, ``hf://<org>/<repo>[@<revision>][/<path>]``, as its parts. The revision runs to the next ``/``, so one
    holding a ``/`` is written percent-encoded, as the Hub's own ``hf://`` paths take it (``@refs%2Fpr%2F3``). Raises
    BundleError."""
    org, _, rest = text.removeprefix(SCHEME).partition("/")
    name, _, path = rest.partition("/")
    name, at, revision = name.partition("@")
    parts = [part for part in path.split("/") if part]
    if not (_NAME.fullmatch(org) and _NAME.fullmatch(name)) or (at and not revision):
        raise BundleError([f"{text}: name a Hub dataset's bundle as {_FORM}"])
    if any(part in (".", "..") for part in parts):
        raise BundleError([f"{text}: the path in the repo can't hold '.' or '..'"])
    return HubReference(f"{org}/{name}", unquote(revision) or None, "/".join(parts) or None)


def fetch_bundle(text: str, *, on_progress: Callable[[str], None] | None = None) -> HubBundle:
    """The bundle ``text`` names, downloaded unless this machine has it. ``on_progress`` is given the line that says
    which repo, commit and bundle it is, before anything is downloaded. Raises BundleError."""
    reference = parse_reference(text)
    try:
        return _fetch(reference, HfApi(), on_progress or (lambda line: None))
    except HfHubHTTPError as e:
        error = BundleError([f"{text}: {str(e).strip() or type(e).__name__}"])
        if isinstance(e, RepositoryNotFoundError):
            error.add_note("a private or gated repo needs a token with access: log in with `hf auth login`, or set "
                           "HF_TOKEN")
        raise error from e


def _fetch(reference: HubReference, api: HfApi, say: Callable[[str], None]) -> HubBundle:
    sha = reference.revision if reference.revision and _SHA.fullmatch(reference.revision) else api.repo_info(
        reference.repo, repo_type="dataset", revision=reference.revision).sha
    folder = cache_root() / "hf" / reference.repo / sha
    with holding_locks([f"{SCHEME}{reference.repo}@{sha}"]):
        commit, card = _commit(api, reference, sha, folder)
        hub = _select(reference, sha, commit.files, card, folder)
        say(hub.source.describe())
        _download(api, hub, commit, folder)
    return hub


def _commit(api: HfApi, reference: HubReference, sha: str, folder: Path) -> tuple[_Commit, dict[str, Any]]:
    """What this machine holds of the commit, and its card's ``agentenv`` table: listed, with the card downloaded,
    the first time."""
    commit = _Commit.read(folder)
    if commit is None:
        commit = _Commit(api.list_repo_files(reference.repo, repo_type="dataset", revision=sha))
        if CARD in commit.files:
            api.hf_hub_download(reference.repo, CARD, repo_type="dataset", revision=sha, local_dir=folder)
        commit.save(folder)
    return commit, _card(folder)


def _select(reference: HubReference, sha: str, files: Sequence[str], card: Mapping[str, Any], folder: Path) -> HubBundle:
    """The bundle ``reference`` names in the commit of ``files``, kept under ``folder``: picked, with its id root,
    where it came from and what the card says it needs."""
    path = _pick(reference, sha, files, card)
    plugins, setup = _needs(reference.repo, card, _bundle_name(reference, path))
    source = HubSource(reference.repo, reference.revision or "main", sha, path)
    declared = _within(path, BUNDLE_TOML) in files
    return HubBundle(folder / path, _id_root(reference, path), source, plugins=() if declared else plugins, setup=setup)


def _download(api: HfApi, hub: HubBundle, commit: _Commit, folder: Path) -> None:
    """Check what the bundle requires, its bundle.toml downloaded alone first when it has one, then download the
    bundle unless the commit's record has it."""
    path = hub.source.path
    declared, downloaded = _within(path, BUNDLE_TOML) in commit.files, path in commit.downloaded
    if declared and not downloaded:
        api.hf_hub_download(hub.source.repo, _within(path, BUNDLE_TOML), repo_type="dataset", revision=hub.source.sha,
                            local_dir=folder)
    _check_requirements(hub, declared)
    if not downloaded:
        api.snapshot_download(hub.source.repo, repo_type="dataset", revision=hub.source.sha, local_dir=folder,
                              allow_patterns=_patterns(path))
        commit.downloaded.append(path)
        commit.save(folder)


def _check_requirements(hub: HubBundle, declared: bool) -> None:
    """Refuse a bundle whose bundle.toml (by a folder's rules) or, without one, card names a plugin not installed,
    the refusal ending with the card's setup command. Raises UnmetRequirements, or BundleError."""
    if declared:
        requires, refused = read_requires(hub.root / BUNDLE_TOML)
        if refused:
            raise BundleError([f"{BUNDLE_TOML}: {problem}" for problem in refused])
        problems, unmet = unmet_problems(requires, BUNDLE_TOML)
    else:
        problems, unmet = unmet_problems(hub.plugins, f"{hub.source.repo}'s card")
    if not problems:
        return
    if not unmet:
        raise BundleError(problems)
    error = UnmetRequirements(problems, unmet)
    if hub.setup_note:
        error.add_note(hub.setup_note)
    raise error


def _card(folder: Path) -> dict[str, Any]:
    """The ``agentenv`` table of the card's metadata, ``{}`` without one."""
    readme = folder / CARD
    table = (metadata_load(readme) or {}).get("agentenv") if readme.is_file() else None
    return table if isinstance(table, dict) else {}


def _needs(repo: str, card: Mapping[str, Any], name: str) -> tuple[tuple[str, ...], str | None]:
    """What the card's ``agentenv: bundles: <name>`` table says the bundle needs: the plugins it requires, and the
    command that sets it up. Raises BundleError for a value of the wrong shape."""
    where = f"{repo}'s card: agentenv: bundles"
    bundles = card.get("bundles") or {}
    if not isinstance(bundles, Mapping):
        raise BundleError([f"{where} must map each bundle's name to what it needs"])
    needs = bundles.get(name) or {}
    if not isinstance(needs, Mapping):
        raise BundleError([f"{where}: {name} must be a table of plugins and setup"])
    plugins, setup = needs.get("plugins") or [], needs.get("setup")
    if not isinstance(plugins, list) or not all(isinstance(plugin, str) for plugin in plugins):
        raise BundleError([f"{where}: {name}: plugins must be a list of pip requirements"])
    if setup is not None and not isinstance(setup, str):
        raise BundleError([f"{where}: {name}: setup must be a command"])
    return tuple(plugins), setup


def _pick(reference: HubReference, sha: str, files: Sequence[str], card: Mapping[str, Any]) -> str:
    """The folder in the repo of the bundle ``reference`` names, ``""`` for the root: its path; else the only bundle
    under ``bundles/``, or the one the card names its default; else the root when it is a bundle. ``bundles/`` comes
    first, since a dataset can keep a ``tasks/`` of data files at its root."""
    where = f"{reference.repo} at {sha[:12]}"
    if reference.path is not None:
        if not _is_bundle(files, reference.path):
            raise BundleError([f"{where}: {reference.path}/ isn't a bundle: it has none of {_kinds()}"])
        return reference.path
    names = sorted({parts[1] for file in files if len(parts := file.split("/")) > 2 and parts[0] == BUNDLES})
    if len(names) == 1:
        return _within(BUNDLES, names[0])
    if card.get("default") in names:
        return _within(BUNDLES, card["default"])
    if names:
        revision = f"@{reference.revision}" if reference.revision else ""
        runs = ", ".join(f"{SCHEME}{reference.repo}{revision}/{BUNDLES}/{name}" for name in names)
        raise BundleError([f"{where} holds several bundles under {BUNDLES}/ and its card names no default "
                           f"(agentenv: default); run one of {runs}"])
    if _is_bundle(files, ""):
        return ""
    raise BundleError([f"{where} holds no bundle: it has no {BUNDLES}/<name>/ folder, and none of {_kinds()} at its "
                       "root"])


def _is_bundle(files: Sequence[str], path: str) -> bool:
    """Whether the folder at ``path`` in the repo holds a kind folder, as a bundle does."""
    prefix = _within(path, "")
    return any(file.startswith(prefix) and file[len(prefix):].split("/", 1)[0] in _KINDS
               and "/" in file[len(prefix):] for file in files)


def _patterns(path: str) -> list[str]:
    """What to download of the bundle at ``path``: its folder, or for the repo's root its kind folders and files."""
    if path:
        return [f"{path}/*"]
    return [*(f"{kind}/*" for kind in sorted(_KINDS)), BUNDLE_TOML, ALIASES_FILE]


def _id_root(reference: HubReference, path: str) -> str:
    """``@hf/<org>/<repo>``, then for a bundle below the root its name."""
    id_root = f"{HF_PREFIX}{reference.repo}" + (f"/{_bundle_name(reference, path)}" if path else "")
    try:
        validate_local_id(id_root)
    except ValueError as e:
        raise BundleError([f"{SCHEME}{reference.repo}/{path}: can't be an id root: {e}"]) from None
    return id_root


def _bundle_name(reference: HubReference, path: str) -> str:
    """The bundle's name, which keys its table in the card and ends its id root: its folder under ``bundles/``, its
    path for a folder elsewhere, or the repo's name for the root."""
    parts = path.split("/")
    return parts[1] if len(parts) == 2 and parts[0] == BUNDLES else path or reference.repo.partition("/")[2]


def _within(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


def _kinds() -> str:
    return ", ".join(f"{kind.value}/" for kind in BundleKind)
