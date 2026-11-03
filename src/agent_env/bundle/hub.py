"""Bundles run straight from a Hugging Face Hub dataset repo, named ``hf://<org>/<repo>[@<revision>][/<path>]``.

The revision, a branch, tag or commit (the main branch by default), resolves to a commit, and the bundle is picked from
that commit's file list before anything is downloaded, so a results dataset's recordings never are. Only the bundle's
files and the repo's card are downloaded, into a plain folder for the commit: the Hub's own cache keeps files as links
into its blob store, and the bundle parser refuses a link that leaves the bundle. A commit never changes, so what was
downloaded for it is reused; a reference pinned to the commit then needs no network at all. The bundle's ids are rooted
at ``@hf/<org>/<repo>``, plus ``/<name>`` for a bundle under ``bundles/``, a local namespace of its own: the id says the
bundle came from the Hub, and a rerun of the same commit reuses what the last one wrote. A gated or private repo is read
with the Hub library's own login (``hf auth login`` or ``HF_TOKEN``).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.repocard import metadata_load

from agent_env.plugins._bundles import ALIASES_FILE
from agent_env.plugins._requirements import unmet_problems
from agent_env.store.ids import HF_PREFIX, validate_local_id

from .parse import BUNDLE_TOML, BundleError, BundleKind
from .requires import UnmetRequirements

SCHEME = "hf://"
BUNDLES = "bundles"
CARD = "README.md"
# The key a run's context records where its bundle came from under, in its metadata.
SOURCE_KEY = "bundle_source"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA = re.compile(r"[0-9a-f]{40}")
_KINDS = frozenset(kind.value for kind in BundleKind)
_FORM = "hf://<org>/<repo>[@<revision>][/<path>]"


@dataclass(frozen=True)
class HubReference:
    repo: str  # <org>/<repo>
    revision: str | None  # None for the main branch
    path: str | None  # the bundle's folder in the repo, when named


@dataclass(frozen=True)
class HubBundle:
    """A bundle downloaded from the Hub: its folder, the id root it runs under, where it came from, and what the repo's
    card says it needs."""

    root: Path
    id_root: str
    source: Mapping[str, str]  # hub, repo, revision, sha, path: what each run's context records
    plugins: tuple[str, ...]  # the card's, for a bundle without a bundle.toml
    setup: str | None  # the card's command that sets the bundle up, shown when a run fails, never run

    def check_plugins(self) -> None:
        """Refuse the run when a plugin the card names isn't installed, naming the command that installs it. Raises
        UnmetRequirements."""
        problems, unmet = unmet_problems(self.plugins, f"{self.source['repo']}'s card")
        if problems:
            raise UnmetRequirements(problems, unmet) if unmet else BundleError(problems)


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


def cache_root() -> Path:
    """Where downloaded commits are kept: ``$XDG_CACHE_HOME/agentenv/hf`` when that is an absolute path, else
    ``~/.cache/agentenv/hf``."""
    base = os.environ.get("XDG_CACHE_HOME", "")
    return (Path(base) if os.path.isabs(base) else Path.home() / ".cache") / "agentenv" / "hf"


def fetch_bundle(text: str, *, on_progress: Callable[[str], None] | None = None) -> HubBundle:
    """The bundle ``text`` names, downloaded unless this machine has it. ``on_progress`` is given the line that says
    which repo, commit and bundle it is, before anything is downloaded. Raises BundleError."""
    reference = parse_reference(text)
    try:
        return _fetch(reference, HfApi(), on_progress or (lambda line: None))
    except httpx.HTTPError as e:
        error = BundleError([f"{text}: {str(e).strip() or type(e).__name__}"])
        if isinstance(e, RepositoryNotFoundError):
            error.add_note("a private or gated repo needs a token with access: log in with `hf auth login`, or set "
                           "HF_TOKEN")
        raise error from e


def describe(source: Mapping[str, str]) -> str:
    """Where a bundle came from, as a run prints it: ``<org>/<repo> at <sha12>, bundle <path>``."""
    return f"{source['repo']} at {source['sha'][:12]}, bundle {source['path'] or 'at its root'}"


def _fetch(reference: HubReference, api: HfApi, say: Callable[[str], None]) -> HubBundle:
    sha = reference.revision if reference.revision and _SHA.fullmatch(reference.revision) else api.repo_info(
        reference.repo, repo_type="dataset", revision=reference.revision).sha
    folder = cache_root() / reference.repo / sha
    commit = _Commit.read(folder)
    if commit is None:
        commit = _Commit(api.list_repo_files(reference.repo, repo_type="dataset", revision=sha))
        if CARD in commit.files:
            api.hf_hub_download(reference.repo, CARD, repo_type="dataset", revision=sha, local_dir=folder)
        commit.save(folder)
    card = _card(folder)
    path = _pick(reference, sha, commit.files, card)
    id_root = _id_root(reference, path)
    source = {"hub": "hf", "repo": reference.repo, "revision": reference.revision or "main", "sha": sha, "path": path}
    say(describe(source))
    if path not in commit.downloaded:
        api.snapshot_download(reference.repo, repo_type="dataset", revision=sha, local_dir=folder,
                              allow_patterns=_patterns(path))
        commit.downloaded.append(path)
        commit.save(folder)
    name = path.rpartition("/")[2] or reference.repo.partition("/")[2]
    needs = (card.get("bundles") or {}).get(name) or {}
    declared = _within(path, BUNDLE_TOML) in commit.files
    return HubBundle(folder / path, id_root, source, plugins=() if declared else tuple(needs.get("plugins") or ()),
                     setup=needs.get("setup"))


def _card(folder: Path) -> dict[str, Any]:
    """The ``agentenv`` table of the card's metadata, ``{}`` without one."""
    readme = folder / CARD
    table = (metadata_load(readme) or {}).get("agentenv") if readme.is_file() else None
    return table if isinstance(table, dict) else {}


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
    """``@hf/<org>/<repo>``, then the bundle's name for one under ``bundles/``, or its path for one elsewhere."""
    parts = path.split("/") if path else []
    if len(parts) == 2 and parts[0] == BUNDLES:
        parts = parts[1:]
    id_root = "/".join([f"{HF_PREFIX}{reference.repo}", *parts])
    try:
        validate_local_id(id_root)
    except ValueError as e:
        raise BundleError([f"{SCHEME}{reference.repo}/{path}: can't be an id root: {e}"]) from None
    return id_root


def _within(folder: str, name: str) -> str:
    return f"{folder}/{name}" if folder else name


def _kinds() -> str:
    return ", ".join(f"{kind.value}/" for kind in BundleKind)
