"""The bundles installed distributions register, through the ``agent_env.bundles`` entry points.

A distribution registers ``<name> = "<package>"`` and ships the bundle folder at ``<package>/<name>/``. The folder is
found without importing any package code: only the top-level package is located, and the rest of the path is walked
as folders. An installed bundle's ids are rooted at its distribution, ``@local/<dist>/<name>``.

A bundle can give its agents short aliases in an ``aliases.toml`` at its root, which a reference names an agent by
when no agent of that id is in the store (``installed_aliases``).

It imports nothing from ``agent_env.bundle``, so a task step can read what's installed without importing the bundle
machinery, whose run imports the task steps.
"""

from __future__ import annotations

import importlib.util
import os
import re
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import EntryPoint
from pathlib import Path

from packaging.utils import canonicalize_name

from agent_env.plugins._discovery import Plugin, claims, identity
from agent_env.plugins._report import INCOMPATIBLE_CORE, INVALID_PLUGIN, NAME_CONFLICT
from agent_env.plugins._requirements import incompatibility
from agent_env.store.ids import LOCAL_PREFIX
from agent_env.utils.paths import os_reason

BUNDLES = "agent_env.bundles"
CORE = "agentenv-framework"
ALIASES_FILE = "aliases.toml"
# A plain id, never a bundle's namespaced one, which holds '@' and '/'.
_ALIAS = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_ALIAS_RULE = "lowercase letters, digits and '-', at most 64 characters, starting and ending with a letter or digit"


@dataclass(frozen=True)
class InstalledBundle:
    """A bundle an installed distribution registers: its folder, or why it can't run."""

    name: str
    package: str  # the distribution, as its metadata spells it
    version: str
    value: str  # the package the entry point names
    root: Path | None
    code: str | None  # the plugin report's code for ``problem``
    problem: str | None

    @property
    def dist(self) -> str:
        return canonicalize_name(self.package)

    @property
    def qualified(self) -> str:
        return f"{self.dist}/{self.name}"

    @property
    def id_root(self) -> str:
        return f"{LOCAL_PREFIX}{self.dist}/{self.name}"


class _Unresolved(Exception):
    pass


def installed_bundles() -> tuple[InstalledBundle, ...]:
    """Every bundle installed distributions register, in name order, each resolved to its folder or with the
    reason it can't run. Raises what reading the installed metadata raises."""
    found = []
    for name, registrations in claims(BUNDLES).items():
        per_package = Counter(plugin.dist for plugin, _ in registrations)
        for plugin, ep in registrations:
            found.append(_installed(name, plugin, ep, repeated=per_package[plugin.dist] > 1))
    return tuple(found)


def _installed(name: str, plugin: Plugin, ep: EntryPoint, *, repeated: bool) -> InstalledBundle:
    package, version = identity(ep)

    def installed(root: Path | None = None, code: str | None = None, problem: str | None = None) -> InstalledBundle:
        return InstalledBundle(name, package, version, plugin.value, root, code, problem)

    if not plugin.dist:
        return installed(code=INVALID_PLUGIN, problem="its distribution's metadata can't be read")
    if repeated:
        return installed(code=NAME_CONFLICT, problem=f"{package} registers the bundle {name!r} more than once")
    if unmet := incompatibility(getattr(ep, "dist", None)):
        return installed(code=INCOMPATIBLE_CORE, problem=f"{package} {unmet}")
    try:
        return installed(root=_folder(ep, name))
    except _Unresolved as e:
        return installed(code=INVALID_PLUGIN, problem=str(e))


def _folder(ep: EntryPoint, name: str) -> Path:
    """The folder ``<package>/<name>`` the entry point names, found without importing the package."""
    # Not ep.module/ep.attr: on a value that doesn't parse they raise AttributeError, or AssertionError from 3.13 on.
    match = ep.pattern.match(ep.value)
    package, attr = (match.group("module"), match.group("attr")) if match else (None, None)
    if package is None or attr:
        raise _Unresolved(f"the entry point's value {ep.value!r} isn't a package name; name the package that holds "
                          f"the {name!r} folder")
    if "/" in name:
        raise _Unresolved(f"the bundle name {name!r} contains '/'")
    top, *rest = package.split(".")
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        raise _Unresolved(f"{top!r} isn't an installed package")
    # A namespace package can have several locations, and an editable install can add ones that aren't folders.
    locations = [Path(location) for location in spec.submodule_search_locations if os.path.isdir(location)]
    if not locations:
        raise _Unresolved(f"the package {top!r} isn't unpacked on disk; reinstall it unpacked")
    for location in locations:
        if (folder := _spelled(location, (*rest, name))) is not None:
            return folder
    raise _Unresolved(f"the package {package!r} has no folder {name!r}")


def _spelled(location: Path, parts: tuple[str, ...]) -> Path | None:
    """``location/parts`` when each part is a folder spelled exactly so, following links without resolving them,
    so a case-insensitive filesystem and a linked folder are judged as they are elsewhere."""
    folder = location
    for part in parts:
        try:
            if part not in os.listdir(folder) or not os.path.isdir(folder / part):
                return None
        except OSError as e:
            raise _Unresolved(f"the folder {folder} can't be read: {os_reason(e)}") from None
        folder /= part
    return folder


@dataclass(frozen=True)
class BundleAlias:
    """An alias an installed bundle gives one of its agents."""

    alias: str
    agent_id: str  # ``<the bundle's id root>/<the agent's folder name>``
    bundle: InstalledBundle

    @property
    def declared_by(self) -> str:
        return f"{self.bundle.dist}'s {self.bundle.name} bundle"


@dataclass(frozen=True)
class InstalledAliases:
    """The aliases installed bundles give their agents: each alias that names an agent, each refused to every bundle
    that declares it, with why, and each bundle's refused claims, with why."""

    aliases: Mapping[str, BundleAlias]
    refused: Mapping[str, str]
    refused_claims: Mapping[str, tuple[str, ...]]  # by the bundle's qualified name

    def target(self, alias: str) -> BundleAlias | None:
        return self.aliases.get(alias)

    def refusal(self, alias: str) -> str | None:
        return self.refused.get(alias)

    def of(self, bundle: InstalledBundle) -> tuple[str, ...]:
        """The aliases that name ``bundle``'s agents, in name order."""
        return tuple(sorted(alias for alias, named in self.aliases.items() if named.bundle == bundle))


def read_aliases(path: Path) -> tuple[dict[str, tuple[str, ...]], list[str]]:
    """The aliases a bundle's ``aliases.toml`` at ``path`` gives its agents, by agent folder name, and its problems; none
    of either when there's no such file, and no aliases when it has a problem."""
    if not path.is_file():
        return {}, []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        return {}, [f"can't be read: {os_reason(e)}"]
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        return {}, [f"isn't valid TOML: {e}"]
    problems = [f"[{key}] isn't a table it takes; it takes [agents]" for key in data if key != "agents"]
    agents = data.get("agents", {})
    if not isinstance(agents, dict):
        return {}, [*problems, "[agents] must be a table of agent folder names to lists of aliases"]
    found: dict[str, tuple[str, ...]] = {}
    named: dict[str, str] = {}
    for name, aliases in agents.items():
        if not (isinstance(aliases, list) and aliases and all(isinstance(alias, str) for alias in aliases)):
            problems.append(f"agents.{name} must be a list of aliases, not {aliases!r}")
            continue
        for alias in aliases:
            if not _ALIAS.match(alias):
                problems.append(f"agents.{name}: {alias!r} isn't an alias: {_ALIAS_RULE}")
            elif alias in named:
                where = "twice" if named[alias] == name else f"for agents.{named[alias]} too"
                problems.append(f"agents.{name}: {alias!r} is listed {where}; an alias names one agent")
            else:
                named[alias] = name
        found[name] = tuple(aliases)
    return ({}, problems) if problems else (found, [])


def installed_aliases(bundles: Sequence[InstalledBundle] | None = None) -> InstalledAliases:
    """The aliases the installed bundles give their agents. agent-env's own bundles' aliases are theirs: another bundle
    that declares one is refused it. An alias two other bundles both declare names neither, so a reference to it is
    refused rather than given one of them. A bundle whose ``aliases.toml`` has a problem gives none, which checking the
    bundle reports."""
    bundles = installed_bundles() if bundles is None else bundles
    claims_by_alias: dict[str, list[BundleAlias]] = {}
    for bundle in bundles:
        if bundle.root is None or bundle.problem:
            continue
        declared, _ = read_aliases(bundle.root / ALIASES_FILE)
        for name, aliases in declared.items():
            for alias in aliases:
                claims_by_alias.setdefault(alias, []).append(BundleAlias(alias, f"{bundle.id_root}/{name}", bundle))
    aliases: dict[str, BundleAlias] = {}
    refused: dict[str, str] = {}
    refused_claims: dict[str, list[str]] = {}
    for alias, claimants in sorted(claims_by_alias.items()):
        core = [claim for claim in claimants if claim.bundle.dist == canonicalize_name(CORE)]
        if core:
            aliases[alias] = core[0]
            for claim in claimants:
                if claim.bundle.dist != core[0].bundle.dist:
                    refused_claims.setdefault(claim.bundle.qualified, []).append(
                        f"{alias!r} is agent-env's own alias, of {core[0].agent_id}, so this bundle can't give it")
        elif len(claimants) == 1:
            aliases[alias] = claimants[0]
        else:
            declared_by = " and ".join(claim.declared_by for claim in claimants)
            refused[alias] = (f"{alias!r} is an alias {declared_by} each give an agent, so it names none of them; "
                              "name the agent by its id")
            for claim in claimants:
                refused_claims.setdefault(claim.bundle.qualified, []).append(
                    f"{alias!r} is another installed bundle's alias too, so neither gets it")
    return InstalledAliases(aliases, refused, {name: tuple(why) for name, why in refused_claims.items()})

