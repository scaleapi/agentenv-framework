"""What a bundle needs beyond core: the plugin packages its optional ``bundle.toml`` requires, checked against what's
installed before the bundle resolves, and who provides each type it uses that core doesn't ship, which a bundle with no
``bundle.toml`` is given one from."""

from __future__ import annotations

import importlib
import json
import os
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import distribution, packages_distributions
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.registry import ARTIFACT_REGISTRY, canonical_type, get_artifact_registry
from agent_env.entity.refs import EntityKind, RefRole, ref_sites
from agent_env.env import registry as env_registry
from agent_env.env.env import Env
from agent_env.env.registry import get_env_registry
from agent_env.plugins import ARTIFACTS, ENVS, TASK_STEPS, _requirements, inventory, load_failures
from agent_env.store.base import NotFoundError
from agent_env.store.document_store import UnknownDocumentTypeError
from agent_env.task_step import registry as task_step_registry
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep

from . import _toml
from ._fs import relative
from .parse import BUNDLE_TOML, BundleError, BundleKind

if TYPE_CHECKING:
    from .resolve import ResolvedBundle, ResolvedEntry

# What core ships, per plugin group: anything else a bundle uses is provided by a plugin, or by configuration.
_BUILTINS = {TASK_STEPS: task_step_registry._builtin_registry, ENVS: env_registry._builtin_registry,
             ARTIFACTS: lambda: ARTIFACT_REGISTRY}
# The entities a step reads whose type a plugin may register.
_GETTERS = {EntityKind.ENV: Env.get, EntityKind.ARTIFACT: Artifact.get}


class UnmetRequirements(BundleError):
    """A bundle whose ``bundle.toml``, or a stored task whose ``requires``, names packages that aren't installed, or
    not at a version it allows, with every other problem found; ``requirements`` are the unmet ones, as written."""

    def __init__(self, problems: Iterable[str], requirements: Sequence[str]):
        super().__init__(problems)
        self.requirements = tuple(requirements)


class Declaring(Enum):
    """What a command says, and does, of the plugin packages a bundle uses that its ``bundle.toml`` doesn't require."""

    WRITE = "write"  # give a bundle without a bundle.toml one; tell one that has one what to add
    WOULD_WRITE = "would write"  # a dry run's: say what WRITE would write
    HINT = "hint"  # tell it what to declare, writing nothing: a folder that isn't the bundle's own
    QUIET = "quiet"  # say nothing: an installed bundle, which whoever runs it can't edit


def declaring_for(root: Path, *, installed: bool, dry_run: bool) -> Declaring:
    """How a command run on the bundle at ``root`` declares its plugins: nothing for an ``installed`` bundle, a hint
    for a folder no file can be created in, and otherwise the file, or in a ``dry_run`` what it would hold."""
    if installed:
        return Declaring.QUIET
    if not os.access(root, os.W_OK | os.X_OK):
        return Declaring.HINT
    return Declaring.WOULD_WRITE if dry_run else Declaring.WRITE


@dataclass(frozen=True)
class Provider:
    """What registers a type core doesn't ship: an installed distribution, at its version, or, with neither, the
    configuration's impls, then with the installed package its class's module comes from, when one ships it."""

    distribution: str | None
    version: str | None
    shipped_by: str | None = None  # "<package> <version>"

    def requirement(self) -> str | None:
        """The pip requirement for it, at least the version installed here without a local label, which a bound can't
        carry, or any version when that isn't one pip compares; None for configuration."""
        if not self.distribution:
            return None
        try:
            return f"{self.distribution}>={Version(self.version).public}"
        except (InvalidVersion, TypeError):
            return self.distribution


@dataclass(frozen=True)
class Use:
    """A type a bundle uses that core doesn't ship: where, what kind of type, its name, and what provides it."""

    where: str
    what: str  # step type, env type or artifact type
    name: str
    provider: Provider


class Providers:
    """What provides each type core doesn't ship, read once from the installed plugins' metadata."""

    def __init__(self) -> None:
        self._origins: dict[tuple[str, str], tuple[str, str, str]] | None = None
        self._packages: Mapping[str, list[str]] | None = None

    def of(self, group: str, name: str, cls: type | None) -> Provider | None:
        """What provides ``name`` of the plugin ``group``, which resolves here to ``cls``: the distribution whose entry
        point names that class, or configuration; None when core ships it. A plugin the registry build skipped
        registers nothing, so it isn't loaded again to compare."""
        if name in _BUILTINS[group]():
            return None
        if self._origins is None:
            self._origins = {(contribution.group, contribution.name): (distribution.name, distribution.version,
                                                                      contribution.value)
                             for distribution in inventory(load=False).distributions
                             for contribution in distribution.contributions}
        distribution, version, value = self._origins.get((group, name), (None, None, None))
        registered = value is not None and cls is not None and name not in load_failures().get(group, {})
        if registered and _names(value, cls):
            return Provider(distribution, version)
        return Provider(None, None, self._shipping(cls))

    def _shipping(self, cls: type | None) -> str | None:
        """The installed package whose files hold ``cls``'s module, at its version; None when no one package does, as
        for a module on the path or an editable install."""
        if cls is None:
            return None
        if self._packages is None:
            self._packages = packages_distributions()
        module = cls.__module__
        names = list(dict.fromkeys(self._packages.get(module.partition(".")[0], ())))
        if len(names) > 1:  # a namespace package: the one whose files hold the module
            files = {f"{module.replace('.', '/')}.py", f"{module.replace('.', '/')}/__init__.py"}
            names = [name for name in names if files & {str(file) for file in distribution(name).files or ()}]
        if len(names) != 1:
            return None
        version = _requirements.installed(canonicalize_name(names[0]))
        return f"{names[0]} {version}" if version else names[0]


def uses(resolved: ResolvedBundle, providers: Providers | None = None) -> list[Use]:
    """Each type the bundle's own tasks, envs and artifacts use that core doesn't ship."""
    providers = providers or Providers()
    return [use for entry in resolved.entries for use in entry_uses(resolved.bundle.root, entry, providers)]


def entry_uses(root: Path, entry: ResolvedEntry, providers: Providers) -> list[Use]:
    """Each type ``entry``, of the bundle at ``root``, uses that core doesn't ship."""
    where = relative(root, entry.entry.path)
    kind = entry.entry.kind
    found = []
    if kind is BundleKind.TASK:
        steps = get_task_step_registry()
        for step in entry.config:
            step_type = step.get("type")
            if isinstance(step_type, str) and (provider := providers.of(TASK_STEPS, step_type,
                                                                        steps.get(step_type))) is not None:
                found.append(Use(f"{where}: step {step.get('id')!r}", "step type", step_type, provider))
    elif kind is BundleKind.ENV and (provider := providers.of(ENVS, entry.entry.type,
                                                              get_env_registry().get(entry.entry.type))) is not None:
        found.append(Use(where, "env type", entry.entry.type, provider))
    elif kind in (BundleKind.ARTIFACT, BundleKind.SKILL):
        name = canonical_type(entry.entry.type)
        if (provider := providers.of(ARTIFACTS, name, get_artifact_registry().get(name))) is not None:
            found.append(Use(where, "artifact type", name, provider))
    return found


def detected(found: Iterable[Use], declared: Iterable[str] = ()) -> list[str]:
    """The requirement for each distinct plugin distribution ``found`` names: as ``declared`` writes it when it names
    that distribution, else at least the version installed here."""
    by_name: dict[str, list[str]] = {}
    for line in declared:
        if (requirement := _requirements.parsed(line)) is not None:
            by_name.setdefault(canonicalize_name(requirement.name), []).append(line)
    requires: set[str] = set()
    for use in found:
        if (requirement := use.provider.requirement()) is not None:
            requires.update(by_name.get(canonicalize_name(use.provider.distribution), [requirement]))
    return sorted(requires)


def task_requires(steps: Iterable[TaskStep], declared: Iterable[str] = ()) -> list[str]:
    """What a stored task made of ``steps`` requires, by the rule registering a bundle's task records: the plugin
    packages behind its own step types and the types of the envs and artifacts its steps read, as the store holds them,
    with those they hold. A package ``declared`` (what the task required before) names keeps its range there; one used
    anew is at least the version installed here; one no longer used is dropped, unless a reference can't be read here
    (an id the run fills in, one the store doesn't hold, a type nothing here registers), when every one is kept."""
    declared = list(declared)
    providers = Providers()
    found, read, unread = [], set(), False
    for step in steps:
        if (provider := providers.of(TASK_STEPS, step.type, get_task_step_registry().get(step.type))) is not None:
            found.append(Use(f"step {step.id!r}", "step type", step.type, provider))
        for site in ref_sites(type(step).entity_refs or (), step.to_dict()):
            if site.ref.role is RefRole.OUTPUT or site.ref.kind not in _GETTERS:
                continue
            key = (site.ref.kind, site.value, site.version if isinstance(site.version, int) else None)
            if key in read:
                continue
            read.add(key)
            try:
                entity = _GETTERS[site.ref.kind](key[1], key[2]) if isinstance(key[1], str) else None
            except (NotFoundError, UnknownDocumentTypeError, ValueError):
                entity = None
            if entity is None:
                unread = True
                continue
            found += [use for held in _held(entity) if (use := _entity_use(f"step {step.id!r}: {site.path}", held,
                                                                           providers)) is not None]
    requires = detected(found, declared)
    if unread:
        requires = sorted({*requires, *(line for line in declared if _requirements.parsed(line) is not None)})
    return requires



def _held(entity: Env | Artifact) -> Iterator[Env | Artifact]:
    """``entity``, then each env and artifact loading it loaded with it, and theirs in turn."""
    seen, queue = set(), [entity]
    while queue:
        held = queue.pop()
        if id(held) in seen:
            continue
        seen.add(id(held))
        yield held
        for value in vars(held).values():
            items = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else (value,)
            queue.extend(item for item in items if isinstance(item, (Env, Artifact)))


def _entity_use(where: str, entity: Env | Artifact, providers: Providers) -> Use | None:
    """The use of ``entity``'s type, when core doesn't ship it."""
    if isinstance(entity, Env):
        provider = providers.of(ENVS, entity.type, get_env_registry().get(entity.type))
        return Use(where, "env type", entity.type, provider) if provider is not None else None
    name = canonical_type(entity.type)
    provider = providers.of(ARTIFACTS, name, get_artifact_registry().get(name))
    return Use(where, "artifact type", name, provider) if provider is not None else None


def declare(resolved: ResolvedBundle, declaring: Declaring) -> list[str]:
    """What to say of the plugin packages the bundle uses that its ``bundle.toml`` doesn't require, and of each type
    only this machine's configuration registers, as ``declaring`` says. Writing, a bundle with no ``bundle.toml`` is
    given one listing them, created whole or not at all and never over one that appeared meanwhile. Otherwise it's
    told what to declare: an existing ``bundle.toml`` is never edited."""
    if declaring is Declaring.QUIET:
        return []
    found = uses(resolved)
    declared = {canonicalize_name(Requirement(line).name) for line in resolved.bundle.requires or ()}
    undeclared = [use for use in found if use.provider.distribution is not None
                  and canonicalize_name(use.provider.distribution) not in declared]
    lines = _configured([use for use in found if use.provider.distribution is None])
    if not undeclared:
        return lines
    requires = detected(undeclared)
    path = resolved.bundle.root / BUNDLE_TOML
    if resolved.bundle.requires is None and declaring is Declaring.WOULD_WRITE:
        return [f"would write {BUNDLE_TOML}: requires = {json.dumps(requires)}", *lines]
    if resolved.bundle.requires is None and declaring is Declaring.WRITE:
        if _create(path, _toml.dumps({"requires": requires})):
            return [f"wrote {BUNDLE_TOML}: requires = {json.dumps(requires)}; commit it so whoever you share the "
                    "bundle with knows what to install", *lines]
    by_distribution: dict[str, list[Use]] = {}
    for use in undeclared:
        by_distribution.setdefault(use.provider.distribution, []).append(use)
    hints = [f"this bundle uses {distribution} {uses_of[0].provider.version} ({_wheres(uses_of)}), which "
             f"{BUNDLE_TOML} doesn't require" for distribution, uses_of in sorted(by_distribution.items())]
    everything = sorted({*(resolved.bundle.requires or ()), *requires})
    hints.append(f"declare them in {BUNDLE_TOML}, so whoever it's shared with knows what to install: "
                 f"requires = {json.dumps(everything)}")
    return [*hints, *lines]


def _create(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` whole, unless something is there by then; whether it was written. It's staged beside
    it under a name of its own, as the umask leaves new files, and linked into place, which fails if ``path`` exists;
    on a filesystem without links, or a full one, nothing is written."""
    staged = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        file = open(staged, "x", encoding="utf-8")
    except OSError:
        return False
    try:
        with file:
            file.write(text)
        os.link(staged, path)
        return True
    except OSError:
        return False
    finally:
        staged.unlink(missing_ok=True)


def _configured(found: Sequence[Use]) -> list[str]:
    """A line for the types ``found`` that only this machine's configuration registers: one per installed package
    their classes come from, then one for those no package ships."""
    by_package: dict[str | None, list[Use]] = {}
    for use in found:
        by_package.setdefault(use.provider.shipped_by, []).append(use)
    lines = [f"this bundle uses {_listing(uses_of)} ({_wheres(uses_of)}), which this machine's configuration registers, "
             f"from {package}; whoever runs it needs that package and its configuration"
             for package, uses_of in sorted(by_package.items(), key=lambda item: item[0] or "") if package]
    if None in by_package:
        lines.append(f"this bundle uses {_listing(by_package[None])} ({_wheres(by_package[None])}), which this "
                     "machine's configuration registers; whoever runs it needs that configuration too")
    return lines


def _listing(found: Sequence[Use]) -> str:
    """Each distinct type ``found`` names, once: ``the step type 'a' and the env type 'b'``."""
    named = list(dict.fromkeys(f"the {use.what} {use.name!r}" for use in found))
    return named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"


def _wheres(found: Sequence[Use]) -> str:
    more = len(found) - 1
    return found[0].where + (f", and {more} more" if more else "")


def _names(value: str, cls: type) -> bool:
    """Whether the entry point ``value``, ``module:attribute``, names ``cls``."""
    module, _, attribute = value.partition(":")
    try:
        named = importlib.import_module(module)
        for part in attribute.split("."):
            named = getattr(named, part)
    except (ImportError, AttributeError):
        return False
    return named is cls
