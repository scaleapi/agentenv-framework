"""What a bundle needs beyond core: the plugin packages its optional ``bundle.toml`` requires, checked against what's
installed before the bundle resolves, and who provides each type it uses that core doesn't ship, which a bundle with no
``bundle.toml`` is given one from."""

from __future__ import annotations

import importlib
import json
import os
import shlex
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from agent_env.artifact.registry import ARTIFACT_REGISTRY, canonical_type, get_artifact_registry
from agent_env.env import registry as env_registry
from agent_env.env.registry import get_env_registry
from agent_env.plugins import ARTIFACTS, ENVS, TASK_STEPS, inventory, load_failures
from agent_env.plugins._requirements import installed
from agent_env.task_step import registry as task_step_registry
from agent_env.task_step.registry import get_task_step_registry

from . import _toml
from ._fs import relative
from .parse import BUNDLE_TOML, BundleError, BundleKind

if TYPE_CHECKING:
    from .resolve import ResolvedBundle

# What core ships, per plugin group: anything else a bundle uses is provided by a plugin, or by configuration.
_BUILTINS = {TASK_STEPS: task_step_registry._builtin_registry, ENVS: env_registry._builtin_registry,
             ARTIFACTS: lambda: ARTIFACT_REGISTRY}


class UnmetRequirements(BundleError):
    """A bundle whose ``bundle.toml`` requires packages that aren't installed, or not at a version it allows, with
    every other problem resolving it found; ``requirements`` are the unmet ones, as it writes them."""

    def __init__(self, problems: Iterable[str], requirements: Sequence[str]):
        super().__init__(problems)
        self.requirements = tuple(requirements)


@dataclass(frozen=True)
class Provider:
    """What registers a type core doesn't ship: an installed distribution, at its version, or, with neither, the
    configuration's impls."""

    distribution: str | None
    version: str | None

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
        return Provider(distribution, version) if registered and _names(value, cls) else Provider(None, None)


def unmet(requires: Iterable[str]) -> list[tuple[str, str]]:
    """Each of ``requires`` that isn't met here, with why: its package isn't installed, is at a version it excludes, or
    is without what the extras it names need. One whose marker is false here doesn't apply."""
    found = []
    for line in requires:
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        have = installed(canonicalize_name(requirement.name))
        if have is None:
            found.append((line, "which isn't installed"))
        elif not _allows(requirement, have):
            found.append((line, f"and {requirement.name} {have} is installed"))
        elif missing := _missing_for_extras(requirement, set()):
            found.append((line, f"and {requirement.name} {have} is installed without {', '.join(missing)}, which its "
                                f"extras need"))
    return found


def _allows(requirement: Requirement, version: str) -> bool:
    """Whether ``requirement`` admits ``version``; one pip can't compare it admits only when it names no versions."""
    try:
        return requirement.specifier.contains(version, prereleases=True)
    except InvalidVersion:
        return not requirement.specifier


def _missing_for_extras(requirement: Requirement, seen: set[tuple[str, str]]) -> list[str]:
    """What the extras ``requirement`` names need of its installed package that isn't installed, or not at a version
    it allows, following an extra that names another of the package's own."""
    name = canonicalize_name(requirement.name)
    missing = []
    for extra in sorted(requirement.extras):
        if (name, extra) in seen:
            continue
        seen.add((name, extra))
        try:
            lines = distribution(name).requires or ()
        except PackageNotFoundError:
            return []
        for line in lines:
            try:
                needed = Requirement(line)
            except InvalidRequirement:
                continue
            if needed.marker is None or not needed.marker.evaluate({"extra": extra}) \
                    or needed.marker.evaluate({"extra": ""}):
                continue
            have = installed(canonicalize_name(needed.name))
            if have is None or not _allows(needed, have):
                missing.append(f"{needed.name}{needed.specifier}")
            else:
                missing.extend(_missing_for_extras(needed, seen))
    return list(dict.fromkeys(missing))


def unmet_problems(requires: Iterable[str]) -> tuple[list[str], list[str]]:
    """The problems ``bundle.toml``'s unmet ``requires`` raise, naming the one command that installs them all, and the
    unmet requirements."""
    found = unmet(requires)
    if not found:
        return [], []
    lines = [line for line, _ in found]
    command = install_command(lines)
    if len(found) == 1:
        line, why = found[0]
        return [f"{BUNDLE_TOML}: requires {line!r}, {why}; install it with: {command}"], lines
    return ([f"{BUNDLE_TOML}: requires {line!r}, {why}" for line, why in found]
            + [f"{BUNDLE_TOML}: install what it requires with: {command}"]), lines


def install_command(requirements: Sequence[str]) -> str:
    """The command that installs ``requirements`` into this environment, through the installer that owns it."""
    return shlex.join(["agent-env", "plugin", "add", *requirements])


def uses(resolved: ResolvedBundle, providers: Providers | None = None) -> list[Use]:
    """Each type the bundle's own tasks, envs and artifacts use that core doesn't ship."""
    providers = providers or Providers()
    steps, envs, artifacts = get_task_step_registry(), get_env_registry(), get_artifact_registry()
    found = []
    for entry in resolved.entries:
        where = relative(resolved.bundle.root, entry.entry.path)
        kind = entry.entry.kind
        if kind is BundleKind.TASK:
            for step in entry.config:
                step_type = step.get("type")
                if isinstance(step_type, str) and (provider := providers.of(TASK_STEPS, step_type,
                                                                            steps.get(step_type))) is not None:
                    found.append(Use(f"{where}: step {step.get('id')!r}", "step type", step_type, provider))
        elif kind is BundleKind.ENV and (provider := providers.of(ENVS, entry.entry.type,
                                                                  envs.get(entry.entry.type))) is not None:
            found.append(Use(where, "env type", entry.entry.type, provider))
        elif kind in (BundleKind.ARTIFACT, BundleKind.SKILL):
            name = canonical_type(entry.entry.type)
            if (provider := providers.of(ARTIFACTS, name, artifacts.get(name))) is not None:
                found.append(Use(where, "artifact type", name, provider))
    return found


def detected(found: Iterable[Use]) -> list[str]:
    """The requirement for each distinct plugin distribution ``found`` names, at least the version installed here."""
    return sorted({requirement for use in found if (requirement := use.provider.requirement()) is not None})


def declare(resolved: ResolvedBundle, *, own: bool, dry_run: bool) -> list[str]:
    """What to say of the plugin packages the bundle uses that its ``bundle.toml`` doesn't require, and of each type
    only this machine's configuration registers. A bundle with no ``bundle.toml`` in a folder of its ``own`` (run by
    path, and writable) is given one listing them, created whole or not at all and never over one that appeared
    meanwhile; a ``dry_run`` says what it would write instead. Any other bundle is told what to declare: an existing
    ``bundle.toml`` is never edited."""
    found = uses(resolved)
    declared = {canonicalize_name(Requirement(line).name) for line in resolved.bundle.requires or ()}
    undeclared = [use for use in found if use.provider.distribution is not None
                  and canonicalize_name(use.provider.distribution) not in declared]
    configured = [use for use in found if use.provider.distribution is None]
    lines = [f"this bundle uses {_listing(configured)} ({_wheres(configured)}), which this machine's configuration "
             "registers; whoever runs it needs that configuration too"] if configured else []
    if not undeclared:
        return lines
    requires = detected(undeclared)
    path = resolved.bundle.root / BUNDLE_TOML
    if resolved.bundle.requires is None and own:
        if dry_run:
            return [f"would write {BUNDLE_TOML}: requires = {json.dumps(requires)}", *lines]
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


def writable(root: Path) -> bool:
    """Whether a file can be created in the folder ``root``."""
    return os.access(root, os.W_OK | os.X_OK)


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
