"""``agent_env.plugins.inventory``: every installed plugin's contributions, and what became of each.

Internal: the public names are re-exported from ``agent_env.plugins``. The registries are built on
a throwaway Config that reads the caller's document, so the Config in use keeps its registries and
its ``load_failures()``, and a conflict is reported rather than raised.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Optional

from agent_env.config import runtime
from agent_env.config import snapshot as config_snapshot
from agent_env.config.errors import ConfigError
from agent_env.plugins import _discovery, _registration

Status = Literal["active", "replaced", "failed", "skipped", "conflict", "blocked", "unloaded"]

# Statuses a group-level error does not override: each already says why the name is missing.
_SETTLED = frozenset({"failed", "skipped", "conflict"})


@dataclass(frozen=True)
class Contribution:
    """One entry point a distribution declares, and what became of it."""

    group: str
    name: str
    value: str
    status: Status
    reason: Optional[str] = None
    replaced_in: Optional[str] = None
    replacement: Optional[str] = None
    conflicts_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class Distribution:
    """An installed distribution that declares entry points in the type groups."""

    name: str
    version: str
    contributions: tuple[Contribution, ...]


@dataclass(frozen=True)
class Inventory:
    """Every installed plugin distribution, and the error each group's real build would raise."""

    distributions: tuple[Distribution, ...]
    # With load=False, only what metadata proves: a conflict, or that no config file was found.
    group_errors: Mapping[str, str]
    # Groups whose installed entry points could not be read, so none of their plugins is listed.
    discovery_errors: Mapping[str, str]
    config_path: Optional[Path]
    config_error: Optional[str]
    loaded: bool


for _public in (Contribution, Distribution, Inventory):
    _public.__module__ = "agent_env.plugins"

def _groups() -> dict[str, tuple[Callable[[], Any], Callable[[runtime.Config], Any]]]:
    """Per group: its built-in names, and how the probe builds it."""
    # Imported here, not at the top: every registry imports agent_env.plugins, and so this module.
    from agent_env.artifact.registry import ARTIFACT_REGISTRY
    from agent_env.env import registry as env_registry
    from agent_env.explorer.plugin import load_plugins
    from agent_env.providers.sandbox_provider import _BUILTIN_SANDBOX_PROVIDERS
    from agent_env.providers.state.env_state_provider import _BUILTIN_STATE_PROVIDERS
    from agent_env.task_step import registry as task_step_registry

    return {
        _registration.ENVS: (env_registry._builtin_registry, lambda probe: probe.env_registry()),
        _registration.ARTIFACTS: (lambda: ARTIFACT_REGISTRY, lambda probe: probe.artifact_registry()),
        _registration.TASK_STEPS: (task_step_registry._builtin_registry, lambda probe: probe.task_step_registry()),
        _registration.SANDBOX_PROVIDERS: (lambda: _BUILTIN_SANDBOX_PROVIDERS, lambda probe: probe.sandbox_registry()),
        _registration.STATE_PROVIDERS: (lambda: _BUILTIN_STATE_PROVIDERS, lambda probe: probe.state_registry()),
        _registration.EXPLORER_PLUGINS: (dict, lambda probe: load_plugins(source=probe)),
    }


def inventory(config: Optional[runtime.Config] = None, *, load: bool = True) -> Inventory:
    """Every installed distribution declaring entry points in the type groups, with a status per
    contribution.

    ``load=False`` reads installed metadata only: no plugin code runs, and a status that needs a
    build (``active``, ``replaced``, ``failed``) is reported as ``unloaded``. ``load=True`` imports
    the plugins, constructs the explorer plugins, and builds each group on a throwaway Config that
    reads ``config``'s document (default: the process Config). ``config`` keeps its registries and
    its ``load_failures()``; like any registry build, this pins its document first if nothing has
    yet. Importing or constructing a plugin can still have process-wide effects, such as a package
    that sets ``AGENT_ENV_CONFIG`` on import.
    """
    base = config or runtime.get_config()
    try:
        document = base._document()
        missing = None
    except ConfigError as exc:
        # No config file was found: every real build raises this before merging anything, so the
        # plugins are checked against an empty document and then blocked.
        document, missing = config_snapshot.Snapshot(path=None, _document={}), exc
    unreadable = missing or document.error
    config_error = None if unreadable is None else str(unreadable)
    # A malformed document is used as it is, so each build raises its parse error where a real one does.
    probe = _registration.InventoryProbe(_snapshot=document) if load else None
    group_errors: dict[str, str] = {}
    discovery_errors: dict[str, str] = {}
    by_distribution: dict[tuple[str, str], list[Contribution]] = {}
    groups = _groups()
    for group, (builtins_of, build) in groups.items():
        try:
            claims = _discovery.claims(group)
        except Exception as exc:
            discovery_errors[group] = _discovery.discovery_error(exc)
            continue
        if not claims:
            continue
        builtins = frozenset(builtins_of())
        registrations = None
        if probe is not None:
            try:
                build(probe)
            except Exception as exc:
                group_errors[group] = f"{type(exc).__name__}: {exc}"
            registrations = probe.registrations.get(group)
            if registrations is not None and registrations.conflicts:
                # A real build raises the first conflict merge met, before anything the probe raised later.
                first, claimants = next(iter(registrations.conflicts.items()))
                group_errors[group] = _registration.conflict_message(first, group, list(claimants))
        else:
            conflicted = [name for name, found in claims.items() if name not in builtins and len(found) > 1]
            if conflicted:
                first = conflicted[0]
                group_errors[group] = _registration.conflict_message(first, group, [p for p, _ in claims[first]])
        if missing is not None:
            group_errors[group] = f"{type(missing).__name__}: {missing}"
        for name, found in claims.items():
            for plugin, ep in found:
                contribution = _classify(group, name, plugin, found, builtins, registrations, document.path)
                if group in group_errors and contribution.status not in _SETTLED:
                    contribution = Contribution(
                        group, name, plugin.value, "blocked", reason=group_errors[group]
                    )
                elif load and contribution.status == "unloaded":
                    # Unreachable while every registry records what it did; visible, not silent, if not.
                    contribution = replace(contribution, reason="its status could not be determined")
                by_distribution.setdefault(_identity(plugin, ep), []).append(contribution)
    order = list(groups)
    distributions = tuple(
        Distribution(dist, version, tuple(sorted(found, key=lambda c: (order.index(c.group), c.name))))
        for (dist, version), found in sorted(by_distribution.items(), key=lambda item: (item[0][0].lower(), item[0][1]))
    )
    return Inventory(
        distributions=distributions,
        group_errors=MappingProxyType(group_errors),
        discovery_errors=MappingProxyType(discovery_errors),
        config_path=document.path,
        config_error=config_error,
        loaded=load,
    )


def _classify(
    group: str,
    name: str,
    plugin: _discovery.Plugin,
    found: list[tuple[_discovery.Plugin, Any]],
    builtins: frozenset[str],
    registrations: Optional[_registration.Registrations],
    config_path: Optional[Path],
) -> Contribution:
    # Same precedence as merge: a built-in name is skipped for every claimant, before conflicts.
    if name in builtins:
        return Contribution(group, name, plugin.value, "skipped", reason="clashes with a built-in")
    if len(found) > 1:
        others = tuple(str(other) for other, _ in found if other != plugin)
        alone = all((other.dist, other.version) == (plugin.dist, plugin.version) for other, _ in found)
        reason = "this package declares the name more than once" if alone else "another installed package registers this name"
        return Contribution(group, name, plugin.value, "conflict", reason=reason, conflicts_with=others)
    if registrations is None:
        return Contribution(group, name, plugin.value, "unloaded")
    if name in registrations.added:
        return Contribution(group, name, plugin.value, "active")
    if name in registrations.released:
        where, replacement = registrations.released[name]
        if replacement is registrations.classes.get(name):
            return Contribution(group, name, plugin.value, "active")
        return Contribution(
            group, name, plugin.value, "replaced",
            replaced_in=f"{where} in {config_path}" if config_path else where,
            replacement=f"{replacement.__module__}:{replacement.__qualname__}",
        )
    if name in registrations.failures:
        reason = registrations.failures[name].removeprefix(f"registered by {plugin} but ")
        return Contribution(group, name, plugin.value, "failed", reason=reason)
    return Contribution(group, name, plugin.value, "unloaded")


def _identity(plugin: _discovery.Plugin, ep: Any) -> tuple[str, str]:
    """The distribution a contribution is grouped under; one with unreadable metadata is told
    apart by where it is installed, so two such packages are not merged."""
    if plugin.dist:
        return plugin.dist, plugin.version
    where = getattr(getattr(ep, "dist", None), "_path", None)
    return f"(unreadable metadata: {Path(where).name})" if where else "(unreadable metadata)", plugin.version
