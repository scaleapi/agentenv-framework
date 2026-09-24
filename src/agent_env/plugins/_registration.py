"""Loading a group's plugins into its registry, and recording each one that did not take effect.

Internal: the public contract is ``agent_env.plugins`` and the README section "Register types
from an installed package". Each registry calls ``merge`` when it is built, so a group's plugins
are imported then and not before.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_env.config import runtime
from agent_env.config.errors import ConfigError
from agent_env.plugins._discovery import Plugin, discover

if TYPE_CHECKING:
    from agent_env.config.runtime import Config, PluginFailures

logger = logging.getLogger(__name__)

ENVS = "agent_env.envs"
ARTIFACTS = "agent_env.artifacts"
TASK_STEPS = "agent_env.task_steps"
SANDBOX_PROVIDERS = "agent_env.sandbox_providers"
STATE_PROVIDERS = "agent_env.state_providers"
EXPLORER_PLUGINS = "agent_env.explorer_plugins"


class PluginConflictError(ConfigError):
    """Two installed distributions register the same name in one group."""


PluginConflictError.__module__ = "agent_env.plugins"


@dataclass
class Registrations:
    """The plugin registrations ``merge`` made in one registry, while that registry is built.

    Only names a plugin still registers remain: config that takes a name over ``release``s it,
    and a check that fails later ``reject``s it.
    """

    registry: dict[str, Any]
    group: str
    added: dict[str, Plugin]
    classes: dict[str, type]
    failures: dict[str, str]  # this group's entry in the Config's PluginFailures
    # Config that took a plugin's name over: name -> (where, the class config registered).
    released: dict[str, tuple[str, type]] = field(default_factory=dict)
    # Names two distributions claim, recorded instead of raised only for an InventoryProbe, in
    # merge's order: the first is the conflict a real build raises.
    conflicts: dict[str, tuple[Plugin, ...]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> Registrations:
        return cls(registry={}, group="", added={}, classes={}, failures={})

    def __iter__(self) -> Iterator[str]:
        return iter(list(self.added))

    def plugin(self, name: str) -> Plugin | None:
        """The plugin that registers ``name``, if one still does."""
        return self.added.get(name)

    def failed(self, name: str) -> bool:
        """Whether a plugin registering ``name`` was skipped."""
        return name in self.failures

    def release(self, name: str, where: str, replacement: type) -> bool:
        """Config at ``where`` takes ``name`` over. Returns whether a plugin registered it; warns
        unless ``replacement`` is the plugin's own class."""
        plugin = self.added.pop(name, None)
        if plugin is None:
            return False
        self.released[name] = (where, replacement)
        if self.classes[name] is not replacement:
            logger.warning("%s replaces the class registered by plugin %s", where, plugin)
        return True

    def reject(self, name: str, reason: str) -> None:
        """Drop ``name``'s registration, recording ``reason`` as for a load failure."""
        plugin = self.added.pop(name)
        del self.registry[name]
        self.failures[name] = _reason(plugin, reason)
        logger.warning("Plugin %s in %s was skipped: %s", plugin, self.group, reason)


def merge(
    registry: dict[str, Any],
    group: str,
    validate: Callable[[str, Any], type],
    *,
    source: Config | None = None,
    entry: Callable[[type], Any] = lambda cls: cls,
) -> Registrations:
    """Add ``group``'s plugins to ``registry``, which holds only the built-ins so far.

    ``validate(name, loaded)`` returns the class to register or raises to reject it; ``entry``
    turns it into the registry's value. Every skipped plugin is recorded on ``source``, the
    Config the registry is being built for. An ``InventoryProbe`` records a conflict instead of
    raising it, and keeps what this build registered.
    """
    added: dict[str, Plugin] = {}
    classes: dict[str, type] = {}
    failures: dict[str, str] = {}
    conflicts: dict[str, tuple[Plugin, ...]] = {}
    probe = source if isinstance(source, InventoryProbe) else None
    for name, claims in discover(group).items():
        if name in registry:
            claimants = "; ".join(str(p) for p, _ in claims)
            failures[name] = _reason(claimants, "clashes with a built-in")
            logger.warning("Plugin %s in %s clashes with a built-in and was skipped", claimants, group)
            continue
        if len(claims) > 1:
            if probe is None:
                raise PluginConflictError(conflict_message(name, group, [p for p, _ in claims]))
            conflicts[name] = tuple(p for p, _ in claims)
            # Recorded as failed too, so config naming the name sees what a real build never reaches.
            failures[name] = _reason("; ".join(str(p) for p, _ in claims), "is claimed more than once")
            continue
        plugin, ep = claims[0]
        try:
            cls = validate(name, ep.load())
        except Exception as exc:
            failures[name] = _reason(plugin, f"failed to load: {exc!r}")
            logger.warning("Plugin %s in %s failed to load and was skipped: %r", plugin, group, exc)
            continue
        registry[name] = entry(cls)
        added[name] = plugin
        classes[name] = cls
    if not runtime.in_nested_registry_build():
        _record(source).replace(group, failures)
    registrations = Registrations(
        registry=registry, group=group, added=added, classes=classes, failures=failures, conflicts=conflicts
    )
    if probe is not None:
        probe.registrations[group] = registrations
    return registrations


@dataclass(eq=False)
class InventoryProbe(runtime.Config):
    """The throwaway Config ``agent_env.plugins.inventory()`` builds registries on.

    ``merge`` records a conflict on it instead of raising, and keeps each group's Registrations.
    The mode belongs to this object rather than the call stack, so a registry that a plugin builds
    on the process Config while being imported stays strict.
    """

    registrations: dict[str, Registrations] = field(default_factory=dict, repr=False)


def conflict_message(name: str, group: str, claimants: list[Plugin]) -> str:
    """The ``PluginConflictError`` text for ``name``, shared with the inventory's report."""
    return (
        f"{len(claimants)} installed plugins register {name!r} in {group}: "
        + "; ".join(str(p) for p in claimants)
        + ". Uninstall all but one; `agent-env plugin list` shows every claim."
    )


def typed_validator(base: type, builtins: Collection[str] = ()) -> Callable[[str, Any], type]:
    """A ``merge`` validator for a group whose classes carry a ``type`` ClassVar: a subclass of
    ``base`` with its own ``type``, registered under exactly that name, since documents are
    written and read by it."""

    def validate(name: str, loaded: Any) -> type:
        cls = require_subclass(loaded, base)
        if cls.type == base.type:
            raise TypeError(f"{cls.__qualname__} does not define its own 'type'")
        if cls.type in builtins:
            raise TypeError(f"{cls.__qualname__} inherits the built-in type {cls.type!r}; give it its own type")
        if cls.type != name:
            raise TypeError(f"{cls.__qualname__} has type {cls.type!r}; its entry point must be named {cls.type!r}")
        return cls

    return validate


def require_subclass(loaded: Any, base: type) -> type:
    if not (isinstance(loaded, type) and issubclass(loaded, base)):
        raise TypeError(f"{loaded!r} is not a subclass of {base.__name__}")
    return loaded


def load_failures(config: Config | None = None) -> dict[str, dict[str, str]]:
    """Every plugin that did not take effect, by group then name, with the reason.

    A plugin is listed when it failed to load, validate or construct, or clashed with a built-in.
    Covers the registries ``config`` (default: the process Config) has built so far, so build
    them first; a deployment that ships its plugins can then assert this is empty at startup.
    ``reset_config()`` starts a new Config, and so a new record.
    """
    return _record(config).snapshot()


def failure_note(group: str, name: str) -> str:
    """``" (…)"`` saying why ``name`` is missing from ``group``'s registry, or ``""``."""
    reason = _record(None).reason(group, name)
    return f" ({reason})" if reason else ""


def _record(config: Config | None) -> PluginFailures:
    return (config or runtime.get_config())._plugin_failures


def _reason(registered_by: object, why: str) -> str:
    return f"registered by {registered_by} but {why}"
