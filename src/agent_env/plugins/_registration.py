"""Loading a group's plugins into its registry, and recording each one that did not take effect.

Internal: the public contract is ``agent_env.plugins`` and the PLUGINS.md section "Register types
from an installed package". Each registry calls ``merge`` when it is built, so a group's plugins
are imported then and not before.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_env.config import runtime
from agent_env.plugins import _report, _requirements
from agent_env.plugins._discovery import Plugin, discover

if TYPE_CHECKING:
    from agent_env.config.runtime import Config, PluginFailures

logger = logging.getLogger(__name__)

ENVS = "agent_env.envs"
ARTIFACTS = "agent_env.artifacts"
TASK_STEPS = "agent_env.task_steps"
SANDBOX_PROVIDERS = "agent_env.sandbox_providers"
STATE_PROVIDERS = "agent_env.state_providers"
ENV_PROVIDERS = "agent_env.env_providers"
EXPLORER_PLUGINS = "agent_env.explorer_plugins"


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
    # The code for each entry in ``failures``.
    codes: dict[str, str] = field(default_factory=dict)
    # Config that took a plugin's name over: name -> (config table, impl as written, the class).
    released: dict[str, tuple[str, str, type]] = field(default_factory=dict)
    # Names more than one entry point claims: left out of the registry, and config cannot take them.
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

    def release(self, name: str, table: str, impl: str, replacement: type) -> bool:
        """Config's ``[table]`` takes ``name`` over with ``impl``, which loaded as ``replacement``.
        Returns whether a plugin registered it; warns unless ``replacement`` is the plugin's own class."""
        plugin = self.added.pop(name, None)
        if plugin is None:
            return False
        self.released[name] = (table, impl, replacement)
        if self.classes[name] is not replacement:
            logger.warning("[%s] impl %r replaces the class registered by plugin %s", table, impl, plugin)
        return True

    def refuse(self, name: str, where: str) -> bool:
        """Whether ``name`` is conflicted, so the config at ``where`` naming it is skipped, with a warning."""
        if name not in self.conflicts:
            return False
        logger.warning("%s names %r and was skipped: %s", where, name, self.failures[name])
        return True

    def reject(self, name: str, code: str, reason: str) -> None:
        """Drop ``name``'s registration, recording ``reason`` as for a load failure."""
        plugin = self.added.pop(name)
        del self.registry[name]
        self.failures[name] = _reason(plugin, reason)
        self.codes[name] = code
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
    Config the registry is being built for. A name more than one entry point claims is left out,
    and every other name still loads. An ``InventoryProbe`` keeps what this build registered.
    """
    added: dict[str, Plugin] = {}
    classes: dict[str, type] = {}
    failures: dict[str, str] = {}
    codes: dict[str, str] = {}
    conflicts: dict[str, tuple[Plugin, ...]] = {}
    probe = source if isinstance(source, InventoryProbe) else None
    for name, claims in discover(group).items():
        if name in registry:
            claimants = "; ".join(str(p) for p, _ in claims)
            failures[name] = _reason(claimants, "clashes with a built-in")
            codes[name] = _report.BUILTIN_NAME
            logger.warning("Plugin %s in %s clashes with a built-in and was skipped", claimants, group)
            continue
        # A claim whose requirements exclude this agent-env cannot load, so it claims nothing.
        usable, unmet_first = [], None
        for plugin, ep in claims:
            if (unmet := _requirements.incompatibility(getattr(ep, "dist", None))) is None:
                usable.append((plugin, ep))
                continue
            logger.warning("Plugin %s in %s was skipped: it %s", plugin, group, unmet)
            unmet_first = unmet_first or _reason(plugin, unmet)
        if not usable:
            failures[name] = unmet_first
            codes[name] = _report.INCOMPATIBLE_CORE
            continue
        if len(usable) > 1:
            conflicts[name] = tuple(p for p, _ in usable)
            failures[name] = conflict_message(name, group, list(conflicts[name]))
            codes[name] = _report.NAME_CONFLICT
            logger.warning("Plugin name %r in %s was skipped: %s", name, group, failures[name])
            continue
        plugin, ep = usable[0]
        code = _report.LOAD_FAILED
        try:
            loaded = ep.load()
            code = _report.INVALID_PLUGIN
            cls = validate(name, loaded)
        except (Exception, SystemExit) as exc:
            failures[name] = _reason(plugin, f"failed to load: {exc!r}")
            codes[name] = code
            logger.warning("Plugin %s in %s failed to load and was skipped: %r", plugin, group, exc)
            continue
        registry[name] = entry(cls)
        added[name] = plugin
        classes[name] = cls
    if not runtime.in_nested_registry_build():
        _record(source).replace(group, failures)
    registrations = Registrations(
        registry=registry, group=group, added=added, classes=classes, failures=failures, codes=codes,
        conflicts=conflicts,
    )
    if probe is not None:
        probe.registrations[group] = registrations
    return registrations


@dataclass(eq=False)
class InventoryProbe(runtime.Config):
    """The throwaway Config ``agent_env.plugins.inventory()`` builds registries on: ``merge`` keeps
    each group's Registrations on it."""

    registrations: dict[str, Registrations] = field(default_factory=dict, repr=False)


def conflict_message(name: str, group: str, claimants: list[Plugin]) -> str:
    """Why ``name`` is left out of ``group``: every claimant, and how to settle it."""
    listed = "; ".join(str(p) for p in claimants)
    first = claimants[0]
    if first.dist and all((p.dist, p.version) == (first.dist, first.version) for p in claimants):
        # Removing the package removes every claim, so only its author can settle it.
        return (
            f"{first.dist} {first.version} registers {name!r} in {group} {len(claimants)} times: {listed}. "
            "Report it to the package's author."
        )
    return (
        f"{len(claimants)} installed plugins register {name!r} in {group}: {listed}. "
        "Remove all but one with `agent-env plugin remove <package>`."
    )


def typed_validator(
    base: type, builtins: Collection[str] = (), required: Collection[str] = ()
) -> Callable[[str, Any], type]:
    """A ``merge`` validator for a group whose classes carry a ``type`` ClassVar: a subclass of
    ``base`` with its own ``type``, registered under exactly that name, since documents are
    written and read by it, and implementing what ``unimplemented`` checks."""

    def validate(name: str, loaded: Any) -> type:
        cls = require_subclass(loaded, base)
        if cls.type == base.type:
            raise TypeError(f"{cls.__qualname__} does not define its own 'type'")
        if cls.type in builtins:
            raise TypeError(f"{cls.__qualname__} inherits the built-in type {cls.type!r}; give it its own type")
        if cls.type != name:
            raise TypeError(f"{cls.__qualname__} has type {cls.type!r}; its entry point must be named {cls.type!r}")
        if problem := unimplemented(cls, base, required):
            raise TypeError(problem)
        return cls

    return validate


def require_subclass(loaded: Any, base: type) -> type:
    if not (isinstance(loaded, type) and issubclass(loaded, base)):
        raise TypeError(f"{loaded!r} is not a subclass of {base.__name__}")
    return loaded


def unimplemented(cls: type, base: type, required: Collection[str] = ()) -> str | None:
    """What ``cls`` still has to implement to work as a ``base``, if anything: its abstract methods,
    and each of ``required``, a method ``base`` defines only to raise, that it inherits unchanged or
    overrides with a plain function where ``base`` has a classmethod, which is called on the class."""
    missing = set(getattr(cls, "__abstractmethods__", ()))
    unbound = []
    for name in required:
        mine, theirs = inspect.getattr_static(cls, name), inspect.getattr_static(base, name)
        if _function(mine) is _function(theirs):
            missing.add(name)
        elif isinstance(theirs, classmethod) and not isinstance(mine, (classmethod, staticmethod)):
            unbound.append(name)
    problems = []
    if missing:
        names = sorted(missing)
        listed = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
        problems.append(f"{cls.__qualname__} must implement {listed}")
    problems += [f"{cls.__qualname__}.{name} must be a classmethod" for name in sorted(unbound)]
    return "; ".join(problems) or None


def _function(found: Any) -> Any:
    """``found`` unwrapped from a classmethod or staticmethod."""
    return getattr(found, "__func__", found)


def load_failures(config: Config | None = None) -> dict[str, dict[str, str]]:
    """Every plugin that did not take effect, by group then name, with the reason.

    A plugin is listed when it failed to load, validate or construct, clashed with a built-in, or
    claims a name another entry point also claims.
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
