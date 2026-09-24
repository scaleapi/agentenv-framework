"""Entry-point loaders for CLI plugins.

``agent_env.cli_plugins`` entries must resolve to top-level click commands/groups;
``agent_env.cli_root_options`` entries to ``click.Option`` instances that attach to the
root ``agent-env`` group. Broken plugins warn and are skipped; core names win over plugins;
two plugins claiming the same root flag abort startup. Full contract in the README
("Extending the CLI (plugins)").
"""

import sys
from contextlib import redirect_stdout
from importlib.metadata import EntryPoint, entry_points
from typing import Any
from weakref import WeakKeyDictionary

import click

from agent_env.plugins._inventory import Contribution
from agent_env.plugins._discovery import discovery_error, dist_name, dist_version, sort_key

CLI_PLUGINS_GROUP = "agent_env.cli_plugins"
CLI_ROOT_OPTIONS_GROUP = "agent_env.cli_root_options"


class RootOptionConflictError(RuntimeError):
    """Two installed plugins register the same root option flag."""


# What each loader did, per root group, so `agent-env plugin` can report it without reloading.
_OUTCOMES: WeakKeyDictionary[click.Group, dict[tuple[str, str], list[Contribution]]] = WeakKeyDictionary()
# Each entry-point group whose installed metadata could not be read, per root group.
_DISCOVERY_ERRORS: WeakKeyDictionary[click.Group, dict[str, str]] = WeakKeyDictionary()


def cli_contributions(group: click.Group) -> dict[tuple[str, str], list[Contribution]]:
    """The CLI entry points loaded onto ``group``, by (distribution, version), with their outcome."""
    return {key: list(found) for key, found in _OUTCOMES.get(group, {}).items()}


def cli_discovery_errors(group: click.Group) -> dict[str, str]:
    """The CLI entry-point groups that could not be read while loading onto ``group``, with why."""
    return dict(_DISCOVERY_ERRORS.get(group, {}))


def load_cli_plugins(group: click.Group) -> None:
    """Register installed ``agent_env.cli_plugins`` entry points onto ``group``."""
    added_by: dict[str, str] = {}
    for ep, origin, plugin in _discover(group, CLI_PLUGINS_GROUP, "CLI plugin"):
        try:
            command = _load(ep)
        except Exception as exc:
            _warn(f"{origin} failed to load and was skipped: {exc!r}")
            _record(group, ep, CLI_PLUGINS_GROUP, "failed", f"failed to load: {exc!r}")
            continue
        if not isinstance(command, click.Command):
            _warn(f"{origin} is not a click.Command and was skipped")
            _record(group, ep, CLI_PLUGINS_GROUP, "failed", "is not a click.Command")
            continue
        name = command.name or ep.name
        if name in group.commands:
            _warn(f"{origin} clashes with existing command {name!r} and was skipped")
            owner = f" from plugin {added_by[name]}" if name in added_by else ""
            _record(group, ep, CLI_PLUGINS_GROUP, "skipped", f"clashes with existing command {name!r}{owner}")
            continue
        try:
            group.add_command(command, name=name)
        except Exception as exc:
            _warn(f"{origin} could not be registered and was skipped: {exc!r}")
            _record(group, ep, CLI_PLUGINS_GROUP, "failed", f"could not be registered: {exc!r}")
            continue
        added_by[name] = plugin
        _record(group, ep, CLI_PLUGINS_GROUP, "active")


def load_cli_root_options(group: click.Group) -> None:
    """Attach installed ``agent_env.cli_root_options`` entry points to ``group``.

    Each must be an optional ``click.Option`` with ``expose_value=False``: it is parsed before
    the subcommand and acts through its callback (which click also invokes with ``None`` when
    the flag is absent), never through the root callback's arguments. A flag core already owns
    is skipped with a warning; a flag two plugins claim raises ``RootOptionConflictError``, so a
    collision never silently picks a winner.
    """
    core_flags = set(group.context_settings.get("help_option_names") or ["--help"])
    core_flags.update(flag for param in group.params for flag in (*param.opts, *param.secondary_opts))
    owners: dict[str, str] = {}
    for ep, origin, plugin in _discover(group, CLI_ROOT_OPTIONS_GROUP, "CLI root option"):
        try:
            option = _load(ep)
        except Exception as exc:
            _warn(f"{origin} failed to load and was skipped: {exc!r}")
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "failed", f"failed to load: {exc!r}")
            continue
        if not isinstance(option, click.Option):
            _warn(f"{origin} is not a click.Option and was skipped")
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "failed", "is not a click.Option")
            continue
        if option.expose_value or option.required:
            _warn(f"{origin} must be optional with expose_value=False and was skipped")
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "failed", "must be optional with expose_value=False")
            continue
        if option in group.params:
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "active", "the same option object is already attached")
            continue
        flags = (*option.opts, *option.secondary_opts)
        clash = sorted(set(flags) & core_flags)
        if clash:
            _warn(f"{origin} clashes with core root option {clash[0]!r} and was skipped")
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "skipped", f"clashes with core root option {clash[0]!r}")
            continue
        conflict = sorted(flag for flag in flags if flag in owners)
        if conflict:
            raise RootOptionConflictError(
                f"root option {conflict[0]!r} is registered by two plugins, {owners[conflict[0]]} and {plugin}; "
                "uninstall one of them"
            )
        owners.update(dict.fromkeys(flags, plugin))
        group.params.append(option)
        _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "active")


def _discover(group: click.Group, group_name: str, label: str) -> list[tuple[EntryPoint, str, str]]:
    """``(entry point, origin for messages, plugin label)`` per installed entry point, by name."""
    try:
        eps = sorted(entry_points(group=group_name), key=sort_key)
    except Exception as exc:
        reason = discovery_error(exc)
        _warn(f"{label} discovery failed; none loaded: {reason}")
        _DISCOVERY_ERRORS.setdefault(group, {})[group_name] = reason
        return []
    found = []
    for ep in eps:
        dist = dist_name(ep)
        plugin = f"{ep.name!r}" + (f" from {dist!r}" if dist else "")
        found.append((ep, f"{label} {plugin}", plugin))
    return found


def _load(ep: EntryPoint) -> Any:
    # What a plugin prints while it is imported must not land in a command's stdout, e.g. --json.
    with redirect_stdout(sys.stderr):
        return ep.load()


def _record(group: click.Group, ep: EntryPoint, kind: str, status: str, reason: str | None = None) -> None:
    key = (dist_name(ep), dist_version(ep))
    contribution = Contribution(kind, ep.name, getattr(ep, "value", "") or "", status, reason=reason)
    _OUTCOMES.setdefault(group, {}).setdefault(key, []).append(contribution)


def _warn(message: str) -> None:
    # Logging isn't configured yet at import time; warn on stderr instead.
    click.echo(f"Warning: {message}", err=True)
