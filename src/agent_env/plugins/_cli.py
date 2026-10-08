"""Entry-point loaders for CLI plugins.

``agent_env.cli_plugins`` entries must resolve to top-level click commands/groups;
``agent_env.cli_root_options`` entries to ``click.Option`` instances that attach to the
root ``agent-env`` group. Broken plugins warn and are skipped; core names win over plugins;
two different root options on the same flag are both left off and reported as a conflict.
Full contract in PLUGINS.md ("CLI plugins, root options, explorer routes").
"""

import sys
from contextlib import redirect_stdout
from importlib.metadata import EntryPoint, entry_points
from typing import Any
from weakref import WeakKeyDictionary

import click
from click.core import ParameterSource

from agent_env.plugins import _report, _requirements
from agent_env.plugins._discovery import discovery_error, dist_name, identity, sort_key
from agent_env.plugins._inventory import Claimant, Contribution, Diagnostic

CLI_PLUGINS_GROUP = "agent_env.cli_plugins"
CLI_ROOT_OPTIONS_GROUP = "agent_env.cli_root_options"


# What each loader did, per root group, so `agent-env plugin` can report it without reloading.
_OUTCOMES: WeakKeyDictionary[click.Group, dict[tuple[str, str], list[Contribution]]] = WeakKeyDictionary()
# Each entry-point group whose installed metadata could not be read, per root group.
_DISCOVERY_ERRORS: WeakKeyDictionary[click.Group, dict[str, Diagnostic]] = WeakKeyDictionary()
# Where each guarded root option came from, for the warning and the record when its callback fails.
_SOURCES: WeakKeyDictionary[click.Option, tuple[EntryPoint, str]] = WeakKeyDictionary()


def cli_contributions(group: click.Group) -> dict[tuple[str, str], list[Contribution]]:
    """The CLI entry points loaded onto ``group``, by (distribution, version), with their outcome."""
    return {key: list(found) for key, found in _OUTCOMES.get(group, {}).items()}


def cli_discovery_errors(group: click.Group) -> dict[str, Diagnostic]:
    """The CLI entry-point groups that could not be read while loading onto ``group``, with why."""
    return dict(_DISCOVERY_ERRORS.get(group, {}))


def load_cli_plugins(group: click.Group) -> None:
    """Register installed ``agent_env.cli_plugins`` entry points onto ``group``."""
    added_by: dict[str, tuple[str, EntryPoint]] = {}
    for ep, origin, plugin in _discover(group, CLI_PLUGINS_GROUP, "CLI plugin"):
        if _incompatible(group, ep, CLI_PLUGINS_GROUP, origin):
            continue
        try:
            command = _load(ep)
        except (Exception, SystemExit) as exc:
            _warn(f"{origin} failed to load and was skipped: {exc!r}")
            _record(group, ep, CLI_PLUGINS_GROUP, "failed", code=_report.LOAD_FAILED,
                    reason=f"failed to load: {exc!r}")
            continue
        if not isinstance(command, click.Command):
            _warn(f"{origin} is not a click.Command and was skipped")
            _record(group, ep, CLI_PLUGINS_GROUP, "failed", code=_report.INVALID_PLUGIN,
                    reason="is not a click.Command")
            continue
        # The entry-point name, as in every group: what the report, --help and the user all use.
        name = ep.name
        if name in group.commands:
            _warn(f"{origin} clashes with existing command {name!r} and was skipped")
            if name in added_by:
                owner, owner_ep = added_by[name]
                _record(group, ep, CLI_PLUGINS_GROUP, "skipped", code=_report.NAME_CONFLICT,
                        reason=f"clashes with existing command {name!r} from plugin {owner}",
                        conflicts_with=(_claimant(owner_ep),))
            else:
                _record(group, ep, CLI_PLUGINS_GROUP, "skipped", code=_report.BUILTIN_NAME,
                        reason=f"clashes with existing command {name!r}")
            continue
        try:
            group.add_command(command, name=name)
        except Exception as exc:
            _warn(f"{origin} could not be registered and was skipped: {exc!r}")
            _record(group, ep, CLI_PLUGINS_GROUP, "failed", code=_report.INVALID_PLUGIN,
                    reason=f"could not be registered: {exc!r}")
            continue
        added_by[name] = (plugin, ep)
        _record(group, ep, CLI_PLUGINS_GROUP, "active")


def load_cli_root_options(group: click.Group) -> None:
    """Attach installed ``agent_env.cli_root_options`` entry points to ``group``.

    Each must be an optional ``click.Option`` with ``expose_value=False``: it is parsed before
    the subcommand and acts through its callback (which click also invokes with ``None`` when
    the flag is absent), never through the root callback's arguments. A flag core already owns
    is skipped with a warning. Different options on the same flag are all left off and recorded
    as a conflict: a collision never silently picks a winner, and the CLI still starts, so
    `plugin remove` can fix it. One option object exported twice, even by two packages, is one
    callback, so it is attached once.
    """
    core_flags = set(group.context_settings.get("help_option_names") or ["--help"])
    core_flags.update(flag for param in group.params for flag in (*param.opts, *param.secondary_opts))
    candidates: list[tuple[EntryPoint, str, str, click.Option]] = []
    for ep, origin, plugin in _discover(group, CLI_ROOT_OPTIONS_GROUP, "CLI root option"):
        if _incompatible(group, ep, CLI_ROOT_OPTIONS_GROUP, origin):
            continue
        try:
            option = _load(ep)
        except (Exception, SystemExit) as exc:
            _warn(f"{origin} failed to load and was skipped: {exc!r}")
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "failed", code=_report.LOAD_FAILED,
                    reason=f"failed to load: {exc!r}")
            continue
        if not isinstance(option, click.Option):
            _warn(f"{origin} is not a click.Option and was skipped")
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "failed", code=_report.INVALID_PLUGIN,
                    reason="is not a click.Option")
            continue
        if option.expose_value or option.required:
            _warn(f"{origin} must be optional with expose_value=False and was skipped")
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "failed", code=_report.INVALID_PLUGIN,
                    reason="must be optional with expose_value=False")
            continue
        clash = sorted(set(_flags(option)) & core_flags)
        if clash:
            _warn(f"{origin} clashes with core root option {clash[0]!r} and was skipped")
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "skipped", code=_report.BUILTIN_NAME,
                    reason=f"clashes with core root option {clash[0]!r}")
            continue
        candidates.append((ep, origin, plugin, option))
    for ep, origin, _, option in candidates:
        # Other option objects on one of its flags; the same object listed twice is not a rival.
        rivals = [(other_ep, other_plugin, set(_flags(other)) & set(_flags(option)))
                  for other_ep, _, other_plugin, other in candidates if other is not option]
        rivals = [(other_ep, other_plugin, shared) for other_ep, other_plugin, shared in rivals if shared]
        if rivals:
            flags = ", ".join(repr(flag) for flag in sorted(set().union(*(shared for *_, shared in rivals))))
            reason = f"root option {flags} is also registered by {' and '.join(p for _, p, _ in rivals)}"
            _warn(f"{origin} was not attached: {reason}")
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "conflict", code=_report.NAME_CONFLICT, reason=reason,
                    conflicts_with=tuple(_claimant(other_ep) for other_ep, _, _ in rivals))
            continue
        if option in group.params:
            _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "active", code=_report.ALREADY_ATTACHED,
                    reason="the same option object is already attached")
            continue
        _guard(option, ep, origin)
        group.params.append(option)
        _record(group, ep, CLI_ROOT_OPTIONS_GROUP, "active")


def _incompatible(group: click.Group, ep: EntryPoint, kind: str, origin: str) -> bool:
    """Record and warn about ``ep`` when its distribution's requirements exclude the installed agent-env."""
    unmet = _requirements.incompatibility(getattr(ep, "dist", None))
    if unmet is None:
        return False
    _warn(f"{origin} was skipped: it {unmet}")
    _record(group, ep, kind, "failed", code=_report.INCOMPATIBLE_CORE, reason=unmet)
    return True


def _flags(option: click.Option) -> tuple[str, ...]:
    return (*option.opts, *option.secondary_opts)


def _claimant(ep: EntryPoint) -> Claimant:
    return Claimant(*identity(ep), getattr(ep, "value", "") or "")


def _guard(option: click.Option, ep: EntryPoint, origin: str) -> None:
    """Wrap ``option``'s callback so that, if it raises while the flag is absent, it is skipped and
    recorded as failed: click calls it on every invocation, so it would otherwise stop every
    command, `plugin remove` included. A flag the user gave still raises. What it prints with the
    flag absent goes to stderr, so it cannot corrupt a command's `--json`. The option object stays
    the plugin's own, so code holding it still finds it in the root group's params."""
    _SOURCES[option] = (ep, origin)
    callback = option.callback
    if callback is None or getattr(callback, "_agent_env_guard", False):
        return

    def call(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
        given = ctx.get_parameter_source(param.name) if param.name else None
        absent = given in (None, ParameterSource.DEFAULT)
        try:
            if not absent:
                return callback(ctx, param, value)
            with redirect_stdout(sys.stderr):
                return callback(ctx, param, value)
        except (Exception, SystemExit) as exc:
            source, where = _SOURCES.get(param, (ep, origin))
            if not absent:
                if isinstance(exc, (click.ClickException, click.exceptions.Exit, click.exceptions.Abort)):
                    raise
                via = f" (set by ${param.envvar})" if given == ParameterSource.ENVIRONMENT and param.envvar else ""
                raise click.ClickException(f"{where}{via} failed: {exc!r}") from exc
            _warn(f"{where} failed with its flag absent and was skipped: {exc!r}")
            _mark(ctx.find_root().command, source, f"its callback failed with the flag absent: {exc!r}")
            return None

    call._agent_env_guard = True  # type: ignore[attr-defined]
    option.callback = call


def _mark(group: click.Group, ep: EntryPoint, reason: str) -> None:
    """Record an attached root option as failed after all."""
    found = _OUTCOMES.get(group, {}).get(identity(ep), [])
    for i, c in enumerate(found):
        if (c.group, c.name) == (CLI_ROOT_OPTIONS_GROUP, ep.name):
            found[i] = Contribution(c.group, c.name, c.value, "failed", code=_report.LOAD_FAILED, reason=reason)


def _discover(group: click.Group, group_name: str, label: str) -> list[tuple[EntryPoint, str, str]]:
    """``(entry point, origin for messages, plugin label)`` per installed entry point, by name."""
    try:
        eps = sorted(entry_points(group=group_name), key=sort_key)
    except Exception as exc:
        reason = discovery_error(exc)
        _warn(f"{label} discovery failed; none loaded: {reason}")
        _DISCOVERY_ERRORS.setdefault(group, {})[group_name] = Diagnostic(_report.ENTRY_POINTS_UNREADABLE, reason)
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


def _record(
    group: click.Group, ep: EntryPoint, kind: str, status: str, *, code: str | None = None, reason: str | None = None,
    conflicts_with: tuple[Claimant, ...] = (),
) -> None:
    contribution = Contribution(
        kind, ep.name, getattr(ep, "value", "") or "", status, code=code, reason=reason, conflicts_with=conflicts_with
    )
    _OUTCOMES.setdefault(group, {}).setdefault(identity(ep), []).append(contribution)


def _warn(message: str) -> None:
    # Logging isn't configured yet at import time; warn on stderr instead.
    click.echo(f"Warning: {message}", err=True)
