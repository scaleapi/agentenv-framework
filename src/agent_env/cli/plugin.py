"""`agent-env plugin` — the installed plugins, what each contributes, and whether it took effect."""

import dataclasses
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stdout
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions
from pathlib import Path
from typing import Optional

import click

from agent_env.bundle import BundleError
from agent_env.bundle.installed import BUNDLES, InstalledBundle, checked, installed_bundles, run_name
from agent_env.cli import _installers, _plugin_changes
from agent_env.cli._installers import FORCEABLE, InstallerError
from agent_env.plugins import (
    ARTIFACTS,
    ENV_PROVIDERS,
    ENVS,
    EXPLORER_PLUGINS,
    SANDBOX_PROVIDERS,
    STATE_PROVIDERS,
    TASK_STEPS,
    Claimant,
    Contribution,
    Diagnostic,
    Distribution,
    Inventory,
    inventory,
)
from agent_env.plugins._cli import CLI_PLUGINS_GROUP, CLI_ROOT_OPTIONS_GROUP, cli_contributions, cli_discovery_errors
from agent_env.plugins._discovery import discovery_error
from agent_env.plugins._report import (
    ENTRY_POINTS_UNREADABLE,
    FORMAT_VERSION,
    INVALID_PLUGIN,
    NAME_CONFLICT,
    QUALIFIED_ONLY,
)

# A contribution with one of these statuses did not take effect, so `check` fails on it.
PROBLEMS = ("conflict", "blocked", "failed", "skipped")
_ORDER = (*PROBLEMS, "replaced", "unloaded")
# `check` always loads, so an `unloaded` contribution there is one whose status could not be determined.
_CHECK_FAILS = (*PROBLEMS, "unloaded")
_KINDS = {
    ENVS: ("env", "envs"),
    ARTIFACTS: ("artifact", "artifacts"),
    TASK_STEPS: ("task step", "task steps"),
    SANDBOX_PROVIDERS: ("sandbox provider", "sandbox providers"),
    STATE_PROVIDERS: ("state provider", "state providers"),
    ENV_PROVIDERS: ("environment provider", "environment providers"),
    EXPLORER_PLUGINS: ("explorer plugin", "explorer plugins"),
    CLI_PLUGINS_GROUP: ("CLI command", "CLI commands"),
    CLI_ROOT_OPTIONS_GROUP: ("root option", "root options"),
    BUNDLES: ("bundle", "bundles"),
}
_CORE = "agentenv-framework"
# `list` names a package's contributions up to this many, and counts them by kind beyond it.
_NAMED_UP_TO = 3


@dataclasses.dataclass(frozen=True)
class Report:
    """What the `plugin` commands print: the inventory, merged with the CLI's own plugins."""

    core_version: str
    environment: str
    location: Path
    inventory: Inventory
    distributions: tuple[Distribution, ...]
    # Entry-point groups, type and CLI, whose installed metadata could not be read.
    discovery_errors: Mapping[str, Diagnostic]


def _json(payload: object) -> str:
    def encode(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"cannot serialize {type(value).__name__} in a plugin report")
    return json.dumps(payload, indent=2, default=encode)


def environment(prefix: Path, base_prefix: Path) -> tuple[str, Path]:
    """How agent-env was installed, and where: the answer decides where a plugin has to go."""
    env = _installers.detect(prefix, base_prefix, Path(sys.executable))
    return env.kind, env.location


def _version(name: str) -> Optional[str]:
    try:
        return distribution(name).version
    except PackageNotFoundError:
        return None


def _core_version() -> str:
    for name in (_CORE, *packages_distributions().get("agent_env", [])):
        if (found := _version(name)) is not None:
            return found
    return "unknown"


@contextmanager
def _quiet() -> Iterator[None]:
    """The report states each failure once; the loaders' own warnings would repeat it."""
    logger = logging.getLogger("agent_env")
    previous = logger.level
    if not logger.isEnabledFor(logging.DEBUG):
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def collect(root: click.Group, *, load: bool) -> Report:
    # A plugin that prints while it is imported must not corrupt `--json` on stdout.
    with _quiet(), redirect_stdout(sys.stderr):
        inv = inventory(load=load)
        bundles, bundle_errors = _bundle_contributions(load=load)
    merged: dict[tuple[str, str], list[Contribution]] = {
        (d.name, d.version): list(d.contributions) for d in inv.distributions
    }
    for key, found in cli_contributions(root).items():
        merged.setdefault(key, []).extend(found)
    for key, found in bundles.items():
        merged.setdefault(key, []).extend(found)
    kind, location = environment(Path(sys.prefix), Path(sys.base_prefix))
    return Report(
        core_version=_core_version(),
        environment=kind,
        location=location,
        inventory=inv,
        distributions=tuple(
            Distribution(name, version, tuple(found))
            for (name, version), found in sorted(merged.items(), key=lambda item: (item[0][0].lower(), item[0][1]))
        ),
        discovery_errors={**inv.discovery_errors, **cli_discovery_errors(root), **bundle_errors},
    )


def _bundle_contributions(*, load: bool) -> tuple[dict[tuple[str, str], list[Contribution]], dict[str, Diagnostic]]:
    """Each installed bundle as a contribution of its package, and the discovery error if there is one."""
    try:
        bundles = installed_bundles()
    except Exception as e:
        return {}, {BUNDLES: Diagnostic(ENTRY_POINTS_UNREADABLE, discovery_error(e))}
    found: dict[tuple[str, str], list[Contribution]] = {}
    for bundle in bundles:
        found.setdefault((bundle.package, bundle.version), []).append(_bundle_contribution(bundle, bundles, load=load))
    return found, {}


def _bundle_contribution(bundle: InstalledBundle, bundles: tuple[InstalledBundle, ...], *, load: bool) -> Contribution:
    """Whether ``bundle`` runs, by the checks ``agent-env run`` makes before it reads a store. Without loading
    plugins, only that the folder parses."""
    namesakes = tuple(Claimant(other.package, other.version, other.value)
                      for other in bundles if other.name == bundle.name and other is not bundle)
    if bundle.problem:
        status = "conflict" if bundle.code == NAME_CONFLICT else "failed"
        return Contribution(BUNDLES, bundle.name, bundle.value, status, code=bundle.code, reason=bundle.problem,
                            conflicts_with=namesakes if status == "conflict" else ())
    try:
        checked(bundle, load=load)
    except BundleError as e:
        return Contribution(BUNDLES, bundle.name, bundle.value, "failed", code=INVALID_PLUGIN,
                            reason=f"it isn't a valid bundle: {e.summary}")
    if run_name(bundle, bundles) != bundle.name:
        return Contribution(BUNDLES, bundle.name, bundle.value, "active", code=QUALIFIED_ONLY,
                            reason=f"another package also installs {bundle.name!r}, so run this one as "
                                   f"{bundle.qualified!r}", conflicts_with=namesakes)
    return Contribution(BUNDLES, bundle.name, bundle.value, "active")


def _as_dict(report: Report) -> dict:
    """The document `list --json` prints; `show` and `check` add to it. See PLUGINS.md "Plugin report format"."""
    error = report.inventory.config_error
    return {
        "format_version": FORMAT_VERSION,
        "agent_env": {"version": report.core_version, "environment": report.environment, "location": report.location},
        "config": {"path": report.inventory.config_path, "error": dataclasses.asdict(error) if error else None},
        "loaded": report.inventory.loaded,
        "group_errors": {group: dataclasses.asdict(e) for group, e in report.inventory.group_errors.items()},
        "discovery_errors": {group: dataclasses.asdict(e) for group, e in report.discovery_errors.items()},
        "plugins": [dataclasses.asdict(d) for d in report.distributions],
    }


def _coded(text: str, code: Optional[str]) -> str:
    return f"{text} [{code}]" if code else text


def _provides(contributions: tuple[Contribution, ...]) -> str:
    if len(contributions) <= _NAMED_UP_TO:
        return ", ".join(f"{_KINDS[c.group][0]} {c.name}" for c in contributions)
    counts = Counter(c.group for c in contributions)
    return ", ".join(
        f"{n} {_KINDS[group][0] if n == 1 else _KINDS[group][1]}" for group, n in counts.items()
    )


def _summary(contributions: tuple[Contribution, ...]) -> str:
    counts = Counter(c.status for c in contributions if c.status != "active")
    if not counts:
        return "ok"
    if set(counts) == {"unloaded"}:
        return "not loaded"
    return ", ".join(f"{counts[status]} {status}" for status in _ORDER if status in counts)


def _unreadable(report: Report) -> list[str]:
    # One unparseable entry_points.txt breaks every group alike, so each distinct error is one line.
    by_error: dict[Diagnostic, list[str]] = {}
    for group, error in report.discovery_errors.items():
        by_error.setdefault(error, []).append(group)
    return [
        _coded(f"installed entry points could not be read for {', '.join(groups)}: {error.reason}", error.code)
        for error, groups in by_error.items()
    ]


def _header(report: Report) -> list[str]:
    lines = [f"agent-env {report.core_version} · {report.environment} at {report.location}"]
    inv = report.inventory
    lines.append(f"config: {inv.config_path}" if inv.config_path else "config: (none)")
    if inv.config_error:
        lines.append(_coded(f"(error) {inv.config_error.reason}", inv.config_error.code))
    lines += [f"(error) {line}" for line in _unreadable(report)]
    return lines


def render_list(report: Report) -> str:
    lines = _header(report)
    lines.append("")
    if not report.distributions:
        lines.append("No plugins installed.")
        return "\n".join(lines)
    rows = [("PACKAGE", "VERSION", "PROVIDES", "STATUS")] + [
        (d.name or "(unreadable metadata)", d.version, _provides(d.contributions), _summary(d.contributions))
        for d in report.distributions
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    lines += [
        f"{name:<{widths[0]}}  {version:<{widths[1]}}  {provides:<{widths[2]}}  {status}"
        for name, version, provides, status in rows
    ]
    for group, error in report.inventory.group_errors.items():
        lines += ["", _coded(f"(error) nothing in {group} loads: {error.reason}", error.code)]
    if not report.inventory.loaded:
        lines += ["", "--no-load: type plugins were not imported (CLI plugins load when the CLI starts)."]
    return "\n".join(lines)


def _outcome(c: Contribution) -> str:
    return _coded(f"{c.status}: {c.reason}", c.code) if c.reason else c.status


def config_effect(name: str) -> str:
    """Whether importing ``name``'s entry points sets AGENT_ENV_CONFIG, checked in a fresh
    interpreter: this process has already imported the CLI plugins, so it cannot tell."""
    try:
        value = _plugin_changes.config_set_by(name)
    except _plugin_changes.ProbeError as exc:
        return f"not checked: {exc}"
    return f"importing it sets AGENT_ENV_CONFIG to {value}" if value else "importing it leaves AGENT_ENV_CONFIG unset"


def _requires_core(name: str) -> list[str]:
    """The requirements naming agent-env."""
    found = []
    for requirement in distribution(name).requires or []:
        project = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if project and _normalized(project.group(1)) == _normalized(_CORE):
            found.append(requirement)
    return found


def render_show(report: Report, dist: Distribution, effect: Optional[str]) -> str:
    lines = [f"{dist.name} {dist.version}"]
    try:
        requires = _requires_core(dist.name)
        location = distribution(dist.name).locate_file("")
    except PackageNotFoundError:
        requires, location = [], None
    if dist.name == _CORE:
        lines.append("  agent-env itself")
    else:
        lines.append(f"  requires: {', '.join(requires) or 'no agent-env requirement declared'} "
                     f"(installed {report.core_version})")
    if location is not None:
        lines.append(f"  location: {location}")
    if effect is not None:
        lines.append(f"  config:   {effect}")
    lines.append("")
    rows = [(_KINDS[c.group][0], c.name, c.value, _outcome(c)) for c in dist.contributions]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    lines += [f"  {kind:<{widths[0]}}  {name:<{widths[1]}}  {value:<{widths[2]}}  {outcome}"
              for kind, name, value, outcome in rows]
    for group in dict.fromkeys(c.group for c in dist.contributions):
        if (error := report.inventory.group_errors.get(group)) is not None:
            lines += ["", _coded(f"  (error) nothing in {group} loads: {error.reason}", error.code)]
    return "\n".join(lines)


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _root() -> click.Group:
    return click.get_current_context().find_root().command


@click.group()
def plugin():
    """Inspect, add and remove installed plugins."""


@plugin.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable.")
@click.option("--no-load", is_flag=True, help="Read installed metadata only; import no type plugin.")
def list_command(as_json: bool, no_load: bool):
    """List every installed plugin package, what it provides, and whether it took effect."""
    report = collect(_root(), load=not no_load)
    click.echo(_json(_as_dict(report)) if as_json else render_list(report))


@plugin.command()
@click.argument("package")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable.")
@click.option("--no-load", is_flag=True, help="Read installed metadata only; import no type plugin.")
def show(package: str, as_json: bool, no_load: bool):
    """Show one plugin PACKAGE: each contribution, its status and why."""
    report = collect(_root(), load=not no_load)
    matches = [d for d in report.distributions if _normalized(d.name) == _normalized(package)]
    if not matches:
        known = ", ".join(d.name for d in report.distributions) or "none"
        raise click.ClickException(f"no installed plugin package named {package!r} (installed: {known})")
    dist = matches[0]
    effect = None if no_load else config_effect(dist.name)
    if as_json:
        click.echo(_json({**_as_dict(report), "plugins": [dataclasses.asdict(dist)], "config_effect": effect}))
        return
    click.echo(render_show(report, dist, effect))


@plugin.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable.")
@click.pass_context
def check(ctx: click.Context, as_json: bool):
    """Load every plugin; exit 1 if any contribution did not take effect, or if the config or the
    installed entry points cannot be read.

    A replaced contribution is not a failure: config chose it. Meant for CI and image builds.
    """
    report = collect(_root(), load=True)
    problems = [(d, c) for d in report.distributions for c in d.contributions if c.status in _CHECK_FAILS]
    error = report.inventory.config_error
    unread = dict(report.discovery_errors)
    reasons = (
        ([f"{len(problems)} plugin contribution(s) did not take effect"] if problems else [])
        + (["the config could not be read"] if error else [])
        + (["installed entry points could not be read"] if unread else [])
    )
    if as_json:
        click.echo(_json({
            **_as_dict(report),
            "ok": not reasons,
            "problems": [{"package": d.name, "version": d.version, **dataclasses.asdict(c)} for d, c in problems],
        }))
        if reasons:
            ctx.exit(1)
        return
    for d, c in problems:
        click.echo(f"{d.name} {d.version}: {_KINDS[c.group][0]} {c.name}: {_outcome(c)}")
    if error:
        click.echo(_coded(f"config: {error.reason}", error.code))
    for line in _unreadable(report):
        click.echo(line)
    if reasons:
        raise click.ClickException("; ".join(reasons))
    total = sum(len(d.contributions) for d in report.distributions)
    replaced = sum(c.status == "replaced" for d in report.distributions for c in d.contributions)
    note = f"; {replaced} replaced by config" if replaced else ", all in effect"
    click.echo(f"ok: {len(report.distributions)} plugin package(s), {total} contribution(s){note}")


@plugin.command()
@click.argument("specs", nargs=-1, required=True)
@click.option("--installer", type=click.Choice(sorted(FORCEABLE)), help="Use this installer, not the detected one.")
@click.option("--index-url", help="The index the installer resolves from, passed through to it.")
@click.option("--dry-run", is_flag=True, help="Show what would run, and change nothing.")
@click.option("--yes", "-y", is_flag=True, help="Do not ask before changing the environment.")
@click.option("--keep", is_flag=True, help="Keep the change even when a plugin does not take effect.")
@click.pass_context
def add(ctx: click.Context, specs: tuple[str, ...], installer: Optional[str], index_url: Optional[str],
        dry_run: bool, yes: bool, keep: bool):
    """Install plugin packages through the installer that owns this environment, then check them.

    SPECS are whatever that installer accepts: names with versions, paths, URLs. If a new plugin
    does not take effect, the environment is restored as it was, unless --keep.
    """
    try:
        code = _plugin_changes.add(list(specs), installer=installer, index_url=index_url, dry_run=dry_run, yes=yes,
                                   keep=keep, tools=_plugin_changes.Tools())
    except InstallerError as exc:
        raise click.ClickException(str(exc)) from exc
    ctx.exit(code)


@plugin.command()
@click.argument("packages", nargs=-1, required=True)
@click.option("--installer", type=click.Choice(sorted(FORCEABLE)), help="Use this installer, not the detected one.")
@click.option("--dry-run", is_flag=True, help="Show what would run, and change nothing.")
@click.option("--yes", "-y", is_flag=True, help="Do not ask before changing the environment.")
@click.option("--force", is_flag=True, help="Remove even when something still depends on or names the package.")
@click.option("--check-usage", is_flag=True, help="Also count stored documents in a remote document store.")
@click.pass_context
def remove(ctx: click.Context, packages: tuple[str, ...], installer: Optional[str], dry_run: bool, yes: bool,
           force: bool, check_usage: bool):
    """Uninstall plugin PACKAGES through the installer that owns this environment.

    A package that `plugin add` installed next to a plugin, such as a pin it needs, can be removed
    too. Refused, unless --force, while another package requires it, the config file names what it
    provides, or stored documents use its types (checked for a local store; --check-usage for a
    remote one).
    """
    try:
        code = _plugin_changes.remove(list(packages), installer=installer, dry_run=dry_run, yes=yes, force=force,
                                      check_usage=check_usage, tools=_plugin_changes.Tools())
    except InstallerError as exc:
        raise click.ClickException(str(exc)) from exc
    ctx.exit(code)
