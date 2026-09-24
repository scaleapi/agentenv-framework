"""`agent-env plugin` — the installed plugins, what each contributes, and whether it took effect."""

import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stdout
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions
from pathlib import Path
from typing import Optional

import click

from agent_env.plugins import (
    ARTIFACTS,
    ENVS,
    EXPLORER_PLUGINS,
    SANDBOX_PROVIDERS,
    STATE_PROVIDERS,
    TASK_STEPS,
    Contribution,
    Distribution,
    Inventory,
    inventory,
)
from agent_env.plugins._cli import CLI_PLUGINS_GROUP, CLI_ROOT_OPTIONS_GROUP, cli_contributions, cli_discovery_errors

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
    EXPLORER_PLUGINS: ("explorer plugin", "explorer plugins"),
    CLI_PLUGINS_GROUP: ("CLI command", "CLI commands"),
    CLI_ROOT_OPTIONS_GROUP: ("root option", "root options"),
}
_CORE = "agentenv-framework"
_RETIRED = "agent-env"
# `list` names a package's contributions up to this many, and counts them by kind beyond it.
_NAMED_UP_TO = 3
# How long `show` waits for a package's entry points to import in the fresh interpreter.
_PROBE_TIMEOUT_S = 120
# Loads one distribution's entry points in a fresh interpreter and writes AGENT_ENV_CONFIG after
# to the file in argv[2]. The other CLI plugins are hidden, because importing agent_env.cli would
# otherwise load them all and credit their import effects to this one.
_CONFIG_PROBE = """
import importlib.metadata as md, json, os, sys
real = md.entry_points
def entry_points(**params):
    if params.get("group") in ("agent_env.cli_plugins", "agent_env.cli_root_options"):
        return md.EntryPoints(())
    return real(**params)
md.entry_points = entry_points
os.environ.pop("AGENT_ENV_CONFIG", None)
for ep in md.distribution(sys.argv[1]).entry_points:
    if ep.group.startswith("agent_env."):
        try:
            ep.load()
        except Exception:
            pass
with open(sys.argv[2], "w") as out:
    json.dump(os.environ.get("AGENT_ENV_CONFIG"), out)
"""


@dataclasses.dataclass(frozen=True)
class Report:
    """What the `plugin` commands print: the inventory, merged with the CLI's own plugins."""

    core_version: str
    environment: str
    location: Path
    retired_version: Optional[str]
    inventory: Inventory
    distributions: tuple[Distribution, ...]
    # Entry-point groups, type and CLI, whose installed metadata could not be read.
    discovery_errors: Mapping[str, str]


def _json(payload: object) -> str:
    def encode(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"cannot serialize {type(value).__name__} in a plugin report")
    return json.dumps(payload, indent=2, default=encode)


def environment(prefix: Path, base_prefix: Path) -> tuple[str, Path]:
    """How agent-env was installed, and where: the answer decides where a plugin has to go."""
    if (prefix / "uv-receipt.toml").is_file():
        return "uv tool", prefix
    if (prefix / "pipx_metadata.json").is_file():
        return "pipx", prefix
    root = prefix.parent
    if prefix.name == ".venv" and (root / "pyproject.toml").is_file() and (root / "uv.lock").is_file():
        return "uv project", root
    if prefix != base_prefix:
        return "virtualenv", prefix
    return "system Python", prefix


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
    merged: dict[tuple[str, str], list[Contribution]] = {
        (d.name, d.version): list(d.contributions) for d in inv.distributions
    }
    for key, found in cli_contributions(root).items():
        merged.setdefault(key, []).extend(found)
    kind, location = environment(Path(sys.prefix), Path(sys.base_prefix))
    # Both installed: they unpack to the same tree, and whichever wrote last is what runs.
    retired = _version(_RETIRED) if _version(_CORE) is not None else None
    return Report(
        core_version=_core_version(),
        environment=kind,
        location=location,
        retired_version=retired,
        inventory=inv,
        distributions=tuple(
            Distribution(name, version, tuple(found))
            for (name, version), found in sorted(merged.items(), key=lambda item: (item[0][0].lower(), item[0][1]))
        ),
        discovery_errors={**inv.discovery_errors, **cli_discovery_errors(root)},
    )


def _as_dict(report: Report) -> dict:
    return {
        "agent_env": {"version": report.core_version, "environment": report.environment, "location": report.location},
        "retired_agent_env": report.retired_version,
        "config": {"path": report.inventory.config_path, "error": report.inventory.config_error},
        "loaded": report.inventory.loaded,
        "group_errors": dict(report.inventory.group_errors),
        "discovery_errors": dict(report.discovery_errors),
        "plugins": [dataclasses.asdict(d) for d in report.distributions],
    }


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
    by_error: dict[str, list[str]] = {}
    for group, error in report.discovery_errors.items():
        by_error.setdefault(error, []).append(group)
    return [
        f"installed entry points could not be read for {', '.join(groups)}: {error}"
        for error, groups in by_error.items()
    ]


def _header(report: Report) -> list[str]:
    lines = [f"agent-env {report.core_version} · {report.environment} at {report.location}"]
    inv = report.inventory
    lines.append(f"config: {inv.config_path}" if inv.config_path else "config: (none)")
    if inv.config_error:
        lines.append(f"(error) {inv.config_error}")
    lines += [f"(error) {line}" for line in _unreadable(report)]
    if report.retired_version:
        lines.append(
            f"(warning) the retired {_RETIRED} {report.retired_version} distribution is also installed: it writes the "
            f"same agent_env package as {_CORE}, so whichever installed last wins. Uninstall both, then reinstall "
            f"{_CORE}."
        )
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
        lines += ["", f"(error) nothing in {group} loads: {error}"]
    if not report.inventory.loaded:
        lines += ["", "--no-load: type plugins were not imported (CLI plugins load when the CLI starts)."]
    return "\n".join(lines)


def _outcome(c: Contribution) -> str:
    if c.status == "replaced":
        return f"replaced by {c.replaced_in} ({c.replacement})"
    if c.status == "conflict":
        return f"conflict: also registered by {'; '.join(c.conflicts_with)}"
    return f"{c.status}: {c.reason}" if c.reason else c.status


def config_effect(name: str) -> str:
    """Whether importing ``name``'s entry points sets AGENT_ENV_CONFIG, checked in a fresh
    interpreter: this process has already imported the CLI plugins, so it cannot tell."""
    env = {k: v for k, v in os.environ.items() if k != "AGENT_ENV_CONFIG"}
    with tempfile.TemporaryDirectory() as scratch:
        result = Path(scratch) / "config.json"
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _CONFIG_PROBE, name, str(result)],
                capture_output=True, text=True, env=env, timeout=_PROBE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"not checked: importing it took more than {_PROBE_TIMEOUT_S}s"
        try:
            value = json.loads(result.read_text())
        except (OSError, ValueError):
            last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit status {proc.returncode}"
            return f"not checked: {last}"
    return f"importing it sets AGENT_ENV_CONFIG to {value}" if value else "importing it leaves AGENT_ENV_CONFIG unset"


def _requires_core(name: str) -> list[str]:
    """The requirements naming agent-env, under any spelling of its distribution names."""
    wanted = {_normalized(_CORE), _normalized(_RETIRED)}
    found = []
    for requirement in distribution(name).requires or []:
        project = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if project and _normalized(project.group(1)) in wanted:
            found.append(requirement)
    return found


def render_show(report: Report, dist: Distribution, effect: Optional[str]) -> str:
    lines = [f"{dist.name} {dist.version}"]
    try:
        requires = _requires_core(dist.name)
        location = distribution(dist.name).locate_file("")
    except PackageNotFoundError:
        requires, location = [], None
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
        if group in report.inventory.group_errors:
            lines += ["", f"  (error) nothing in {group} loads: {report.inventory.group_errors[group]}"]
    return "\n".join(lines)


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _root() -> click.Group:
    return click.get_current_context().find_root().command


@click.group()
def plugin():
    """Inspect installed plugins."""


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
            "ok": not reasons,
            "config_error": error,
            "discovery_errors": unread,
            "group_errors": dict(report.inventory.group_errors),
            "problems": [{"package": d.name, "version": d.version, **dataclasses.asdict(c)} for d, c in problems],
        }))
        if reasons:
            ctx.exit(1)
        return
    for d, c in problems:
        click.echo(f"{d.name} {d.version}: {_KINDS[c.group][0]} {c.name}: {_outcome(c)}")
    if error:
        click.echo(f"config: {error}")
    for line in _unreadable(report):
        click.echo(line)
    if reasons:
        raise click.ClickException("; ".join(reasons))
    total = sum(len(d.contributions) for d in report.distributions)
    replaced = sum(c.status == "replaced" for d in report.distributions for c in d.contributions)
    note = f"; {replaced} replaced by config" if replaced else ", all in effect"
    click.echo(f"ok: {len(report.distributions)} plugin package(s), {total} contribution(s){note}")
