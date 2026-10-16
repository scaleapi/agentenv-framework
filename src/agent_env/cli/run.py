"""``agent-env run``: run the tasks and evals of a bundle, a folder or one an installed package provides."""

from __future__ import annotations

import os
import shlex
import sys
import time
import traceback
from pathlib import Path

import click
from click.core import ParameterSource

from agent_env.bundle import (BundleError, BundleKind, BundleRun, DryRun, Outcome, RunInterrupted, TaskRun,
                              dry_run_bundle, run_bundle)
from agent_env.bundle.installed import InstalledBundle, checked, find_bundle, installed_bundles, run_name
from agent_env.plugins._discovery import discovery_error
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox
from agent_env.task.interrupts import Interrupts
from agent_env.task.teardown import RecordedSandbox, TeardownReport, kind, recorded_sandboxes, sandbox_ids
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.teardown_sandboxes import TORN_DOWN_KEY

_COLORS = {Outcome.PASSED: "green", Outcome.BELOW_ONE: "yellow", Outcome.UNSCORED: "yellow", Outcome.FAILED: "red",
           Outcome.CANCELLED: "yellow"}
_ENV_ENDPOINTS = (("mcp", "mcp_url"), ("gateway", "gateway_url"), ("pgweb", "db_web_url"), ("db-mcp", "db_mcp_url"),
                  ("vnc", "vnc_url"), ("expires", "expires_at_utc"))
_DRY_RUN = "Dry run: nothing is written or run."
_SHOWN_CRITERIA = 5
_REASON_CHARS = 160


@click.command()
@click.argument("bundle", required=False)
@click.option("--task", "tasks", multiple=True, help="Run this task, by name or id. Repeatable.")
@click.option("--eval", "evals", multiple=True, help="Run this eval's tasks, by name or id. Repeatable.")
@click.option("--model", default=None, help="The model the agent runs on. The judge keeps its own.")
@click.option("--sandbox", default=None,
              help="The sandbox provider envs, agents, sandboxes and the judge deploy on; comma-separated for a "
                   "fallback chain.")
@click.option("--keep", is_flag=True,
              help="Keep the sandboxes up after the runs, print their endpoints, and tear them down on Ctrl-C.")
@click.option("--dry-run", is_flag=True,
              help="Check the bundle as the run does and show what it would write and run; write and run nothing.")
@click.option("--fail-under", type=click.FloatRange(0, 1), default=None, metavar="SCORE",
              help="Exit 1 when a task scored below SCORE, or recorded no score; --fail-under 1 fails on any miss.")
@click.pass_context
def run(ctx: click.Context, bundle: str | None, tasks: tuple[str, ...], evals: tuple[str, ...], model: str | None,
        sandbox: str | None, keep: bool, dry_run: bool, fail_under: float | None):
    """Run a bundle's tasks and evals, or list the installed bundles.

    BUNDLE is a folder, or the name of a bundle an installed package provides (PACKAGE/NAME when several
    packages provide that name). Without BUNDLE, it lists the installed bundles.

    With no --task or --eval, it runs every eval or, in a bundle without evals, every task. Each task runs
    once, at most four at a time, and the run exits 1 if any of them failed. A task that scored below 1 lists
    the criteria that failed and why, and exits 1 only under --fail-under, which also fails a task that
    recorded no score. With --verbose, a run that raised prints its traceback.

    --dry-run makes the checks the run makes before its first task, reading the stores as the run does, and
    prints what it would write and run, writing and running nothing. It exits 1 on a problem the run would
    stop at, and 0 otherwise; --model, --keep and --fail-under change nothing it shows. It takes no lock, so
    another run can write first and change what it shows.

    Each run's sandboxes are torn down as it ends; its instance and outputs stay. --keep holds them up
    until Ctrl-C instead. Ctrl-C or SIGTERM mid-run cancels the runs, tears them down, prints what ran and
    exits 130 (143 for SIGTERM); a second one stops the teardown and prints what is still up.
    """
    if bundle is None:
        given = [param.opts[0] for param in ctx.command.params if isinstance(param, click.Option)
                 and ctx.get_parameter_source(param.name) not in (None, ParameterSource.DEFAULT)]
        if given:
            *rest, last = given
            raise click.UsageError(f"{', '.join(rest)} and {last} need a BUNDLE to run" if rest
                                   else f"{last} needs a BUNDLE to run")
        _list_installed()
        return
    root, id_root = _locate(bundle)
    verbose = ctx.find_root().params.get("verbose")
    try:
        if dry_run:
            _echo(_DRY_RUN)
            dry = dry_run_bundle(root, tasks=tasks, evals=evals, sandbox=sandbox, on_progress=_echo, id_root=id_root)
            _report_dry_run(dry)
            return
        result = run_bundle(root, tasks=tasks, evals=evals, model=model, sandbox=sandbox, on_progress=_echo,
                            id_root=id_root, keep=keep)
    except BundleError as e:
        _note_an_installed_namesake(e, bundle, root, id_root)
        raise
    except RunInterrupted as stop:
        with Interrupts():
            _summarize(stop.result, verbose)
            _report_teardown(stop.result)
        raise SystemExit(128 + stop.signum) from None
    with Interrupts() as interrupts:
        _summarize(result, verbose)
        if keep:
            result = _hold(result, interrupts)
        _report_teardown(result)
    below = result.below(fail_under) if fail_under is not None else ()
    if below:
        _report_below(result, below, fail_under)
    if result.failed or below:
        raise SystemExit(1)


def _echo(message: str | None = None, err: bool = False) -> None:
    """``click.echo``, which drops the line once its stream's reader has gone, as when piped into head, so the run
    still finishes and tears down."""
    try:
        click.echo(message, err=err)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, (sys.stderr if err else sys.stdout).fileno())
        os.close(devnull)


def _report_below(result: BundleRun, below: tuple[TaskRun, ...], threshold: float) -> None:
    shown = ", ".join(f"{result.path(run.entry)} ({min(run.scores.values()):g})" if run.scores
                      else f"{result.path(run.entry)} (no score)" for run in below)
    _echo(f"\nBelow --fail-under {threshold:g}: {shown}")


def _path_like(text: str) -> bool:
    """Text that names a folder even when there is none there."""
    return text.startswith((".", "/", "~"))


def _locate(text: str) -> tuple[Path, str | None]:
    """The folder ``text`` names, and the id root of an installed bundle's. An existing folder wins, and text
    that starts like a path is always one; anything else names an installed bundle."""
    if _path_like(text) or os.path.isdir(text):
        return Path(text), None
    found = find_bundle(text, _installed())
    return found.root, found.id_root


def _note_an_installed_namesake(error: BundleError, text: str, root: Path, id_root: str | None) -> None:
    """A folder that isn't a bundle can hide the installed bundle the user meant. The note names it by its
    qualified name, which only helps when that isn't what was typed and isn't a folder too."""
    if id_root is not None or _path_like(text) or _has_kind_folder(root):
        return
    try:
        found = find_bundle(text)
    except Exception:
        return
    if os.path.isdir(found.qualified):
        return
    error.add_note(f"an installed bundle is also named {text!r}: run it with agent-env run {found.qualified}")


def _has_kind_folder(root: Path) -> bool:
    return any(os.path.isdir(root / kind.value) for kind in BundleKind)


def _installed() -> tuple[InstalledBundle, ...]:
    try:
        return installed_bundles()
    except Exception as e:
        raise click.ClickException(f"the installed bundles can't be read: {discovery_error(e)}") from e


def _list_installed() -> None:
    bundles = _installed()
    if not bundles:
        _echo("No bundles are installed. agent-env run PATH runs a folder.")
        return
    header = ("NAME", "PACKAGE", "CONTENTS", "DESCRIPTION")
    rows = [(run_name(bundle, bundles), f"{bundle.package} {bundle.version}", *_contents(bundle)) for bundle in bundles]
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(3)]

    def echo(name: str, package: str, contents: str, description: str) -> None:
        _echo(f"{name:<{widths[0]}}  {package:<{widths[1]}}  {contents:<{widths[2]}}  {description}".rstrip())

    echo(*header)
    for bundle, row in zip(bundles, rows):
        echo(*row)
        if bundle.root is not None:
            _echo(f"{'':<{widths[0]}}  {bundle.root}")
    _echo("\nRun one with agent-env run NAME. To change one, copy its folder and run the copy.")


def _contents(bundle: InstalledBundle) -> tuple[str, str]:
    """What ``bundle`` holds and its description, or why it can't run."""
    if bundle.problem:
        return "invalid", bundle.problem
    try:
        parsed = checked(bundle)
    except BundleError as e:
        return "invalid", e.summary
    counts = [_count(parsed.entries, kind, noun) for kind, noun in ((BundleKind.TASK, "task"), (BundleKind.EVAL, "eval"))]
    return ", ".join(count for count in counts if count), parsed.description or ""


def _count(entries, kind: BundleKind, noun: str) -> str:
    n = sum(entry.kind is kind for entry in entries)
    return f"{n} {noun}{'s' if n != 1 else ''}" if n else ""


def _summarize(result: BundleRun, verbose: bool) -> None:
    _echo()
    _echo("Tasks:")
    for run in result.runs:
        _echo(click.style(f"  {_described(result, run)}", fg=_COLORS[run.outcome]))
        for line in _criteria_lines(run):
            _echo(f"    {line}")
    if result.evals:
        _echo("Evals:")
    for eval_run in result.evals:
        passed, failed, cancelled = (sum(run.outcome is outcome for run in eval_run.runs)
                                     for outcome in (Outcome.PASSED, Outcome.FAILED, Outcome.CANCELLED))
        line = f"  {result.path(eval_run.entry)} v{eval_run.version}: {passed} of {len(eval_run.runs)} passed"
        rest = [f"{failed} failed" if failed else "", f"{cancelled} cancelled" if cancelled else ""]
        _echo(", ".join([line, *filter(None, rest)]))
    for entry in result.skipped:
        _echo(f"{result.path(entry)} isn't named by any eval, so it didn't run; run it with "
              f"--task {shlex.quote(entry.name)}")
    if verbose:
        _print_tracebacks(result)


def _criteria_lines(run: TaskRun) -> list[str]:
    """A line for each criterion the run failed, its step, the criterion and the reason it gave, up to
    ``_SHOWN_CRITERIA``, then a count of the rest."""
    failed = run.failed_criteria
    lines = []
    for criterion in failed[:_SHOWN_CRITERIA]:
        reason = " ".join(criterion.reason.split())
        if len(reason) > _REASON_CHARS:
            reason = reason[:_REASON_CHARS - 1].rstrip() + "…"
        lines.append(f"{criterion.step}: {criterion.criterion}" + (f": {reason}" if reason else ""))
    if len(failed) > _SHOWN_CRITERIA:
        lines.append(f"and {len(failed) - _SHOWN_CRITERIA} more")
    return lines


def _report_dry_run(dry: DryRun) -> None:
    materialization = dry.materialization
    plan = materialization.plan
    _echo()
    if plan.store_refs:
        _echo("Store refs:")
    for ref in plan.store_refs:
        if ref.version is None:
            _echo(f"  {ref.kind} {ref.id} v{plan.store_latest[ref.kind, ref.id]}, the latest")
        else:
            _echo(f"  {ref.kind} {ref.id} v{ref.version}")
    if images := dry.images():
        _echo("Images:")
    for line in images:
        _echo(f"  {line}")
    if dry.infra:
        _echo("Would build first:")
    for build in dry.infra:
        _echo(f"  {build}")
    _echo("Would run:")
    for entry in dry.runs:
        _echo(f"  {dry.path(entry)} v{materialization.version_of('task', entry.id)}")
    if plan.evals:
        _echo("Evals:")
    for entry in plan.evals:
        named = ", ".join(dry.path(ref.local) for ref in entry.references)
        _echo(f"  {dry.path(entry.entry)} v{materialization.version_of('eval', entry.entry.id)}: {named}")
    for entry in dry.skipped:
        _echo(f"{dry.path(entry)} isn't named by any eval, so it wouldn't run; run it with "
              f"--task {shlex.quote(entry.name)}")
    if materialization.not_preflighted:
        _echo("Not preflighted, since each reads what the run would write first:")
    for write, step in materialization.not_preflighted:
        _echo(f"  {dry.path(write.source.entry)}: step {step.id!r} ({step.type})")
    _echo(_DRY_RUN)


def _print_tracebacks(result: BundleRun) -> None:
    for run in result.runs:
        if run.error is not None:
            _echo(f"\n{result.path(run.entry)} raised:", err=True)
            _echo("".join(traceback.format_exception(run.error)).rstrip(), err=True)


def _described(result: BundleRun, run: TaskRun) -> str:
    if not run.started:
        return f"{result.path(run.entry)} v{run.task.version}: didn't start"
    return (f"{result.path(run.entry)} v{run.task.version}: {_outcome(run)}, {run.duration:.1f}s, "
            f"{f'instance {run.instance_id}' if run.instance_id else 'no instance recorded'}")


def _outcome(run: TaskRun) -> str:
    if run.outcome is Outcome.FAILED:
        if not run.failed_steps:
            return f"failed: {type(run.error).__name__}: {run.error}"
        return "failed at " + "; ".join(f"step {failure['step_id']!r}: {failure['error_type']}: {failure['error']}"
                                        for failure in run.failed_steps)
    scores = ", ".join(f"{step_id}: {score:g}" for step_id, score in run.scores.items())
    return f"{run.outcome.value} ({scores})" if scores else run.outcome.value


def _hold(result: BundleRun, interrupts: Interrupts) -> BundleRun:
    """Print what the runs keep up, wait for Ctrl-C or SIGTERM, then tear it down; another one stops that."""
    kept = [run for run in result.runs if run.torn_down is None]
    count = sum(len(TeardownReport.skipped(run.context).left) for run in kept)
    if not count:
        _echo("\nNothing to keep up: no run left a sandbox.")
        return result
    _echo("\nKept up:")
    for run in kept:
        lines = _kept_up(run.context)
        if lines:
            _echo(f"  {result.path(run.entry)}")
            for line in lines:
                _echo(f"    {line}")
    them = "it" if count == 1 else "them"
    _echo(f"\nHolding {_sandboxes(count)} up; Ctrl-C tears {them} down.")
    while not interrupts.count:
        time.sleep(0.2)
    if interrupts.count > 1:
        return result
    _echo(f"\nTearing down {_sandboxes(count)} (Ctrl-C again to stop now)")
    try:
        return result.teardown()
    except RunInterrupted as stopped:
        return stopped.result


def _kept_up(context: TaskStepContext) -> list[str]:
    """Each sandbox still up: the records behind it, the endpoints of those it is the sandbox of, and a local
    one's folder."""
    torn_down = set(context.metadata.get(TORN_DOWN_KEY) or [])
    records = _records(context)
    lines = []
    for sandbox in recorded_sandboxes(context):
        if sandbox.sandbox_id in torn_down:
            continue
        behind = [(label, record, endpoints) for label, record, endpoints in records
                  if sandbox.sandbox_id in sandbox_ids(record)]
        lines.append(f"{sandbox.sandbox_id}  {kind(sandbox)}  {', '.join(label for label, _, _ in behind)}")
        for _, record, endpoints in behind:
            if getattr(record, "sandbox_id", None) == sandbox.sandbox_id:
                lines.extend(f"  {label}  {value}" for label, value in endpoints if value)
        if folder := LocalSandbox.find_work_dir(sandbox.sandbox_id):
            lines.append(f"  folder  {folder}")
    return lines


def _records(context: TaskStepContext) -> list[tuple[str, object, list[tuple[str, str | None]]]]:
    """Every deployed record in ``context``, labelled, with the endpoints it records."""
    records = []
    for sandbox in context.deployed_sandboxes:
        ports = [(f"port {port}", url) for port, url in (sandbox.tunnel_urls or {}).items()]
        records.append((f"sandbox {sandbox.sandbox_name}", sandbox,
                        [*ports, ("vnc", sandbox.vnc_url), ("expires", sandbox.expires_at_utc)]))
    for env in context.deployed_envs:
        endpoints = [(label, getattr(env, field, None)) for label, field in _ENV_ENDPOINTS]
        sites = [(f"site {name}", url) for name, url in (getattr(env, "website_frontend_urls", None) or {}).items()]
        records.append((f"env {env.env_id}", env, [*endpoints, *sites]))
    for agent in context.deployed_agents:
        records.append((f"agent {agent.agent_name}", agent, [("a2a", agent.a2a_url or agent.api_url)]))
    return records


def _report_teardown(result: BundleRun) -> None:
    reports = [run.torn_down or TeardownReport.skipped(run.context) for run in result.runs]
    terminated = sum(len(report.terminated) for report in reports)
    still_up = [sandbox for report in reports for sandbox in report.still_up
                if sandbox.sandbox_type != "local" or LocalSandbox.find_work_dir(sandbox.sandbox_id)]
    if terminated or still_up:
        _echo()
    if terminated:
        _echo(f"Tore down {_sandboxes(terminated)}.")
    if still_up:  # on stderr, which still reaches the user when stdout is piped away
        _echo(click.style(f"Still up, {_sandboxes(len(still_up))}:", fg="yellow"), err=True)
        for sandbox in still_up:
            _echo(f"  {_still_up(sandbox)}", err=True)


def _still_up(sandbox: RecordedSandbox) -> str:
    folder = LocalSandbox.find_work_dir(sandbox.sandbox_id)
    return "  ".join([sandbox.sandbox_id, kind(sandbox), *([str(folder)] if folder else [])])


def _sandboxes(n: int) -> str:
    return f"{n} sandbox{'es' if n != 1 else ''}"
