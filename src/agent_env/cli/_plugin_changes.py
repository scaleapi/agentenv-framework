"""`agent-env plugin add` and `remove`: plan, preview, run through the installer, verify, roll back.

Every check that needs installed state runs in a fresh interpreter of the environment, because
this process has already imported its plugins and cannot see a change to them.
"""

import difflib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import unquote, urlparse

import click

from agent_env.cli import _installers
from agent_env.cli._installers import Environment, InstallerError, Plan, ToolReceipt, normalize
from agent_env.cli.utils import interruptible
from agent_env.config import get_config
from agent_env.config.loader import interpolate
from agent_env.config.paths import state_root
from agent_env.store.routing import RoutingDocumentStore
from agent_env.plugins import ARTIFACTS, ENV_PROVIDERS, ENVS, SANDBOX_PROVIDERS, STATE_PROVIDERS, TASK_STEPS
from agent_env.plugins._report import FORMAT_VERSION
from agent_env.store.document_store.document_store import Filter, In

try:
    import fcntl
except ImportError:  # Windows, which agent-env does not target: changes run without the lock.
    fcntl = None

# Every installed distribution: its version and source, what this environment needs of it, its
# agent_env entry points, and for a plugin the modules it owns.
_SNAPSHOT_PROBE = r"""
import importlib.metadata as md, json, re, sys
try:
    from packaging.requirements import Requirement
except Exception:
    Requirement = None
def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()
def needed(requirement, extra=""):
    if Requirement is None:
        return not extra and "extra ==" not in requirement.replace('"', "").replace("'", "")
    try:
        marker = Requirement(requirement).marker
    except Exception:
        return not extra
    return marker is None or marker.evaluate({"extra": extra})
def extras_of(requirement):
    try:
        return sorted(norm(e) for e in Requirement(requirement).extras) if Requirement else []
    except Exception:
        return []
def modules(dist, values):
    found = set()
    for f in dist.files or []:
        if f.suffix == ".py" and f.parts and not f.parts[0].startswith(".."):
            dotted = list(f.parts[:-1]) + ([] if f.stem == "__init__" else [f.stem])
            if dotted and all(part.isidentifier() for part in dotted):
                found.add(".".join(dotted))
    if not found:
        # An editable install lists no source files; its entry points still name its modules.
        for value in values:
            module = value.split(":", 1)[0].strip()
            found.update(filter(None, {module, module.rpartition(".")[0]}))
    return sorted(found)
out = {}
for dist in md.distributions():
    try:
        name = dist.metadata["Name"]
        if not name:
            continue
        requires, wants, extras = [], {}, {}
        provided = [norm(e) for e in dist.metadata.get_all("Provides-Extra") or []]
        for requirement in dist.requires or []:
            match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
            if not match:
                continue
            dep = norm(match.group(1))
            if needed(requirement):
                requires.append(dep)
                if extras_of(requirement):
                    wants[dep] = sorted(set(wants.get(dep, [])) | set(extras_of(requirement)))
            for extra in provided:
                if not needed(requirement) and needed(requirement, extra):
                    asked = extras.setdefault(extra, {}).setdefault(dep, [])
                    asked[:] = sorted(set(asked) | set(extras_of(requirement)))
        points = [ep for ep in dist.entry_points if ep.group.startswith("agent_env.")]
        direct = dist.read_text("direct_url.json")
        # requires: what it always needs; wants: the extras it asks of those; extras: what each of
        # its own extras adds, with the extras it asks of each.
        entry = {"name": name, "version": dist.version, "requires": requires, "wants": wants, "extras": extras,
                 "entry_points": [[ep.group, ep.name] for ep in points],
                 "direct_url": json.loads(direct) if direct else None}
        if points:
            entry["modules"] = modules(dist, [ep.value for ep in points])
    except Exception:
        continue
    out[norm(name)] = entry
json.dump(out, sys.stdout)
"""
_OK = ("active", "replaced")
_UNNAMED_PIPX_HINT = ("pipx leaves a package it already has as it is. To upgrade an installed plugin from a URL or a "
                      "directory, name it: `agent-env plugin add 'NAME @ URL'`.")
# How long the fresh-interpreter snapshot, the `plugin show` check and the resolver dry run may take.
_SNAPSHOT_TIMEOUT_S = 120
_INSPECT_TIMEOUT_S = 300
_PREVIEW_TIMEOUT_S = 600

# How long `show` and `remove` wait for a package's entry points to import in the fresh interpreter.
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
    # A bundle is a folder; agent-env never imports its package.
    if ep.group.startswith("agent_env.") and ep.group != "agent_env.bundles":
        try:
            ep.load()
        except Exception:
            pass
with open(sys.argv[2], "w") as out:
    json.dump(os.environ.get("AGENT_ENV_CONFIG"), out)
"""

Snapshot = dict[str, dict]
Runner = Callable[[tuple[str, ...], Optional[Path]], int]


def run(command: tuple[str, ...], cwd: Optional[Path] = None) -> int:
    """Run one installer command, its output going straight to the terminal.

    The installer starts with Ctrl-C ignored: stopped halfway, it leaves packages without their
    metadata, which no installer can read or roll back. Ctrl-C reaches agent-env, which lets it
    finish and then rolls back; a second Ctrl-C stops it at once. It keeps the terminal, so its own
    prompts, such as git's for credentials, still work.
    """
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        proc = subprocess.Popen(list(command), cwd=cwd)
    finally:
        signal.signal(signal.SIGINT, previous)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        click.echo("Interrupted; letting the installer finish before restoring (Ctrl-C again stops it now).", err=True)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            _wait_out(proc)
        raise


def _wait_out(proc: subprocess.Popen) -> None:
    while True:
        try:
            proc.wait()
            return
        except KeyboardInterrupt:
            proc.kill()


def broken() -> set[Path]:
    """Package records in this environment without their METADATA: what an installer stopped
    halfway leaves. No installer can read them, so every later change fails until they are gone."""
    roots = {Path(sysconfig.get_path("purelib")), Path(sysconfig.get_path("platlib"))}
    return {record for root in roots if root.is_dir() for record in root.glob("*.dist-info")
            if not (record / "METADATA").is_file()}


def snapshot(python: Path) -> Snapshot:
    """The environment's installed distributions, read by a fresh interpreter."""
    proc = subprocess.run(
        [str(python), "-c", _SNAPSHOT_PROBE], capture_output=True, text=True, timeout=_SNAPSHOT_TIMEOUT_S
    )
    if proc.returncode != 0:
        raise InstallerError(f"could not read the installed packages: {proc.stderr.strip() or proc.returncode}")
    return json.loads(proc.stdout)


def inspect(python: Path, dist: str) -> dict:
    """`plugin show DIST --json`, run by a fresh interpreter without the current folder on its path, so the
    installed code is what loads."""
    proc = subprocess.run(
        [str(python), "-P", "-m", "agent_env.cli", "plugin", "show", dist, "--json"],
        capture_output=True, text=True, timeout=_INSPECT_TIMEOUT_S,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit status {proc.returncode}"
        return {"error": detail}
    try:
        return json.loads(proc.stdout)
    except ValueError as exc:
        return {"error": f"`plugin show {dist} --json` printed no JSON report ({exc})"}


class ProbeError(Exception):
    """A fresh-interpreter check that could not tell."""


def config_set_by(name: str) -> Optional[str]:
    """The AGENT_ENV_CONFIG that importing ``name``'s entry points sets, or None. Checked in a
    fresh interpreter: this process has already imported the CLI plugins, so it cannot tell."""
    env = {k: v for k, v in os.environ.items() if k != "AGENT_ENV_CONFIG"}
    with tempfile.TemporaryDirectory() as scratch:
        result = Path(scratch) / "config.json"
        try:
            proc = subprocess.run(
                [sys.executable, "-P", "-c", _CONFIG_PROBE, name, str(result)],
                capture_output=True, text=True, env=env, timeout=_PROBE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise ProbeError(f"importing it took more than {_PROBE_TIMEOUT_S}s") from None
        try:
            return json.loads(result.read_text())
        except (OSError, ValueError):
            last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit status {proc.returncode}"
            raise ProbeError(last) from None


def installs(name: str, path: Path) -> bool:
    """Whether ``path`` is one of the files installed with ``name``, so uninstalling it deletes
    the file. An editable install records none of its sources, so it never qualifies."""
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    target = Path(path).resolve()
    return any(Path(dist.locate_file(file)).resolve() == target for file in dist.files or [])


def current_environment(installer: Optional[str] = None) -> Environment:
    """The environment this agent-env runs in, and the installer that owns it."""
    env = _installers.detect(
        Path(sys.prefix), Path(sys.base_prefix), Path(sys.executable),
        has_pip=importlib.util.find_spec("pip") is not None,
        project_environment=os.environ.get("UV_PROJECT_ENVIRONMENT"), cwd=Path.cwd(),
    )
    env = _installers.forced(env, installer)
    return _installers.with_refusal(
        env, stdlib=Path(sysconfig.get_path("stdlib")), site_packages=Path(sysconfig.get_path("purelib"))
    )


@dataclass
class Tools:
    """The side-effecting pieces, swapped out by the unit tests."""

    run: Runner = run
    snapshot: Callable[[Path], Snapshot] = snapshot
    inspect: Callable[[Path, str], dict] = inspect
    config_set_by: Callable[[str], Optional[str]] = config_set_by
    installs: Callable[[str, Path], bool] = installs
    broken: Callable[[], set[Path]] = broken


@dataclass
class _Rollback:
    """What a change restores when it does not take effect."""

    env: Environment
    before: Snapshot
    files: dict[Path, bytes]
    broken: set[Path] = field(default_factory=set)
    receipt: Optional[ToolReceipt] = None
    injected: set[str] = field(default_factory=set)


def _save(env: Environment, before: Snapshot, tools: Tools) -> _Rollback:
    saved = _Rollback(env, before, _files(env), tools.broken())
    if env.kind == _installers.UV_TOOL:
        saved.receipt = ToolReceipt.read(env.prefix / "uv-receipt.toml")
    if env.kind == _installers.PIPX:
        saved.injected = _injected(env)
    return saved


def add(specs: list[str], *, installer: Optional[str], index_url: Optional[str], dry_run: bool, yes: bool,
        keep: bool, tools: Tools) -> int:
    env = current_environment(installer)
    with _locked(env):
        plan = _installers.add_plan(env, specs, index_url=index_url)
        planned = _files(env)
        _describe(plan)
        if not plan.runs:
            return 0
        if env.kind in (_installers.VIRTUALENV, _installers.SYSTEM) and not _preview(env, specs, index_url):
            return 1
        if dry_run:
            return 0
        if not _proceed(yes):
            click.echo("Nothing changed.")
            return 1
        _still_as_planned(env, planned)
        saved = _save(env, tools.snapshot(env.python), tools)
        unnamed = env.kind == _installers.PIPX and None in {_installers.requirement_name(s) for s in specs}
        hint = _UNNAMED_PIPX_HINT if unnamed else None
        return _apply(saved, lambda: _install_and_verify(plan, saved, specs, tools, unchanged_hint=hint), tools,
                      keep=keep)


def _still_as_planned(env: Environment, planned: dict[Path, bytes]) -> None:
    """The plan was built from the installer's record before the prompt; if something else changed
    the record while it waited, running it would undo that change."""
    if _files(env) != planned:
        raise InstallerError(
            "the installer's record of this environment changed while this waited to be confirmed, so the plan is "
            "out of date; nothing was changed. Run the command again."
        )


def _install_and_verify(plan: Plan, saved: _Rollback, specs: list[str], tools: Tools, *,
                        unchanged_hint: Optional[str] = None) -> list[str]:
    """Run the installer and check what it added. Empty when the change took effect."""
    env = plan.environment
    for command in plan.commands:
        if tools.run(command, plan.cwd) != 0:
            return ["the installer failed"]
    after = tools.snapshot(env.python)
    _show_changes(saved.before, after)
    if env.kind == _installers.UV_PROJECT:
        for path in _record(env, lock=False):
            _show_diff(saved.files, path)
    if lost := _lost(saved.before, after):
        return [f"the installer removed {', '.join(lost)}, which this change did not ask for"]
    if not _changed(saved.before, after):
        if _files(env) == saved.files:
            click.echo("Nothing changed: the installer left every package as it was.")
            if unchanged_hint:
                click.echo(unchanged_hint)
            return []
        # Only the installer's record changed, so what it names must be a plugin that is here.
        named = [key for key in map(_installers.requirement_name, specs) if key in after]
        if not any(after[key]["entry_points"] for key in named):
            return ["none of the packages it names declares an agent_env entry point, so it is not an agent-env plugin"]
        for path in _record(env, lock=False) if env.kind != _installers.UV_PROJECT else ():
            _show_diff(saved.files, path)
        click.echo("Recorded. The installed packages did not change.")
        return []
    problems = _core_replaced(saved.before, after) or _verify(env, saved.before, after, tools)
    if not problems:
        click.echo("Added. `agent-env plugin list` shows every installed plugin.")
    return problems


def _lost(before: Snapshot, after: Snapshot, expected: frozenset[str] = frozenset()) -> list[str]:
    """Plugins the installer removed that the change did not name."""
    return sorted(before[key]["name"] for key in before
                  if key not in after and key not in expected and before[key]["entry_points"])


def _core_replaced(before: Snapshot, after: Snapshot) -> list[str]:
    """agent-env itself installed from a URL, git or a directory it did not come from: only a spec
    that built as agent-env does that, and a bare URL or directory cannot be refused before it
    builds. An upgrade from the index, which a plugin's requirements may need, is not refused."""
    found = []
    for key in _installers.CORE:
        new = after.get(key)
        if new and new.get("direct_url") and not _same(before.get(key), new):
            found.append(f"{new['name']} now comes from {new['direct_url'].get('url')}: a spec built as agent-env "
                         "itself, which `plugin add` does not replace. Upgrade it with the installer that owns the "
                         "environment.")
    return found


def remove(names: list[str], *, installer: Optional[str], dry_run: bool, yes: bool, force: bool, check_usage: bool,
           tools: Tools) -> int:
    env = current_environment(installer)
    with _locked(env):
        installed = tools.snapshot(env.python)
        wanted = _installers.recorded(env)
        targets = [normalize(name) for name in names]
        for name, key in zip(names, targets):
            if key in _installers.CORE:
                raise InstallerError(f"{name} is agent-env itself; uninstall it with the installer that owns it")
            if key not in installed:
                raise InstallerError(f"{name} is not installed in this environment")
            # A package `plugin add` recorded next to a plugin, such as a pin the plugin needs, can
            # go the way it came; anything else without entry points belongs to what requires it.
            if not installed[key]["entry_points"] and key not in (wanted or ()):
                raise InstallerError(
                    f"{installed[key]['name']} declares no agent_env entry points, so it is not a plugin"
                )
        also = _taken_along(env, installed, targets, wanted)
        if also:
            click.echo(f"This also removes {', '.join(installed[key]['name'] for key in also)}: nothing that stays "
                       f"needs {'it' if len(also) == 1 else 'them'}.")
        checked = targets + also
        blockers = _dependents(installed, checked) + _config_references(installed, checked, tools)
        blockers += _stored_usage(installed, checked, check_remote=check_usage)
        for blocker in blockers:
            click.echo(f"{'warning' if force else 'blocked'}: {blocker}", err=True)
        if blockers and not force:
            raise InstallerError("not removed; fix the above, or pass --force to remove anyway")
        plan = _installers.remove_plan(env, [installed[key]["name"] for key in targets],
                                       also=[installed[key]["name"] for key in also])
        planned = _files(env)
        _describe(plan)
        if left := _left_behind(installed, checked, wanted):
            one = len(left) == 1
            click.echo(f"note: {', '.join(left)} {'stays' if one else 'stay'} installed: the installer records "
                       f"{'it' if one else 'them'} on {'its' if one else 'their'} own, though only what this removes "
                       f"needed {'it' if one else 'them'}. `agent-env plugin remove {' '.join(left)}` removes "
                       f"{'it' if one else 'them'} too.", err=True)
        if not plan.runs or dry_run:
            return 0
        if not _proceed(yes):
            click.echo("Nothing changed.")
            return 1
        _still_as_planned(env, planned)
        saved = _save(env, installed, tools)
        return _apply(saved, lambda: _uninstall(plan, saved, targets, also, tools), tools)


Wanted = dict[str, frozenset[str]]


def _needed(installed: Snapshot, roots: Wanted) -> set[str]:
    """``roots`` and everything they require as installed, following the extras each asks for."""
    found, seen, todo = set(), set(), list(roots.items())
    while todo:
        key, extras = todo.pop()
        if key not in installed or (key, extras) in seen:
            continue
        seen.add((key, extras))
        found.add(key)
        dist = installed[key]
        wants = dist.get("wants") or {}
        todo += [(dep, frozenset(wants.get(dep, ()))) for dep in dist["requires"]]
        for extra in extras:
            todo += [(dep, frozenset(asked)) for dep, asked in ((dist.get("extras") or {}).get(extra) or {}).items()]
    return found


def _roots(installed: Snapshot, wanted: Wanted, without: list[str] = ()) -> Wanted:
    """What the installer keeps installed on its own, agent-env included, less ``without``."""
    roots = {key: extras for key, extras in wanted.items() if key in installed and key not in without}
    return {**{key: frozenset() for key in _installers.CORE if key in installed}, **roots}


def _taken_along(env: Environment, installed: Snapshot, targets: list[str], wanted: Optional[Wanted]) -> list[str]:
    """Other plugins that go with ``targets``: needed before, and by nothing the installer keeps
    after. A uv tool's rebuild and a uv project's lock drop them; pip and pipx leave them."""
    if wanted is None or env.kind not in (_installers.UV_TOOL, _installers.UV_PROJECT):
        return []
    gone = _needed(installed, _roots(installed, wanted)) - _needed(installed, _roots(installed, wanted, targets))
    return sorted(key for key in gone - set(targets) if installed[key]["entry_points"])


def _left_behind(installed: Snapshot, removed: list[str], wanted: Optional[Wanted]) -> list[str]:
    """Packages the record lists on their own that only ``removed`` needed, such as the pins added
    with a plugin. They stay, since the record asks for them."""
    if not wanted:
        return []
    needed_by_removed = _needed(installed, {key: wanted.get(key, frozenset()) for key in removed})
    return sorted(installed[key]["name"] for key in wanted
                  if key in installed and key not in removed and key not in _installers.CORE
                  and not installed[key]["entry_points"] and key in needed_by_removed
                  and key not in _needed(installed, _roots(installed, wanted, [*removed, key])))


def _uninstall(plan: Plan, saved: _Rollback, targets: list[str], also: list[str], tools: Tools) -> list[str]:
    for command in plan.commands:
        if tools.run(command, plan.cwd) != 0:
            return ["the installer failed"]
    after = tools.snapshot(plan.environment.python)
    _show_changes(saved.before, after)
    left = [saved.before[key]["name"] for key in targets if key in after]
    if left:
        return [f"still installed after the installer ran: {', '.join(left)}"]
    if lost := _lost(saved.before, after, frozenset(targets + also)):
        names = " ".join([saved.before[key]["name"] for key in targets] + lost)
        return [f"the installer also removed {', '.join(lost)}, which this change did not name; "
                f"`agent-env plugin remove {names}` removes them together"]
    click.echo("Removed.")
    return []


def _apply(saved: _Rollback, change: Callable[[], list[str]], tools: Tools, *, keep: bool = False) -> int:
    """Make the change, and put the environment back as it was unless it took effect."""
    with interruptible():  # so SIGTERM reaches the rollback
        try:
            problems = change()
        except KeyboardInterrupt:
            click.echo("Interrupted; restoring the environment as it was.", err=True)
            _restore(saved, tools)
            return 130
        except Exception as exc:
            problems = [f"could not finish the change: {exc}"]
    if not problems:
        return 0
    for problem in problems:
        click.echo(f"problem: {problem}", err=True)
    if keep:
        click.echo("Kept as installed (--keep).", err=True)
        return 1
    click.echo("Restoring the environment as it was.", err=True)
    _restore(saved, tools)
    return 1


@contextmanager
def _locked(env: Environment) -> Iterator[None]:
    """One `plugin add` or `remove` at a time per environment, held from reading the environment to
    the result so neither runs a plan the other has made stale. The lock lives outside the
    environment, which the installer may recreate, in a directory only this user can write to."""
    if fcntl is None:
        yield
        return
    path = _lock_path(env)
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise InstallerError(f"cannot take the lock {path}: {exc.strerror or exc}") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise InstallerError(
                "another `agent-env plugin add` or `remove` is changing this environment; run this after it finishes"
            ) from None
        yield
    finally:
        os.close(fd)


def _lock_path(env: Environment) -> Path:
    digest = hashlib.sha256(str(env.prefix.resolve()).encode()).hexdigest()[:16]
    return state_root() / "locks" / f"plugin-{digest}.lock"


@contextmanager
def _uninterrupted() -> Iterator[None]:
    """Ctrl-C and SIGTERM wait until the restore finishes; stopping halfway would leave the
    environment in neither state."""
    previous = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _describe(plan: Plan) -> None:
    env = plan.environment
    click.echo(f"agent-env: {env.kind} at {env.location}")
    if plan.note:
        click.echo(plan.note)
    for command in plan.commands:
        prefix = f"cd {shlex.quote(str(plan.cwd))} && " if plan.cwd and not plan.runs else ""
        click.echo(f"{'Run' if not plan.runs else 'Will run'}: {prefix}{shlex.join(command)}")


def _proceed(yes: bool) -> bool:
    return yes or click.confirm("Proceed?", default=False)


def _preview(env: Environment, specs: list[str], index_url: Optional[str]) -> bool:
    """A resolver dry run, for the installers that have one. False when it cannot resolve."""
    flags = ("--index-url", index_url) if index_url else ()
    if env.has_pip:
        command = [str(env.python), "-m", "pip", "install", "--dry-run", *flags, *specs]
    elif shutil.which("uv"):
        command = ["uv", "pip", "install", "--dry-run", "--python", str(env.python), *specs]
        command += ["--default-index", index_url] if index_url else []
    else:
        return True
    proc = subprocess.run(command, capture_output=True, text=True, timeout=_PREVIEW_TIMEOUT_S)
    lines = [ln.strip() for ln in (proc.stdout + proc.stderr).splitlines()]
    if proc.returncode != 0:
        click.echo("The installer cannot resolve this:", err=True)
        for line in lines[-8:]:
            click.echo(f"  {line}", err=True)
        return False
    plan = [ln for ln in lines if ln.startswith(("Would install", "+ ", "- "))]
    for line in plan:
        core = " (agent-env itself)" if _names_in(line) & set(_installers.CORE) else ""
        click.echo(f"  {line}{core}")
    return True


def _names_in(line: str) -> set[str]:
    """The packages a resolver's dry-run line names: `+ name==1.0` or `+ name @ url` from uv, and
    `Would install name-1.0 other-2.0` from pip."""
    if line.startswith(("+ ", "- ")):
        return {_installers.requirement_name(line[2:].strip())} - {None}
    return {normalize(word.rsplit("-", 1)[0]) for word in line.split()[2:]}


def _record(env: Environment, *, lock: bool = True) -> list[Path]:
    """The files in which the installer records this environment."""
    if env.kind == _installers.UV_PROJECT:
        return [*_installers.project_files(env), *([env.location / "uv.lock"] if lock else [])]
    return {
        _installers.UV_TOOL: [env.prefix / "uv-receipt.toml"],
        _installers.PIPX: [env.prefix / "pipx_metadata.json"],
    }.get(env.kind, [])


def _files(env: Environment) -> dict[Path, bytes]:
    """The installer's record of this environment, kept byte for byte to restore."""
    return {path: path.read_bytes() for path in _record(env) if path.is_file()}


def _injected(env: Environment) -> set[str]:
    metadata = json.loads((env.prefix / "pipx_metadata.json").read_text())
    return {normalize(name) for name in metadata.get("injected_packages") or {}}


def _same(a: Optional[dict], b: Optional[dict]) -> bool:
    """Whether two installs are the same version from the same source. The requested git revision
    is left out, since a rollback reinstalls by commit; a hash counts only when both installs
    recorded one, since uv records none."""
    if a is None or b is None:
        return a is b
    da, db = a.get("direct_url") or {}, b.get("direct_url") or {}
    if (a["version"], _origin(da)) != (b["version"], _origin(db)):
        return False
    ha, hb = _sha256(da), _sha256(db)
    return ha is None or hb is None or ha == hb


def _origin(direct: dict) -> tuple:
    return (direct.get("url"), (direct.get("vcs_info") or {}).get("commit_id"),
            bool((direct.get("dir_info") or {}).get("editable")), direct.get("subdirectory"))


def _sha256(direct: dict) -> Optional[str]:
    return ((direct.get("archive_info") or {}).get("hashes") or {}).get("sha256")


def _changed(before: Snapshot, after: Snapshot) -> list[str]:
    return sorted(key for key in set(before) | set(after) if not _same(before.get(key), after.get(key)))


def _show_changes(before: Snapshot, after: Snapshot) -> None:
    for key in _changed(before, after):
        old, new = before.get(key), after.get(key)
        core = " (agent-env itself)" if key in _installers.CORE else ""
        if old is None:
            click.echo(f"  + {new['name']} {new['version']}{core}")
        elif new is None:
            click.echo(f"  - {old['name']} {old['version']}{core}")
        elif old["version"] != new["version"]:
            click.echo(f"  ~ {new['name']} {old['version']} -> {new['version']}{core}")
        else:
            click.echo(f"  ~ {new['name']} {new['version']}, from a different source{core}")


def _show_diff(files: dict[Path, bytes], path: Path) -> None:
    if path not in files or not path.is_file():
        return
    old = files[path].decode().splitlines(keepends=True)
    new = path.read_text().splitlines(keepends=True)
    for line in difflib.unified_diff(old, new, f"{path.name} (before)", f"{path.name} (after)"):
        click.echo(line.rstrip("\n"))


def _verify(env: Environment, before: Snapshot, after: Snapshot, tools: Tools) -> list[str]:
    """Why the change did not take effect, or nothing."""
    plugins = [key for key in _changed(before, after) if key in after and after[key]["entry_points"]]
    if all(key in _installers.CORE for key in plugins):
        return ["none of the packages it installed declares an agent_env entry point, so it is not an agent-env plugin"]
    problems = []
    for key in plugins:
        name = after[key]["name"]
        report = tools.inspect(env.python, name)
        if "error" in report:
            problems.append(f"{name}: {report['error']}")
            continue
        # A report from before format 1 has no version but the same shape, so it is read as format 1.
        version = report.get("format_version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            problems.append(f"{name}: after the change agent-env reports plugins in format {version!r}, but this "
                            f"agent-env reads format {FORMAT_VERSION}; upgrade agent-env first, then add the plugin")
            continue
        if not isinstance(report.get("plugins"), list) or not report["plugins"]:
            problems.append(f"{name}: `plugin show {name} --json` reported no plugin package")
            continue
        for plugin in report["plugins"]:
            for c in plugin["contributions"]:
                status = c["status"]
                why = f" ({c['reason']})" if c.get("reason") else ""
                click.echo(f"  {name}: {c['group'].removeprefix('agent_env.')} {c['name']} {status}{why}")
                if status not in _OK:
                    problems.append(f"{name}: {c['name']} is {status}{why}")
        if report.get("config_effect"):
            click.echo(f"  {name}: {report['config_effect']}")
    return problems


def _pinned(entry: dict) -> list[str]:
    """Installer arguments that reinstall ``entry`` exactly: the same version from the same source."""
    direct = entry.get("direct_url")
    if not direct:
        return [f"{entry['name']}=={entry['version']}"]
    url = direct["url"]
    if direct.get("dir_info", {}).get("editable"):
        return ["-e", unquote(urlparse(url).path) if url.startswith("file:") else url]
    vcs = direct.get("vcs_info")
    if vcs:
        url = f"{vcs['vcs']}+{url}@{vcs.get('commit_id') or vcs.get('requested_revision')}"
    subdirectory = f"#subdirectory={direct['subdirectory']}" if direct.get("subdirectory") else ""
    return [f"{entry['name']} @ {url}{subdirectory}"]


def _restore(saved: _Rollback, tools: Tools) -> None:
    with _uninterrupted():
        _restore_now(saved, tools)


def _restore_now(saved: _Rollback, tools: Tools) -> None:
    env = saved.env
    # Records an installer stopped halfway left behind: no installer can read them, so the restore
    # itself would fail on them. What they stood for is reinstalled below.
    for record in sorted(tools.broken() - saved.broken):
        shutil.rmtree(record, ignore_errors=True)
    now = tools.snapshot(env.python)
    if not _changed(saved.before, now):
        pass  # the change never reached the packages; only the record may need putting back
    elif env.kind == _installers.UV_TOOL and saved.receipt is not None:
        # The old requirements alone keep whatever newer versions still satisfy them.
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as constraints:
            constraints.write("".join(f"{e['name']}=={e['version']}\n" for e in saved.before.values()
                                      if not e.get("direct_url")))
        try:
            tools.run(("uv", "tool", "install", *saved.receipt.install_args(), "--constraints", constraints.name), None)
        finally:
            Path(constraints.name).unlink(missing_ok=True)
    else:
        injected = sorted(_injected(env) - saved.injected) if env.kind == _installers.PIPX else []
        if injected:
            tools.run(("pipx", "uninject", "--leave-deps", env.prefix.name, *injected), None)
            now = tools.snapshot(env.python)
        if added := [now[key]["name"] for key in now if key not in saved.before]:
            tools.run(_pip(env, "uninstall", *added), None)
        pinned = [arg for key, entry in saved.before.items() if not _same(now.get(key), entry)
                  for arg in _pinned(entry)]
        if pinned:
            tools.run(_pip(env, "install", "--no-deps", *pinned), None)
    for path in _record(env):
        if path in saved.files:
            path.write_bytes(saved.files[path])
        elif path.exists():
            path.unlink()
    drift = _changed(saved.before, tools.snapshot(env.python))
    drift += sorted(record.name for record in tools.broken() - saved.broken)
    if not drift:
        click.echo("Restored.", err=True)
        return
    click.echo(f"Could not restore exactly; these differ from before: {', '.join(drift)}", err=True)
    repair = {
        _installers.UV_TOOL: f"uv tool upgrade --reinstall {env.prefix.name}",
        _installers.PIPX: f"pipx reinstall {env.prefix.name}",
        _installers.UV_PROJECT: f"uv sync --inexact --reinstall --project {shlex.quote(str(env.location))}"
                                + (f" --package {env.member[0]}" if env.member else ""),
    }.get(env.kind)
    if repair:
        click.echo(f"`{repair}` reinstalls what the installer records.", err=True)


def _pip(env: Environment, action: str, *args: str) -> tuple[str, ...]:
    if env.kind != _installers.PIPX:
        return _installers.pip_command(env, action, *args)
    if action == "uninstall":
        return ("pipx", "runpip", env.prefix.name, "uninstall", "-y", *args)
    metadata = json.loads((env.prefix / "pipx_metadata.json").read_text())
    return ("pipx", "runpip", env.prefix.name, action, *(metadata["main_package"].get("pip_args") or []), *args)


def _dependents(installed: Snapshot, targets: list[str]) -> list[str]:
    found = []
    for key in targets:
        users = sorted(installed[d]["name"] for d in installed if key in installed[d]["requires"] and d not in targets)
        if users:
            found.append(f"{installed[key]['name']} is required by {', '.join(users)}")
    return found


def _contributed(installed: Snapshot, targets: list[str]) -> dict[str, set[str]]:
    """What these packages provide that no remaining package also provides."""
    kept = {(group, name) for key, dist in installed.items() if key not in targets
            for group, name in dist["entry_points"]}
    names: dict[str, set[str]] = {}
    for key in targets:
        for group, name in installed[key]["entry_points"]:
            if (group, name) not in kept:
                names.setdefault(group, set()).add(name)
    return names


def _owns(modules: set[str], impl: object) -> bool:
    if not isinstance(impl, str):
        return False
    module = impl.split(":", 1)[0].strip()
    return any(module == owned or module.startswith(f"{owned}.") for owned in modules)


def _config_references(installed: Snapshot, targets: list[str], tools: Tools) -> list[str]:
    """Where the config file still names something only these packages provide. Nothing, when
    the file in effect is one a package selects on import and installs: it goes with that package."""
    contributed = _contributed(installed, targets)
    modules = {module for key in targets for module in installed[key].get("modules", [])}
    config = get_config()
    try:
        sandbox, state, artifacts = config.section("sandbox"), config.section("state"), config.section("artifacts")
        impl_lists = {"envs": config.section("envs"), "task_steps": config.section("task_steps"),
                      "artifacts": artifacts, "explorer.plugins": config.section("explorer", "plugins")}
        stores = ("document", "object", "image", "secret", "runner")
        tables = {name: config.trace_section(name).value for name in stores}
    except Exception as exc:
        return [f"the config file cannot be read to check what it names: {exc}"]
    found: list[str] = []
    providers = contributed.get(SANDBOX_PROVIDERS, set())
    for key in ("default", "agent_default"):
        value = sandbox.get(key)
        if isinstance(value, str) and providers & {token.strip() for token in value.split(",")}:
            found.append(f"[sandbox] {key} = {value!r} names its sandbox provider")
    for group, section, table in ((SANDBOX_PROVIDERS, sandbox, "sandbox"), (STATE_PROVIDERS, state, "state")):
        for name, settings in (section.get("providers") or {}).items():
            if name in contributed.get(group, set()):
                found.append(f"[{table}.providers.{name}] configures its provider")
            elif isinstance(settings, dict) and _owns(modules, settings.get("impl")):
                found.append(f"[{table}.providers.{name}] impl {settings['impl']!r} is from it")
    for alias, target in (artifacts.get("type_aliases") or {}).items():
        if target in contributed.get(ARTIFACTS, set()):
            found.append(f"[artifacts] type_aliases maps {alias!r} to its type {target!r}")
    for table, section in impl_lists.items():
        for impl in section.get("impls") or []:
            if _owns(modules, impl):
                found.append(f"[{table}] impls names {impl!r} from it")
    for name, table in tables.items():
        if isinstance(table, dict) and _owns(modules, table.get("impl")):
            where = "runner" if name == "runner" else f"stores.{name}"
            found.append(f"[{where}] impl {table['impl']!r} is from it")
    owner = _config_owner(config.config_path(), [installed[key]["name"] for key in targets], tools) if found else None
    if owner:
        click.echo(f"note: {owner} installs the config file in effect and selects it when it is imported, "
                   "so what it names goes with it", err=True)
        return []
    return found


def _config_owner(path: Optional[Path], names: list[str], tools: Tools) -> Optional[str]:
    """Which of these packages both installs ``path`` and sets AGENT_ENV_CONFIG to it when it is
    imported. Selecting a file alone is not enough: a shared file stays, and discovery may still
    pick it once the package is gone."""
    if path is None:
        return None
    for name in names:
        if not tools.installs(name, path):
            continue
        try:
            selected = tools.config_set_by(name)
        except ProbeError:
            continue
        if selected and Path(selected).resolve() == Path(path).resolve():
            return name
    return None


def _local_namespace_documents_exist(config) -> bool:
    store = config.get_document_store()
    return isinstance(store, RoutingDocumentStore) and store.local.path.is_file()


def _stored_usage(installed: Snapshot, targets: list[str], *, check_remote: bool) -> list[str]:
    """Stored documents of types only these packages provide: always for a local store, with
    --check-usage for a remote one. Reads a local database only if it already exists."""
    contributed = _contributed(installed, targets)
    envs, artifacts, steps, providers = (contributed.get(g, set()) for g in (ENVS, ARTIFACTS, TASK_STEPS, ENV_PROVIDERS))
    if not (envs or artifacts or steps or providers):
        return []
    config = get_config()
    try:
        section = config.trace_section("document").value
        local = str(section.get("impl", "")).endswith("LocalSqliteDocumentStore")
        path = interpolate(section.get("config") or {}).get("path", "") if local else ""
    except Exception as exc:
        return [f"the document store's config cannot be read to check stored documents: {exc}"]
    if local and not Path(path).is_file() and not _local_namespace_documents_exist(config):
        return []
    if not local and not check_remote:
        click.echo("note: the document store is remote, so stored documents were not checked (--check-usage does)",
                   err=True)
        return []
    try:
        store = config.get_document_store()
        found = []
        for collection, types, what in (("envs", envs, "env"), ("artifacts", artifacts, "artifact")):
            if types:
                count = store.count_distinct(collection, Filter().where("type", In(sorted(types))))
                if count:
                    found.append(f"{count} stored {what}(s) use its types ({', '.join(sorted(types))})")
        if steps:
            # Step types sit inside each task's steps list, which a filter cannot reach.
            ids = {task.get("id") for task in store.query("tasks", Filter())
                   if any(isinstance(step, dict) and step.get("type") in steps for step in task.get("steps") or [])}
            if ids:
                found.append(f"{len(ids)} stored task(s) use its step types ({', '.join(sorted(steps))})")
        if providers:
            # An env declares the provider it deploys through, and an instance records the one that deployed it.
            names = In(sorted(providers))
            for collection, id_field, what in (("envs", "id", "env"), ("env_instances", "instance_id", "env instance")):
                count = store.count_distinct(collection, Filter().where("env_provider_type", names), id_field=id_field)
                if count:
                    found.append(f"{count} stored {what}(s) use its environment providers ({', '.join(sorted(providers))})")
        return found
    except Exception as exc:
        return [f"the document store cannot be read to check stored documents: {exc!r}"]
