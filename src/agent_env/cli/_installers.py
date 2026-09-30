"""How agent-env was installed, and the commands that add or remove a package there.

agent-env never installs anything itself: it builds the command for the installer that owns its
environment, so that installer's next rebuild (`uv tool upgrade`, `uv sync`, `pipx reinstall`)
keeps the change. Nothing here runs a command.
"""

import json
import os
import re
import shlex
import shutil
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, unquote, urlparse

UV_TOOL = "uv tool"
PIPX = "pipx"
UV_PROJECT = "uv project"
VIRTUALENV = "virtualenv"
SYSTEM = "system Python"
POETRY = "Poetry project"
PDM = "PDM project"
HATCH = "Hatch environment"

# The installers `--installer` can force, by the value it takes.
FORCEABLE = {"uv-tool": UV_TOOL, "pipx": PIPX, "uv-project": UV_PROJECT, "pip": VIRTUALENV}
# agent-env itself, the protocol package it pins exactly, and the command's own name, which a spec may
# use for agent-env; `plugin add` and `remove` never change them.
CORE = ("agentenv-framework", "agentenv-framework-protocol", "agentenv-protocol", "agent-env")

_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


class InstallerError(Exception):
    """This change cannot go through the environment's installer; the message says what to do."""


@dataclass(frozen=True)
class Environment:
    """Where agent-env is installed, and which installer owns it."""

    kind: str
    location: Path  # the tool or pipx venv, the project root, or the venv itself
    prefix: Path
    python: Path
    has_pip: bool = True
    refusal: Optional[str] = None  # set when agent-env must not change this environment
    # A uv workspace whose root has no [project]: the member that declares agent-env, and its directory.
    member: Optional[tuple[str, Path]] = None


@dataclass(frozen=True)
class Plan:
    """What `plugin add` or `plugin remove` runs, in order, or only prints for the user to run."""

    environment: Environment
    commands: tuple[tuple[str, ...], ...]
    runs: bool = True
    cwd: Optional[Path] = None
    note: Optional[str] = None


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> Optional[str]:
    """The distribution a requirement names: from ``name`` or ``name @ url``, else from a wheel
    or sdist file name. None for a directory or a VCS spec, whose name only a build tells."""
    head = spec.split("@", 1)[0].strip() if "@" in spec else spec
    if not any(c in head for c in "/\\") and not head.endswith((".whl", ".tar.gz", ".zip")):
        match = _NAME.match(head)
        return normalize(match.group(0)) if match else None
    filename = unquote(urlparse(spec).path if "://" in spec else spec).rstrip("/").rsplit("/", 1)[-1]
    if filename.endswith(".whl"):
        return normalize(filename.split("-", 1)[0])
    for suffix in (".tar.gz", ".zip"):
        if filename.endswith(suffix) and "-" in filename:
            return normalize(filename[: -len(suffix)].rsplit("-", 1)[0])
    return None


def detect(prefix: Path, base_prefix: Path, python: Path, *, has_pip: bool = True,
           project_environment: Optional[str] = None, cwd: Optional[Path] = None) -> Environment:
    """The installer that owns the environment at ``prefix``. ``project_environment`` is
    UV_PROJECT_ENVIRONMENT, which puts a uv project's environment outside its `.venv`; the project
    is then the one uv finds from ``cwd``."""
    def env(kind: str, location: Path, **more) -> Environment:
        return Environment(kind, location, prefix, python, has_pip, **more)

    if (prefix / "uv-receipt.toml").is_file():
        return env(UV_TOOL, prefix)
    if (prefix / "pipx_metadata.json").is_file():
        return env(PIPX, prefix)
    if project_environment:
        project, found_here = _project_using(prefix, project_environment, cwd)
        if project is not None and found_here:
            return _uv_project(env, project)
        if project is not None or (os.path.isabs(project_environment)
                                   and _same_path(Path(project_environment), prefix)):
            owner = f"the uv project at {project}" if project else "a uv project"
            return env(VIRTUALENV, prefix, refusal=(
                f"UV_PROJECT_ENVIRONMENT makes this the environment of {owner}, and its next `uv sync` would drop "
                "what agent-env installs here; run this from the project's directory"
            ))
    root = prefix.parent
    if prefix.name == ".venv" and (root / "pyproject.toml").is_file():
        if (root / "uv.lock").is_file():
            return _uv_project(env, root)
        tools = set(_toml(root / "pyproject.toml").get("tool", {}))
        if "poetry" in tools or (root / "poetry.lock").is_file():
            return env(POETRY, root)
        if "pdm" in tools or (root / "pdm.lock").is_file():
            return env(PDM, root)
    if hatch := next((d for d in prefix.parents if (d / "pyproject.toml").is_file() and _hatch_env(d, prefix)), None):
        return env(HATCH, hatch)
    parts = prefix.parts
    if "pypoetry" in parts and "virtualenvs" in parts:
        return env(POETRY, prefix)
    if "hatch" in parts and "env" in parts:
        return env(HATCH, prefix)
    return env(VIRTUALENV if prefix != base_prefix else SYSTEM, prefix)


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _project_using(prefix: Path, project_environment: str, cwd: Optional[Path]) -> tuple[Optional[Path], bool]:
    """The uv project whose UV_PROJECT_ENVIRONMENT is ``prefix``, and whether agent-env may change
    it from ``cwd``: it is the project uv finds from there, or there is none. A value relative to
    its project, as uv reads it, also finds a project the environment lies under; a command run
    inside a different project must not change that one."""
    path = Path(project_environment)

    def uses(directory: Path) -> bool:
        return ((directory / "pyproject.toml").is_file() and (directory / "uv.lock").is_file()
                and _same_path(path if path.is_absolute() else directory / path, prefix))

    nearest = next((d for d in ((cwd, *cwd.parents) if cwd else ())
                    if (d / "pyproject.toml").is_file() and (d / "uv.lock").is_file()), None)
    if nearest is not None and uses(nearest):
        return nearest, True
    if path.is_absolute():
        return None, False
    return next((d for d in prefix.parents if uses(d)), None), nearest is None


def _uv_project(make: Callable[..., Environment], root: Path) -> Environment:
    """A uv project at ``root``. A workspace root with no [project] table cannot take a dependency
    itself, so the change goes to the one member that declares agent-env."""
    data = _toml(root / "pyproject.toml")
    if "project" in data or "workspace" not in data.get("tool", {}).get("uv", {}):
        return make(UV_PROJECT, root)
    members = [(name, path) for name, path in _workspace_members(root, data)
               if "agentenv-framework" in _declared(path / "pyproject.toml")]
    if len(members) == 1:
        return make(UV_PROJECT, root, member=members[0])
    return make(UV_PROJECT, root, refusal=(
        f"this uv workspace's root has no [project] table, and {len(members)} of its members declare "
        "agentenv-framework, so agent-env cannot tell which one to change; run `uv add --package <member> ...` "
        "yourself"
    ))


def _workspace_members(root: Path, data: dict) -> list[tuple[str, Path]]:
    workspace = data.get("tool", {}).get("uv", {}).get("workspace", {})
    excluded = {path.resolve() for pattern in workspace.get("exclude") or [] for path in root.glob(pattern)}
    found = []
    for pattern in workspace.get("members") or []:
        for path in sorted(root.glob(pattern)):
            name = _toml(path / "pyproject.toml").get("project", {}).get("name")
            if name and path.resolve() not in excluded:
                found.append((name, path))
    return found


def _hatch_env(root: Path, prefix: Path) -> bool:
    """Whether ``prefix`` is a Hatch environment that the project keeps in its own directory."""
    envs = _toml(root / "pyproject.toml").get("tool", {}).get("hatch", {}).get("envs", {})
    return any(isinstance(settings, dict) and isinstance(settings.get("path"), str)
               and _same_path(root / settings["path"], prefix) for settings in envs.values())


def with_refusal(env: Environment, *, stdlib: Path, site_packages: Path) -> Environment:
    """``env``, refused when its Python is system-managed (PEP 668) or its packages are read-only."""
    if env.kind == SYSTEM and (stdlib / "EXTERNALLY-MANAGED").is_file():
        return replace(env, refusal=(
            "this Python is managed by the system (PEP 668). Install agent-env into its own "
            "environment instead, e.g. `uv tool install agentenv-framework --with <plugin>`"
        ))
    if not os.access(site_packages, os.W_OK):
        return replace(env, refusal=f"{site_packages} is read-only, so nothing can be installed into it")
    return env


def forced(env: Environment, installer: Optional[str]) -> Environment:
    """``env`` with its installer chosen by ``--installer`` instead of detected."""
    if installer is None:
        return env
    kind = FORCEABLE[installer]
    if kind == UV_TOOL and not (env.prefix / "uv-receipt.toml").is_file():
        raise InstallerError(f"--installer uv-tool needs a uv tool; {env.prefix} has no uv-receipt.toml")
    if kind == PIPX and not (env.prefix / "pipx_metadata.json").is_file():
        raise InstallerError(f"--installer pipx needs a pipx venv; {env.prefix} has no pipx_metadata.json")
    location = env.prefix
    if kind == UV_PROJECT:
        location = env.prefix.parent
        if env.prefix.name != ".venv" or not (location / "pyproject.toml").is_file():
            raise InstallerError(
                f"--installer uv-project needs agent-env to run from a project's own .venv; {env.prefix} is not one"
            )
    return replace(env, kind=kind, location=location)


def add_plan(env: Environment, specs: list[str], *, index_url: Optional[str] = None) -> Plan:
    """The commands that install ``specs`` into ``env`` the way its installer keeps them."""
    _refuse(env)
    for spec in specs:
        if requirement_name(spec) in CORE:
            raise InstallerError(f"{spec} is agent-env itself; upgrade it with the installer that owns the environment")
    if env.kind == UV_TOOL:
        receipt = ToolReceipt.read(env.prefix / "uv-receipt.toml")
        return Plan(env, (("uv", "tool", "install", *receipt.install_args(add=specs, default_index=index_url)),))
    if env.kind == PIPX:
        metadata = json.loads((env.prefix / "pipx_metadata.json").read_text())
        injected = {normalize(name) for name in metadata.get("injected_packages") or {}}
        # pipx leaves an injected package as it is unless forced, so an upgrade would do nothing.
        # Only a named spec is forced: an unnamed one could build as the venv's main package,
        # agent-env itself, which a forced inject would reinstall.
        force = ("--force",) if injected & {requirement_name(spec) for spec in specs} else ()
        inject = ("pipx", "inject", env.prefix.name, *specs, *force)
        return Plan(env, (inject + _pipx_args(env) + _flag("--index-url", index_url),))
    if env.kind == UV_PROJECT:
        # `uv add` and `uv sync` sync exactly, which would uninstall extras and anything installed
        # outside the lock, agent-env included; an inexact sync only adds.
        project = ("--project", str(env.location), *_package(env))
        return Plan(env, (
            ("uv", "add", "--no-sync", *project, *specs, *_flag("--default-index", index_url)),
            ("uv", "sync", "--inexact", *project),
        ))
    if env.kind in (VIRTUALENV, SYSTEM):
        return Plan(env, (pip_command(env, "install", *specs, *_flag("--index-url", index_url)),))
    return _printed(env, "add", specs)


def _package(env: Environment) -> tuple[str, ...]:
    return ("--package", env.member[0]) if env.member else ()


def remove_plan(env: Environment, names: list[str], *, also: list[str] = ()) -> Plan:
    """The commands that uninstall ``names`` from ``env`` the way its installer keeps them. ``also``
    are the plugins only ``names`` needed: a uv tool's rebuild drops them by itself, and a uv
    project's lock does, so they are uninstalled with them."""
    _refuse(env)
    if env.kind == UV_TOOL:
        receipt = ToolReceipt.read(env.prefix / "uv-receipt.toml")
        return Plan(env, (("uv", "tool", "install", *receipt.install_args(drop=names)),))
    if env.kind == PIPX:
        main = json.loads((env.prefix / "pipx_metadata.json").read_text())["main_package"]["package"]
        if normalize(main) in {normalize(n) for n in names}:
            raise InstallerError(
                f"{main} is this pipx venv's own package; `pipx uninstall {env.prefix.name}` removes it"
            )
        # Without --leave-deps pipx also uninstalls whatever nothing else requires, agent-env included.
        return Plan(env, (("pipx", "uninject", "--leave-deps", env.prefix.name, *names),))
    if env.kind == UV_PROJECT:
        removes = []
        for name in names:
            places = _places(env, name)
            if not places:
                raise InstallerError(f"the project does not declare {name}, so `uv remove` cannot drop it; it is "
                                     "installed for another package, or outside the lock")
            removes += [("uv", "remove", "--no-sync", "--project", str(env.location),
                         *(("--package", package) if package else ()), *flags, name) for package, flags in places]
        return Plan(env, (*removes, ("uv", "pip", "uninstall", "--python", str(env.python), *names, *also)))
    if env.kind in (VIRTUALENV, SYSTEM):
        return Plan(env, (pip_command(env, "uninstall", *names),))
    return _printed(env, "remove", names)


def recorded(env: Environment) -> Optional[dict[str, frozenset[str]]]:
    """The packages the installer's record lists as wanted, rather than pulled in by others, each
    with the extras it asks for; None when it keeps no record. A uv tool's or project's rebuild
    keeps these and what they need."""
    if env.kind == UV_TOOL:
        return {normalize(entry["name"]): frozenset(map(normalize, entry.get("extras") or ()))
                for entry in ToolReceipt.read(env.prefix / "uv-receipt.toml").requirements}
    if env.kind == PIPX:
        metadata = json.loads((env.prefix / "pipx_metadata.json").read_text())
        names = [metadata["main_package"]["package"], *(metadata.get("injected_packages") or {})]
        return {normalize(name): frozenset() for name in names}
    if env.kind == UV_PROJECT:
        wanted: dict[str, frozenset[str]] = {}
        for _, path in _projects(env.location):
            data = _toml(path / "pyproject.toml")
            if name := data.get("project", {}).get("name"):
                wanted.setdefault(normalize(name), frozenset())
            for requirement, _ in _declarations(data):
                name, extras = _requirement_extras(requirement)
                if name:
                    wanted[name] = wanted.get(name, frozenset()) | extras
        return wanted
    return None


def project_files(env: Environment) -> list[Path]:
    """The pyproject.toml files of a uv project and its workspace members."""
    return [path / "pyproject.toml" for _, path in _projects(env.location)]


def _projects(root: Path) -> list[tuple[Optional[str], Path]]:
    """The uv project at ``root`` (None when it is a virtual workspace root) and its workspace
    members, by name, with their directories."""
    data = _toml(root / "pyproject.toml")
    return [(None, root)] + _workspace_members(root, data)


def _declarations(data: dict) -> list[tuple[str, tuple[str, ...]]]:
    """Every dependency a pyproject.toml declares, with the `uv remove` flags that name its group."""
    project = data.get("project", {})
    places = [((), project.get("dependencies") or [])]
    places += [(("--optional", extra), items) for extra, items in (project.get("optional-dependencies") or {}).items()]
    places += [(("--group", group), items) for group, items in (data.get("dependency-groups") or {}).items()]
    places += [(("--dev",), data.get("tool", {}).get("uv", {}).get("dev-dependencies") or [])]
    return [(item, flags) for flags, items in places for item in items if isinstance(item, str)]


def _places(env: Environment, name: str) -> list[tuple[Optional[str], tuple[str, ...]]]:
    """Where the project declares ``name``: the workspace member (None for the root project) and
    the group, as `uv remove` flags."""
    key = normalize(name)
    return [(package, flags) for package, path in _projects(env.location)
            for requirement, flags in _declarations(_toml(path / "pyproject.toml"))
            if requirement_name(requirement) == key]


_EXTRAS = re.compile(r"\s*[A-Za-z0-9][A-Za-z0-9._-]*\s*\[([^\]]*)\]")


def _requirement_extras(requirement: str) -> tuple[Optional[str], frozenset[str]]:
    match = _EXTRAS.match(requirement)
    extras = {normalize(e.strip()) for e in match.group(1).split(",") if e.strip()} if match else set()
    return requirement_name(requirement), frozenset(extras)


def _declared(pyproject: Path) -> set[str]:
    """Every dependency a pyproject.toml declares, in any group."""
    return {requirement_name(requirement) for requirement, _ in _declarations(_toml(pyproject))} - {None}


def _refuse(env: Environment) -> None:
    if env.refusal:
        raise InstallerError(f"agent-env will not change this environment: {env.refusal}")


def _flag(flag: str, value: Optional[str]) -> tuple[str, ...]:
    return (flag, value) if value else ()


def pip_command(env: Environment, action: str, *args: str) -> tuple[str, ...]:
    """pip inside the environment when it has one, else `uv pip` aimed at its interpreter."""
    if env.has_pip:
        return (str(env.python), "-m", "pip", action, *(("-y",) if action == "uninstall" else ()), *args)
    if shutil.which("uv"):
        return ("uv", "pip", action, "--python", str(env.python), *args)
    raise InstallerError(f"{env.prefix} has neither pip nor uv to {action} with")


def _pipx_args(env: Environment) -> tuple[str, ...]:
    """The pip arguments the venv was created with, so an injected package resolves the same way."""
    metadata = json.loads((env.prefix / "pipx_metadata.json").read_text())
    pip_args = metadata["main_package"].get("pip_args") or []
    return ("--pip-args", shlex.join(pip_args)) if pip_args else ()


def _printed(env: Environment, action: str, names: list[str]) -> Plan:
    if env.kind in (POETRY, PDM):
        tool = "poetry" if env.kind == POETRY else "pdm"
        return Plan(env, ((tool, action, *names),), runs=False, cwd=env.location,
                    note=f"{env.kind.split()[0]} owns this environment, so run this in the project yourself.")
    verb = "Add" if action == "add" else "Remove"
    return Plan(env, (), runs=False, cwd=env.location, note=(
        f"Hatch has no command for this. {verb} `{' '.join(names)}` in the environment's dependencies in "
        "pyproject.toml, then recreate it with `hatch env prune`."
    ))


def _toml(path: Path) -> dict:
    try:
        return tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


# `[tool.options]` keys a receipt can hold that agent-env passes back to `uv tool install`.
_RECEIPT_VALUES = {
    "index-url": "--default-index", "default-index": "--default-index", "extra-index-url": "--index",
    "find-links": "--find-links", "index-strategy": "--index-strategy", "keyring-provider": "--keyring-provider",
    "prerelease": "--prerelease", "resolution": "--resolution", "exclude-newer": "--exclude-newer",
    "link-mode": "--link-mode",
}
_RECEIPT_FLAGS = {
    "no-index": "--no-index", "compile-bytecode": "--compile-bytecode", "no-build-isolation": "--no-build-isolation",
    "no-sources": "--no-sources", "no-build": "--no-build",
}
_RECEIPT_REQUIREMENT_KEYS = frozenset({
    "name", "extras", "marker", "specifier", "url", "path", "directory", "editable", "git", "rev", "tag", "branch",
    "subdirectory",
})


@dataclass(frozen=True)
class ToolReceipt:
    """A uv tool's `uv-receipt.toml`: the requirements it was installed with, and its options.

    Re-running `uv tool install` forgets any option not passed again, so every change passes the
    receipt's Python and options back; anything that cannot be passed back refuses the change.
    """

    requirements: tuple[dict, ...]
    python: Optional[str]
    options: dict
    unsupported: tuple[str, ...] = ()

    @classmethod
    def read(cls, path: Path) -> "ToolReceipt":
        try:
            tool = tomllib.loads(path.read_text())["tool"]
        except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
            raise InstallerError(f"cannot read the uv tool receipt at {path}: {exc}") from exc
        requirements = tuple(tool.get("requirements") or ())
        if not requirements:
            raise InstallerError(f"the uv tool receipt at {path} lists no requirements")
        kept_elsewhere = ("constraints", "overrides", "build-constraint-dependencies")
        unsupported = tuple(key for key in kept_elsewhere if tool.get(key))
        tool_name = normalize(requirements[0]["name"])
        if any(normalize(ep.get("from", tool_name)) != tool_name for ep in tool.get("entrypoints") or []):
            unsupported += ("executables from another package (--with-executables-from)",)
        return cls(requirements, tool.get("python"), dict(tool.get("options") or {}), unsupported)

    def install_args(self, *, add: list[str] = (), drop: list[str] = (),
                     default_index: Optional[str] = None) -> tuple[str, ...]:
        """`uv tool install` arguments for this tool, with ``add`` joined and ``drop`` left out.
        ``default_index`` replaces the receipt's own default index, which uv takes only once."""
        if self.unsupported:
            raise InstallerError(
                f"the uv tool was installed with {', '.join(self.unsupported)}, which agent-env cannot pass back to "
                "`uv tool install`; make this change with uv yourself"
            )
        tool, *extras = self.requirements
        dropped = {normalize(n) for n in drop}
        if normalize(tool["name"]) in dropped:
            raise InstallerError(f"{tool['name']} is the tool itself; `uv tool uninstall {tool['name']}` removes it")
        listed = {normalize(e["name"]) for e in extras}
        missing = dropped - listed
        if missing:
            raise InstallerError(
                f"the tool was not installed `--with` {', '.join(sorted(missing))}, so it is a dependency of "
                "another package there"
            )
        # A bare name that is listed already asks for nothing new: its entry stays, pin and all.
        unchanged = {normalize(spec.strip()) for spec in add if _NAME.fullmatch(spec.strip())} & listed
        replaced = {requirement_name(spec) for spec in add} - {None} - unchanged
        _check_source(tool, fix="reinstall the tool with uv")
        editable = _editable(tool)
        args: list[str] = ["-e", editable] if editable else [_spec(tool)]
        for entry in extras:
            if normalize(entry["name"]) not in dropped | replaced:
                _check_source(entry)
                editable = _editable(entry)
                args += ["--with-editable", editable] if editable else ["--with", _spec(entry)]
        for spec in add:
            if normalize(spec.strip()) not in unchanged:
                args += ["--with", spec]
        if self.python:
            args += ["--python", self.python]
        return tuple(args) + self._option_args(default_index)

    def _option_args(self, default_index: Optional[str]) -> tuple[str, ...]:
        args: list[str] = []
        indexes: list[tuple[str, Optional[str], bool]] = []
        for key, value in self.options.items():
            if key in _RECEIPT_FLAGS:
                args += [_RECEIPT_FLAGS[key]] if value else []
            elif key in _RECEIPT_VALUES and _RECEIPT_VALUES[key] in ("--default-index", "--index"):
                for item in value if isinstance(value, list) else [value]:
                    indexes.append((str(item), None, _RECEIPT_VALUES[key] == "--default-index"))
            elif key in _RECEIPT_VALUES:
                for item in value if isinstance(value, list) else [value]:
                    args += [_RECEIPT_VALUES[key], str(item)]
            elif key == "index":
                for index in value:
                    if isinstance(index, dict):
                        indexes.append((index["url"], index.get("name"), bool(index.get("default"))))
                    else:
                        indexes.append((str(index), None, False))
            else:
                raise InstallerError(
                    f"the uv tool receipt sets {key!r}, which agent-env cannot pass back to `uv tool install`; "
                    "make this change with uv yourself"
                )
        return tuple(args) + _index_args(indexes, default_index)


def _index_args(indexes: list[tuple[str, Optional[str], bool]], default_index: Optional[str]) -> tuple[str, ...]:
    """Each index once, and one default. uv records an index from its config and the same index
    passed as a flag side by side, and refuses `--default-index` twice, so indexes are merged by
    URL, and a name is kept so the next install matches the configured index instead of adding
    another. ``default_index`` replaces the receipt's default."""
    merged: dict[str, tuple[Optional[str], bool]] = {}
    for url, name, default in indexes:
        known_name, known_default = merged.get(url, (None, False))
        merged[url] = (known_name or name, known_default or default)
    spelled = {url: f"{name}={url}" if name else url for url, (name, _) in merged.items()}
    chosen = default_index or next((spelled[url] for url, (_, default) in merged.items() if default), None)
    args: list[str] = []
    for url, (_, default) in merged.items():
        if not (default and (default_index or spelled[url] == chosen)):
            args += ["--index", spelled[url]]
    return tuple(args) + (("--default-index", chosen) if chosen else ())


def _check_source(entry: dict, fix: Optional[str] = None) -> None:
    """Refuse up front when a requirement comes from a local file or directory that is gone: uv
    cannot rebuild the tool without it, whatever the change is."""
    url = entry.get("url") or ""
    local = entry.get("path") or entry.get("directory") or _editable(entry) or (
        unquote(urlparse(url).path) if url.startswith("file:") else None)
    if local and not Path(local).exists():
        name = entry["name"]
        fix = fix or (f"add it again from where it is now (`agent-env plugin add '{name} @ file:///path/to/it'`), or "
                      f"remove it (`agent-env plugin remove {name}`)")
        raise InstallerError(f"the uv tool installs {name} from {local}, which no longer exists, so uv cannot rebuild "
                             f"it; {fix}")


def _editable(entry: dict) -> Optional[str]:
    """The path of an editable requirement: uv writes `editable = "<path>"`."""
    value = entry.get("editable")
    if isinstance(value, str):
        return value
    return str(entry.get("directory") or entry.get("path")) if value else None


def _spec(entry: dict) -> str:
    """A receipt requirement as the PEP 508 spec `uv tool install` accepts."""
    unknown = set(entry) - _RECEIPT_REQUIREMENT_KEYS
    if unknown:
        raise InstallerError(
            f"the uv tool receipt's requirement {entry.get('name')!r} uses {', '.join(sorted(unknown))}, which "
            "agent-env cannot pass back to `uv tool install`; make this change with uv yourself"
        )
    name = entry["name"] + (f"[{','.join(entry['extras'])}]" if entry.get("extras") else "")
    marker = f" ; {entry['marker']}" if entry.get("marker") else ""
    if entry.get("git"):
        # uv keeps the ref and subdirectory in the URL's query: `https://host/repo?rev=v1`.
        parsed = urlparse(entry["git"])
        query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
        ref = next((v for k in ("rev", "tag", "branch") if (v := entry.get(k) or query.get(k))), None)
        subdirectory = entry.get("subdirectory") or query.get("subdirectory")
        url = f"git+{parsed._replace(query='', fragment='').geturl()}" + (f"@{ref}" if ref else "")
        url += f"#subdirectory={subdirectory}" if subdirectory else ""
        return f"{name} @ {url}{marker}"
    if entry.get("url"):
        subdirectory = f"#subdirectory={entry['subdirectory']}" if entry.get("subdirectory") else ""
        return f"{name} @ {entry['url']}{subdirectory}{marker}"
    if entry.get("path") or entry.get("directory"):
        return f"{name} @ {Path(entry.get('path') or entry['directory']).resolve().as_uri()}{marker}"
    return f"{name}{entry.get('specifier', '')}{marker}"
