"""Real installers, offline: `agent-env plugin add` and `remove` in each environment setup.

Every run builds agentenv-framework and agentenv-framework-protocol from the checkout, so it tests the
change under review. Their dependencies and build backend come from a wheelhouse downloaded
once per `uv.lock` and cached under `.cache/`; that download is the only step that uses the
network. After it, the build and every pip, uv and pipx install run with the package index
switched off, against temporary tool, state and home directories, so nothing touches the
developer's own tools or config.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import platform
import shutil
import site
import subprocess
import sys
import threading
import tomllib
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import pytest

from tst.util.capabilities import missing_capability_reason

REPO = Path(__file__).resolve().parents[2]
PROJECTS = (REPO, REPO / "packages" / "agentenv-protocol")
PYTHON = f"{sys.version_info[0]}.{sys.version_info[1]}"
# pip and flit_core come along for pipx's shared pip and for building a directory spec offline.
_EXTRA_WHEELS = ("pip", "flit_core")
# What `uv build` needs for the checkout, so building it is offline too.
# How long one `agent-env` command in a test may take; an install resolves ~100 wheels.
CLI_TIMEOUT_S = 600
_BUILD_REQUIRES = tuple(sorted({requirement for project in PROJECTS for requirement in
                                tomllib.loads((project / "pyproject.toml").read_text())["build-system"]["requires"]}))


def _tool(name: str) -> Optional[str]:
    return shutil.which(name)


def requires_tool(name: str):
    return pytest.mark.skipif(_tool(name) is None, reason=missing_capability_reason(f"{name}_installed"))


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def _checked(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    proc = _run(command, **kwargs)
    if proc.returncode != 0:
        raise AssertionError(f"{' '.join(command)} exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return proc


@contextmanager
def filling(root: Path) -> Iterator[bool]:
    """Hold ``root``'s lock and say whether it still needs filling; mark it complete after.

    Test workers share the cache, so one fills it while the others wait on the lock, and none
    reads a directory another is still writing.
    """
    root.mkdir(parents=True, exist_ok=True)
    done = root / ".complete"
    with open(root / ".lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        empty = not done.exists()
        yield empty
        if empty:
            done.touch()


def locked_requirements() -> str:
    return _checked(["uv", "export", "--frozen", "--no-dev", "--no-emit-workspace", "--no-hashes",
                     "--format", "requirements-txt", "-q"], cwd=REPO).stdout


@pytest.fixture(scope="session")
def dependencies() -> Path:
    """agentenv-framework's locked dependencies as wheels for this interpreter and platform."""
    wanted = "\n".join((*_EXTRA_WHEELS, *_BUILD_REQUIRES)).encode()
    lock = hashlib.sha256((REPO / "uv.lock").read_bytes() + wanted).hexdigest()[:16]
    root = REPO / ".cache" / "installer-wheelhouse" / f"{lock}-py{PYTHON}-{sys.platform}-{platform.machine()}"
    with filling(root) as empty:
        if empty:
            requirements = root / "requirements.txt"
            requirements.write_text(locked_requirements() + "".join(f"{name}\n" for name in _EXTRA_WHEELS))
            pip_wheel = ["uvx", "--python", sys.executable, "--from", "pip", "pip", "wheel", "-q", "-w", str(root),
                         "--index-url", "https://pypi.org/simple"]
            public = {**os.environ, "PIP_CONFIG_FILE": os.devnull}
            _checked([*pip_wheel, "--no-deps", "-r", str(requirements)], env=public)
            # The build backend with its own dependencies, which the lock does not list.
            _checked([*pip_wheel, *_BUILD_REQUIRES], env=public)
    return root


@pytest.fixture(scope="session")
def checkout_wheels(tmp_path_factory, dependencies) -> Path:
    out = tmp_path_factory.mktemp("checkout-wheels")
    offline = {**os.environ, "UV_OFFLINE": "1", "UV_NO_CONFIG": "1", "UV_FIND_LINKS": str(dependencies)}
    for project in PROJECTS:
        _checked(["uv", "build", "--wheel", "-q", "-o", str(out), str(project)], cwd=REPO, env=offline)
    return out


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


@pytest.fixture(scope="session")
def simple_index(tmp_path_factory, dependencies, checkout_wheels) -> Iterator[str]:
    """The wheelhouse as a PEP 503 index on localhost, for an index configured the way a user
    configures a private one. Only this process's own server is contacted."""
    root = tmp_path_factory.mktemp("simple-index")
    projects: dict[str, list[Path]] = {}
    for wheel in [*dependencies.glob("*.whl"), *checkout_wheels.glob("*.whl")]:
        projects.setdefault(wheel.name.split("-", 1)[0].lower().replace("_", "-"), []).append(wheel)
    for project, wheels in projects.items():
        page = root / "simple" / project
        page.mkdir(parents=True)
        for wheel in wheels:
            (page / wheel.name).symlink_to(wheel)
        links = "".join(f'<a href="{wheel.name}">{wheel.name}</a>\n' for wheel in wheels)
        (page / "index.html").write_text(f"<!DOCTYPE html><html><body>\n{links}</body></html>\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/simple/"
    finally:
        server.shutdown()


@pytest.fixture(scope="session")
def uv_cache(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("uv-cache")


def build_wheel(dest: Path, name: str, version: str, modules: dict[str, str],
                entry_points: Optional[dict[str, dict[str, str]]] = None, requires: tuple[str, ...] = ()) -> Path:
    """A pure-Python wheel written directly, so the toy plugins need no build backend."""
    dist = name.replace("-", "_")
    info = f"{dist}-{version}.dist-info"
    files = {path: text.encode() for path, text in modules.items()}
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    files[f"{info}/METADATA"] = (metadata + "".join(f"Requires-Dist: {r}\n" for r in requires)).encode()
    files[f"{info}/WHEEL"] = (b"Wheel-Version: 1.0\nGenerator: agent-env-tests\nRoot-Is-Purelib: true\n"
                              b"Tag: py3-none-any\n")
    if entry_points:
        files[f"{info}/entry_points.txt"] = "".join(
            f"[{group}]\n" + "".join(f"{key} = {value}\n" for key, value in points.items())
            for group, points in entry_points.items()
        ).encode()
    rows = []
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"{path},sha256={digest},{len(data)}")
    files[f"{info}/RECORD"] = ("\n".join([*rows, f"{info}/RECORD,,"]) + "\n").encode()
    wheel = dest / f"{dist}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return wheel


@dataclass
class Setup:
    """One environment with agent-env installed the way its installer does it."""

    kind: str
    python: Path
    cli_path: Path
    records: list[Path]
    env: dict[str, str]
    work: Path
    project: Optional[Path] = None
    _rebuild: list[list[str]] = field(default_factory=list)

    def cli(self, *args: str, input: Optional[str] = None, timeout: int = CLI_TIMEOUT_S,
            env: Optional[dict[str, str]] = None) -> subprocess.CompletedProcess:
        return _run([str(self.cli_path), *args], env={**self.env, **(env or {})}, cwd=self.work, input=input,
                    timeout=timeout)

    def plugins(self) -> dict[str, dict]:
        """`plugin list --json`, keyed by package name."""
        proc = self.cli("plugin", "list", "--json")
        assert proc.returncode == 0, proc.stderr
        return {dist["name"]: dist for dist in json.loads(proc.stdout)["plugins"]}

    def installed(self) -> list[tuple[str, str, str]]:
        """Every installed distribution with its version and recorded source."""
        probe = ("import importlib.metadata as m, json; print(json.dumps(sorted("
                 "(d.metadata['Name'].lower(), d.version, d.read_text('direct_url.json') or '') "
                 "for d in m.distributions())))")
        return [tuple(row) for row in json.loads(_checked([str(self.python), "-c", probe], env=self.env).stdout)]

    def state(self) -> tuple[list, dict[str, bytes]]:
        return self.installed(), {str(p): p.read_bytes() for p in self.records if p.is_file()}

    def rebuild(self) -> None:
        """What the installer itself does to recreate the environment."""
        for command in self._rebuild:
            _checked(command, env=self.env, cwd=self.project or self.work)


def _platform_marker() -> str:
    return (f"sys_platform == '{sys.platform}' and platform_machine == '{platform.machine()}' "
            f"and python_version == '{PYTHON}'")


@pytest.fixture
def hermetic(tmp_path, dependencies, checkout_wheels, uv_cache) -> dict[str, str]:
    """Only what the installers need: no index, no user config, temporary homes."""
    for name in ("home", "work", "state", "config", "cache", "data"):
        (tmp_path / name).mkdir()
    return {
        "PATH": os.environ.get("PATH", ""),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "HOME": str(tmp_path / "home"),
        # Keeps a tool installed with `pip install --user` importable under the temporary HOME;
        # virtualenvs ignore the user site, so the environments under test do not see it.
        "PYTHONUSERBASE": site.getuserbase(),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": f"{checkout_wheels} {dependencies}",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "UV_NO_CONFIG": "1",
        "UV_OFFLINE": "1",
        "UV_FIND_LINKS": f"{checkout_wheels},{dependencies}",
        "UV_PYTHON_DOWNLOADS": "never",
        "UV_CACHE_DIR": str(uv_cache),
        "UV_TOOL_DIR": str(tmp_path / "uv-tools"),
        "UV_TOOL_BIN_DIR": str(tmp_path / "uv-bin"),
        "PIPX_HOME": str(tmp_path / "pipx"),
        "PIPX_BIN_DIR": str(tmp_path / "pipx-bin"),
        "PIPX_MAN_DIR": str(tmp_path / "pipx-man"),
    }


def make_setup(kind: str, tmp_path: Path, env: dict[str, str]) -> Setup:
    work = tmp_path / "work"
    if kind == "venv":
        venv = tmp_path / "venv"
        _checked(["uv", "venv", "-q", "--seed", "--python", sys.executable, str(venv)], env=env)
        _checked([str(venv / "bin" / "python"), "-m", "pip", "install", "-q", "agentenv-framework"], env=env)
        frozen = tmp_path / "frozen.txt"
        # A venv has no record of its own; its owner rebuilds it from a requirements list.
        script = (f"'{venv}/bin/python' -m pip freeze > '{frozen}' && rm -rf '{venv}' && "
                  f"uv venv -q --seed --python '{sys.executable}' '{venv}' && "
                  f"'{venv}/bin/python' -m pip install -q -r '{frozen}'")
        rebuild = [["sh", "-c", script]]
        return Setup(kind, venv / "bin" / "python", venv / "bin" / "agent-env", [], env, work, _rebuild=rebuild)
    if kind in ("poetry", "pdm", "hatch"):
        # Print-only owners: agent-env names the command and runs nothing, so their tools need not
        # be installed; only the layout they leave behind matters.
        if kind == "hatch":
            venv = tmp_path / "hatch" / "env" / "virtual" / "demo" / "a1b2c3" / "demo"
            root = tmp_path / "hatch-project"
        else:
            root = tmp_path / f"{kind}-project"
            venv = root / ".venv"
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text(f'[project]\nname = "demo"\nversion = "0.1.0"\n\n[tool.{kind}]\n')
        if kind != "hatch":
            (root / f"{kind}.lock").write_text("")
        _checked(["uv", "venv", "-q", "--seed", "--python", sys.executable, str(venv)], env=env)
        _checked([str(venv / "bin" / "python"), "-m", "pip", "install", "-q", "agentenv-framework"], env=env)
        return Setup(kind, venv / "bin" / "python", venv / "bin" / "agent-env", [root / "pyproject.toml"], env, work,
                     project=root)
    if kind == "uv-tool-indexed":
        # A uv tool whose index comes from the user's uv.toml, not from flags: no find-links, and
        # the config is read.
        config = Path(env["XDG_CONFIG_HOME"]) / "uv" / "uv.toml"
        config.parent.mkdir(parents=True)
        config.write_text(f'[[index]]\nname = "local"\nurl = "{env.pop("SIMPLE_INDEX")}"\ndefault = true\n')
        for name in ("UV_NO_CONFIG", "UV_OFFLINE", "UV_FIND_LINKS"):
            env.pop(name)
        _checked(["uv", "tool", "install", "-q", "agentenv-framework", "--python", sys.executable], env=env)
        prefix = Path(env["UV_TOOL_DIR"]) / "agentenv-framework"
        return Setup(kind, prefix / "bin" / "python", Path(env["UV_TOOL_BIN_DIR"]) / "agent-env",
                     [prefix / "uv-receipt.toml"], env, work)
    if kind == "uv-tool":
        _checked(["uv", "tool", "install", "-q", "agentenv-framework", "--python", sys.executable], env=env)
        prefix = Path(env["UV_TOOL_DIR"]) / "agentenv-framework"
        return Setup(kind, prefix / "bin" / "python", Path(env["UV_TOOL_BIN_DIR"]) / "agent-env",
                     [prefix / "uv-receipt.toml"], env, work,
                     _rebuild=[["uv", "tool", "upgrade", "-q", "--reinstall", "agentenv-framework"]])
    if kind == "pipx":
        _checked(["pipx", "install", "--quiet", "agentenv-framework", "--python", sys.executable], env=env)
        prefix = Path(env["PIPX_HOME"]) / "venvs" / "agentenv-framework"
        return Setup(kind, prefix / "bin" / "python", Path(env["PIPX_BIN_DIR"]) / "agent-env",
                     [prefix / "pipx_metadata.json"], env, work,
                     _rebuild=[["pipx", "reinstall", "--quiet", "agentenv-framework", "--python", sys.executable]])
    if kind == "uv-project":
        project = tmp_path / "project"
        project.mkdir()
        # An offline wheelhouse holds this platform's wheels only, so the lock covers only it.
        (project / "pyproject.toml").write_text(
            f'[project]\nname = "demo"\nversion = "0.1.0"\nrequires-python = ">={PYTHON}"\ndependencies = []\n\n'
            f'[tool.uv]\nenvironments = ["{_platform_marker()}"]\n'
        )
        _checked(["uv", "add", "-q", "--python", sys.executable, "agentenv-framework"], env=env, cwd=project)
        venv = project / ".venv"
        return Setup(kind, venv / "bin" / "python", venv / "bin" / "agent-env",
                     [project / "pyproject.toml", project / "uv.lock"], env, work, project=project,
                     _rebuild=[["uv", "sync", "-q"]])
    raise ValueError(kind)
