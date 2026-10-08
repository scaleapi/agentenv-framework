"""The clean-install gate: build both distributions as the release does, check that the wheel ships every example
and its entry point, install the two wheels into a fresh venv from public PyPI, run a shipped bundle twice by name,
and check what the runs left in the local store.

Every child process gets an environment built from an allowlist, so no config file, credential, package index or
installed package of the calling shell can make the gate pass. It builds in place from the checkout: hatchling
ignores ``.gitignore`` when the build root's own path matches one of its patterns, as a copy under ``/tmp`` does, so
it also refuses a tracked example file that ``.gitignore`` matches, wherever the checkout is.
"""

from __future__ import annotations

import argparse
import configparser
import os
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = "src/agent_env/examples"
BUNDLES = "agent_env.bundles"
PYPI = "https://pypi.org/simple"
CHECK = Path(__file__).with_name("check_clean_install.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="an empty or missing folder with no space in its path")
    parser.add_argument("--bundle", default="hello", help="the shipped bundle to run (default: hello)")
    args = parser.parse_args()
    work = args.work.resolve()
    if any(c.isspace() for c in str(work)):
        return _fail(f"--work {work} contains a space; the local sandbox can't run under such a path")
    if work.exists() and any(work.iterdir()):
        return _fail(f"--work {work} isn't empty; the gate needs a fresh store")
    uv, git = shutil.which("uv"), shutil.which("git")
    if uv is None or git is None:
        return _fail("the gate needs uv and git on PATH")
    for name in ("dist", "home", "tmp", "state", "config", "cache", "data", "sandboxes", "cwd"):
        (work / name).mkdir(parents=True, exist_ok=True)
    venv = work / "venv"
    env = _hermetic(work, venv / "bin")

    _step("Build both distributions as the release does (sdist, then the wheel from it)")
    for package in ("packages/agentenv-protocol", "."):
        _run([uv, "tool", "run", "--python", sys.executable, "--from", "build", "pyproject-build",
              "--outdir", str(work / "dist"), package], env=env, cwd=REPO)
    protocol = _one(work / "dist", "agentenv_framework_protocol-*.whl")
    framework = _one(work / "dist", "agentenv_framework-*.whl")

    _step(f"Check {framework.name} ships every tracked example and its entry points")
    tracked = _listed([git, "ls-files", "-z", EXAMPLES], env)
    ignored = _listed([git, "ls-files", "-z", "--cached", "--ignored", "--exclude-standard", EXAMPLES], env)
    declared = tomllib.loads((REPO / "pyproject.toml").read_text())["project"].get("entry-points", {}).get(BUNDLES, {})
    problems = [f"::error file={path}::{path} is tracked but .gitignore matches it, so hatch leaves it out of the wheel"
                for path in ignored]
    problems += wheel_problems(framework, [path for path in tracked if path not in ignored], declared)
    if problems:
        print("\n".join(problems))
        return 1

    _step("Install the two wheels into a fresh venv, resolving everything else from public PyPI")
    _run([uv, "venv", "--python", sys.executable, str(venv)], env=env, cwd=work)
    _run([uv, "pip", "install", "--python", str(venv / "bin" / "python"), "--compile-bytecode", str(protocol),
          str(framework)], env=env, cwd=work)

    _step(f"List the installed bundles, then run {args.bundle} twice by name")
    agent_env, python = str(venv / "bin" / "agent-env"), str(venv / "bin" / "python")
    listing = _run([agent_env, "run"], env=env, cwd=work / "cwd", capture=True).stdout
    rows = [re.split(r"\s{2,}", line) for line in listing.splitlines()]
    if not any(row[0] == args.bundle and len(row) > 2 and row[2] != "invalid" for row in rows):
        return _fail(f"`agent-env run` doesn't list {args.bundle} as a valid bundle")
    _run([agent_env, "run", args.bundle], env=env, cwd=work / "cwd")
    _run([python, str(CHECK), "snapshot", str(work / "after-first-run.json")], env=env, cwd=work / "cwd")
    _run([agent_env, "run", args.bundle], env=env, cwd=work / "cwd")

    _step("Check the runs through the local store")
    return subprocess.run([python, str(CHECK), "verify", "--bundle", args.bundle, "--runs", "2", "--since",
                           str(work / "after-first-run.json")], env=env, cwd=work / "cwd").returncode


def wheel_problems(wheel: Path, tracked: list[str], declared: dict[str, str]) -> list[str]:
    """Why ``wheel`` doesn't ship the tracked example files under ``src/`` and the declared bundle entry points."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        entry_points = next((name for name in names if name.endswith(".dist-info/entry_points.txt")), None)
        shipped = configparser.ConfigParser(interpolation=None)
        shipped.optionxform = str
        if entry_points is not None:
            shipped.read_string(archive.read(entry_points).decode())
    problems = [f"::error file={path}::{path} is tracked but not in {wheel.name} (hatch leaves out .gitignore'd names)"
                for path in tracked if path.removeprefix("src/") not in names]
    found = dict(shipped[BUNDLES]) if shipped.has_section(BUNDLES) else {}
    if found != declared:
        problems.append(f"::error file=pyproject.toml::{wheel.name} registers the {BUNDLES} entry points {found}, "
                        f"but pyproject.toml declares {declared}")
    return problems


def _hermetic(work: Path, venv_bin: Path) -> dict[str, str]:
    """An environment with nothing from the calling shell but its locale: no AWS, agent-env or package-index
    settings, and every home and cache folder under ``work``."""
    return {
        "PATH": os.pathsep.join([str(venv_bin), "/usr/bin", "/bin"]),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "HOME": str(work / "home"),
        "TMPDIR": str(work / "tmp"),
        "XDG_STATE_HOME": str(work / "state"),
        "XDG_CONFIG_HOME": str(work / "config"),
        "XDG_CACHE_HOME": str(work / "cache"),
        "XDG_DATA_HOME": str(work / "data"),
        "AGENT_ENV_LOCAL_SANDBOX_DIR": str(work / "sandboxes"),
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_INDEX_URL": PYPI,
        "UV_DEFAULT_INDEX": PYPI,
        "UV_NO_CONFIG": "1",
        "UV_CACHE_DIR": str(work / "cache" / "uv"),
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
    }


def _run(command: list[str], *, env: dict[str, str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess:
    """Run ``command``, its output streaming to this process's, or captured and then printed with ``capture``."""
    sys.stdout.flush()
    result = subprocess.run(command, env=env, cwd=cwd, text=True, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT)
    if capture:
        print(result.stdout, end="", flush=True)
    if result.returncode != 0:
        print(f"::error::{Path(command[0]).name} exited {result.returncode}: {' '.join(command[1:])}")
        sys.exit(1)
    return result


def _listed(command: list[str], env: dict[str, str]) -> list[str]:
    output = subprocess.run(command, env=env, cwd=REPO, check=True, capture_output=True, text=True).stdout
    return [path for path in output.split("\0") if path]


def _one(folder: Path, pattern: str) -> Path:
    [found] = sorted(folder.glob(pattern))
    return found


def _step(title: str) -> None:
    print(f"\n== {title}", flush=True)


def _fail(message: str) -> int:
    print(f"::error::{message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
