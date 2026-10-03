"""The open-source user journey, replayed in fresh containers with no network.

A new user on `python:3.12-slim` installs agent-env as a uv tool, adds two plugins, runs a task
that uses their steps, and removes them; the README's plugin commands run on the way, so the
docs cannot drift from the CLI. A second container uses Ubuntu's system Python, which carries
the PEP 668 marker, and `plugin add` must refuse it. Linux wheels for the dependencies are
downloaded once, inside a container, and cached; the journey itself runs with `--network none`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from tst.installer.conftest import (
    _EXTRA_WHEELS,
    CLI_TIMEOUT_S,
    REPO,
    _checked,
    _run,
    filling,
    locked_requirements,
)
from tst.installer.toys import BROWSER, GRADER, browser_wheel, grader_wheel
from tst.util.capabilities import missing_capability_reason


def _docker_works() -> bool:
    return shutil.which("docker") is not None and _run(["docker", "info"]).returncode == 0


pytestmark = [
    pytest.mark.installer,
    pytest.mark.skipif(not _docker_works(), reason=missing_capability_reason("docker_daemon")),
]

# From ECR Public, pinned by digest like the tst/data images: no Docker Hub pull limits on shared
# runner IPs, and the gate cannot move without a change here. Both are multi-arch indexes.
SLIM = ("mirror.gcr.io/library/python:3.12-slim"
        "@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf")
UBUNTU = ("mirror.gcr.io/library/ubuntu:24.04"
          "@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3")
# The Python both images run, which the Linux wheels are built for.
JOURNEY_PYTHON = "3.12"
# Ubuntu 24.04's system Python is 3.12, the version the wheelhouse is built for, and it carries
# the PEP 668 marker.
PEP668_DOCKERFILE = (
    f"FROM {UBUNTU}\n"
    "RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip "
    "&& rm -rf /var/lib/apt/lists/*\n"
)
_TASK = [
    {"id": "open", "type": "toy_navigate", "url": "https://example.com"},
    {"id": "grade", "type": "toy_grade", "depends_on": [{"task_step_id": "open"}]},
]


@pytest.fixture(scope="session")
def linux_wheels() -> Path:
    """The locked dependencies, plus uv, as wheels for Linux on this Docker's architecture."""
    arch = _checked(["docker", "version", "--format", "{{.Server.Arch}}"]).stdout.strip()
    requirements = locked_requirements() + "".join(f"{name}\n" for name in (*_EXTRA_WHEELS, "uv"))
    key = hashlib.sha256(requirements.encode()).hexdigest()[:16]
    root = REPO / ".cache" / "installer-wheelhouse" / f"{key}-py{JOURNEY_PYTHON}-linux-{arch}"
    with filling(root) as empty:
        if empty:
            (root / "requirements.txt").write_text(requirements)
            _checked(["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
                      "-v", f"{root}:/out", SLIM, "python", "-m", "pip", "wheel", "--no-deps", "-q",
                      "--disable-pip-version-check", "-r", "/out/requirements.txt", "-w", "/out"])
    return root


class Container:
    """One running container, and `docker exec` into it."""

    def __init__(self, image: str, mounts: dict[Path, str], env: dict[str, str]):
        self.name = f"agent-env-journey-{uuid.uuid4().hex[:10]}"
        self.env = env
        volumes = [arg for host, inside in mounts.items() for arg in ("-v", f"{host}:{inside}")]
        _checked(["docker", "run", "-d", "--name", self.name, "--network", "none", *volumes, image,
                  "sleep", "infinity"])

    def run(self, *command: str) -> subprocess.CompletedProcess:
        env = [arg for key, value in self.env.items() for arg in ("-e", f"{key}={value}")]
        return _run(["docker", "exec", "-w", "/work", *env, self.name, *command], timeout=CLI_TIMEOUT_S)

    def ok(self, *command: str) -> subprocess.CompletedProcess:
        proc = self.run(*command)
        assert proc.returncode == 0, f"{' '.join(command)}:\n{proc.stdout}\n{proc.stderr}"
        return proc

    def remove(self) -> None:
        _run(["docker", "rm", "-f", self.name])


@dataclass
class Journey:
    mounts: dict[Path, str]
    work: Path
    grader: str
    browser: str


@pytest.fixture
def journey(tmp_path, checkout_wheels, linux_wheels) -> Journey:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    browser, grader = browser_wheel(plugins), grader_wheel(plugins)
    work = tmp_path / "work"
    work.mkdir()
    work.chmod(0o777)  # the container's user writes its stores and outputs here
    (work / "task.json").write_text(json.dumps(_TASK))
    mounts = {linux_wheels: "/wheels:ro", checkout_wheels: "/checkout:ro", plugins: "/plugins:ro", work: "/work"}
    return Journey(mounts, work, f"/plugins/{grader.name}", f"/plugins/{browser.name}")


def _readme_plugin_commands() -> list[list[str]]:
    """The commands in the README's "Manage plugins" block, as a user would type them."""
    section = (REPO / "README.md").read_text().split("### Manage plugins", 1)[1]
    block = section.split("```bash\n", 1)[1].split("```", 1)[0]
    lines = [line.split("#", 1)[0].split() for line in block.splitlines() if line.startswith("agent-env plugin")]
    assert [line[2] for line in lines] == ["list", "show", "check", "add", "remove"], lines
    return lines


def test_a_new_user_installs_agent_env_adds_plugins_runs_a_task_and_removes_them(journey):
    grader, browser = journey.grader, journey.browser
    offline = {"UV_OFFLINE": "1", "UV_FIND_LINKS": "/checkout,/wheels", "UV_TOOL_BIN_DIR": "/usr/local/bin"}
    container = Container(SLIM, journey.mounts, offline)
    try:
        no_network = container.run("python", "-c", "import socket; socket.create_connection(('pypi.org', 443), 5)")
        assert no_network.returncode != 0, "the journey container must not reach the network"
        container.ok("python", "-m", "pip", "install", "-q", "--no-index", "--find-links", "/wheels", "uv")
        container.ok("uv", "tool", "install", "agentenv-framework")
        rows = container.ok("agent-env", "plugin", "list").stdout.split("STATUS\n", 1)[1]
        assert re.fullmatch(r"agentenv-framework +\S+ +bundle hello +ok\n", rows), rows

        added = container.ok("agent-env", "plugin", "add", browser, grader, "--yes")
        assert "toy_navigate active" in added.stdout and "toy_grade active" in added.stdout

        # The README's commands, verbatim but for their placeholders; --yes because there is no terminal.
        for line in _readme_plugin_commands():
            command = [{"PACKAGE": GRADER, "SPEC...": grader}.get(word, word) for word in line]
            container.ok(*command, *(["--yes"] if line[2] in ("add", "remove") else []))
        assert GRADER not in container.ok("agent-env", "plugin", "list").stdout
        container.ok("agent-env", "plugin", "add", grader, "--yes")

        container.ok("agent-env", "task", "create", "task.json", "--id", "journey")
        container.ok("agent-env", "task", "run", "--id", "journey", "--output-dir", "/work/out")
        (context,) = (journey.work / "out").glob("*.json")
        metadata = json.loads(context.read_text())["metadata"]
        assert metadata["visited"] == ["https://example.com"]
        assert metadata["verifications"]["grade"]["score"] == 1.0

        blocked = container.run("agent-env", "plugin", "remove", BROWSER, "--yes")
        assert blocked.returncode == 1 and "stored task(s) use its step types (toy_navigate)" in blocked.stderr
        container.ok("agent-env", "plugin", "remove", BROWSER, "--yes", "--force")

        gone = container.run("agent-env", "task", "create", "task.json", "--id", "again")
        assert gone.returncode != 0 and "Unknown task step type: toy_navigate" in gone.stdout + gone.stderr
    finally:
        container.remove()


def test_a_system_python_with_the_pep_668_marker_is_refused(journey, tmp_path):
    image = f"agent-env-installer-pep668:{hashlib.sha256(PEP668_DOCKERFILE.encode()).hexdigest()[:12]}"
    if _run(["docker", "image", "inspect", image]).returncode != 0:
        context = tmp_path / "pep668"
        context.mkdir()
        (context / "Dockerfile").write_text(PEP668_DOCKERFILE)
        _checked(["docker", "build", "-q", "-t", image, str(context)])
    container = Container(image, journey.mounts, {})
    try:
        # Only the setup overrides the marker, so there is an agent-env in the system Python to ask.
        container.ok("python3", "-m", "pip", "install", "-q", "--break-system-packages", "--no-index",
                     "--find-links", "/checkout", "--find-links", "/wheels", "agentenv-framework")

        refused = container.run("agent-env", "plugin", "add", journey.grader, "--yes")

        assert refused.returncode == 1
        assert "managed by the system (PEP 668)" in refused.stderr
        assert "uv tool install agentenv-framework --with <plugin>" in refused.stderr
    finally:
        container.remove()
