"""The agent's folder is its build context: the Dockerfile copies only its own files and those shared_sources.toml takes
from agent-env's sources, installs the SDK at the version its tool names are recorded for and the protocol at or above
its floor, which a release doesn't pin, and keeps the agent's state under /data whatever the uid; install.sh carries
the same modules."""

from __future__ import annotations

import ast
import re
import tomllib

from tst.unit.agents.antigravity.agent import AGENT_DIR, antigravity_bridge

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()
SOURCES = AGENT_DIR.parents[2]
SHARED = tomllib.loads((AGENT_DIR / "shared_sources.toml").read_text())
INSTALL_SH = (AGENT_DIR / "install.sh").read_text()


def _copied() -> list[str]:
    return [name for line in re.findall(r"^COPY (?!--from=)(.+) \S+$", DOCKERFILE, re.MULTILINE)
            for name in line.split()]


def _source(name: str):
    """Where the build context's ``name`` comes from: the folder, or a folder shared_sources.toml takes from agent-env's
    own sources at that name."""
    first, _, rest = name.partition("/")
    return SOURCES / SHARED[first] / rest if first in SHARED else AGENT_DIR / name


def _in_context(name: str) -> bool:
    path = _source(name)
    return path.is_dir() if name.endswith("/") else path.is_file()


def test_every_file_the_image_copies_is_in_the_folder():
    copied = _copied()

    assert copied and all(_in_context(name) for name in copied), copied
    # install.sh and run.sh are install/v1's: a build context pruned to what the Dockerfile reads keeps them so.
    assert {"a2a_server.py", "antigravity_bridge.py", "antigravity_date.py", "antigravity_kit.py", "entrypoint.sh",
            "install.sh", "run.sh", "requirements.txt"} <= set(copied)


def _imported(path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_image_and_an_install_carry_every_module_the_server_imports():
    """The folder's own modules and those the build context takes from agent-env's sources, the egress rule, the clock
    page's format and the model relay among them, each by its name in the context."""
    shared = {path.stem: f"{name}/{path.name}" for name, source in SHARED.items()
              for path in (SOURCES / source).glob("*.py")}
    modules = {**shared, **{path.stem: path.name for path in AGENT_DIR.glob("*.py")}}
    needed, pending = set(), ["a2a_server"]
    while pending:
        if (name := modules[pending.pop()]) not in needed:
            needed.add(name)
            pending += _imported(_source(name)) & set(modules)
    installed = set(re.search(r'^AGENT_FILES="([^"]+)"$', INSTALL_SH, re.MULTILINE).group(1).split())

    assert {"agent_clock/agent_egress.py", "clock/clockpage.py", "model_relay/model_relay.py",
            "antigravity_date.py"} <= needed
    assert needed <= set(_copied()) and needed <= installed, (needed, installed)


def test_the_requirements_are_the_protocol_floor_the_pinned_sdk_the_relays_http_and_the_ca_bundle():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert requirements == ["agentenv-framework-protocol[agent]>=0.1.643",
                            f"google-antigravity=={antigravity_bridge.SDK_VERSION}", "aiohttp>=3.9,<4", "certifi"]
    assert "pip install --no-cache-dir -r requirements.txt" in DOCKERFILE
    assert "-r app/requirements.txt" in INSTALL_SH


def test_the_image_starts_through_cmd_which_modal_passes_its_env_to():
    assert 'CMD ["/usr/local/bin/start-agent"]' in DOCKERFILE
    assert not re.search(r"^ENTRYPOINT", DOCKERFILE, re.MULTILINE)


def test_the_agents_home_is_fixed_under_data_whatever_the_uid():
    env = dict(re.findall(r"(\w+)=(\S+)", re.search(r"^ENV (.+)$", DOCKERFILE, re.MULTILINE).group(1)))

    assert (env["HOME"], env["A2A_PORT"], env["PYTHONPATH"]) == ("/data", "8000", "/opt/agent")
    assert "poppler-utils" in DOCKERFILE
    assert "ANTIGRAVITY_HOME=%s" in INSTALL_SH and "HOME=%s" in INSTALL_SH


def test_the_harness_follows_the_clock_only_through_what_it_starts():
    """The harness is a static binary libfaketime can't reach: the image preloads the library for what it starts, and
    the server runs on real time, as every clocked agent's does."""
    assert "/etc/ld.so.preload" in DOCKERFILE
    assert 'AGENTENV_REAL_TIME="1"' in (AGENT_DIR / "root.conf").read_text()
