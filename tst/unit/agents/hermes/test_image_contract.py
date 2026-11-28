"""The agent's folder is its build context: the Dockerfile copies every module the server imports, the plugin Hermes
loads among them, and installs the protocol at or above the floor its atif/1 adapter needs, which a release doesn't
pin."""
from __future__ import annotations

import ast
import re

from tst.unit.agents.hermes.agent import AGENT_DIR

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()


def _copied() -> set[str]:
    return {name for line in re.findall(r"^COPY (.+) \S+$", DOCKERFILE, re.MULTILINE) for name in line.split()}


def _imported(path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_image_carries_every_module_the_server_imports():
    modules = {path.stem for path in AGENT_DIR.glob("*.py")}
    needed, pending = set(), ["a2a_server"]
    while pending:
        if (name := pending.pop()) not in needed:
            needed.add(name)
            pending += _imported(AGENT_DIR / f"{name}.py") & modules

    assert "hermes_atif" in needed
    assert {f"{name}.py" for name in needed} | {"agentenv_plugin", "entrypoint.sh"} <= _copied()


def test_the_protocol_floor_is_the_one_the_adapter_needs():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert "agentenv-framework-protocol[agent]>=0.1.670" in requirements
