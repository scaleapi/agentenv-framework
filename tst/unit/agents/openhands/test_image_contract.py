"""The agent's folder is its build context: the Dockerfile copies every module the server imports, which install/v1's
build context is packed from, and the image installs the protocol at or above the floor its ATIF adapter is written
for, which a release doesn't pin."""
from __future__ import annotations

import ast
import re

from tst.unit.agents.openhands.agent import AGENT_DIR

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()


def _copied() -> list[str]:
    return [name for line in re.findall(r"^COPY (?!--from=)(.+) \S+$", DOCKERFILE, re.MULTILINE)
            for name in line.split()]


def _imported(name: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse((AGENT_DIR / name).read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_image_copies_every_module_the_server_imports():
    modules = {path.stem: path.name for path in AGENT_DIR.glob("*.py")}
    needed, pending = set(), ["a2a_server"]
    while pending:
        if (name := modules[pending.pop()]) not in needed:
            needed.add(name)
            pending += _imported(name) & set(modules)

    assert {"openhands_agent.py", "openhands_atif.py", "openhands_runtime.py", "conversation_run.py"} <= needed
    assert needed <= set(_copied()), needed
    assert all((AGENT_DIR / name).is_file() for name in _copied())


def test_the_image_installs_the_protocol_at_or_above_the_adapters_floor():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert "agentenv-framework-protocol[agent]>=0.1.670" in requirements
