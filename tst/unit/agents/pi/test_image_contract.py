"""The agent's folder is its build context, and the Dockerfile's COPY line names what install/v1 packs as well: the image
and an install carry every module the server imports, the ATIF adapter among them, and install the protocol alone, at or
above the floor the adapter is written against."""

from __future__ import annotations

import ast
import re

from tst.unit.agents.pi.agent import AGENT_DIR
from tst.util.build_context import sent

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()


def _copied() -> list[str]:
    return [name for line in re.findall(r"^COPY (?!--from=)(.+) \S+$", DOCKERFILE, re.MULTILINE)
            for name in line.split()]


def _imported(path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_image_and_an_install_carry_every_module_the_server_imports():
    modules = {path.stem: path.name for path in AGENT_DIR.glob("*.py")}
    needed, pending = set(), ["a2a_server"]
    while pending:
        if (name := modules[pending.pop()]) not in needed:
            needed.add(name)
            pending += _imported(AGENT_DIR / name) & set(modules)

    assert {"pi_agent.py", "pi_atif.py"} <= needed <= set(_copied()), needed
    assert all(sent(AGENT_DIR, name) == AGENT_DIR / name for name in _copied())


def test_the_image_installs_the_protocol_alone_at_the_adapters_floor():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert requirements == ["agentenv-framework-protocol[agent]>=0.1.670"]
