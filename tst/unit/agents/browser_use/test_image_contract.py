"""The agent's folder is its build context: the Dockerfile copies the server, every module of the folder the server
imports (its atif/1 adapter among them) and agent.py, which browser-use's own environment runs, and installs the
protocol at or above its floor, which a release doesn't pin."""

import ast
import re

from tst.unit.agents.browser_use.agent import AGENT_DIR

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


def test_every_file_the_image_copies_is_in_the_folder_and_the_servers_modules_are_among_them():
    modules = {path.stem for path in AGENT_DIR.glob("*.py")}
    copied = _copied()

    assert all((AGENT_DIR / name).is_file() for name in copied), copied
    assert {f"{module}.py" for module in _imported("a2a_server.py") & modules} | {"a2a_server.py", "agent.py"} \
        <= set(copied)
    assert "browser_use_atif.py" in copied


def test_the_protocol_floor_is_the_one_the_atif_adapter_is_tested_against():
    requirements = [line for line in (AGENT_DIR / "requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]

    assert requirements == ["agentenv-framework-protocol[agent]>=0.1.670"]
