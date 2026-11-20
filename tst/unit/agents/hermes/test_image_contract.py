"""The agent's folder is its build context: the Dockerfile copies only files the context sends, the image carries every
module the server imports in the folder it runs the server from, and its build fails, rather than a turn on an armed
clock, when the pinned Hermes' launcher isn't one the bridge reads."""

from __future__ import annotations

import ast
import re

from agent_env.utils.docker.context import BuildContext
from tst.unit.agents.hermes.agent import AGENT_DIR
from tst.unit.agents.test_agent_clock_images import _copied_to

DOCKERFILE = (AGENT_DIR / "Dockerfile").read_text()
SENT = {entry.name: entry.path for entry in BuildContext.of(AGENT_DIR, AGENT_DIR / "Dockerfile").entries}
SERVER_DIR = "/opt/agent"


def _imported(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_every_file_the_image_copies_is_in_its_build_context():
    copied = list(_copied_to(DOCKERFILE))

    assert copied and all(name in SENT or any(sent.startswith(name.rstrip("/") + "/") for sent in SENT)
                          for name in copied), copied


def test_the_image_carries_every_module_the_server_imports_beside_it():
    """The folder's own modules and those the context takes from agent-env's sources, the egress rule and the clock
    page's format among them, each copied into the folder uvicorn imports the server from."""
    modules = {name.rsplit("/", 1)[-1].removesuffix(".py"): name for name in SENT
               if name.endswith(".py") and name.count("/") <= 1 and not name.startswith("agentenv_plugin/")}
    needed, pending = set(), ["a2a_server"]
    while pending:
        if (name := modules[pending.pop()]) not in needed:
            needed.add(name)
            pending += _imported(SENT[name].read_text()) & set(modules)
    folders = _copied_to(DOCKERFILE)

    assert {"hermes_bridge.py", "hermes_kit.py", "agent_clock/agent_egress.py", "clock/clockpage.py"} <= needed
    assert all(SERVER_DIR in folders.get(name, set()) for name in needed), (needed, folders)
    # Beside the bridge too, where its PLUGIN_DIR is: each profile's plugins/agentenv links to it.
    assert SERVER_DIR in folders["agentenv_plugin"]


def test_the_build_reads_the_pinned_hermes_launcher_as_an_armed_turn_does():
    assert re.search(r"^RUN .*python3 -c 'import hermes_bridge; hermes_bridge\.launcher_command\(\)'$", DOCKERFILE,
                     re.MULTILINE)
