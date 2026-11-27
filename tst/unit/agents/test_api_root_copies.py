"""Every copy of core's ``api_root``, in an agent or the key holder, whose images can't import core, answers as core's
does. Each copy is read from its file's source, since some agents' modules import packages only their images install."""

import ast
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

import agent_env
from agent_env.config.model import api_root

SOURCES = Path(agent_env.__file__).parent
AGENTS = SOURCES / "agents" / "agents"
KEY_HOLDER = SOURCES / "utils" / "key_holder" / "key_holder.py"
URLS = [
    "https://proxy.example", "https://proxy.example/", "https://proxy.example/v1", "https://proxy.example/v1/",
    "https://proxy.example/v1/v1", "https://proxy.example/prefix", "https://proxy.example/prefix/v1",
    "http://172.17.0.1:4000/v1", "https://proxy.example/v1?team=a", " https://proxy.example/v1 ",
    "HTTPS://Proxy.Example/V1", "https://proxy.example/v10", "",
]


def _copies() -> dict[str, object]:
    """Each agent file's ``_api_root``, run from its source with only what it imports from ``urllib.parse``."""
    copies = {}
    for path in [*sorted(AGENTS.glob("*/*.py")), KEY_HOLDER]:
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_api_root":
                namespace = {"urlsplit": urlsplit, "urlunsplit": urlunsplit}
                exec(compile(ast.Module([node], type_ignores=[]), str(path), "exec"), namespace)
                copies[str(path.relative_to(SOURCES))] = namespace["_api_root"]
    return copies


COPIES = _copies()


def test_the_agents_that_append_their_own_path_each_have_a_copy():
    assert {"agents/agents/browser-use/agent.py", "agents/agents/claude-code/claude_bridge.py",
            "agents/agents/codex/codex_config.py", "agents/agents/gemini/a2a_server.py",
            "utils/key_holder/key_holder.py"} <= set(COPIES)


@pytest.mark.parametrize("copy", sorted(COPIES))
def test_a_copy_answers_as_core_does(copy):
    assert [COPIES[copy](url) for url in URLS] == [api_root(url) for url in URLS]
