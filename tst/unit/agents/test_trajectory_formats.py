"""Every bundled agent declares its trajectory's native format, so its card offers it as ``formats``, ``native`` and
``default``, the SDK holds its log and returned native trajectories to it, and every read it serves names it. An agent
that left ``native`` out would keep serving trajectories its card can't describe. One with an ATIF adapter offers atif/1
too, after native."""
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
from agentenv_protocol.a2a_agent import TRAJECTORY_V1

import agent_env

AGENTS_DIR = Path(agent_env.__file__).parent / "agents" / "agents"
#: Each bundled agent's native format, and where its tests import the module holding its agent class.
#: The agents with an ATIF adapter, whose cards offer atif/1 beside native.
SERVES_ATIF = {"claude-code"}
AGENTS = {
    "antigravity": ("antigravity-sdk/1", "tst.unit.agents.antigravity.agent", "a2a_server"),
    "claude-code": ("claude-code-stream-json/1", "tst.unit.agents.claude_code.agent", "a2a_server"),
    "codex": ("codex-exec-json/1", "tst.unit.agents.codex.agent", "a2a_server"),
    "gemini": ("gemini-stream-json/1", "tst.unit.agents.gemini.agent", "a2a_server"),
    "hermes": ("hermes-hook-events/1", "tst.unit.agents.hermes.agent", "a2a_server"),
    "letta-code": ("letta-code-events/1", "tst.unit.agents.letta_code.agent", "a2a_server"),
    "opencode": ("opencode-events/1", "tst.unit.agents.opencode.agent", "a2a_server"),
    "openhands": ("openhands-events/v1", "tst.unit.agents.openhands.agent", "openhands_agent"),
    "pi": ("pi-json-events/v1", "tst.unit.agents.pi.agent", "pi_agent"),
    "proxy": ("agentenv-proxy/v1", "tst.unit.agents.proxy.agent", "a2a_server"),
    "pydantic-ai": ("pydantic-ai-messages/v1", "tst.unit.agents.pydantic_ai.agent", "pydantic_ai_agent"),
}


def _trajectory(module) -> object:
    """The trajectory activation of the agent class the module itself declares."""
    (agent,) = [value for value in vars(module).values() if isinstance(value, type)
                and value.__module__ == module.__name__ and "_agentenv_a2a_definition" in vars(value)]
    (activation,) = [a for a in vars(agent)["_agentenv_a2a_definition"].extensions if a.definition is TRAJECTORY_V1]
    return activation


def test_every_bundled_agent_is_listed():
    folders = sorted(path.parent.name for path in AGENTS_DIR.glob("*/requirements.txt"))
    assert folders == sorted([*AGENTS, "browser-use"])


@pytest.mark.parametrize("agent", sorted(AGENTS))
def test_a_bundled_agent_declares_its_native_trajectory_format(agent):
    native, loader, attribute = AGENTS[agent]
    activation = _trajectory(getattr(importlib.import_module(loader), attribute))
    assert activation.options["native"] == native
    formats = [native, "atif/1"] if agent in SERVES_ATIF else [native]
    assert dict(activation.wire_params) == {"formats": formats, "native": native, "default": native}


def test_browser_use_declares_its_native_trajectory_format():
    # browser-use's own test stubs its runtime before importing it, so the module is read here by its path.
    name = "browser_use_a2a_server"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, AGENTS_DIR / "browser-use" / "a2a_server.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    assert _trajectory(module).options["native"] == module.TRAJECTORY_FORMAT == "browser-use-messages/1"
