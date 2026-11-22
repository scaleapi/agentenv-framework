"""Every bundled agent that installs through install/v1 takes its control token from the installer: its declaration
lists ``control_token`` in ``optional_params`` (and ``optional_params_host`` for a host install), and its install reads
the protocol's ``CONTROL_TOKEN_ENV``, which the installer exports. An agent whose install spelled it otherwise would
install fine and serve its extension routes open."""

import re
from pathlib import Path

import pytest
from agentenv_protocol import CONTROL_TOKEN_ENV

import agent_env

AGENTS = Path(agent_env.__file__).parent / "agents" / "agents"


def _sources(agent: str) -> dict[str, str]:
    return {path.name: path.read_text() for path in (AGENTS / agent).iterdir() if path.suffix in (".py", ".sh")}


INSTALLABLE = sorted(path.name for path in AGENTS.iterdir()
                     if path.is_dir() and any("required_params=" in src for src in _sources(path.name).values()))


def test_the_bundled_agents_that_install_are_the_ones_this_test_knows():
    assert INSTALLABLE == ["claude-code", "codex", "gemini", "letta-code", "opencode", "openhands", "pi", "proxy",
                           "pydantic-ai"]


@pytest.mark.parametrize("agent", INSTALLABLE)
def test_an_installed_agent_takes_its_control_token_under_the_protocols_variable(agent):
    sources = _sources(agent)
    (declaration,) = [src for src in sources.values() if "required_params=" in src]

    assert 'optional_params=["control_token"]' in declaration
    if "required_params_host=" in declaration:
        assert 'optional_params_host=["control_token"]' in declaration
    # The installer exports the variable; the install reads it, as ${...} in a shell (${{...}} in a format template).
    assert any(re.search(r"\$\{+" + CONTROL_TOKEN_ENV + r"\b", src) for src in sources.values())
