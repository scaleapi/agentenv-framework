"""An agent that installs on a VM's host declares where it works there (install/v1's ``workdir_host``), and that is
the folder its own host install works in, so a proxy installed in its place works where it would."""

import pytest

from tst.unit.agents.claude_code.agent import a2a_server as claude_code
from tst.unit.agents.codex.agent import a2a_server as codex
from tst.unit.agents.gemini.agent import a2a_server as gemini
from tst.unit.agents.opencode.agent import a2a_server as opencode

INSTALL_V1 = "urn:agentenv:install/v1"


@pytest.mark.parametrize("server", [claude_code, codex, gemini, opencode], ids=["claude-code", "codex", "gemini", "opencode"])
def test_an_agent_declares_the_host_folder_its_own_host_install_works_in(server):
    card = server.AGENT_CARD.model_dump(mode="json", by_alias=True, exclude_none=True)
    (params,) = [ext["params"] for ext in card["capabilities"]["extensions"] if ext["uri"] == INSTALL_V1]

    assert params["install_commands_host"][-1].split()[-1] == params["workdir_host"]
    assert params["user_host"] == "root"
