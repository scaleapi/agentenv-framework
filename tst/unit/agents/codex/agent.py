"""The Codex agent's modules, imported as its image runs them: by their own names, from its folder and from the folders
its build context takes as agent_clock/ and clock/, whose modules the image copies beside the bridge."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import agent_env
from tst.util.build_context import sent

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "codex"
IMPORTED = [str(AGENT_DIR), str(sent(AGENT_DIR, "agent_clock/agent_egress.py").parent),
            str(sent(AGENT_DIR, "clock/clockpage.py").parent)]
PYTHONPATH = os.pathsep.join(IMPORTED)
sys.path[:0] = IMPORTED
codex_bridge = importlib.import_module("codex_bridge")
codex_config = importlib.import_module("codex_config")
codex_kit = importlib.import_module("codex_kit")
agent_egress = importlib.import_module("agent_egress")
# By its path: Claude Code's a2a_server, imported by the same name, may be loaded already.
_spec = importlib.util.spec_from_file_location("codex_a2a_server", AGENT_DIR / "a2a_server.py")
a2a_server = importlib.util.module_from_spec(_spec)
# Registered first: pydantic resolves its config's annotations through the module's entry.
sys.modules[_spec.name] = a2a_server
_spec.loader.exec_module(a2a_server)
