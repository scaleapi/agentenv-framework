"""The Claude Code agent's modules, imported as its image runs them: by their own names, from its folder and from the
folders its build context takes as agent_clock/ and clock/, whose modules the image copies beside the bridge."""

import importlib
import os
import sys
from pathlib import Path

import agent_env
from tst.util.build_context import sent

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "claude-code"
IMPORTED = [str(AGENT_DIR), str(sent(AGENT_DIR, "agent_clock/agent_egress.py").parent),
            str(sent(AGENT_DIR, "clock/clockpage.py").parent)]
PYTHONPATH = os.pathsep.join(IMPORTED)
sys.path[:0] = IMPORTED
a2a_server = importlib.import_module("a2a_server")
tar_changelog = importlib.import_module("tar_changelog")
claude_bridge = importlib.import_module("claude_bridge")
kit = importlib.import_module("kit")
agent_egress = importlib.import_module("agent_egress")
claude_code_atif = importlib.import_module("claude_code_atif")
