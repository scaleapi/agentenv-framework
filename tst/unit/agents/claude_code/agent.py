"""The Claude Code agent's modules, imported as its image runs them: by their own names, from its folder and from the
folder its build context takes as clock/, whose modules the image copies beside the bridge, both on the path."""

import importlib
import os
import sys
from pathlib import Path

import agent_env
from tst.util.build_context import sent

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "claude-code"
CLOCK_DIR = sent(AGENT_DIR, "clock/agent_egress.py").parent
PYTHONPATH = os.pathsep.join([str(AGENT_DIR), str(CLOCK_DIR)])
sys.path[:0] = [str(AGENT_DIR), str(CLOCK_DIR)]
a2a_server = importlib.import_module("a2a_server")
tar_changelog = importlib.import_module("tar_changelog")
claude_bridge = importlib.import_module("claude_bridge")
kit = importlib.import_module("kit")
agent_egress = importlib.import_module("agent_egress")
