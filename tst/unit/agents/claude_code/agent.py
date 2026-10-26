"""The Claude Code agent's modules, imported from its folder as its image runs them: by their own names, from the
folder on the path."""

import importlib
import sys
from pathlib import Path

import agent_env

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "claude-code"
sys.path.insert(0, str(AGENT_DIR))
a2a_server = importlib.import_module("a2a_server")
tar_changelog = importlib.import_module("tar_changelog")
claude_bridge = importlib.import_module("claude_bridge")
kit = importlib.import_module("kit")
