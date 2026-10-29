"""The pi agent's modules, imported from its folder as its image runs them: by their own names, from the folder on the
path."""

import importlib
import importlib.util
import sys
from pathlib import Path

import agent_env

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "pi"
sys.path.insert(0, str(AGENT_DIR))
changelog = importlib.import_module("changelog")
peers = importlib.import_module("peers")
pi_agent = importlib.import_module("pi_agent")
# By its path: Claude Code's a2a_server, imported by the same name, may be loaded already.
_spec = importlib.util.spec_from_file_location("pi_a2a_server", AGENT_DIR / "a2a_server.py")
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)
