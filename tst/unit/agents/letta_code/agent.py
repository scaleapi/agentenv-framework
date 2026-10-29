"""The Letta Code agent's modules, imported from its folder as its image runs them: by their own names, from the folder
on the path."""

import importlib
import importlib.util
import sys
from pathlib import Path

import agent_env

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "letta-code"
sys.path.insert(0, str(AGENT_DIR))
letta_code_bridge = importlib.import_module("letta_code_bridge")
letta_code_kit = importlib.import_module("letta_code_kit")
# By its path: another agent's a2a_server, imported by the same name, may be loaded already.
_spec = importlib.util.spec_from_file_location("letta_code_a2a_server", AGENT_DIR / "a2a_server.py")
a2a_server = importlib.util.module_from_spec(_spec)
# Registered first: pydantic resolves its config's annotations through the module's entry.
sys.modules[_spec.name] = a2a_server
_spec.loader.exec_module(a2a_server)
