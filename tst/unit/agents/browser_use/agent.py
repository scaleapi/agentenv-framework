"""The Browser Use agent's server and its atif/1 adapter, imported as its image runs them: the adapter by its own name,
from the agent's folder, which the image copies beside the server. The folder goes last on the path, so its
``a2a_server`` and ``agent`` never stand in for another agent's of the same name."""

import importlib
import importlib.util
import sys
from pathlib import Path

import agent_env

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "browser-use"
sys.path.append(str(AGENT_DIR))
browser_use_atif = importlib.import_module("browser_use_atif")
# By its path: another agent's a2a_server, imported by the same name, may be loaded already.
_spec = importlib.util.spec_from_file_location("browser_use_a2a_server", AGENT_DIR / "a2a_server.py")
a2a_server = importlib.util.module_from_spec(_spec)
# Registered first: pydantic resolves its config's annotations through the module's entry.
sys.modules[_spec.name] = a2a_server
_spec.loader.exec_module(a2a_server)
