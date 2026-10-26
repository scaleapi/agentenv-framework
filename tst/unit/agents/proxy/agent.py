"""The proxy's a2a_server, imported from its folder by its path: Claude Code's, imported by the same name, may be
loaded already."""

import importlib.util
from pathlib import Path

import agent_env

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "proxy"
_spec = importlib.util.spec_from_file_location("proxy_a2a_server", AGENT_DIR / "a2a_server.py")
a2a_server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a2a_server)
