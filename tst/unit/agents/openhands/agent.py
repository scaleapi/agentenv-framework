"""The OpenHands agent's A2A module and its run driver, imported from its folder as its image runs them: by their own
names, from the folder on the path. Its runtime module needs the OpenHands SDK, which only the agent's image installs,
so tests give the agent a fake runtime instead."""

import importlib
import sys
from pathlib import Path
from typing import Final

import agent_env

AGENT_DIR: Final[Path] = Path(agent_env.__file__).parent / "agents" / "agents" / "openhands"
sys.path.insert(0, str(AGENT_DIR))
openhands_agent = importlib.import_module("openhands_agent")
conversation_run = importlib.import_module("conversation_run")
