"""The Pydantic AI agent's A2A module, imported from its folder as its image runs it: by its own name, from the folder on
the path. Its runtime module needs Pydantic AI, which only the agent's image installs, so tests give the agent a fake
runtime instead."""

import importlib
import sys
from pathlib import Path
from typing import Final

import agent_env

AGENT_DIR: Final[Path] = Path(agent_env.__file__).parent / "agents" / "agents" / "pydantic-ai"
sys.path.insert(0, str(AGENT_DIR))
pydantic_ai_agent = importlib.import_module("pydantic_ai_agent")
