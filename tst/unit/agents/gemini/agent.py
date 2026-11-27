"""The Gemini agent's modules, imported as its image runs them: by their own names, from its folder and from the folder
its build context takes as tail_follow/, whose module the image copies beside the bridge."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import agent_env
from tst.util.build_context import sent

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "gemini"
IMPORTED = [str(AGENT_DIR), str(sent(AGENT_DIR, "tail_follow/tail_follow.py").parent)]
PYTHONPATH = os.pathsep.join(IMPORTED)
sys.path[:0] = IMPORTED
gemini_config = importlib.import_module("gemini_config")
gemini_kit = importlib.import_module("gemini_kit")
gemini_changelog = importlib.import_module("gemini_changelog")
gemini_bridge = importlib.import_module("gemini_bridge")
gemini_transcript = importlib.import_module("gemini_transcript")
gemini_atif = importlib.import_module("gemini_atif")
# By its path: another agent's a2a_server, imported by the same name, may be loaded already.
_spec = importlib.util.spec_from_file_location("gemini_a2a_server", AGENT_DIR / "a2a_server.py")
a2a_server = importlib.util.module_from_spec(_spec)
# Registered first: pydantic resolves its config's annotations through the module's entry.
sys.modules[_spec.name] = a2a_server
_spec.loader.exec_module(a2a_server)
