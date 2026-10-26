"""Serves the OpenHands agent, configured from its environment. ``AGENT_CARD`` is what agent-env's install/v1 reads
from ``/app/a2a_server.py``."""

import os
from pathlib import Path
from typing import Final

from a2a.types import AgentCard
from starlette.applications import Starlette

from openhands_agent import DEFAULT_MODEL_ENV, Endpoint, OpenHandsAgent
from openhands_runtime import SdkRuntime

agent: Final[OpenHandsAgent] = OpenHandsAgent(
    SdkRuntime(),
    endpoint=Endpoint.from_environ(os.environ),
    default_model=os.environ.get(DEFAULT_MODEL_ENV),
    workspace=Path(os.environ.get("OPENHANDS_A2A_WORKSPACE", "/workspace")),
)
app: Final[Starlette] = agent.create_app()
AGENT_CARD: Final[AgentCard] = app.state.agentenv_a2a.card

if __name__ == "__main__":
    agent.serve()
