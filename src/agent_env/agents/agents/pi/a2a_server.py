"""Serves the pi agent. ``AGENT_CARD`` is what agent-env's install/v1 reads from ``/app/a2a_server.py``."""

import os
from pathlib import Path

from pi_agent import PiAgent

agent = PiAgent(workspace=Path(os.environ.get("PI_A2A_WORKSPACE", "/workspace")))
app = agent.create_app()
AGENT_CARD = app.state.agentenv_a2a.card

if __name__ == "__main__":
    agent.serve()
