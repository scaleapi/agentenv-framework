"""Serves the pi agent. ``AGENT_CARD`` is what agent-env's install/v1 reads from ``/app/a2a_server.py``."""

from pi_agent import PiAgent

agent = PiAgent()
app = agent.create_app()
AGENT_CARD = app.state.agentenv_a2a.card

if __name__ == "__main__":
    agent.serve()
