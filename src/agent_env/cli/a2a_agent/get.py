import json

import click

from agent_env.a2a_agent import A2AAgent


@click.command()
@click.option("--id", "agent_id", required=True, help="A2A agent id")
@click.option("--version", "agent_version", default=None, type=int, help="Agent version (defaults to latest)")
def get(agent_id: str, agent_version: int | None):
    """Get an A2A agent and output its definition as JSON."""

    agent = A2AAgent.get(agent_id, agent_version)
    click.echo(json.dumps(agent.to_dict(), indent=2))
