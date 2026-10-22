import json

import click

from agent_env.a2a_agent import A2AAgent
from agent_env.plugins._bundles import installed_aliases


@click.command()
@click.option("--id", "agent_id", required=True, help="A2A agent id, or an alias an installed bundle gives an agent")
@click.option("--version", "agent_version", default=None, type=int, help="Agent version (defaults to latest)")
def get(agent_id: str, agent_version: int | None):
    """Get an A2A agent and output its definition as JSON: the agent a step naming that id deploys, so an alias gets
    the agent it names."""

    agent = A2AAgent.resolve(agent_id, agent_version)
    if agent.id != agent_id and (named := installed_aliases().target(agent_id)) is not None:
        click.echo(f"{agent_id} → {agent.id} (the alias {named.declared_by} gives it)", err=True)
    click.echo(json.dumps(agent.to_dict(), indent=2))
