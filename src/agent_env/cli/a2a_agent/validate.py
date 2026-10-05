import asyncio
import json
import sys

import click

from agent_env.a2a_agent import A2AAgent


def run_validation(agent: A2AAgent, litellm_api_key: str | None = None) -> dict:
    """Run validation and display results. Returns verifications dict."""
    click.echo(f"Validating agent: id={agent.id} version={agent.version} image={agent.docker_image_artifact.id}")

    verifications = asyncio.run(agent.validate(on_progress=click.echo, litellm_api_key=litellm_api_key))

    # Re-fetch agent to get the persisted metadata
    agent = A2AAgent.get(agent.id, agent.version)

    for key in ("validated_agent_card", "validated_a2a_protocol", "validated_a2a_extensions", "validated_data_extensions"):
        value = agent.metadata.get(key, {})
        click.echo(click.style(f"\n{key}:", bold=True))
        click.echo(json.dumps(value, indent=2))

    return verifications


@click.command()
@click.option("--id", "agent_id", required=True, help="A2A agent id")
@click.option("--version", "agent_version", default=None, type=int, help="Agent version (defaults to latest)")
@click.option("--litellm-api-key", type=str, default=None, help="LiteLLM API key for agent prompts")
@click.option("--sandbox", default=None,
              help="Sandbox backend(s): a built-in or a [sandbox.providers] name from .agentenv/config.toml; "
                   "comma-separated for a fallback chain. Defaults to [sandbox].agent_default when omitted.")
def validate(agent_id: str, agent_version: int | None, litellm_api_key: str | None, sandbox: str):
    """Deploy an A2A agent, validate its agent card and MCP extension."""
    from agent_env.providers import build_sandbox_provider, set_agent_sandbox_provider
    from agent_env.store.base import NotFoundError

    if sandbox:
        try:
            provider = build_sandbox_provider(sandbox)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="'--sandbox'")
        set_agent_sandbox_provider(provider)
    click.echo(f"Sandbox backend: {sandbox or 'config default'}")

    try:
        agent = A2AAgent.get(agent_id, agent_version)
    except NotFoundError:
        click.echo(click.style(f"Error: A2AAgent '{agent_id}' not found", fg="red"), err=True)
        sys.exit(1)

    verifications = run_validation(agent, litellm_api_key=litellm_api_key)

    if not verifications.get("a2a_agent_mcp", {}).get("passed", False):
        sys.exit(1)
