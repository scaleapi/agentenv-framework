"""The A2A agent validator deploys the mail-mcp example's env by its alias, `mail`, writing it first when the store has
no env of that id, with namespace routing on throughout so the alias's @local env resolves, in the CLI or a service."""

import asyncio
import contextvars
import importlib
import threading

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent import validator as validator_module
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError
from agent_env.cli import cli
from agent_env.env import store as env_store_module
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.plugins._bundles import InstalledAliases
from agent_env.store.routing import namespace_routing_enabled

# By its path: the package exports the command under the module's name.
validate_module = importlib.import_module("agent_env.cli.a2a_agent.validate")


@pytest.fixture
def written(monkeypatch, local_stores):
    """The bundles the validator writes, as (folder, id_root)."""
    calls = []
    monkeypatch.setattr(validator_module, "_write_bundle",
                        lambda root, **kwargs: calls.append((root.name, kwargs["id_root"])))
    return calls


def test_the_validator_deploys_the_mail_mcp_examples_env_by_its_alias():
    assert A2AAgent.VALIDATION_ENV_ID == "mail"


def test_the_example_its_alias_names_is_written_when_the_store_has_no_env_of_that_id(written):
    lines = []

    validator_module.write_validation_env(lines.append)

    assert written == [("mail-mcp", "@local/agentenv-framework/mail-mcp")]
    assert lines == ["mail → @local/agentenv-framework/mail-mcp/mail (the alias agentenv-framework's mail-mcp bundle "
                     "gives it): writing it"]


def test_an_env_the_store_held_under_that_id_before_is_the_one_deployed(written, monkeypatch):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="mail/env_image", description="d", image_name="ghcr.io/acme/mail:1", tar_gz_s3_url="s3://b/x"))
    with monkeypatch.context() as before_the_alias:
        before_the_alias.setattr(env_store_module, "installed_aliases",
                                 lambda table: InstalledAliases({}, {}, {}, table))
        MCPServerEnv.put(id="mail", docker_image_artifact=image, environment_name="mail")

    validator_module.write_validation_env(lambda line: None)

    assert written == []


@pytest.mark.asyncio
async def test_validation_writes_the_env_then_runs_with_namespace_routing_on_throughout(monkeypatch):
    seen = []
    monkeypatch.setattr(validator_module, "write_validation_env",
                        lambda on_progress: seen.append(("write", namespace_routing_enabled())))

    async def run(agent, on_progress, litellm_api_key):
        seen.append(("run", namespace_routing_enabled()))
        return {"done": True}

    monkeypatch.setattr(validator_module.A2AAgentValidator, "_run", staticmethod(run))

    assert await validator_module.A2AAgentValidator.validate(object()) == {"done": True}
    assert seen == [("write", True), ("run", True)] and not namespace_routing_enabled()


@pytest.mark.asyncio
async def test_the_writers_progress_reaches_the_callback_on_the_event_loop_in_the_callers_context(monkeypatch):
    """A service's callback can need both, as a Temporal activity's heartbeat does."""
    caller = contextvars.ContextVar("caller")
    caller.set("activity")
    loop_thread, lines = threading.get_ident(), []
    monkeypatch.setattr(validator_module, "write_validation_env", lambda on_progress: on_progress("writing it"))

    async def run(agent, on_progress, litellm_api_key):
        return {}

    monkeypatch.setattr(validator_module.A2AAgentValidator, "_run", staticmethod(run))

    def on_progress(line):
        asyncio.get_running_loop()
        lines.append((line, threading.get_ident() == loop_thread, caller.get(None)))

    await validator_module.A2AAgentValidator.validate(object(), on_progress=on_progress)

    assert lines == [("writing it", True, "activity")]


def test_the_cli_says_why_the_env_couldnt_be_written(monkeypatch, local_stores, cli_routing):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="agent-image", description="d", image_name="ghcr.io/acme/agent:1", tar_gz_s3_url="s3://b/x"))
    A2AAgent.put(id="mine", docker_image_artifact=image)

    def refuse(agent, litellm_api_key=None):
        raise BundleError(["tasks/smoke.json: no docker here"])

    monkeypatch.setattr(validate_module, "run_validation", refuse)

    result = CliRunner().invoke(cli, ["a2a-agent", "validate", "--id", "mine"])

    assert result.exit_code == 1 and "Error: tasks/smoke.json: no docker here" in result.output
