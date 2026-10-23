"""`agent-env a2a-agent validate` writes the env it deploys, the mail-mcp example's by its alias, when the store has
none of that id."""

import importlib

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError
from agent_env.cli import cli
from agent_env.env.envs.mcp_server import MCPServerEnv

# By its path: the package exports the command under the module's name.
validate_module = importlib.import_module("agent_env.cli.a2a_agent.validate")


@pytest.fixture
def written(monkeypatch, local_stores, cli_routing):
    """The bundles validate writes, as (root, id_root, the sandbox each is checked on)."""
    calls = []
    monkeypatch.setattr(validate_module, "write_bundle", lambda root, **kwargs: calls.append(
        (root.name, kwargs["id_root"], kwargs.get("sandbox"))))
    return calls


def test_the_validator_deploys_the_mail_mcp_examples_env_by_its_alias():
    assert A2AAgent.VALIDATION_ENV_ID == "mail-mcp"


def test_validate_writes_the_example_its_alias_names_on_the_env_backend_when_the_store_has_no_env_of_that_id(written):
    """Checked where the validation deploys the env, the configured env backend, not on --sandbox, the agents'."""
    validate_module._write_validation_env()

    assert written == [("mail-mcp", "@local/agentenv-framework/mail-mcp", None)]


def test_the_stores_env_of_that_id_is_the_one_validate_deploys(written):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="mail-mcp/env_image", description="d", image_name="ghcr.io/acme/mail:1", tar_gz_s3_url="s3://b/x"))
    MCPServerEnv.put(id="mail-mcp", docker_image_artifact=image, environment_name="mail")

    validate_module._write_validation_env()

    assert written == []


def test_stores_that_arent_local_need_an_env_of_that_id_and_are_told_how_to_put_one(written, monkeypatch):
    monkeypatch.setattr(validate_module, "stores_are_local", lambda: False)

    with pytest.raises(BundleError, match="the store has no env 'mail-mcp', the one validate deploys, and validate "
                                          "writes agentenv-framework's mail-mcp bundle's only into local stores; put "
                                          r"it with `agent-env env mcp-server put --id mail-mcp --dockerfile .*"
                                          r"mail-mcp/envs/mail/Dockerfile`"):
        validate_module._write_validation_env()
    assert written == []


def test_the_cli_writes_the_env_before_it_validates(written, monkeypatch):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="agent-image", description="d", image_name="ghcr.io/acme/agent:1", tar_gz_s3_url="s3://b/x"))
    A2AAgent.put(id="mine", docker_image_artifact=image)
    order = []
    monkeypatch.setattr(validate_module, "_write_validation_env", lambda: order.append("write"))
    monkeypatch.setattr(validate_module, "run_validation", lambda agent, litellm_api_key=None: order.append(
        "validate") or {"a2a_agent_mcp": {"passed": True}})

    result = CliRunner().invoke(cli, ["a2a-agent", "validate", "--id", "mine", "--sandbox", "local"])

    assert result.exit_code == 0, result.output
    assert order == ["write", "validate"]
