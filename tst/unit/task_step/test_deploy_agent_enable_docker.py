"""deploy_agent `enable_docker`: config persistence, the host-socket security
contract, and the VM-only guard.

That the agent gets a *working* isolated daemon end-to-end is covered by
test_enable_docker_e2e.py against a real sandbox VM.
"""

from unittest.mock import AsyncMock, patch

import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep


def _agent() -> A2AAgent:
    return A2AAgent(
        id="a",
        version=None,
        docker_image_artifact=DockerImageArtifact(
            id="img", description="t", image_name="img:1", tar_gz_s3_url="s3://b/x.tgz"
        ),
    )


def test_enable_docker_persists_through_config():
    set_on = DeployAgentTaskStep(id="d", version=None, enable_docker=True)
    assert DeployAgentTaskStep.from_dict(set_on.to_dict()).enable_docker is True
    assert DeployAgentTaskStep(id="d2", version=None).enable_docker is False
    legacy = {"id": "d3", "type": "deploy_agent", "version": 1}
    assert DeployAgentTaskStep.from_dict(legacy).enable_docker is False


@pytest.mark.asyncio
async def test_enable_docker_never_exposes_host_socket():
    # Security contract: never bind-mount the VM socket; redirect the agent to the
    # isolated daemon. The sandbox (VM) is the external boundary we assert against.
    agent = _agent()
    agent._sandbox = AsyncMock()
    await agent._run_container("img:1", 8000, {"LITELLM_API_KEY": "k"}, enable_docker=True)
    launch = agent._sandbox.exec_script.await_args.args[0]
    assert "/var/run/docker.sock" not in launch
    assert "DOCKER_HOST" in launch


@pytest.mark.asyncio
async def test_enable_docker_refuses_non_vm_sandbox():
    # Docker is VM-only; asking for it on a non-VM sandbox must fail loudly rather
    # than deploy an agent that silently has no Docker.
    agent = _agent()
    sandbox = AsyncMock()
    sandbox.mode = "modal"
    # boto3 (AWS creds) and get_config (secrets) are the external seams deploy()
    # hits before the guard; stub them so the test is hermetic.
    with patch("agent_env.a2a_agent.a2a_agent.boto3.Session") as session, patch(
        "agent_env.config.get_config"
    ):
        session.return_value.get_credentials.return_value = None
        with pytest.raises(ValueError, match="VM sandbox"):
            await agent.deploy(
                sandbox=sandbox,
                enable_docker=True,
                env_vars={"LITELLM_API_KEY": "k", "LITELLM_BASE_URL": "http://x"},
            )
