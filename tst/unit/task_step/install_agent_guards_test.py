"""Same-sandbox tunnel-collision guard for multi-agent host installs (no I/O in the guard), and preflight's refusal of an
agent with no build context to install."""
import pytest
from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep


def _step(agent_name):
    return InstallAgentTaskStep(id="s", version=None, sandbox_name="vm", a2a_agent_id="agent-x", agent_name=agent_name)


def _ctx(agent_name="solver-a", api_url="https://t-8000.gw", sandbox_id="vm-1"):
    ctx = TaskStepContext()
    ctx.deployed_agents.append(DeployedAgent(agent_name=agent_name, api_url=api_url, sandbox_id=sandbox_id))
    return ctx


def test_tunnel_url_collision_rejected():
    with pytest.raises(RuntimeError, match="already serves"):
        _step("solver-b")._check_tunnel_collision(_ctx(), "vm-1", "https://t-8000.gw", 8000)


def test_distinct_port_passes():
    _step("solver-b")._check_tunnel_collision(_ctx(), "vm-1", "https://t-8010.gw", 8010)


def test_other_sandbox_ignored():
    _step("solver-b")._check_tunnel_collision(_ctx(sandbox_id="vm-2"), "vm-1", "https://t-8000.gw", 8000)


# What preflight refuses at save: install/v1 copies an agent's source files from its image's build context

def _agent_with(**image_fields):
    image = get_artifact_store().put_document(DockerImageArtifact(id="agent-x-image", description="d", **image_fields))
    return A2AAgent.put(id="agent-x", docker_image_artifact=image)


def test_preflight_refuses_an_agent_whose_image_has_no_build_context(local_stores):
    _agent_with(image_name="ghcr.io/team/agent@sha256:" + "0" * 64)

    assert _step("solver").preflight() == [
        "install_agent 's': Agent agent-x v1 has no build_context_object_url on its docker_image_artifact; install/v1 "
        "needs the build context (agent source files) to install into the task container"]


def test_preflight_passes_an_agent_with_a_build_context(local_stores):
    _agent_with(image_name="local/agent-x-0123456789ab:v1", build_context_object_url="s3://bucket/ctx.tar.gz")

    assert _step("solver").preflight() == []


def test_preflight_reports_an_agent_the_store_doesnt_hold(local_stores):
    (problem,) = _step("solver").preflight()

    assert problem.startswith("install_agent 's': agent 'agent-x' can't be loaded: ")
