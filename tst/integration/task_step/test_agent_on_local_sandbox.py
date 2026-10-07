"""An agent deployed onto a local deploy_sandbox sandbox answers at the port that sandbox published, two such agents
run at once, and teardown takes each down. Builds the echo agent (tst/data/a2a_agent); needs Docker and the local
registry.
"""

import asyncio
import shutil
import subprocess
import uuid

import httpx
import pytest

from agent_env.a2a_agent.a2a_agent import DEFAULT_A2A_PORT
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from tst.util.a2a_test_agent import put_test_agent
from tst.util.capabilities import missing_capability_reason

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]


@pytest.fixture
def local_backends(monkeypatch, tmp_path):
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    configure()
    reset_artifact_store()
    try:
        yield tmp_path
    finally:
        reset_artifact_store()
        reset_config()


@pytest.mark.asyncio
async def test_agents_placed_on_local_sandboxes_answer_at_their_own_ports(local_backends):
    agent = put_test_agent(f"placed-agent-{uuid.uuid4().hex[:8]}")

    async def place(name: str) -> TaskStepContext:
        context = TaskStepContext(instance_id=f"placed-{name}-{uuid.uuid4().hex[:8]}")
        context = await DeploySandboxTaskStep(
            id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local",
            exposed_ports=[DEFAULT_A2A_PORT],
        ).execute(context)
        return await DeployAgentTaskStep(
            id="agent", version=None, env_ids=[], a2a_agent_id=agent.id, agent_name="solver", sandbox_name="box",
        ).execute(context)

    contexts = await asyncio.gather(place("a"), place("b"), return_exceptions=True)
    try:
        assert not [c for c in contexts if isinstance(c, BaseException)], contexts
        urls = [c.deployed_agents[0].a2a_url for c in contexts]
        assert urls == [c.deployed_sandboxes[0].tunnel_urls[str(DEFAULT_A2A_PORT)] for c in contexts]
        assert len(set(urls)) == 2
        async with httpx.AsyncClient() as client:
            assert [(await client.get(f"{url}/.well-known/agent.json", timeout=30)).status_code for url in urls] == [
                200, 200]
    finally:
        reports = [await teardown_run(c) for c in contexts if isinstance(c, TaskStepContext)]

    assert not any(r.still_up for r in reports)
    for context in contexts:
        sandbox_id = context.deployed_sandboxes[0].sandbox_id
        assert not subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^agent-{sandbox_id}$"],
                                  capture_output=True, text=True).stdout.strip()
