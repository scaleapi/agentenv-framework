"""An agent installed on a Modal VM's host, played from this machine through the proxy installed in its place: the
shape of a task that loads its inputs onto the VM, installs a harness there and checks the outputs it leaves."""

import asyncio
import tempfile
import uuid
from pathlib import Path

import modal
import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact import DockerImageArtifact, FileArtifact
from agent_env.task import EpisodeSpec, Task
from tst.util.capabilities import skip_without_remote_sandbox

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_remote_sandbox("modal_vm")]

# An agent the proxy replaces needs only a card declaring where it works on a VM host.
REPLACED_SERVER = """
class _Card:
    def model_dump(self):
        return {"name": "replaced", "capabilities": {"extensions": [{"uri": "urn:agentenv:install/v1", "params": {
            "workdir_host": "/opt/agent-{agent_name}/workspace", "user_host": "agent-{agent_name}",
            "install_commands_host": ["false"], "a2a_port": 8000}}]}}


AGENT_CARD = _Card()
"""
REPLACED_DOCKERFILE = """FROM mirror.gcr.io/library/python:3.12-slim
WORKDIR /app
COPY a2a_server.py .
"""


@pytest.mark.asyncio
async def test_an_agent_installed_on_a_vm_host_is_played_through_a_proxy_installed_in_its_place():
    uid = uuid.uuid4().hex[:8]
    greeting = FileArtifact.put_bytes(id=f"greeting-vm-{uid}", description="proxy on a VM", filename="greeting.txt",
                                      content=b"hello", content_type="text/plain")
    task = Task.from_dict({"id": f"copy-vm-{uid}", "steps": [
        {"id": "vm", "type": "deploy_sandbox", "sandbox_name": "vm", "sandbox_mode": "vm", "sandbox_type": "modal_vm",
         "exposed_ports": [8000], "cpu": 2, "memory_mb": 8192, "ttl_seconds": 1800},
        {"id": "load", "type": "load_artifact", "sandbox_name": "vm", "artifact_id": greeting.id,
         "destination_path": "/opt/agent-solver/workspace"},
        {"id": "install", "type": "install_agent", "sandbox_name": "vm", "a2a_agent_id": _replaced(uid).id,
         "agent_name": "solver"},
        {"id": "solve", "type": "prompt_agent", "agent_name": "solver", "prompt_id": "solve",
         "prompt": "Copy greeting.txt to outputs/answer.txt."},
        {"id": "check", "type": "verify_sandbox", "sandbox_name": "vm", "base_dir": "/opt/agent-solver/workspace",
         "criteria": [{"type": "probe_file_contains", "criterion": "the greeting is copied",
                       "paths": ["outputs/answer.txt"], "expected": "^hello$"}]},
        {"id": "check-agent", "type": "verify_sandbox", "agent_name": "solver", "base_dir": "/opt/agent-solver/workspace",
         "criteria": [{"type": "probe_file_exists", "criterion": "the answer is on the agent's host",
                       "paths": ["outputs/answer.txt"]}]},
    ]})

    episode = await EpisodeSpec.from_task(task).start()
    try:
        turn = await episode.next_turn()
        async with turn.tools() as tools:
            assert [tool.name for tool in tools.list()] == ["bash"]
            assert (await tools.call("bash", {"command": "pwd; whoami; cat greeting.txt"})).text == (
                "/opt/agent-solver/workspace\nagent-solver\nhello")
            await tools.call("bash", {"command": "mkdir -p outputs && cp greeting.txt outputs/answer.txt"})
        await turn.end("Copied.")
        assert await episode.result() == {"check": 1.0, "check-agent": 1.0}
    finally:
        await episode.abandon("the test ended")
    [vm] = Task.get_instance(episode.id).context["deployed_sandboxes"]
    sandbox = await modal.Sandbox.from_id.aio(vm["sandbox_id"])
    async with asyncio.timeout(60):
        while await sandbox.poll.aio() is None:
            await asyncio.sleep(2)


def _replaced(uid: str) -> A2AAgent:
    with tempfile.TemporaryDirectory() as build_dir:
        context = Path(build_dir)
        (context / "a2a_server.py").write_text(REPLACED_SERVER)
        (context / "Dockerfile").write_text(REPLACED_DOCKERFILE)
        image = DockerImageArtifact.put_context(f"replaced-agent-image-{uid}", description="proxy on a VM",
                                                context_path=str(context), dockerfile_path="Dockerfile", platform=None)
    return A2AAgent.put(id=f"replaced-agent-{uid}", docker_image_artifact=image)
