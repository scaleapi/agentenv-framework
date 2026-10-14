"""A server env on Modal: its one MCP server in a Modal container, with no gateway.

A load is timed by the size of the payload it staged, measured in the server's container through the deploy's own
handle and a restored one; and the server's state reads as JSON whether ``data/get`` answers with data or with a file
bundle. Needs the resolved config to reach Modal and an image store Modal can pull from; skipped otherwise.
"""

import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path

import modal
import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.artifact import DockerImageArtifact, EnvironmentArtifact, FileArtifact
from agent_env.env import legacy_protocol
from agent_env.env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.gateway import constants
from agent_env.task import EpisodeSpec, Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.teardown_sandboxes import TeardownSandboxesTaskStep
from tst.util.capabilities import skip_without_remote_sandbox
from tst.util.image_cache import build_or_reuse

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_remote_sandbox("modal")]

_REPO = Path(__file__).resolve().parents[4]


@pytest.mark.asyncio
async def test_a_modal_load_is_timed_by_its_payload_and_the_servers_state_reads_as_json_from_a_file_export(monkeypatch):
    staged_sizes, timeout_for = [], constants.data_plane_load_timeout_s
    monkeypatch.setattr(constants, "data_plane_load_timeout_s", lambda size: staged_sizes.append(size) or timeout_for(size))
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-modal-{uid}", docker_image_artifact=_items_image(),
                           environment_name="items", env_provider_type="server")
    deployed = await env.deploy(sandbox_type="modal", ttl_seconds=900)
    try:
        artifact = _items_artifact(uid)
        await env.load_environment_artifact(artifact)
        await (await Env.from_instance_id(deployed.instance_id)).load_environment_artifact(artifact)

        assert staged_sizes == [len(artifact.get_file_artifact().load())] * 2
        items = {"items": ["snap-x", "snap-y"]}
        assert await legacy_protocol.service_state(deployed, None, "items") == items
        base_url = await legacy_protocol.v1_base_url(deployed, None, "items")
        card = await protocol_v1.get_card(base_url)
        await protocol_v1.invoke_extension(base_url, card, "urn:agentenv:export-as-file/v1", {"enabled": True})
        assert [part.kind for part in (await protocol_v1.get_data(base_url)).parts] == ["file"]
        assert await legacy_protocol.service_state(deployed, None, "items") == items
    finally:
        await TeardownSandboxesTaskStep(id="teardown", version=None, env_ids=[env.id]).execute(
            TaskStepContext(deployed_envs=[deployed])
        )


VERIFY_CRATE = """
from agentenv_protocol import client


async def verify(mcp_url):
    items = (await client.get_data(mcp_url.removesuffix("/mcp"))).parts[0].data["items"]
    return [{"id": "crate", "result": "crate" in items}]
"""


@pytest.mark.asyncio
async def test_an_episodes_env_on_modal_is_played_from_this_machine_and_no_ending_leaves_it_running():
    """A loop on this machine plays an agent whose env the task deploys on Modal: the env's tools reached over its
    tunnel as the agent's role, the score read from the env's state there, and every episode, played, abandoned or
    forgotten past its turn's deadline, leaving its sandbox stopped."""
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-episode-{uid}", docker_image_artifact=_items_context(uid),
                           environment_name="items", env_provider_type="server")
    verify = FileArtifact.put_bytes(id=f"verify-crate-modal-{uid}", description="episode on Modal",
                                    filename="verify.py", content=VERIFY_CRATE.encode(), content_type="text/x-python")
    task = Task.from_dict({"id": f"crate-modal-{uid}", "steps": [
        {"id": "env", "type": "deploy_env", "env_id": env.id, "sandbox_type": "modal", "ttl_seconds": 900},
        {"id": "load", "type": "load_artifact", "env_id": env.id, "artifact_id": _items_artifact(uid).id},
        {"id": "policy", "type": "deploy_agent", "agent_name": "policy", "env_ids": [env.id],
         "role": "alice@example.com"},
        {"id": "play", "type": "prompt_agent", "agent_name": "policy", "prompt_id": "play", "prompt": "Add a crate."},
        {"id": "check", "type": "env_outcome_verifier", "env_id": env.id, "file_artifact_id": verify.id,
         "verifier_id": "crate"},
    ]})
    spec = EpisodeSpec.from_task(task)
    started = []

    async def start(spec):
        started.append(episode := await spec.start())
        return episode

    try:
        played = await start(spec)
        turn = await played.next_turn()
        assert turn.mcp_urls[env.id].startswith("https://")
        async with turn.tools() as tools:
            await tools.call("items_add_item", {"item": "crate"})
            assert json.loads((await tools.call("items_whoami")).text)["role"] == "alice@example.com"
        await turn.end("Added a crate.")
        assert await played.result() == {"check": 1.0}

        abandoned = await start(spec)
        await abandoned.next_turn()
        await abandoned.abandon("test")
        assert Task.get_instance(abandoned.id).status == "cancelled"

        forgotten = await start(EpisodeSpec.from_task(task, turn_timeout=3))
        await forgotten.next_turn()
        with pytest.raises(TimeoutError, match="wasn't ended within 3s"):
            await forgotten.next_turn()

        for episode in (played, abandoned, forgotten):
            [sandbox_id] = [entry["sandbox_id"] for entry in Task.get_instance(episode.id).context["deployed_envs"]]
            await _stopped(sandbox_id)
    finally:
        for episode in started:
            await episode.abandon("the test ended")

async def _stopped(sandbox_id: str) -> None:
    sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
    async with asyncio.timeout(60):
        while await sandbox.poll.aio() is None:
            await asyncio.sleep(2)


def _items_context(uid: str) -> DockerImageArtifact:
    """The items server as a build context, which Modal builds where it runs, so no registry Modal can pull from is
    needed."""
    data = _REPO / "tst" / "data" / "agentenv_mcp"
    with tempfile.TemporaryDirectory() as build_dir:
        context = Path(build_dir)
        shutil.copytree(_REPO / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol", context / "agentenv_protocol",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("server.py", "Dockerfile", "seed.json"):
            shutil.copy(data / name, context / name)
        return DockerImageArtifact.put_context(f"server-items-episode-image-{uid}", description="episode on Modal",
                                               context_path=str(context), dockerfile_path="Dockerfile")


def _items_image() -> DockerImageArtifact:
    """The in-memory items server (card 'items'), built for linux/amd64 with the vendored agentenv_protocol."""
    data = _REPO / "tst" / "data" / "agentenv_mcp"
    with tempfile.TemporaryDirectory() as build_dir:
        context = Path(build_dir)
        shutil.copytree(_REPO / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol", context / "agentenv_protocol",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("server.py", "Dockerfile", "seed.json"):
            shutil.copy(data / name, context / name)
        return build_or_reuse(artifact_id="server-items-modal", description="server env on Modal",
                              dockerfile=context / "Dockerfile", context=context, tag="server-items-modal")


def _items_artifact(uid: str) -> EnvironmentArtifact:
    return EnvironmentArtifact.put(
        id=f"server-items-modal-data-{uid}",
        environment_name="items",
        file_artifact=FileArtifact.put_bytes(id=f"server-items-modal-file-{uid}", description="server env on Modal",
                                             filename="items.json", content=json.dumps({"items": ["snap-x", "snap-y"]}).encode(),
                                             content_type="application/json"),
    )
