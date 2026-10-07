"""A server env on Modal: its one MCP server in a Modal container, with no gateway.

A load is timed by the size of the payload it staged, measured in the server's container through the deploy's own
handle and a restored one; and the server's state reads as JSON whether ``data/get`` answers with data or with a file
bundle. Needs the resolved config to reach Modal and an image store Modal can pull from; skipped otherwise.
"""

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.artifact import DockerImageArtifact, EnvironmentArtifact, FileArtifact
from agent_env.env import legacy_protocol
from agent_env.env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.gateway import constants
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
