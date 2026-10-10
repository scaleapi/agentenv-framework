"""snapshot_env on the local backend, end to end, through a parts grant.

A MultiEnv's items server takes ``write_object``, so the snapshot hands it an object of its own on the local store; the
server uploads its bundle from its container, the snapshot registers that object as the server's export, and loading the
snapshot back restores the state it held. Requires a docker daemon; spins up a throwaway ``registry:2`` and skips if it
can't start.
"""

import io
import json
import logging
import uuid
import zipfile

import pytest
from agentenv_protocol import DataPart
from agentenv_protocol import client as protocol_v1

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.env import legacy_protocol
from agent_env.env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import snapshot_env as snapshot_mod
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _docker, _seed_bootstrap_envs, local_stack  # noqa: F401  (fixture)

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]


@pytest.mark.asyncio
async def test_a_snapshot_of_a_server_that_takes_parts_registers_its_upload_and_loads_back(local_stack, caplog):
    _seed_bootstrap_envs()
    uid = uuid.uuid4().hex[:8]
    items = MCPServerEnv.put(id=f"snap-items-{uid}", environment_name="items",
                             docker_image_artifact=_put_items_image(f"snap-items-{uid}"))
    env = MultiEnv.put(id=f"snap-{uid}", mcp_server_envs=[items], name="suite")

    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    work_dir = env._sandbox.work_dir
    try:
        base_url = await legacy_protocol.v1_base_url(deployed, deployed.gateway_url, "items")
        await protocol_v1.add_data(base_url, [DataPart(data={"items": ["snap-x", "snap-y"]})])

        step = SnapshotEnvTaskStep(id="snap", version=None, snapshot_id=f"snap-universe-{uid}")
        with caplog.at_level(logging.INFO, logger=snapshot_mod.logger.name):
            context = await step.execute(TaskStepContext(deployed_envs=[deployed]))

        assert "items exported through parts" in caplog.text
        universe = EnvironmentUniverseArtifact.get(context.metadata["env_snapshotted_universes"]["snap"]["id"])
        (bundle,) = universe.get_file_artifacts().values()
        assert "/agentenv-snapshots/" in bundle.object_url and bundle.object_url.endswith("/items.zip")
        assert (bundle.filename, bundle.content_type) == ("items.zip", "application/zip")
        with zipfile.ZipFile(io.BytesIO(bundle.load())) as zf:
            assert json.loads(zf.read("items.json")) == {"items": ["snap-x", "snap-y"]}

        await protocol_v1.add_data(base_url, [DataPart(data={"items": ["after-the-snapshot"]})])
        await (await Env.from_instance_id(deployed.instance_id)).load_environment_universe_artifact(universe)
        assert (await protocol_v1.get_data(base_url)).model_dump()["parts"][0]["data"] == {"items": ["snap-x", "snap-y"]}
    finally:
        _docker("compose", "--project-directory", str(work_dir), "down", "-v", "--remove-orphans")
