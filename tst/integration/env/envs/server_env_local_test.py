"""A server env on the local backend: its one MCP server in a LocalSandbox container, with no gateway and no Modal.

The env deploys, a later process restores its record and loads data into the server, and the teardown step, which
rebuilds each sandbox from disk, removes the container. A load is timed by the size of the payload it staged, through
the deploy's own handle and a restored one; the server's state reads as JSON whether ``data/get`` answers with data or
with a file bundle; and handed an object to upload to, the server sends its bundle from its container in parts over the
local grant server. Requires a docker daemon; spins up a throwaway ``registry:2`` and skips if it can't start.
"""

import io
import json
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest
from agentenv_protocol import DataPart, uploaded_object
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.transfers import PARTS_IN_FLIGHT, HttpPartsPutGrant, WriteObject, part_ranges

from agent_env.artifact import DockerImageArtifact, EnvironmentArtifact, FileArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config, set_image_store
from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedSandboxEnv, Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.gateway import constants
from agent_env.store import get_config
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.teardown_sandboxes import TORN_DOWN_KEY, TeardownSandboxesTaskStep

pytestmark = pytest.mark.integration

_REPO = Path(__file__).resolve().parents[4]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _wait_ready(url: str, codes: tuple[int, ...], timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code in codes:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture
def local_stack(monkeypatch, tmp_path):
    """Local stores and a throwaway registry; yields the container names the test registers, removed afterwards."""
    if _docker("info").returncode != 0:
        pytest.skip("docker daemon not available")
    reg_port = _free_port()
    reg_name = f"server-e2e-registry-{uuid.uuid4().hex[:8]}"
    if _docker("run", "-d", "--rm", "--name", reg_name, "-p", f"127.0.0.1:{reg_port}:5000", "registry:2").returncode != 0:
        pytest.skip("could not start registry:2")
    host = f"localhost:{reg_port}"
    if not _wait_ready(f"http://{host}/v2/", (200, 401)):
        _docker("rm", "-f", reg_name)
        pytest.skip("local registry did not become ready")

    owned: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    for var in ("DOCUMENT", "OBJECT", "IMAGE"):
        monkeypatch.setenv(f"AGENT_ENV_{var}_STORE", "local")
    configure()
    set_image_store(LocalRegistryImageStore(host))
    reset_artifact_store()
    try:
        yield owned
    finally:
        for container_name in owned:
            _docker("rm", "-f", container_name)
        reset_artifact_store()
        reset_config()
        _docker("rm", "-f", reg_name)


@pytest.mark.asyncio
async def test_a_server_env_deploys_restores_loads_and_tears_down_on_local_containers(local_stack):
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-{uid}", docker_image_artifact=_put_items_image(f"server-items-{uid}"),
                           environment_name="items", env_provider_type="server")

    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    container = f"agent-{deployed.sandbox_id}"
    local_stack.append(container)
    assert type(deployed) is DeployedSandboxEnv and deployed.sandbox_type == "local"
    assert deployed.environment_card["name"] == "items" and deployed.mcp_url.startswith("http://127.0.0.1:")

    restored = await Env.from_instance_id(deployed.instance_id)
    await restored.load_environment_artifact(_items_artifact(uid))
    base_url = await legacy_protocol.v1_base_url(deployed, None, "items")
    assert (await protocol_v1.get_data(base_url)).model_dump()["parts"][0]["data"] == {"items": ["snap-x", "snap-y"]}

    step = TeardownSandboxesTaskStep(id="teardown", version=None, env_ids=[env.id])
    context = await step.execute(TaskStepContext(deployed_envs=[deployed]))
    assert context.metadata[TORN_DOWN_KEY] == [deployed.sandbox_id]
    assert _docker("ps", "-aq", "--filter", f"name=^/{container}$").stdout.strip() == ""


@pytest.mark.asyncio
async def test_a_load_is_timed_by_its_payload_and_the_servers_state_reads_as_json_from_a_file_export(local_stack, monkeypatch):
    staged_sizes, timeout_for = [], constants.data_plane_load_timeout_s
    monkeypatch.setattr(constants, "data_plane_load_timeout_s", lambda size: staged_sizes.append(size) or timeout_for(size))
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-{uid}", docker_image_artifact=_put_items_image(f"server-items-{uid}"),
                           environment_name="items", env_provider_type="server")
    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    local_stack.append(f"agent-{deployed.sandbox_id}")
    artifact = _items_artifact(uid)

    await env.load_environment_artifact(artifact)  # the deploy's container-mode handle
    await (await Env.from_instance_id(deployed.instance_id)).load_environment_artifact(artifact)  # a restored one

    assert staged_sizes == [len(artifact.get_file_artifact().load())] * 2
    items = {"items": ["snap-x", "snap-y"]}
    assert await legacy_protocol.service_state(deployed, None, "items") == items
    base_url = await legacy_protocol.v1_base_url(deployed, None, "items")
    card = await protocol_v1.get_card(base_url)
    await protocol_v1.invoke_extension(base_url, card, "urn:agentenv:export-as-file/v1", {"enabled": True})
    assert [part.kind for part in (await protocol_v1.get_data(base_url)).parts] == ["file"]
    assert await legacy_protocol.service_state(deployed, None, "items") == items


@pytest.mark.asyncio
async def test_a_server_handed_an_object_uploads_its_bundle_in_parts_from_its_container(local_stack):
    uid = uuid.uuid4().hex[:8]
    env = MCPServerEnv.put(id=f"server-items-{uid}", docker_image_artifact=_put_items_image(f"server-items-{uid}"),
                           environment_name="items", env_provider_type="server")
    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    local_stack.append(f"agent-{deployed.sandbox_id}")
    base_url = await legacy_protocol.v1_base_url(deployed, None, "items")
    items = [f"item-{n:03d}-{uid}" for n in range(60)]
    await protocol_v1.add_data(base_url, [DataPart(data={"items": items})])
    assert "http-put-parts" in protocol_v1.write_kinds(await protocol_v1.get_card(base_url))

    # Part URLs on the local grant server, more than the bundle needs: the server uses the first ones its size takes.
    store, part_bytes, prefix = get_config().get_object_store(), 100, f"parts-e2e/{uid}"
    puts = [store.issue_write_grant(store.object_url(f"{prefix}/{n}"), media_type="application/zip",
                                    max_bytes=part_bytes, expires_in=900) for n in range(1, 41)]
    write_object = WriteObject(
        media_type="application/zip",
        max_bytes=part_bytes * len(puts),
        write=HttpPartsPutGrant(kind="http-put-parts", part_bytes=part_bytes, urls=[put.url for put in puts],
                                expires_at=min(put.expires_at for put in puts), headers={"Content-Type": "application/zip"}),
    )

    uploaded = uploaded_object((await protocol_v1.get_data(base_url, write_object=write_object)).parts[0])

    ranges = part_ranges(uploaded.size_bytes, part_bytes)
    assert PARTS_IN_FLIGHT < len(ranges) < len(puts) and uploaded.sha256 is None
    parts = [store.get(store.object_url(f"{prefix}/{n}")) for n in range(1, len(ranges) + 1)]
    assert [len(part) for part in parts] == [length for _, length in ranges]
    assert not store.exists(f"{prefix}/{len(ranges) + 1}")
    with zipfile.ZipFile(io.BytesIO(b"".join(parts))) as bundle:
        assert json.loads(bundle.read("items.json")) == {"items": items}


def _put_items_image(artifact_id: str) -> DockerImageArtifact:
    """The in-memory items server (card 'items'), built with the vendored agentenv_protocol and pushed to the registry."""
    data = _REPO / "tst" / "data" / "agentenv_mcp"
    with tempfile.TemporaryDirectory() as build_dir:
        bd = Path(build_dir)
        shutil.copytree(_REPO / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol", bd / "agentenv_protocol",
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("server.py", "Dockerfile", "seed.json"):
            shutil.copy(data / name, bd / name)
        build = _docker("build", "-t", artifact_id, str(bd))
        assert build.returncode == 0, f"items build failed: {build.stderr[-2000:]}"
    return DockerImageArtifact.put(id=artifact_id, description="server env e2e", image_name=artifact_id)


def _items_artifact(uid: str) -> EnvironmentArtifact:
    return EnvironmentArtifact.put(
        id=f"server-items-data-{uid}",
        environment_name="items",
        file_artifact=FileArtifact.put_bytes(id=f"server-items-file-{uid}", description="server env e2e", filename="items.json",
                                             content=json.dumps({"items": ["snap-x", "snap-y"]}).encode(), content_type="application/json"),
    )
