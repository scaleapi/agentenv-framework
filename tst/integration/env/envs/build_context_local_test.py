"""Images a VM sandbox builds from their build contexts, on the local backend's, which runs this machine's Docker: one
that is only a build context, and one whose tar.gz is built for another platform than the VM's, which the VM builds from
the context beside it instead of loading. Each deploys an MCP server env through the gateway and reads its server's
card. Requires a docker daemon; spins up a throwaway ``registry:2`` and skips if it can't start.
"""

import logging
import shutil
import uuid
from pathlib import Path

import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.artifact import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env import legacy_protocol
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.providers.sandbox_providers import sandbox
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox
from agent_env.utils.docker_build import normalize_platform
from tst.integration.env.envs.server_env_local_test import local_stack as local_stack
from tst.integration.store.local_parallel_deploy_test import _docker, _seed_bootstrap_envs

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

_REPO = Path(__file__).resolve().parents[4]
_OTHER_PLATFORM = {"linux/amd64": "linux/arm64", "linux/arm64": "linux/amd64"}


@pytest.fixture
def vm_platform(monkeypatch, caplog):
    """The local sandbox records no platform, its Docker being this machine's; given this machine's, as a remote VM
    records its own, it rebuilds an image recorded for another one as a remote VM does."""
    platform = normalize_platform(f"linux/{_docker('info', '--format', '{{.Architecture}}').stdout.strip()}")
    if platform not in _OTHER_PLATFORM:
        pytest.skip(f"no other platform to record for a {platform} host")
    monkeypatch.setattr(LocalSandbox, "platform", platform)
    caplog.set_level(logging.INFO, logger=sandbox.__name__)
    return platform


@pytest.mark.asyncio
async def test_an_image_that_is_only_a_build_context_is_built_on_the_vm(local_stack, vm_platform, tmp_path, caplog):
    _seed_bootstrap_envs()
    uid = uuid.uuid4().hex[:8]
    image = DockerImageArtifact.put_context(f"ctx-items-{uid}", description="build context e2e",
                                            context_path=str(_items_context(tmp_path / "items")),
                                            dockerfile_path="Dockerfile", platform=vm_platform)

    await _deploys_and_answers(image, uid)

    assert f"Building {image.image_name} from its build context, for {vm_platform}..." in caplog.messages


@pytest.mark.asyncio
async def test_an_image_whose_tar_gz_is_for_another_platform_is_built_from_the_context_beside_it(
        local_stack, vm_platform, tmp_path, caplog):
    _seed_bootstrap_envs()
    uid = uuid.uuid4().hex[:8]
    context = _items_context(tmp_path / "items")
    tag = f"ctx-items-{uid}"
    build = _docker("build", "-t", tag, str(context))
    assert build.returncode == 0, f"items build failed: {build.stderr[-2000:]}"
    stored = DockerImageArtifact.put(id=tag, description="build context e2e", image_name=tag,
                                     build_context_path=str(context), dockerfile_path="Dockerfile")
    # As a machine of the other platform would have written it, with a tar.gz a VM that loaded it would fail on.
    image = get_artifact_store().put_document(stored.model_copy(update={
        "id": f"{tag}-other", "version": 0, "platform": _OTHER_PLATFORM[vm_platform],
        "tar_gz_object_url": "file:///nonexistent/items.tar.gz"}))
    assert image.rebuilt_for(vm_platform)

    await _deploys_and_answers(image, uid)

    assert f"Building {image.image_name} from its build context, for {vm_platform}..." in caplog.messages


async def _deploys_and_answers(image: DockerImageArtifact, uid: str) -> None:
    """Deploy an items server env of ``image`` through the gateway on the local sandbox, read its card, and close it."""
    env = MCPServerEnv.put(id=f"ctx-items-env-{uid}", docker_image_artifact=image, environment_name="items")
    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    work_dir = env._sandbox.work_dir
    try:
        base_url = await legacy_protocol.v1_base_url(deployed, deployed.gateway_url, "items")
        assert (await protocol_v1.get_card(base_url))["name"] == "items"
        await env.close()
    finally:
        _docker("compose", "--project-directory", str(work_dir), "down", "-v", "--remove-orphans")


def _items_context(root: Path) -> Path:
    """The in-memory items server's build context (card 'items'), with the vendored agentenv_protocol."""
    data = _REPO / "tst" / "data" / "agentenv_mcp"
    shutil.copytree(_REPO / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol", root / "agentenv_protocol",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("server.py", "Dockerfile", "seed.json"):
        shutil.copy(data / name, root / name)
    return root
