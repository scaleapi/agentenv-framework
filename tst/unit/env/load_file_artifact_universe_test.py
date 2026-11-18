"""An env's default file-universe load: onto its VM host when its sandbox is a VM, into its container when its sandbox is
the server's own container, as a lone server's is on Modal, which has no VM host to stage on."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox
from agent_env.providers.sandbox_providers.sandbox import VmSandbox

FILES = {"a.txt": "s3://bucket/a.txt", "sub/b.txt": "s3://bucket/b.txt"}


def _universe():
    files = {name: SimpleNamespace(object_url=url) for name, url in FILES.items()}
    return SimpleNamespace(id="docs", version=3, get_file_artifacts=lambda: files)


def _env(sandbox) -> MCPServerEnv:
    env = MCPServerEnv(id="mcp-docs", version=1, docker_image_artifact=MagicMock(), environment_name="docs",
                       env_provider_type="server")
    env._sandbox = sandbox
    return env


@pytest.mark.asyncio
async def test_a_lone_servers_container_on_modal_is_written_directly():
    sandbox = AsyncMock(spec=ModalSandbox)
    sandbox.mode = "container"

    result = await _env(sandbox).load_file_artifact_universe(_universe(), "/app/loaded/")

    assert sorted(c.args for c in sandbox.exec.await_args_list) == [
        ("mkdir", "-p", "/app/loaded"), ("mkdir", "-p", "/app/loaded/sub"),
    ]
    assert sorted(c.args for c in sandbox.write_file_from_object.await_args_list) == [
        ("s3://bucket/a.txt", "/app/loaded/a.txt"), ("s3://bucket/b.txt", "/app/loaded/sub/b.txt"),
    ]
    assert result.destination_path == "/app/loaded"
    assert result.files == {"a.txt": "/app/loaded/a.txt", "sub/b.txt": "/app/loaded/sub/b.txt"}


@pytest.mark.asyncio
async def test_a_lone_servers_container_on_a_vm_is_written_through_its_container():
    sandbox = AsyncMock(spec=VmSandbox)
    sandbox.mode = "container"
    sandbox.container_name = "env-local-1"

    result = await _env(sandbox).load_file_artifact_universe(_universe())

    assert sorted(c.args[0] for c in sandbox.exec_script.await_args_list) == [
        "docker exec -u 0 env-local-1 mkdir -p /tmp/file_artifacts",
        "docker exec -u 0 env-local-1 mkdir -p /tmp/file_artifacts/sub",
    ]
    assert sorted(c.args for c in sandbox.write_file_from_object.await_args_list) == [
        ("s3://bucket/a.txt", "/tmp/file_artifacts/a.txt"), ("s3://bucket/b.txt", "/tmp/file_artifacts/sub/b.txt"),
    ]
    sandbox.load_object_file.assert_not_awaited()
    assert result.files == {"a.txt": "/tmp/file_artifacts/a.txt", "sub/b.txt": "/tmp/file_artifacts/sub/b.txt"}


@pytest.mark.asyncio
async def test_a_vm_is_loaded_on_its_host():
    sandbox = AsyncMock(spec=VmSandbox)
    sandbox.mode = "vm"

    result = await _env(sandbox).load_file_artifact_universe(_universe(), "/app/loaded")

    sandbox.exec_script.assert_awaited_once_with("mkdir -p /app/loaded && mkdir -p /app/loaded/sub")
    assert [c.args for c in sandbox.load_object_file.await_args_list] == [
        ("s3://bucket/a.txt", "/app/loaded/a.txt"), ("s3://bucket/b.txt", "/app/loaded/sub/b.txt"),
    ]
    sandbox.write_file_from_object.assert_not_awaited()
    sandbox.exec.assert_not_awaited()
    assert result.files == {"a.txt": "/app/loaded/a.txt", "sub/b.txt": "/app/loaded/sub/b.txt"}
