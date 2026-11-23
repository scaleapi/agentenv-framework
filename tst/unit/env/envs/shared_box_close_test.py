"""No way an env, its provider or a run's teardown closes reaches Modal's terminate for a shared box: each is refused,
logged, and, in a teardown, reported as not terminated."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import modal
import pytest

from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.envs._deployment import close_deployed, close_replaced
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.sandbox_providers.modal_sandbox import SHARED_BOX_TAG, ModalSandbox, ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox, ModalVmSandboxProvider
from agent_env.task import teardown
from agent_env.task.teardown import RecordedSandbox
from agent_env.task_step.context import TaskStepContext


@pytest.fixture
def box(monkeypatch) -> MagicMock:
    """The Modal sandbox of shared box ``box-1``, which ``from_id`` finds, and Modal clients that never connect."""
    sb = MagicMock(object_id="sb-box")
    sb.get_tags.aio = AsyncMock(return_value={SHARED_BOX_TAG: "box-1"})
    sb.terminate.aio = AsyncMock()
    sb.tunnels.aio = AsyncMock(return_value={})
    sb.exec.aio = AsyncMock(side_effect=RuntimeError("no exec here"))
    monkeypatch.setattr(modal.Sandbox, "from_id", MagicMock(aio=AsyncMock(return_value=sb)))
    for cls in (ModalVmSandboxProvider, ModalSandboxProvider):
        monkeypatch.setattr(cls, "_get_client", AsyncMock(return_value=MagicMock()))
    return sb


HANDLES = {
    "created vm": lambda sb: ModalVmSandbox(sb, {}, shared_box_id="box-1"),
    "reattached vm": lambda sb: ModalVmSandbox(sb, {}),
    "reattached container": lambda sb: ModalSandbox(sb, {}),
}


def _mcp() -> MCPServerEnv:
    return MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=None, environment_name="email")


def _website() -> WebsiteEnv:
    return WebsiteEnv(id="web-email", version=1, backend_docker_image_artifact=None,
                      frontend_docker_image_artifact=None, environment_name="email")


def _multi() -> MultiEnv:
    return MultiEnv(id="crm-suite", version=1, mcp_server_envs=[_mcp()])


def _computer() -> ComputerEnv:
    return ComputerEnv("desk", 1, "desk", MagicMock(), MagicMock())


async def _env_close(make_env, handle):
    env = make_env()
    env._env_provider, env._sandbox = None, handle
    await env.close()


async def _close_replaced(handle):
    env = _mcp()
    env._replaced = [(None, handle)]
    await close_replaced(env)


async def _env_provider_close(handle):
    provider = EnvironmentGatewayProvider()
    provider._container_sandboxes = [handle]
    await provider.close()


HANDLE_PATHS = {
    "MCPServerEnv.close": lambda handle: _env_close(_mcp, handle),
    "WebsiteEnv.close": lambda handle: _env_close(_website, handle),
    "MultiEnv.close": lambda handle: _env_close(_multi, handle),
    "ComputerEnv.close": lambda handle: _env_close(_computer, handle),
    "close_replaced": _close_replaced,
    "EnvProvider.close": _env_provider_close,
}


@pytest.mark.parametrize("handle", HANDLES.values(), ids=HANDLES.keys())
@pytest.mark.parametrize("close", HANDLE_PATHS.values(), ids=HANDLE_PATHS.keys())
@pytest.mark.asyncio
async def test_closing_a_handle_to_a_shared_box_leaves_it_running(box, close, handle):
    await close(handle(box))

    box.terminate.aio.assert_not_called()


@pytest.mark.parametrize("sandbox_type", ["modal_vm", "modal"])
@pytest.mark.asyncio
async def test_close_deployed_of_a_plugins_deployment_on_a_shared_box_leaves_it_running(box, monkeypatch, sandbox_type):
    env = _mcp()
    env._env_provider = object()
    monkeypatch.setattr(MCPServerEnv, "from_deployed_env", AsyncMock(return_value=env))
    deployed = DeployedGatewayEnv(env_id="mcp-email", env_version=1, gateway_url="http://g", sandbox_id="sb-box",
                                  sandbox_type=sandbox_type)

    await close_deployed(deployed, MCPServerEnv)

    box.terminate.aio.assert_not_called()


@pytest.mark.parametrize("sandbox_type, default", [
    ("modal_vm", None), ("modal", None), (None, ModalVmSandboxProvider), (None, ModalSandboxProvider)],
    ids=["by id on modal_vm", "by id on modal", "reattached on modal_vm", "reattached on modal"])
@pytest.mark.asyncio
async def test_a_teardown_reports_a_shared_box_not_terminated_and_leaves_it_running(box, monkeypatch, sandbox_type,
                                                                                    default):
    if default is not None:
        monkeypatch.setitem(teardown._DEFAULT_PROVIDERS, "env", default)
    context = TaskStepContext()
    sandbox = RecordedSandbox("sb-box", sandbox_type, "env")

    report = await teardown.tear_down(context, [sandbox], [], timeout=5)

    box.terminate.aio.assert_not_called()
    assert report.terminated == ()
    assert [(failed, "shared box 'box-1'" in why) for failed, why in report.failed] == [(sandbox, True)]
    assert teardown.TORN_DOWN_KEY not in context.metadata
