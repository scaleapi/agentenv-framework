"""An env reads its stack's layout from the built-in provider that deployed it, and a slot's artifact loads stage inside that slot."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.env.envs._deployment import stack_layout_of
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_providers.env_provider import EnvironmentProvider
from agent_env.providers.env_providers.stack_layout import DEFAULT_STACK_LAYOUT, StackLayout


class _PluginProvider(EnvironmentProvider):
    type = "plugin_layout"

    async def deploy(self, env, sandbox_provider, **options):
        raise NotImplementedError

    async def close(self):
        pass


def _slot_provider(index: int) -> EnvironmentGatewayProvider:
    provider = EnvironmentGatewayProvider()
    provider._stack_layout = StackLayout.for_slot(index)
    return provider


def _vm_env(provider) -> SimpleNamespace:
    sandbox = MagicMock(mode="vm", load_object_file=AsyncMock(), docker_cp=AsyncMock(), exec_script=AsyncMock(return_value=""))
    provider._get_container_id = AsyncMock(return_value="cid")
    return SimpleNamespace(_sandbox=sandbox, _env_provider=provider, environment_name="crm")


def _scripts(env: SimpleNamespace) -> list[str]:
    return [c.args[0] for c in env._sandbox.exec_script.await_args_list]


def test_a_builtin_provider_gives_the_layout_it_deployed():
    assert stack_layout_of(_slot_provider(2)) == StackLayout.for_slot(2)
    assert stack_layout_of(EnvironmentGatewayProvider()) == DEFAULT_STACK_LAYOUT


@pytest.mark.parametrize("provider", [None, _PluginProvider()])
def test_a_plugin_provider_or_none_gives_the_default_layout(provider):
    assert stack_layout_of(provider) == DEFAULT_STACK_LAYOUT


@pytest.mark.asyncio
async def test_a_slot_mcp_server_stages_its_artifact_in_the_slot_dir():
    env = _vm_env(_slot_provider(1))
    await MCPServerEnv._copy_artifact_into_container(env, SimpleNamespace(object_url="s3://b/mail.json"), "mail.json")

    staged = env._sandbox.load_object_file.await_args.args[1]
    assert staged.startswith("/app/s1/_artifact_staging/crm-")
    assert "mkdir -p /app/s1/_artifact_staging" in _scripts(env)


@pytest.mark.asyncio
async def test_a_slot_website_copies_and_removes_its_artifact_through_the_slot_compose_project():
    env = _vm_env(_slot_provider(1))
    await WebsiteEnv._copy_artifact_into_container(env, SimpleNamespace(object_url="s3://b/shop.json"), "shop.json")
    await WebsiteEnv._remove_from_container(env, "shop.json")

    compose_scripts = [s for s in _scripts(env) if s.startswith("docker compose")]
    assert compose_scripts and all(s.startswith("docker compose -f /app/s1/docker-compose.yml -p s1 ") for s in compose_scripts)
