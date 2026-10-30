"""The snapshot reload edits the compose file the deploy wrote: it starts the servicedb from the snapshot and leaves
every other service as deployed, whatever config names now."""

from __future__ import annotations

import shlex
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from agent_env.artifact import DockerImageArtifact
from agent_env.env.envs.multi_env import MultiEnv, snapshot_compose
from agent_env.env.envs.service_db import ServiceDBConfig
from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
from agent_env.providers import EnvironmentGatewayProvider, MCPServerConfig, WebsiteConfig
from agent_env.providers.env_providers.constants import DOCKER_COMPOSE_PATH
from agent_env.providers.env_state import LocalPostgresStateProvider


class _Stop(Exception):
    """Abort once the gateway is waited on; the rest needs a real sandbox."""


def _deployed_compose(browser="website-browser"):
    """The compose file a deploy with a website writes, its browser named ``browser``."""
    return EnvironmentGatewayProvider().create_docker_compose(
        mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack"),
                     MCPServerConfig(image="browser:v1", environment_name=browser, card_port=18766)],
        website_configs=[WebsiteConfig(backend_image="shop-be", frontend_image="shop-fe", environment_name="shop")],
        gateway_image="gateway:v38", gateway_port=AGENT_ENV_GATEWAY_MCP_PORT,
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig(
            db_web_image="db-web:v1", db_mcp_image="db-mcp:v1")),
        state_instance=LocalPostgresStateProvider.default_instance(), mcp_server_name="crm")


def _env(compose):
    env = MultiEnv(id="crm-suite", version=1, mcp_server_envs=[], name="crm")
    env.mcp_server_envs = [MagicMock(environment_name="slack")]
    env.website_envs = [MagicMock(environment_name="shop")]
    env._sandbox = MagicMock(mode="vm", load_docker_images=AsyncMock(), host_ips=("127.0.0.1",),
                             host_port=lambda port: port + 1000,
                             exec_script=AsyncMock(side_effect=lambda script, **k: compose + "\n" if script.startswith("cat ") else ""),
                             exec_with_output=AsyncMock(return_value=(0, "", "")))
    return env


async def _reload(env):
    db_image = MagicMock(spec=DockerImageArtifact, image_name="snap:1")
    snapshot = MagicMock(db_image_artifact_id="snap", db_image_artifact_version=1)
    with patch("agent_env.artifact.Artifact.get", return_value=db_image), \
         patch("agent_env.env.env.Env.get", return_value=MagicMock()) as get, \
         patch("agent_env.config.get_config", return_value=MagicMock(default_service_db_env_id="db-id")), \
         patch("agent_env.providers.env_state.LocalPostgresStateProvider"), \
         patch.object(EnvironmentGatewayProvider, "_compose_up", AsyncMock()) as up, \
         patch.object(EnvironmentGatewayProvider, "_wait_for_gateway", side_effect=_Stop) as wait:
        with pytest.raises(_Stop):
            await env._load_from_snapshot(snapshot)
    return get, up, wait


def _written(env):
    (written,) = [argv[2] for argv in map(shlex.split, (call.args[0] for call in env._sandbox.exec_script.await_args_list))
                  if argv[:1] == ["printf"]]
    return written


@pytest.mark.asyncio
@pytest.mark.parametrize("browser", ["website-browser", "browseruse-mcp"])
async def test_the_reload_starts_the_servicedb_from_the_snapshot_and_keeps_every_other_service_as_deployed(browser):
    deployed = _deployed_compose(browser)
    env = _env(deployed)

    get, _, _ = await _reload(env)

    assert _written(env) == snapshot_compose(deployed, "snap:1")  # byte for byte, with no blank line added
    before, after = yaml.safe_load(deployed)["services"], yaml.safe_load(_written(env))["services"]
    assert after["servicedb"]["image"] == "snap:1" and "volumes" not in after["servicedb"]
    assert {name: service for name, service in after.items() if name != "servicedb"} == {
        name: service for name, service in before.items() if name != "servicedb"}
    assert [call.args[0] for call in get.call_args_list] == ["db-id"]  # the store it connects to; no image is read


@pytest.mark.asyncio
async def test_the_reload_recreates_every_service_but_the_servicedb_the_browser_included():
    env = _env(_deployed_compose("browseruse-mcp"))

    _, up, wait = await _reload(env)

    (args,) = [call.args[1] for call in up.await_args_list]
    assert set(shlex.split(args.removeprefix("--force-recreate "))) == {
        "slack", "browseruse-mcp", "shop-website-backend", "shop-website-frontend", "gateway", "pgweb", "db-mcp"}
    assert wait.call_args.args[1] == AGENT_ENV_GATEWAY_MCP_PORT + 1000


@pytest.mark.asyncio
async def test_the_reload_writes_its_compose_as_one_argument():
    env = _env(_deployed_compose().replace("  slack:\n", "  slack:\n    labels:\n      note: \"$(touch pwned) 'q' `x`\"\n", 1))

    await _reload(env)

    assert "$(touch pwned) 'q' `x`" in _written(env)
    scripts = [call.args[0] for call in env._sandbox.exec_script.await_args_list]
    assert any(argv[:2] == ["printf", "%s\\n"] and argv[3:] == [">", DOCKER_COMPOSE_PATH]
               for argv in map(shlex.split, scripts))
    assert not any("<<" in script for script in scripts)


def test_a_second_reload_swaps_the_image_again():
    once = snapshot_compose(_deployed_compose(), "snap:1")
    assert yaml.safe_load(snapshot_compose(once, "snap:2"))["services"]["servicedb"]["image"] == "snap:2"


@pytest.mark.parametrize("compose, problem", [
    ("services:\n  slack:\n    image: mcp-slack\n", "names a servicedb image 0 times"),
    ("services:\n  servicedb:\n    image: a\n  servicedb:\n    image: b\n", "names a servicedb image 2 times"),
])
def test_a_compose_file_without_exactly_one_servicedb_image_is_refused(compose, problem):
    with pytest.raises(RuntimeError, match=problem):
        snapshot_compose(compose, "snap:1")


def test_a_snapshot_image_that_would_change_what_the_file_says_is_refused():
    with pytest.raises(ValueError):
        snapshot_compose(_deployed_compose(), "snap:1\n  evil:\n    image: x")
