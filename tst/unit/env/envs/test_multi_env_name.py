"""A MultiEnv's optional `name` is what agents see its MCP server under: stored only when set,
handed to the gateway on deploy, and settable from the CLI."""

from __future__ import annotations

import re
from types import SimpleNamespace

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.cli.env.multi import multi
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.gateway.constants import random_mcp_server_name
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from tst.unit.env.envs.test_multi_env_vm_sizing import _deployed_kwargs
from tst.util.sandbox_providers import mock_provider


def test_name_is_stored_only_when_set_and_round_trips():
    assert "name" not in MultiEnv(id="e", version=1, mcp_server_envs=[]).to_dict()
    doc = MultiEnv(id="e", version=1, mcp_server_envs=[], name="crm").to_dict()
    assert doc["name"] == "crm"
    with patch("agent_env.env.env.Env.get"):
        assert MultiEnv.from_dict(doc).name == "crm"
        assert MultiEnv.from_dict({"id": "e", "version": 1}).name is None


@pytest.mark.parametrize("bad", ["", "my crm", "crm\n"])
def test_name_must_be_non_empty_without_whitespace(bad):
    with pytest.raises(ValueError):
        MultiEnv(id="e", version=1, mcp_server_envs=[], name=bad)


@pytest.mark.asyncio
async def test_deploy_hands_the_name_to_the_gateway():
    assert (await _deployed_kwargs(MultiEnv(id="e", version=1, mcp_server_envs=[], name="crm")))["mcp_server_name"] == "crm"
    assert re.fullmatch(r"env\d{4}", (await _deployed_kwargs(MultiEnv(id="e", version=1, mcp_server_envs=[])))["mcp_server_name"])


def test_cli_put_passes_name():
    server = MagicMock(type="mcp_server", id="slack", version=3)
    with patch("agent_env.cli.env.multi.Env.get", return_value=server), \
         patch("agent_env.cli.env.multi.detect_base_metadata", return_value={}), \
         patch("agent_env.cli.env.multi.MultiEnv.put", return_value=MagicMock(id="crm-suite", version=1)) as put:
        result = CliRunner().invoke(multi, ["put", "--id", "crm-suite", "--mcp-server", "slack", "--name", "crm"])
    assert result.exit_code == 0, result.output
    assert put.call_args.kwargs["name"] == "crm"


def test_random_default_is_env_plus_four_digits():
    draws = {random_mcp_server_name() for _ in range(20)}
    assert all(re.fullmatch(r"env\d{4}", d) for d in draws) and len(draws) > 1


def test_deployed_env_records_the_gateway_name():
    base = dict(env_id="e", env_version=1, gateway_url="g", mcp_url="m", db_web_url=None, sandbox_id="s")
    assert DeployedEnv.from_dict({**base, "mcp_server_name": "env4821"}).mcp_server_name == "env4821"
    assert DeployedEnv.from_dict(base).mcp_server_name is None


@pytest.mark.asyncio
async def test_a_deploy_records_the_mcp_url_and_name_its_gateway_card_gives():
    card = {"name": "crm", "additionalInterfaces": [{"url": "/mcp", "transport": "mcp"}]}
    result = SimpleNamespace(gateway_url="https://gw.example", mcp_url="https://gw.example/mcp", db_web_url=None, db_mcp_url=None,
                             website_frontend_urls=None, environment_card=card, environment_card_read_at_utc="t",
                             env_state_instance_ids=[], mcp_server_name="crm")

    async def deploy_gateway(self, sandbox_provider, **kwargs):
        self._sandbox = SimpleNamespace(sandbox_id="gw", type="modal_vm")
        return result

    with patch.object(EnvironmentGatewayProvider, "_deploy_gateway", deploy_gateway), \
         patch("agent_env.providers.env_providers.env_gateway_provider._probe_tools", AsyncMock()), \
         patch("agent_env.env.env.Env.get"), \
         patch("agent_env.providers.get_env_sandbox_provider", MagicMock(return_value=mock_provider())), \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", AsyncMock(return_value=None)), \
         patch("agent_env.env.envs._deployment.register_env_instance", side_effect=lambda deployed, ttl: deployed):
        record = await MultiEnv(id="e", version=1, mcp_server_envs=[], name="crm").deploy()
    assert (type(record), record.env_provider_type, record.mcp_url, record.mcp_server_name) == (DeployedGatewayEnv, "gateway", "https://gw.example/mcp", "crm")
