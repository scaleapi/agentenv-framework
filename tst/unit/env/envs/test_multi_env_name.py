"""A MultiEnv's optional `name` is what agents see its MCP server under: stored only when set,
handed to the gateway on deploy, and settable from the CLI."""

from __future__ import annotations

import re

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.cli.env.multi import multi
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.gateway.constants import random_mcp_server_name
from tst.unit.env.envs.test_multi_env_vm_sizing import _deployed_kwargs


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
    assert (await _deployed_kwargs(MultiEnv(id="e", version=1, mcp_server_envs=[])))["mcp_server_name"] is None


def test_cli_put_passes_name():
    server = MagicMock(type="mcp_server", id="slack", version=3)
    with patch("agent_env.cli.env.multi.Env.get", return_value=server), \
         patch("agent_env.cli.env.multi.detect_base_metadata", return_value={}), \
         patch("agent_env.cli.env.multi.MultiEnv.put", return_value=MagicMock(id="crm-suite", version=1)) as put:
        result = CliRunner().invoke(multi, ["put", "--id", "crm-suite", "--mcp-server", "slack", "--name", "crm", "--skip-validation"])
    assert result.exit_code == 0, result.output
    assert put.call_args.kwargs["name"] == "crm"


def test_random_default_is_env_plus_four_digits():
    draws = {random_mcp_server_name() for _ in range(20)}
    assert all(re.fullmatch(r"env\d{4}", d) for d in draws) and len(draws) > 1


def test_deployed_env_records_the_gateway_name():
    base = dict(env_id="e", env_version=1, gateway_url="g", mcp_url="m", db_web_url=None, sandbox_id="s")
    assert DeployedEnv.from_dict({**base, "mcp_server_name": "env4821"}).mcp_server_name == "env4821"
    assert DeployedEnv.from_dict(base).mcp_server_name is None
