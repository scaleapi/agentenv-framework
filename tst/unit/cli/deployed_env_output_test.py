"""The CLI prints a record's gateway lines only when a gateway fronts it, and a gateway record's output is unchanged."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.cli.env import env as env_cli
from agent_env.cli.task.run import _format_env
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, DeployedSandboxEnv
from agent_env.env.gateway.console_session import LINK_PARAM, ConsoleSession

_FRONTED = DeployedGatewayEnv(env_id="e1", env_version=1, gateway_url="https://gw.example", mcp_url="https://gw.example/mcp",
                              sandbox_id="gw", instance_id="i1")
_BARE = DeployedSandboxEnv(env_id="e1", env_version=1, mcp_url="https://srv.example/mcp", sandbox_id="srv", instance_id="i1")
_HOSTED = DeployedEnv(env_id="e1", env_version=1, env_provider_type="hosted", mcp_url="https://hosted.example/mcp", instance_id="i1")


@pytest.mark.parametrize("deployed, gateway_line", [(_FRONTED, True), (_BARE, False)], ids=["gateway", "no-gateway"])
def test_env_deploy_prints_the_gateway_url_only_for_a_gateway(deployed, gateway_line):
    async def fake_deploy(**kwargs):
        return deployed

    with patch("agent_env.cli.env.deploy.Env") as Env, patch("agent_env.providers.build_sandbox_provider"):
        Env.get.return_value = MagicMock(id="e1", version=1, type="mcp_server", deploy=fake_deploy)
        result = CliRunner().invoke(env_cli, ["deploy", "--id", "e1"])
    assert result.exit_code == 0, result.output
    assert f"Env MCP Url: {deployed.mcp_url}" in result.output
    assert ("Env Gateway Url: https://gw.example" in result.output) is gateway_line


@pytest.mark.parametrize("deployed, gateway_line", [(_FRONTED, True), (_BARE, False)], ids=["gateway", "no-gateway"])
def test_env_get_instance_prints_the_gateway_url_only_for_a_gateway(deployed, gateway_line):
    with patch("agent_env.env.store.get_env_instance_store") as store:
        store.return_value.get.return_value = deployed
        result = CliRunner().invoke(env_cli, ["get-instance", "--id", "i1"])
    assert result.exit_code == 0, result.output
    assert f"Sandbox ID: {deployed.sandbox_id}" in result.output
    assert ("Env Base Url: https://gw.example" in result.output) is gateway_line


def test_task_run_prints_a_gateway_record_as_before_and_omits_a_missing_gateway():
    assert _format_env(_FRONTED).splitlines()[4:8] == [
        "  gateway_url: https://gw.example", "  mcp_url: https://gw.example/mcp", "  db_web_gateway_url: None",
        "  db_mcp_gateway_url: None"]
    assert "gateway_url" not in _format_env(_BARE) and "  mcp_url: https://srv.example/mcp" in _format_env(_BARE)


_CONSOLES = DeployedGatewayEnv(env_id="e1", env_version=1, gateway_url="https://gw.example",
                               mcp_url="https://gw.example/mcp", sandbox_id="gw", instance_id="i1",
                               db_web_url="https://tunnel-8081.example/", db_mcp_url="https://tunnel-18767.example/mcp",
                               db_web_gateway_url="https://gw.example/svc/pgweb/",
                               db_mcp_gateway_url="https://gw.example/svc/db-mcp/mcp")


def test_the_cli_prints_the_sql_consoles_at_the_gateway_and_never_their_tunnels():
    async def fake_deploy(**kwargs):
        return _CONSOLES

    with patch("agent_env.cli.env.deploy.Env") as Env, patch("agent_env.providers.build_sandbox_provider"):
        Env.get.return_value = MagicMock(id="e1", version=1, type="mcp_server", deploy=fake_deploy)
        deployed = CliRunner().invoke(env_cli, ["deploy", "--id", "e1"]).output
    with patch("agent_env.env.store.get_env_instance_store") as store:
        store.return_value.get.return_value = _CONSOLES
        got = CliRunner().invoke(env_cli, ["get-instance", "--id", "i1"]).output
    for output in (deployed, got, _format_env(_CONSOLES)):
        assert "https://gw.example/svc/pgweb/" in output and "https://gw.example/svc/db-mcp/mcp" in output
        assert "tunnel-" not in output



def test_the_cli_prints_a_pgweb_link_a_browser_can_open_and_never_the_token():
    token = "s3cret-token"
    record = DeployedGatewayEnv(env_id="e1", env_version=1, gateway_url="https://gw.example",
                                mcp_url="https://gw.example/mcp", sandbox_id="gw", instance_id="i1",
                                db_web_gateway_url="https://gw.example/svc/pgweb/", control_token=token)

    async def fake_deploy(**kwargs):
        return record

    with patch("agent_env.cli.env.deploy.Env") as Env, patch("agent_env.providers.build_sandbox_provider"):
        Env.get.return_value = MagicMock(id="e1", version=1, type="mcp_server", deploy=fake_deploy)
        deployed = CliRunner().invoke(env_cli, ["deploy", "--id", "e1"]).output
    with patch("agent_env.env.store.get_env_instance_store") as store:
        store.return_value.get.return_value = record
        got = CliRunner().invoke(env_cli, ["get-instance", "--id", "i1"]).output
    for output in (deployed, got):
        assert token not in output
        links = [w for w in output.split() if w.startswith(f"https://gw.example/svc/pgweb/?{LINK_PARAM}=")]
        assert len(links) == 1, output
        assert ConsoleSession(token).opens(links[0].split(f"{LINK_PARAM}=", 1)[1])

def test_a_record_outside_our_sandboxes_prints_no_sandbox():
    with patch("agent_env.env.store.get_env_instance_store") as store:
        store.return_value.get.return_value = _HOSTED
        result = CliRunner().invoke(env_cli, ["get-instance", "--id", "i1"])
    assert result.exit_code == 0, result.output
    assert "Env MCP Url: https://hosted.example/mcp" in result.output and "Sandbox ID" not in result.output
    assert "sandbox" not in _format_env(_HOSTED) and "  mcp_url: https://hosted.example/mcp" in _format_env(_HOSTED)
