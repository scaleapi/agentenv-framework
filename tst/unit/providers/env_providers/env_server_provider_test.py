"""EnvironmentServerProvider: one MCP server in its own container on any sandbox provider, with no gateway in front and no
state store, whose own card is the env card and whose record has no gateway. The container is faked; the base _run() runs for real."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.env_providers import env_server_provider
from agent_env.providers.env_providers.env_provider import build_env_provider, record_class_for
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

_SERVER_CARD = {"name": "slack", "additionalInterfaces": [{"url": "/custom-mcp", "transport": "mcp"}]}
_SERVER_URL = "https://mcp-slack.fake.host"
_ENV = SimpleNamespace(id="slack-bare", version=2, environment_name="slack", docker_image_artifact=SimpleNamespace(image_name="mcp-slack"))


class _Container:
    type = "fake"

    def __init__(self, sandbox_id: str, port: int, url: str):
        self.sandbox_id, self.tunnel_urls, self.terminated = sandbox_id, {port: url}, False

    async def terminate(self) -> None:
        self.terminated = True


@pytest.mark.asyncio
async def test_a_deploy_returns_a_sandbox_record_of_the_servers_own_card_and_its_one_container():
    run = await _run_server_deploy()
    [server], record = run.created, run.result
    assert type(record) is DeployedSandboxEnv and not hasattr(record, "gateway_url")
    assert (record.env_id, record.env_version, record.env_provider_type) == ("slack-bare", 2, "server")
    assert (record.environment_card_url, record.environment_card) == (f"{_SERVER_URL}{WELL_KNOWN_PATH}", _SERVER_CARD)
    assert record.environment_card_read_at_utc is not None
    assert (record.mcp_url, record.mcp_server_name) == (f"{_SERVER_URL}/custom-mcp", "slack")  # from the card
    assert (record.sandbox_id, record.sandbox_type, record.sandbox_ids) == (server.sandbox_id, "fake", {"mcp_server": {"slack": server.sandbox_id}})
    assert run.gp.environment_sandbox("slack") is server
    run.gp._wait_for_tunnel.assert_awaited_once_with(_SERVER_URL, timeout=180)
    run.tool_names.assert_awaited_once_with(f"{_SERVER_URL}/custom-mcp")  # the card's MCP path, not the /mcp convention


@pytest.mark.asyncio
async def test_a_deploy_starts_only_the_server_on_sqlite_with_its_sizing():
    run = await _run_server_deploy(cpu=2.0, memory_mb=4096, attribution={"project_id": "0123456789abcdef01234567"})
    [call] = run.calls
    assert (call["image_name"], call["port"], call["cpu"], call["memory"], call["timeout"]) == ("mcp-slack", 18765, 2.0, 4096, 60)
    assert call["attribution"] == {"project_id": "0123456789abcdef01234567"} and "i6pn" not in call
    assert call["env"] == {"ENVIRONMENT_NAME": "slack", "MCP_HOST": "0.0.0.0", "DATABASE_URL": "sqlite:////tmp/slack.sqlite"}


@pytest.mark.asyncio
@pytest.mark.parametrize("card, tools, error", [
    (None, ["slack_list"], "did not serve its env card at https://mcp-slack.fake.host/.well-known/agent-env.json in time; an env deployed without a gateway must serve its own card"),
    ({"name": "mail"}, ["slack_list"], "'slack' serves a card named 'mail'; an env deployed without a gateway must serve its card under its environment_name"),
    (_SERVER_CARD, [], "'slack' lists no MCP tools at"),
    (_SERVER_CARD, ["slack_send", "slack_list", "slack_send"], "'slack' lists duplicate MCP tool names at https://mcp-slack.fake.host/custom-mcp: slack_send"),
], ids=["no-card", "card-named-otherwise", "zero-tools", "duplicate-tools"])
async def test_a_deploy_that_fails_tears_the_server_down(card, tools, error):
    run = await _run_server_deploy(card=card, tools=tools)
    assert isinstance(run.result, RuntimeError) and error in str(run.result)
    assert len(run.created) == 1 and run.created[0].terminated
    assert run.gp._container_sandboxes == [] and run.gp._environment_sandboxes == {}


@pytest.mark.asyncio
async def test_a_server_whose_tunnel_hangs_fails_readiness_on_the_clock(monkeypatch):
    """Each probe attempt can hang for its own timeout, so the attempt count alone doesn't bound readiness."""
    monkeypatch.setattr(env_server_provider, "_READINESS_TIMEOUT_S", 0.05)
    run = await _run_server_deploy(wait_for_tunnel=_never_answers)
    assert isinstance(run.result, RuntimeError) and "must serve its own card" in str(run.result)
    assert run.created[0].terminated


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_names, cause", [
    (lambda: _never_answers, "TimeoutError"),
    (lambda: _hangs_in_cleanup, "TimeoutError"),
    (lambda: AsyncMock(side_effect=ExceptionGroup("unhandled errors in a TaskGroup", [ConnectionError("refused")])), "ConnectionError('refused')"),
], ids=["never-answers", "hangs-in-cleanup", "transport-error"])
async def test_a_tools_list_that_fails_fails_the_deploy_readably_within_its_cap(monkeypatch, tool_names, cause):
    """The MCP client's session-closing DELETE runs in cleanup, so the cap must hold there too."""
    monkeypatch.setattr(env_server_provider, "_TOOLS_GATE_TIMEOUT_S", 0.05)
    start = time.monotonic()
    run = await _run_server_deploy(tool_names=tool_names())
    assert time.monotonic() - start < 10
    assert isinstance(run.result, RuntimeError)
    assert str(run.result).startswith(f"'slack' did not list its MCP tools at {_SERVER_URL}/custom-mcp: ") and cause in str(run.result)
    assert run.created[0].terminated


@pytest.mark.asyncio
async def test_a_chain_falls_back_past_a_member_that_fails_and_logs_both_attempts(caplog):
    caplog.set_level(logging.INFO, logger="agent_env.providers.env_providers.env_provider")
    run = await _run_server_deploy(ChainedSandboxProvider([_FakeSandboxProvider(), _FakeSandboxProvider()]), wait_for_tunnel=[None, _SERVER_CARD])
    [first, second] = run.created
    assert run.result.sandbox_id == second.sandbox_id and first.terminated and not second.terminated
    assert run.gp._container_sandboxes == [second] and run.gp.environment_sandbox("slack") is second
    lines = [m for m in caplog.messages if m.startswith("env_deploy_provider_attempt ")]
    assert len(lines) == 2 and " status=failure " in lines[0] and " status=success " in lines[1]


@pytest.mark.asyncio
async def test_a_chain_whose_members_all_fail_names_each_error():
    run = await _run_server_deploy(ChainedSandboxProvider([_FakeSandboxProvider(error=RuntimeError("no capacity")), _FakeSandboxProvider()]),
                                   card={"name": "mail"})
    assert isinstance(run.result, RuntimeError) and str(run.result).startswith("All 2 chained providers failed to deploy: ")
    assert "RuntimeError('no capacity')" in str(run.result) and "serves a card named 'mail'" in str(run.result)
    assert all(c.terminated for c in run.created)


@pytest.mark.parametrize("name, cls", [("gateway", EnvironmentGatewayProvider), ("server", EnvironmentServerProvider)])
def test_build_env_provider_builds_a_new_provider_of_the_declared_type(name, cls):
    first, second = build_env_provider(name), build_env_provider(name)
    assert type(first) is cls and first is not second


def test_build_env_provider_rejects_an_unknown_type():
    with pytest.raises(ValueError, match=r"Unknown env_provider_type: 'vm' \(expected one of \['gateway', 'server'\]\)"):
        build_env_provider("vm")


@pytest.mark.asyncio
async def test_its_records_load_back_as_sandbox_records():
    record = (await _run_server_deploy()).result
    assert record_class_for("server") is DeployedSandboxEnv
    assert DeployedEnv.from_dict(dataclasses.asdict(record)) == record


@pytest.mark.asyncio
async def test_it_has_no_changelog_triggers_to_install():
    assert await EnvironmentServerProvider().install_changelog_triggers("slack") is None


async def _never_answers(_url: str, **_kwargs) -> list[str]:
    await asyncio.sleep(60)
    return []


async def _hangs_in_cleanup(_url: str) -> list[str]:
    try:
        await asyncio.sleep(60)
    finally:
        await asyncio.sleep(60)  # like the MCP client's session-closing DELETE, which a one-shot cancel doesn't interrupt
    return []


async def _run_server_deploy(sandbox_provider=None, *, card=_SERVER_CARD, tools=("slack_list", "slack_send"), tool_names=None,
                             wait_for_tunnel=None, **deploy_kwargs) -> SimpleNamespace:
    """deploy() with the container faked: the provider, its card URL (or the error it raised), each create_container's
    kwargs, the containers they returned, and the tools/list fake."""
    gp, calls, created = EnvironmentServerProvider(), [], []
    tool_names = tool_names or AsyncMock(return_value=list(tools))

    async def fake_create_container(provider, **kwargs):
        calls.append(kwargs)
        if provider.error:
            raise provider.error
        created.append(_Container(f"sb-{kwargs['image_name']}-{len(calls)}", kwargs["port"], f"https://{kwargs['image_name']}.fake.host"))
        return created[-1]

    gp._wait_for_tunnel = AsyncMock(side_effect=wait_for_tunnel) if wait_for_tunnel else AsyncMock(return_value=card)
    with patch.object(_FakeSandboxProvider, "create_container", new=fake_create_container), \
         patch.object(env_server_provider, "_tool_names", new=tool_names):
        try:
            result = await gp.deploy(_ENV, sandbox_provider or _FakeSandboxProvider(), ttl_seconds=60, **deploy_kwargs)
        except Exception as e:  # noqa: BLE001  the failure is the result
            result = e
    return SimpleNamespace(gp=gp, result=result, calls=calls, created=created, tool_names=tool_names)


class _FakeSandboxProvider(SandboxProvider):
    """Any sandbox provider: create_container is patched per run, and ``error`` makes it fail."""

    def __init__(self, error: Exception | None = None):
        self.error = error

    async def create_sandbox(self, **kwargs):
        raise AssertionError("the server provider starts containers, not sandboxes")
