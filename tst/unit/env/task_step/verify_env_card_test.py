"""Unit tests for VerifyEnvironmentCardStep."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_env.env.env import DeployedGatewayEnv, DeployedSandboxEnv
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.env_card_validator.verify_env_card import VerifyEnvironmentCardStep


CARD = {
    "name": "AgentEnvGateway",
    "protocolVersion": "1.0",
    "url": "/agentenv",
    "preferredTransport": "JSONRPC",
    "additionalInterfaces": [],
    "capabilities": {
        "extensions": [
            {"uri": "urn:agentenv:disable-tool/v1"},
            {"uri": "urn:agentenv:enable-tool/v1"},
        ],
        "tools": [{"name": "items_add_item", "description": "Add an item.", "inputSchema": {"type": "object"}}],
    },
    "children_environments": [{"name": "items", "url": "/svc/mcp-items/agentenv"}],
}


def _make_step():
    return VerifyEnvironmentCardStep(id="t", version=None, env_id="e1")


def _make_context():
    ctx = TaskStepContext()
    ctx.deployed_envs = [DeployedGatewayEnv(
        env_id="e1", env_version=3, gateway_url="https://gw", mcp_url="https://gw/mcp",
        db_web_url=None, sandbox_id="sb", environment_card_url=f"https://gw{WELL_KNOWN_PATH}",
    )]
    return ctx


def _run(step, ctx, *, card=None, exc=None, env_name="items"):
    fake_env = MagicMock()
    fake_env.metadata = {}
    fake_env.environment_name = env_name
    fake_env.update_metadata = MagicMock()
    get_card = AsyncMock(side_effect=exc) if exc else AsyncMock(return_value=card)
    with patch("agentenv_protocol.client.get_card", get_card), \
         patch("agent_env.env.env.Env.get", return_value=fake_env):
        result_ctx = asyncio.run(step.execute(ctx))
    return result_ctx, fake_env, get_card


class TestExecute:
    def test_available_card(self):
        result_ctx, env, get_card = _run(_make_step(), _make_context(), card=CARD)
        get_card.assert_awaited_once_with("https://gw", headers={})
        update = env.update_metadata.call_args[0][0]
        assert update["environment_card"] == CARD
        v = update["validated_environment_card"]
        assert v["accessible"] is True
        assert v["children_count"] == 1
        assert v["extensions"] == ["urn:agentenv:disable-tool/v1", "urn:agentenv:enable-tool/v1"]
        assert v["tools"] == ["items_add_item"]
        assert "data_operations" not in v
        assert all(f["present"] for f in v["required_fields"].values())
        assert result_ctx.metadata["verifications"]["environment_card"]["validated_environment_card"] == v

    def test_a_server_record_persists_the_servers_own_card(self):
        ctx = TaskStepContext()
        ctx.deployed_envs = [DeployedSandboxEnv(env_id="e1", env_version=3, env_provider_type="server",
                                                environment_card_url=f"https://srv{WELL_KNOWN_PATH}", sandbox_id="srv")]
        leaf = {**{k: v for k, v in CARD.items() if k != "children_environments"}, "name": "items"}
        _, env, get_card = _run(_make_step(), ctx, card=leaf)
        get_card.assert_awaited_once_with("https://srv", headers={})
        update = env.update_metadata.call_args[0][0]
        assert update["environment_card"] == leaf and update["validated_environment_card"]["accessible"] is True

    def test_extensions_without_uri_are_filtered_out(self):
        card = {**CARD, "capabilities": {"extensions": [
            {"uri": "urn:agentenv:disable-tool/v1"},
            {"name": "no-uri-extension"},
            {"uri": ""},
        ]}}
        _, env, _ = _run(_make_step(), _make_context(), card=card)
        v = env.update_metadata.call_args[0][0]["validated_environment_card"]
        assert v["extensions"] == ["urn:agentenv:disable-tool/v1"]
        assert v["tools"] == []

    def test_tools_recorded_and_nameless_filtered(self):
        card = {**CARD, "capabilities": {"tools": [
            {"name": "items_add_item"},
            {"description": "no-name-tool"},
            {"name": ""},
        ]}}
        _, env, _ = _run(_make_step(), _make_context(), card=card)
        v = env.update_metadata.call_args[0][0]["validated_environment_card"]
        assert v["tools"] == ["items_add_item"]
        assert v["extensions"] == []

    def test_missing_required_field(self):
        card = {k: v for k, v in CARD.items() if k != "url"}
        _, env, _ = _run(_make_step(), _make_context(), card=card)
        v = env.update_metadata.call_args[0][0]["validated_environment_card"]
        assert v["accessible"] is True
        assert v["required_fields"]["url"]["present"] is False
        assert v["required_fields"]["name"]["present"] is True

    def test_inaccessible_records_and_does_not_raise(self):
        result_ctx, env, _ = _run(_make_step(), _make_context(), exc=RuntimeError("connect timeout"))
        update = env.update_metadata.call_args[0][0]
        assert "environment_card" not in update
        assert update["validated_environment_card"]["accessible"] is False
        assert "connect timeout" in update["validated_environment_card"]["error"]
        assert "environment_card" in result_ctx.metadata["verifications"]

    def test_missing_deployed_env_raises(self):
        step = VerifyEnvironmentCardStep(id="t", version=None, env_id="other")
        try:
            _run(step, _make_context(), card=CARD)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "not found in context.deployed_envs" in str(e)

    def test_name_check_matches_when_registered_name_served(self):
        _, env, _ = _run(_make_step(), _make_context(), card=CARD, env_name="items")
        v = env.update_metadata.call_args[0][0]["validated_environment_card"]
        assert v["name_check"] == {"registered_name": "items", "served_card_names": ["items"], "matches": True}

    def test_name_check_flags_drift(self):
        _, env, _ = _run(_make_step(), _make_context(), card=CARD, env_name="items-renamed")
        v = env.update_metadata.call_args[0][0]["validated_environment_card"]
        assert v["name_check"]["matches"] is False
        assert v["name_check"]["registered_name"] == "items-renamed"
        assert v["name_check"]["served_card_names"] == ["items"]

    def test_name_check_ok_with_extra_infra_children(self):
        card = {**CARD, "children_environments": [{"name": "items"}, {"name": "website-browser"}]}
        _, env, _ = _run(_make_step(), _make_context(), card=card, env_name="items")
        v = env.update_metadata.call_args[0][0]["validated_environment_card"]
        assert v["name_check"]["matches"] is True

    def test_name_check_skipped_without_environment_name(self):
        _, env, _ = _run(_make_step(), _make_context(), card=CARD, env_name=None)
        v = env.update_metadata.call_args[0][0]["validated_environment_card"]
        assert v["name_check"] is None


class TestWire:
    @pytest.mark.parametrize("gateway_url", ["https://gw", "https://gw/", "https://sandbox.example.com/sandbox/sandbox-vm-1-18765"])
    def test_reads_the_card_at_its_address(self, gateway_url):
        base = gateway_url.rstrip("/")
        ctx = TaskStepContext(deployed_envs=[DeployedGatewayEnv(
            env_id="e1", env_version=3, gateway_url=gateway_url, mcp_url=f"{base}/mcp", db_web_url=None, sandbox_id="sb",
            environment_card_url=f"{gateway_url}{WELL_KNOWN_PATH}",
        )])
        sent = []

        def handle(request):
            sent.append(request)
            return httpx.Response(200, json=CARD)

        real = httpx.AsyncClient
        fake_env = MagicMock(metadata={}, environment_name="items")
        with patch.object(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle))), \
             patch("agent_env.env.env.Env.get", return_value=fake_env):
            asyncio.run(_make_step().execute(ctx))
        assert [(r.method, str(r.url)) for r in sent] == [("GET", f"{base}{WELL_KNOWN_PATH}")]
        assert fake_env.update_metadata.call_args[0][0]["validated_environment_card"]["accessible"] is True


class TestSerialization:
    def test_roundtrip(self):
        step = _make_step()
        d = step.to_dict()
        assert d["env_id"] == "e1"
        assert d["type"] == "verify_env_card"
        restored = VerifyEnvironmentCardStep.from_dict(d)
        assert restored.env_id == "e1"
        assert restored.id == "t"
