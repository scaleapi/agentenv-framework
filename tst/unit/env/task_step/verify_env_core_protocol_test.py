"""Unit tests for VerifyCoreEnvironmentProtocolStep."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.env_card_validator.verify_env_core_protocol import VerifyCoreEnvironmentProtocolStep


CARD = {
    "name": "AgentEnvGateway",
    "capabilities": {},
    "children_environments": [{"name": "items", "url": "/svc/mcp-items/agentenv"}],
}


def _make_step():
    return VerifyCoreEnvironmentProtocolStep(id="t", version=None, env_id="e1")


def _make_context():
    ctx = TaskStepContext()
    ctx.deployed_envs = [DeployedEnv(
        env_id="e1", env_version=3, gateway_url="https://gw", mcp_url="https://gw/mcp",
        db_web_url=None, sandbox_id="sb",
    )]
    return ctx


def _run(step, ctx, *, card=None, card_exc=None, probe=True):
    fake_env = MagicMock()
    fake_env.metadata = {}
    fake_env.update_metadata = MagicMock()
    get_card = AsyncMock(side_effect=card_exc) if card_exc else AsyncMock(return_value=card)
    with patch("agentenv_protocol.client.get_card", get_card), \
         patch("agent_env.env.env.Env.get", return_value=fake_env), \
         patch.object(VerifyCoreEnvironmentProtocolStep, "_probe_operation", AsyncMock(return_value=probe)):
        result_ctx = asyncio.run(step.execute(ctx))
    return result_ctx, fake_env


class TestExecute:
    def test_legacy_child_without_operations_key(self):
        result_ctx, env = _run(_make_step(), _make_context(), card=CARD, probe=True)
        protocol = env.update_metadata.call_args[0][0]["validated_environment_protocol"]
        assert protocol == {
            "data/reset": {"supported": True},
            "data/add": {"supported": True},
            "data/get": {"supported": True},
        }
        assert result_ctx.metadata["verifications"]["env_core_protocol"] == protocol

    def test_advertised_full_trio(self):
        card = {**CARD, "children_environments": [
            {"name": "items", "capabilities": {"operations": ["data/reset", "data/add", "data/get"]}},
        ]}
        _, env = _run(_make_step(), _make_context(), card=card, probe=True)
        protocol = env.update_metadata.call_args[0][0]["validated_environment_protocol"]
        assert protocol["data/reset"] == {"supported": True}

    def test_no_operations_env(self):
        card = {**CARD, "children_environments": [
            {"name": "items", "capabilities": {"operations": []}},
        ]}
        _, env = _run(_make_step(), _make_context(), card=card, probe=False)
        protocol = env.update_metadata.call_args[0][0]["validated_environment_protocol"]
        assert protocol == {
            "data/reset": {"supported": False},
            "data/add": {"supported": False},
            "data/get": {"supported": False},
        }

    def test_standalone_card_without_children(self):
        card = {"name": "items", "capabilities": {"operations": ["data/reset"]}}
        _, env = _run(_make_step(), _make_context(), card=card, probe=True)
        protocol = env.update_metadata.call_args[0][0]["validated_environment_protocol"]
        assert protocol["data/reset"] == {"supported": True}

    def test_multi_child_reset_indeterminate(self):
        card = {**CARD, "children_environments": [{"name": "a"}, {"name": "b"}]}
        _, env = _run(_make_step(), _make_context(), card=card, probe=None)
        protocol = env.update_metadata.call_args[0][0]["validated_environment_protocol"]
        assert protocol == {
            "data/reset": {"supported": None},
            "data/add": {"supported": None},
            "data/get": {"supported": None},
        }

    def test_card_fetch_failure_still_probes(self):
        _, env = _run(_make_step(), _make_context(), card_exc=RuntimeError("timeout"), probe=True)
        protocol = env.update_metadata.call_args[0][0]["validated_environment_protocol"]
        assert protocol["data/reset"] == {"supported": None}
        assert protocol["data/add"] == {"supported": True}
        assert protocol["data/get"] == {"supported": True}

    def test_missing_deployed_env_raises(self):
        step = VerifyCoreEnvironmentProtocolStep(id="t", version=None, env_id="other")
        try:
            _run(step, _make_context(), card=CARD)
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "not found in context.deployed_envs" in str(e)


def _probe(payload):
    class _Resp:
        def json(self):
            return payload

    class _Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            return _Resp()

    with patch("agent_env.task_step.task_steps.env_card_validator.verify_env_core_protocol.httpx.AsyncClient", _Client):
        return asyncio.run(_make_step()._probe_operation("https://gw", "data/add", {"parts": []}))


class TestProbeOperation:
    def test_method_not_found_means_unregistered(self):
        assert _probe({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "nf"}}) is False

    def test_invalid_params_proves_registered(self):
        assert _probe({"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "ip"}}) is True

    def test_gateway_forward_error_is_indeterminate(self):
        assert _probe({"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "multi"}}) is None

    def test_result_means_registered(self):
        assert _probe({"jsonrpc": "2.0", "id": 1, "result": {"parts": []}}) is True


class TestSerialization:
    def test_roundtrip(self):
        step = _make_step()
        d = step.to_dict()
        assert d["env_id"] == "e1"
        assert d["type"] == "verify_env_core_protocol"
        restored = VerifyCoreEnvironmentProtocolStep.from_dict(d)
        assert restored.env_id == "e1"
        assert restored.id == "t"
