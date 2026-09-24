"""`deploy_agent` resolves an env it did not deploy through the env's own `mcp_url`."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_env.task_step.task_steps import deploy_agent


def _env_get_returns(env):
    return patch.object(deploy_agent.Env, "get", return_value=env)


@pytest.mark.parametrize("raw", ["https://mcp.example.com/mcp", " https://mcp.example.com/mcp\n"])
def test_env_with_a_live_mcp_url_resolves(raw):
    with _env_get_returns(SimpleNamespace(mcp_url=raw)):
        assert deploy_agent._live_mcp_url("live-env") == "https://mcp.example.com/mcp"


@pytest.mark.parametrize(
    "env",
    [
        SimpleNamespace(),
        SimpleNamespace(mcp_url=""),
        SimpleNamespace(mcp_url="   "),
        SimpleNamespace(mcp_url=None),
        SimpleNamespace(mcp_url="file:///tmp/x"),
        SimpleNamespace(mcp_url="mcp.example.com/mcp"),
        MagicMock(),
    ],
    ids=["no-attr", "empty", "blank", "none", "file-scheme", "no-scheme", "mock-attrs-are-not-urls"],
)
def test_env_without_a_live_mcp_url_is_an_error(env):
    with _env_get_returns(env), pytest.raises(RuntimeError, match="not in context.deployed_envs"):
        deploy_agent._live_mcp_url("deployable-env")
