"""Unit tests for InstallAgentTaskStep's workspace_dir plumbing (no I/O)."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps.install_agent import (
    InstallAgentTaskStep,
    _build_param_resolvers,
)


def _resolvers(workspace_dir):
    return _build_param_resolvers(
        container_name="c", work_dir="/tmp/stage", agent_ctx_tar="/tmp/stage/ctx.tar.gz",
        a2a_port=8000, config=object(), agent_name="default-agent", workspace_dir=workspace_dir,
    )


def test_workspace_dir_resolver_returns_value_when_set():
    assert _resolvers("/repo")["workspace_dir"]() == "/repo"


def test_workspace_dir_resolver_raises_when_unset():
    # Mirrors an agent declaring required_param 'workspace_dir' without the
    # step supplying one — the execute() loop turns this KeyError into a clear
    # RuntimeError naming the param.
    with pytest.raises(KeyError, match="workspace_dir"):
        _resolvers(None)["workspace_dir"]()


def test_other_resolvers_unaffected():
    r = _resolvers(None)
    assert r["container"]() == "c"
    assert r["a2a_port"]() == "8000"
    assert "workspace_dir" in r  # always registered, even when value is None


def test_to_dict_from_dict_roundtrips_workspace_dir():
    step = InstallAgentTaskStep(
        id="s", version=None, sandbox_name="h", container_name="c",
        a2a_agent_id="claude-code-cli", agent_name="harbor_agent", workspace_dir="/repo",
    )
    restored = InstallAgentTaskStep.from_dict(step.to_dict())
    assert restored.workspace_dir == "/repo"
    assert restored.agent_name == "harbor_agent"


def test_workspace_dir_defaults_none():
    step = InstallAgentTaskStep(
        id="s", version=None, sandbox_name="h", container_name="c", a2a_agent_id="x",
    )
    assert step.workspace_dir is None
    assert step.to_dict()["workspace_dir"] is None


def test_container_resolver_raises_in_host_mode():
    resolvers = _build_param_resolvers(
        container_name=None, work_dir="/tmp/stage", agent_ctx_tar="/tmp/stage/ctx.tar.gz",
        a2a_port=8000, config=object(), agent_name="default-agent", workspace_dir=None,
    )
    with pytest.raises(KeyError, match="host-mode install"):
        resolvers["container"]()


def test_host_mode_roundtrips_without_container_name():
    step = InstallAgentTaskStep(id="s", version=None, sandbox_name="vm", a2a_agent_id="agent")
    assert step.container_name is None
    restored = InstallAgentTaskStep.from_dict(step.to_dict())
    assert restored.container_name is None
    assert restored.a2a_agent_id == "agent"


def test_a2a_agent_id_required():
    with pytest.raises(ValueError, match="a2a_agent_id"):
        InstallAgentTaskStep(id="s", version=None, sandbox_name="vm")


def test_agent_name_resolver_supplies_step_name():
    resolvers = _build_param_resolvers(
        container_name=None, work_dir="/tmp/w", agent_ctx_tar="/tmp/w/c.tgz",
        a2a_port=8000, config=object(), agent_name="solver-a", workspace_dir=None,
    )
    assert resolvers["agent_name"]() == "solver-a"


class _FakeConfig:
    def get_litellm_api_key(self):
        return "config-key"


def test_litellm_key_resolver_prefers_per_run_override():
    r = _build_param_resolvers(
        container_name="c", work_dir="/tmp/stage", agent_ctx_tar="/tmp/stage/ctx.tar.gz",
        a2a_port=8000, config=_FakeConfig(), agent_name="default-agent", workspace_dir=None,
        litellm_api_key_override="run-key",
    )
    assert r["litellm_api_key"]() == "run-key"


def test_litellm_key_resolver_falls_back_to_config():
    r = _build_param_resolvers(
        container_name="c", work_dir="/tmp/stage", agent_ctx_tar="/tmp/stage/ctx.tar.gz",
        a2a_port=8000, config=_FakeConfig(), agent_name="default-agent", workspace_dir=None,
    )
    assert r["litellm_api_key"]() == "config-key"


def test_a2a_port_roundtrip():
    step = InstallAgentTaskStep(id="s", version=None, sandbox_name="vm", a2a_agent_id="agent", a2a_port=8010)
    assert InstallAgentTaskStep.from_dict(step.to_dict()).a2a_port == 8010
    step2 = InstallAgentTaskStep(id="s", version=None, sandbox_name="vm", a2a_agent_id="agent")
    assert InstallAgentTaskStep.from_dict(step2.to_dict()).a2a_port is None
