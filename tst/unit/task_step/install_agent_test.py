"""Unit tests for InstallAgentTaskStep's workspace_dir plumbing, and what its install logs show (no I/O)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.config.model import ModelConfig
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
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


class _InstallSandbox:
    def __init__(self):
        self.scripts: list[str] = []

    def scoped_name(self, name):
        return name

    def host_port(self, port):
        return port

    async def load_docker_images(self, images):
        return None

    async def load_object_file(self, url, destination):
        return None

    async def exec_script(self, script):
        self.scripts.append(script)
        return ""


def _resolved(value):
    return lambda *args, **kwargs: asyncio.sleep(0, result=value)


@pytest.mark.asyncio
async def test_install_logs_show_the_command_template_not_the_model_key_or_headers(monkeypatch, caplog, local_stores):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-never-logged")
    monkeypatch.setattr(local_stores, "_model_cfg", ModelConfig(headers={"X-Bill-To": "org-never-logged"}))
    sandbox = _InstallSandbox()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda name: SimpleNamespace(get_sandbox=_resolved(sandbox)))
    image = SimpleNamespace(image_name="img", build_context_object_url="s3://bucket/ctx.tar.gz")
    monkeypatch.setattr(A2AAgent, "get", staticmethod(lambda *args: SimpleNamespace(
        id="agent-x", version=1, metadata={}, docker_image_artifact=image)))
    step = InstallAgentTaskStep(id="i", version=None, sandbox_name="vm", a2a_agent_id="agent-x", agent_name="solver")
    monkeypatch.setattr(step, "_extract_install_extension", _resolved({"params": {
        "install_commands_host": ["start --key {litellm_api_key} --headers {litellm_extra_headers}"],
        "required_params_host": ["litellm_api_key", "litellm_extra_headers"], "a2a_port": 8000}}))
    monkeypatch.setattr(step, "_wait_for_agent_card", _resolved({}))
    monkeypatch.setattr(step, "_register_instance", lambda *args: SimpleNamespace(instance_id="solver-1"))
    context = TaskStepContext(deployed_sandboxes=[DeployedSandbox(
        sandbox_name="vm", sandbox_id="sb-1", sandbox_mode="vm", sandbox_type="local",
        tunnel_urls={"8000": "http://127.0.0.1:8000"})])

    with caplog.at_level(logging.INFO, logger="agent_env.task_step.task_steps.install_agent"):
        await step.execute(context)

    [command] = [s for s in sandbox.scripts if s.startswith("start ")]
    assert "sk-never-logged" in command and "org-never-logged" in command
    assert "start --key {litellm_api_key} --headers {litellm_extra_headers}" in caplog.text
    assert "never-logged" not in caplog.text
