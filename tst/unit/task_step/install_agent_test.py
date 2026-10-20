"""Unit tests for InstallAgentTaskStep's workspace_dir plumbing, and what its install logs show (no I/O)."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.config.model import ModelConfig
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.store.secret_store.local_secret_store import LocalSecretStore
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps import install_agent as install_agent_module
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


def test_a_step_naming_no_agent_builds_and_round_trips_without_one():
    step = InstallAgentTaskStep.from_dict({"id": "s", "type": "install_agent", "sandbox_name": "vm"})

    assert step.a2a_agent_id is None
    assert InstallAgentTaskStep.from_dict(step.to_dict()).a2a_agent_id is None


def test_agent_name_resolver_supplies_step_name():
    resolvers = _build_param_resolvers(
        container_name=None, work_dir="/tmp/w", agent_ctx_tar="/tmp/w/c.tgz",
        a2a_port=8000, config=object(), agent_name="solver-a", workspace_dir=None,
    )
    assert resolvers["agent_name"]() == "solver-a"


class _FakeConfig:
    def get_litellm_api_key(self):
        return "config-key"

    def resolve_reference(self, value):
        return value


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


@pytest.mark.asyncio
async def test_a_per_run_key_given_by_reference_is_resolved_through_the_secret_store(monkeypatch, local_stores):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
    local_stores.set_secret_store(LocalSecretStore(values={"team_key": "sk-from-the-store"}, use_env=False))
    sandbox = _InstallSandbox()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda name: SimpleNamespace(get_sandbox=_resolved(sandbox)))
    image = SimpleNamespace(image_name="img", build_context_object_url="s3://bucket/ctx.tar.gz")
    monkeypatch.setattr(A2AAgent, "get", staticmethod(lambda *args: SimpleNamespace(
        id="agent-x", version=1, metadata={}, docker_image_artifact=image)))
    step = InstallAgentTaskStep(id="i", version=None, sandbox_name="vm", a2a_agent_id="agent-x", agent_name="solver")
    monkeypatch.setattr(step, "_extract_install_extension", _resolved({"params": {
        "install_commands_host": ["start --key {litellm_api_key}"],
        "required_params_host": ["litellm_api_key"], "a2a_port": 8000}}))
    monkeypatch.setattr(step, "_wait_for_agent_card", _resolved({}))
    monkeypatch.setattr(step, "_register_instance", lambda *args: SimpleNamespace(instance_id="solver-1"))
    context = TaskStepContext(
        deployed_sandboxes=[DeployedSandbox(
            sandbox_name="vm", sandbox_id="sb-1", sandbox_mode="vm", sandbox_type="local",
            tunnel_urls={"8000": "http://127.0.0.1:8000"})],
        metadata={"user_overrides": {"litellm_api_key": "secret:team_key"}},
    )

    await step.execute(context)

    assert [s for s in sandbox.scripts if s.startswith("start ")] == ["start --key sk-from-the-store"]


def test_a_per_run_reference_is_resolved_only_when_the_install_asks_for_the_key():
    resolved = []

    class _Config(_FakeConfig):
        def resolve_reference(self, value):
            resolved.append(value)
            return "sk-resolved"

    r = _build_param_resolvers(
        container_name="c", work_dir="/tmp/stage", agent_ctx_tar="/tmp/stage/ctx.tar.gz",
        a2a_port=8000, config=_Config(), agent_name="default-agent", workspace_dir=None,
        litellm_api_key_override="secret:team_key",
    )
    assert resolved == []
    assert r["litellm_api_key"]() == "sk-resolved"
    assert resolved == ["secret:team_key"]


@pytest.mark.asyncio
async def test_a_step_naming_no_agent_installs_the_configured_default(monkeypatch, local_stores):
    asked = []

    class _Stop(Exception):
        pass

    def get(agent_id, version=None):
        asked.append(agent_id)
        raise _Stop

    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda name: SimpleNamespace(get_sandbox=_resolved(_InstallSandbox())))
    monkeypatch.setattr(A2AAgent, "get", staticmethod(get))
    step = InstallAgentTaskStep(id="i", version=None, sandbox_name="vm")
    context = TaskStepContext(deployed_sandboxes=[DeployedSandbox(
        sandbox_name="vm", sandbox_id="sb-1", sandbox_mode="vm", sandbox_type="local",
        tunnel_urls={"8000": "http://127.0.0.1:8000"})])

    with pytest.raises(_Stop):
        await step.execute(context)

    assert asked == [local_stores.get_default_a2a_agent_id()]


class _TrustingSandbox:
    """A sandbox that records the CA files it's given."""

    def __init__(self, on_this_machine):
        self.ON_THIS_MACHINE = on_this_machine
        self.copied, self.written = [], {}

    async def copy_trust_dir_into_container(self, trust_dir, container, destination):
        self.copied.append((trust_dir, container, destination))

    async def write_host_file(self, data, vm_path):
        self.written[vm_path] = data


@pytest.fixture
def ca(monkeypatch, tmp_path):
    trust = tmp_path / "trust"
    trust.mkdir()
    (trust / "ca.pem").write_text("ca")
    (trust / "ca-bundle.pem").write_text("bundle")
    monkeypatch.setattr(install_agent_module, "local_grant_trust", lambda: trust)
    return trust


def test_an_install_in_a_container_on_this_machine_gets_the_ca_copied_in_and_the_lines_that_find_it(ca):
    sandbox = _TrustingSandbox(on_this_machine=True)

    lines = asyncio.run(InstallAgentTaskStep._trust_lines(sandbox, "box-task", "/tmp/work"))

    assert sandbox.copied == [(ca, "box-task", "/etc/agentenv")]
    assert lines.splitlines() == ["SSL_CERT_FILE=/etc/agentenv/ca-bundle.pem",
                                  "REQUESTS_CA_BUNDLE=/etc/agentenv/ca-bundle.pem",
                                  "NODE_EXTRA_CA_CERTS=/etc/agentenv/ca.pem"]


def test_an_install_on_a_host_on_this_machine_gets_the_ca_beside_its_work_dir(ca):
    sandbox = _TrustingSandbox(on_this_machine=True)

    lines = asyncio.run(InstallAgentTaskStep._trust_lines(sandbox, None, "/tmp/work dir"))

    assert sandbox.written == {"/tmp/work dir/trust/ca-bundle.pem": b"bundle", "/tmp/work dir/trust/ca.pem": b"ca"}
    assert lines.splitlines()[-1] == "NODE_EXTRA_CA_CERTS='/tmp/work dir/trust/ca.pem'"


def test_an_install_elsewhere_or_without_a_local_ca_gets_no_lines(ca, monkeypatch):
    remote = _TrustingSandbox(on_this_machine=False)
    assert asyncio.run(InstallAgentTaskStep._trust_lines(remote, "box-task", "/tmp/work")) == ""
    monkeypatch.setattr(install_agent_module, "local_grant_trust", lambda: None)
    assert asyncio.run(InstallAgentTaskStep._trust_lines(_TrustingSandbox(True), "box-task", "/tmp/work")) == ""
    assert remote.copied == [] and remote.written == {}


def test_the_trust_lines_resolve_as_the_trust_env_param():
    resolvers = _build_param_resolvers(container_name="c", work_dir="/w", agent_ctx_tar="/w/a.tgz", a2a_port=8000,
                                       config=None, agent_name="a", trust_lines="SSL_CERT_FILE=/x\n")
    assert resolvers["trust_env"]() == "SSL_CERT_FILE=/x\n"
