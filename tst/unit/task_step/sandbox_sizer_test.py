"""A deploy step reserves the size it sets, else what the configured sizer chooses for what it leaves unset, else its
default or the provider's, and says on the sandbox's attribution where each value came from."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.config import ConfigError
from agent_env.env.env import DeployedGatewayEnv
from agent_env.providers.sandbox_providers import sizing
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.sizing import SandboxSize, SandboxSizer
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from tst.util.config import config_with_document


class HalfSizer(SandboxSizer):
    """Sizes every step's memory to ``memory_mb``, and its cpu only when ``cpu`` is given; records what it was asked."""

    asked: list = []

    def __init__(self, memory_mb: int, cpu: float | None = None):
        self.memory_mb, self.cpu = memory_mb, cpu

    def size(self, step, attribution, *, vm):
        HalfSizer.asked.append((step.id, dict(attribution), vm))
        return SandboxSize(cpu=self.cpu, memory_mb=self.memory_mb)


def _config(sizer: dict | None = None):
    return config_with_document({"sandbox": {"sizer": sizer}} if sizer else {})


@pytest.fixture(autouse=True)
def _reset_asked():
    HalfSizer.asked = []


class _Step:
    id = "s"

    def __init__(self, cpu=None, memory_mb=None):
        self.cpu, self.memory_mb = cpu, memory_mb


@pytest.mark.parametrize("sizer, step, default, expected", [
    (None, _Step(), SandboxSize(), (None, None, "default", "default")),
    (None, _Step(), SandboxSize(1.0, 8192), (1.0, 8192, "default", "default")),
    ({"impl": HalfSizer, "config": {"memory_mb": 2048}}, _Step(), SandboxSize(1.0, 8192),
     (1.0, 2048, "default", "sizer")),
    ({"impl": HalfSizer, "config": {"memory_mb": 2048, "cpu": 0.5}}, _Step(), SandboxSize(),
     (0.5, 2048, "sizer", "sizer")),
    ({"impl": HalfSizer, "config": {"memory_mb": 2048, "cpu": 0.5}}, _Step(memory_mb=16384), SandboxSize(),
     (0.5, 16384, "sizer", "step")),
    ({"impl": HalfSizer, "config": {"memory_mb": 2048, "cpu": 0.5}}, _Step(cpu=4.0, memory_mb=16384), SandboxSize(),
     (4.0, 16384, "step", "step")),
], ids=["no-sizer", "no-sizer-step-default", "sizer-memory-only", "sizer-both", "step-memory-wins", "step-sets-both"])
def test_each_value_comes_from_the_step_then_the_sizer_then_the_default(sizer, step, default, expected):
    with patch("agent_env.config.runtime.get_config", return_value=_config(sizer)):
        size, attribution = sizing.resolve(step, {"run_id": "r"}, default, vm=False)

    assert (size.cpu, size.memory_mb, attribution["cpu_source"], attribution["memory_source"]) == expected
    assert attribution["run_id"] == "r"


def test_a_step_that_sets_both_never_asks_the_sizer():
    with patch("agent_env.config.runtime.get_config", return_value=_config({"impl": HalfSizer, "config": {"memory_mb": 1}})):
        sizing.resolve(_Step(cpu=1.0, memory_mb=1024), {}, vm=False)
    assert HalfSizer.asked == []


def test_a_sizer_that_is_not_one_is_refused():
    with patch("agent_env.config.runtime.get_config", return_value=_config({"impl": dict})):
        with pytest.raises(ConfigError, match="not a subclass of SandboxSizer"):
            sizing.resolve(_Step(), {}, vm=False)


@pytest.mark.asyncio
async def test_deploy_env_hands_the_sized_values_and_their_sources_to_the_env():
    env = MagicMock()
    env.deploy = AsyncMock(return_value=DeployedGatewayEnv(env_id="e", env_version=1, gateway_url="http://gw",
                                                           mcp_url="http://mcp", db_web_url=None, sandbox_id="s1"))
    step = DeployEnvTaskStep(id="deploy", version=None, env_id="e", env_version=1, cpu=2.0)
    config = _config({"impl": HalfSizer, "config": {"memory_mb": 2048, "cpu": 0.5}})

    with patch("agent_env.config.runtime.get_config", return_value=config), patch("agent_env.env.env.Env") as Env, \
            patch("agent_env.task_step.task_steps.deploy_env.may_create_vm", return_value=False):
        Env.get.return_value = env
        await step.execute(TaskStepContext(instance_id="run-1"))

    kwargs = env.deploy.await_args.kwargs
    assert (kwargs["cpu"], kwargs["memory_mb"]) == (2.0, 2048)
    assert {k: kwargs["attribution"][k] for k in ("cpu_source", "memory_source", "run_id")} == {
        "cpu_source": "step", "memory_source": "sizer", "run_id": "run-1"}
    assert HalfSizer.asked == [("deploy", {"run_id": "run-1"}, False)]


@pytest.mark.asyncio
async def test_deploy_sandbox_falls_back_to_its_own_default_when_the_sizer_chooses_nothing():
    sandbox = AsyncMock()
    sandbox.sandbox_id, sandbox.mode, sandbox.type = "sb-1", "vm", "modal_vm"
    sandbox.tunnel_urls, sandbox.vnc_url, sandbox.network_policy = {}, None, NetworkPolicy()
    provider = AsyncMock(ON_THIS_MACHINE=False)
    provider.links = (provider,)
    provider.create_vm.return_value = sandbox
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="vm", sandbox_type="any")

    with patch("agent_env.config.runtime.get_config", return_value=_config({"impl": SandboxSizer})), \
            patch("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider", return_value=provider):
        await step.execute(TaskStepContext())

    kwargs = provider.create_vm.call_args.kwargs
    assert (kwargs["cpu"], kwargs["memory"]) == (1.0, 8192)
    assert (kwargs["attribution"]["cpu_source"], kwargs["attribution"]["memory_source"]) == ("default", "default")


@pytest.mark.parametrize("attribution, expected", [
    ({"run_id": "r", "cpu_source": "sizer", "memory_source": "sizer"},
     {"run_id": "r", "cpu_source": "sizer", "memory_source": "provider"}),
    ({"run_id": "r", "memory_source": "step"}, {"run_id": "r", "memory_source": "provider"}),
    ({"run_id": "r"}, {"run_id": "r"}),
    (None, None),
], ids=["sized", "step-memory", "no-sources", "none"])
def test_a_sandbox_made_without_the_steps_memory_says_its_provider_chose_it(attribution, expected):
    assert sizing.with_default_memory(attribution) == expected


class RaisingSizer(SandboxSizer):
    def size(self, step, attribution, *, vm):
        raise RuntimeError("store down")


class WrongShapeSizer(SandboxSizer):
    def size(self, step, attribution, *, vm):
        return {"cpu": 0.5}


@pytest.mark.parametrize("impl", [RaisingSizer, WrongShapeSizer], ids=["raises", "wrong-shape"])
def test_a_sizer_that_fails_leaves_the_default_and_never_fails_the_deploy(impl, caplog):
    with patch("agent_env.config.runtime.get_config", return_value=_config({"impl": impl})):
        size, attribution = sizing.resolve(_Step(), {}, SandboxSize(1.0, 8192), vm=False)

    assert (size, attribution["cpu_source"], attribution["memory_source"]) == (SandboxSize(1.0, 8192), "default",
                                                                               "default")
    assert f"{impl.__name__} could not size step 's'" in caplog.text


def test_a_sizer_named_by_a_string_is_built():
    config = config_with_document({"sandbox": {"sizer": f"{__name__}:RaisingSizer"}})
    assert type(config.sandbox_sizer()) is RaisingSizer


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, create, extra, vm", [
    ("vm", "create_vm", {}, True),
    ("container", "create_container", {"image": "example.com/box:v1", "port": 80}, False),
], ids=["vm", "container"])
async def test_deploy_sandbox_tells_the_sizer_whether_it_deploys_a_vm(mode, create, extra, vm):
    sandbox = AsyncMock()
    sandbox.sandbox_id, sandbox.mode, sandbox.type = "sb-1", mode, "modal"
    sandbox.tunnel_urls, sandbox.vnc_url, sandbox.network_policy = {}, None, NetworkPolicy()
    provider = AsyncMock(ON_THIS_MACHINE=False)
    provider.links = (provider,)
    getattr(provider, create).return_value = sandbox
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode=mode, sandbox_type="any", **extra)

    with patch("agent_env.config.runtime.get_config", return_value=_config({"impl": HalfSizer,
                                                                           "config": {"memory_mb": 2048}})), \
            patch("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider", return_value=provider):
        await step.execute(TaskStepContext())

    assert HalfSizer.asked[0][2] is vm


@pytest.mark.parametrize("spec, vm", [
    ("modal", False), ("modal_vm", True), ("modal,modal_vm", True), ("no_such_backend", True),
], ids=["container", "vm", "chain-with-a-vm", "unknown"])
def test_a_spec_may_create_a_vm_when_any_backend_it_names_does(spec, vm):
    from agent_env.providers.sandbox_providers.sandbox_provider import may_create_vm

    assert may_create_vm(spec) is vm


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_type, vm", [("modal", False), ("modal_vm", True)], ids=["container", "vm"])
async def test_deploy_env_tells_the_sizer_whether_its_sandbox_type_may_be_a_vm(sandbox_type, vm):
    env = MagicMock()
    env.deploy = AsyncMock(return_value=DeployedGatewayEnv(env_id="e", env_version=1, gateway_url="http://gw",
                                                           mcp_url="http://mcp", db_web_url=None, sandbox_id="s1"))
    step = DeployEnvTaskStep(id="deploy", version=None, env_id="e", env_version=1, sandbox_type=sandbox_type)

    with patch("agent_env.config.runtime.get_config", return_value=_config({"impl": HalfSizer,
                                                                           "config": {"memory_mb": 2048}})), \
            patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext(instance_id="run-1"))

    assert HalfSizer.asked == [("deploy", {"run_id": "run-1"}, vm)]


class _Deployed(Exception):
    """Stops a deploy_agent step once it has asked its agent to deploy, with what it asked."""


@pytest.mark.asyncio
async def test_deploy_agent_hands_the_sized_values_and_their_sources_to_the_agent():
    from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep

    agent = MagicMock(metadata={})
    agent.deploy = AsyncMock(side_effect=_Deployed)
    step = DeployAgentTaskStep(id="agent", version=1, env_ids=[], a2a_agent_id="solver", memory_mb=16384,
                               sandbox_type="modal")
    config = _config({"impl": HalfSizer, "config": {"memory_mb": 2048, "cpu": 0.5}})

    with patch("agent_env.config.runtime.get_config", return_value=config), \
            patch("agent_env.a2a_agent.A2AAgent") as A2AAgent, \
            patch.object(DeployAgentTaskStep, "_resolve_agent_ref", return_value=("solver", 1)), \
            patch.object(DeployAgentTaskStep, "_model_endpoint", return_value={}):
        A2AAgent.get.return_value = agent
        with pytest.raises(_Deployed):
            await step.execute(TaskStepContext(instance_id="run-1"))

    kwargs = agent.deploy.await_args.kwargs
    assert (kwargs["cpu"], kwargs["memory"]) == (0.5, 16384)
    assert (kwargs["attribution"]["cpu_source"], kwargs["attribution"]["memory_source"]) == ("sizer", "step")
    assert HalfSizer.asked == [("agent", {"run_id": "run-1"}, False)]


def test_where_a_size_came_from_stays_out_of_names_labels_and_tags():
    from agent_env.providers.sandbox_providers.boat import config as boat
    from agent_env.providers.sandbox_providers.coreweave.provider import sandbox_tags

    attribution = {"project_id": "p", "cpu_source": "sizer", "memory_source": "default"}
    assert "sizer" not in boat.sandbox_name(attribution) and "default" not in boat.sandbox_name(attribution)
    assert not any("source" in key for key in boat.sandbox_labels(attribution, [80], "t"))
    assert not any("source" in tag for tag in sandbox_tags(attribution, "c"))
