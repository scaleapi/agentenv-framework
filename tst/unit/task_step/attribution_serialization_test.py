"""The persisted deploy steps' attribution shape is pinned.

Attribution lives at `metadata["attribution"]` and nowhere else in a step document. Pinned
rather than assumed because `tasks.steps[]` documents are versioned and immutable, and
consumers query them by dotted path: a key that moves returns an empty result set, not an
error.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent_env.attribution import attribution_of
from agent_env.env.env import DeployedGatewayEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep

ATTRIBUTION = {"cost_center": "research", "team": "platform"}


def _deploy_env(**kwargs):
    return DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2, **kwargs)


def _deploy_agent(**kwargs):
    return DeployAgentTaskStep(id="t-1.deploy_agent", version=None, **kwargs)


def _deploy_sandbox(**kwargs):
    return DeploySandboxTaskStep(
        id="t-1.deploy_sandbox", version=None, sandbox_name="sb", sandbox_mode="vm", **kwargs
    )


_EVERY_PERSISTED_STEP = pytest.mark.parametrize(
    "factory",
    [_deploy_env, _deploy_agent, _deploy_sandbox],
    ids=["deploy_env", "deploy_agent", "deploy_sandbox"],
)


@_EVERY_PERSISTED_STEP
def test_attribution_is_written_under_metadata_only(factory):
    doc = factory(metadata={"attribution": ATTRIBUTION}).to_dict()
    assert not any(key in doc for key in ATTRIBUTION), sorted(set(doc) & set(ATTRIBUTION))
    assert doc["metadata"] == {"attribution": ATTRIBUTION}


@_EVERY_PERSISTED_STEP
def test_metadata_persists_and_round_trips(factory):
    metadata = {"attribution": {"cost_center": "research"}, "run_label": "nightly"}
    step = factory(metadata=metadata)
    assert step.to_dict()["metadata"] == metadata
    assert type(step).from_dict(step.to_dict()).metadata == metadata


@_EVERY_PERSISTED_STEP
def test_unset_metadata_is_written_as_an_empty_map(factory):
    assert factory().to_dict()["metadata"] == {}


@_EVERY_PERSISTED_STEP
def test_attribution_of_reads_the_reserved_metadata_key_only(factory):
    step = factory(metadata={"attribution": {"cost_center": "research"}, "run_label": "nightly"})
    assert attribution_of(step) == {"cost_center": "research"}
    assert attribution_of(factory()) == {}


@_EVERY_PERSISTED_STEP
def test_a_top_level_key_in_a_document_is_not_attribution(factory):
    doc = {**factory().to_dict(), **ATTRIBUTION}
    restored = type(factory()).from_dict(doc)
    assert restored.metadata == {}
    assert attribution_of(restored) == {}


# --- the dict is what reaches the env --------------------------------------------


@pytest.mark.asyncio
async def test_deploying_an_env_passes_the_attribution_dict():
    passed = {}

    async def deploy(*, attribution=None, **kwargs):
        passed.update(kwargs, attribution=attribution)
        return DeployedGatewayEnv(
            env_id="art-1", env_version=2, gateway_url="http://gw", mcp_url="http://mcp",
            db_web_url=None, sandbox_id="s1", metadata=None,
        )

    env = MagicMock()
    env.deploy = deploy
    step = _deploy_env(metadata={"attribution": ATTRIBUTION})

    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext())

    assert passed["attribution"] == ATTRIBUTION


@pytest.mark.asyncio
async def test_deploying_an_env_with_no_attribution_passes_an_empty_dict():
    passed = {}

    async def deploy(*, attribution=None, **kwargs):
        passed["attribution"] = attribution
        return DeployedGatewayEnv(
            env_id="art-1", env_version=2, gateway_url="http://gw", mcp_url="http://mcp",
            db_web_url=None, sandbox_id="s1", metadata=None,
        )

    env = MagicMock()
    env.deploy = deploy

    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await _deploy_env().execute(TaskStepContext())

    assert passed["attribution"] == {}
