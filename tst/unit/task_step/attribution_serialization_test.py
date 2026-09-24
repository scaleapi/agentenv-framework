"""The persisted deploy steps' attribution shape is pinned.

Attribution lives at `metadata["attribution"]` and nowhere else in a step document. Pinned
rather than assumed because:

- `tasks.steps[]` documents are versioned and immutable, and consumers query them by
  dotted path. A key that moves returns an empty result set, not an error.
- Documents written before the `metadata` map carry the four flat keys
  (`product`/`customer`/`team`/`project_id`); they are never rewritten, so `from_dict`
  must keep folding them for as long as those documents exist.

`priority` is serialized in the same block but is a scheduling concern, not
attribution, so it is not asserted here.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent_env.attribution import attribution_of
from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep

# A 24-char hex project id, the shape billing backends accept.
ATTRIBUTION = {
    "project_id": "0123456789abcdef01234567",
    "product": "agent-env",
    "customer": "acme",
    "team": "platform",
}
LEGACY_KEYS = tuple(ATTRIBUTION)


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
def test_the_flat_keys_are_no_longer_constructor_arguments(factory):
    with pytest.raises(TypeError, match="project_id"):
        factory(project_id=ATTRIBUTION["project_id"])


@_EVERY_PERSISTED_STEP
def test_to_dict_does_not_emit_the_flat_keys(factory):
    doc = factory(metadata={"attribution": ATTRIBUTION}).to_dict()
    assert not any(key in doc for key in LEGACY_KEYS), sorted(set(doc) & set(LEGACY_KEYS))
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


# --- documents written before `metadata` existed ---------------------------------


@_EVERY_PERSISTED_STEP
def test_from_dict_folds_a_legacy_flat_key_document_into_metadata(factory):
    doc = {**factory().to_dict(), **ATTRIBUTION}
    del doc["metadata"]
    restored = type(factory()).from_dict(doc)
    assert restored.metadata == {"attribution": ATTRIBUTION}
    assert attribution_of(restored) == ATTRIBUTION


@_EVERY_PERSISTED_STEP
def test_from_dict_keeps_other_metadata_when_folding(factory):
    doc = {**factory(metadata={"run_label": "nightly"}).to_dict(), "team": "platform"}
    assert type(factory()).from_dict(doc).metadata == {
        "run_label": "nightly",
        "attribution": {"team": "platform"},
    }


@_EVERY_PERSISTED_STEP
def test_the_metadata_sub_key_wins_over_a_flat_key_on_conflict(factory):
    doc = {
        **factory(metadata={"attribution": {"project_id": "authored", "team": "t"}}).to_dict(),
        "project_id": "flat",
        "product": "p",
    }
    assert attribution_of(type(factory()).from_dict(doc)) == {
        "project_id": "authored", "team": "t", "product": "p",
    }


@_EVERY_PERSISTED_STEP
def test_null_flat_keys_in_a_legacy_document_are_not_attribution(factory):
    """Pre-metadata documents wrote every flat key, null when unset."""
    doc = {**factory().to_dict(), **dict.fromkeys(LEGACY_KEYS, None)}
    restored = type(factory()).from_dict(doc)
    assert restored.metadata == {}
    assert attribution_of(restored) == {}


@_EVERY_PERSISTED_STEP
def test_a_folded_document_is_rewritten_in_the_new_shape(factory):
    doc = {**factory().to_dict(), **ATTRIBUTION}
    rewritten = type(factory()).from_dict(doc).to_dict()
    assert not any(key in rewritten for key in LEGACY_KEYS)
    assert rewritten["metadata"] == {"attribution": ATTRIBUTION}


# --- the dict is what reaches the env --------------------------------------------


@pytest.mark.asyncio
async def test_deploying_an_env_passes_the_attribution_dict():
    passed = {}

    async def deploy(*, attribution=None, **kwargs):
        passed.update(kwargs, attribution=attribution)
        return DeployedEnv(
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
    assert not any(key in passed for key in LEGACY_KEYS), "flat keys still forwarded"


@pytest.mark.asyncio
async def test_deploying_an_env_with_no_attribution_passes_an_empty_dict():
    passed = {}

    async def deploy(*, attribution=None, **kwargs):
        passed["attribution"] = attribution
        return DeployedEnv(
            env_id="art-1", env_version=2, gateway_url="http://gw", mcp_url="http://mcp",
            db_web_url=None, sandbox_id="s1", metadata=None,
        )

    env = MagicMock()
    env.deploy = deploy

    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await _deploy_env().execute(TaskStepContext())

    assert passed["attribution"] == {}
