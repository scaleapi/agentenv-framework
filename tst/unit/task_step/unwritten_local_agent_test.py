"""A step deploying an @local agent the store doesn't hold says that a bundle run writes one, not only that it isn't
found: outside a bundle run, nothing has written it. A step naming an agent by an installed bundle's alias deploys, or
checks, the agent the alias names."""

import pytest

from agent_env.a2a_agent.store import A2AAgentStore
from agent_env.store.base import NotFoundError
from agent_env.task_step.context import DeployedAgent, PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.a2a_agent_validator.verify_a2a_agent_card import VerifyA2AAgentCardStep
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

AGENT = "@local/agentenv-framework/agents/claude-code"
WRITTEN_BY_A_RUN = r"there is no agent '@local/agentenv-framework/agents/claude-code' in the store: an @local agent is " \
                   r"written by `agent-env run` of the bundle that defines it or uses it"


@pytest.fixture(autouse=True)
def empty_store(monkeypatch):
    def missing(self, id, version=None):
        raise NotFoundError(f"A2AAgent {id} not found")

    monkeypatch.setattr(A2AAgentStore, "get", missing)


@pytest.mark.asyncio
async def test_deploying_an_unwritten_local_default_agent_names_the_run_that_writes_one(monkeypatch):
    monkeypatch.setattr("agent_env.config.runtime.Config.get_default_a2a_agent_id", lambda self: AGENT)

    with pytest.raises(NotFoundError, match=WRITTEN_BY_A_RUN + r".*point a2a_agent_id or \[agents\] default_a2a"):
        await DeployAgentTaskStep(id="deploy", version=None, agent_name="solver").execute(TaskStepContext())


@pytest.mark.asyncio
async def test_a_store_agent_that_isnt_there_is_reported_as_the_store_reports_it():
    with pytest.raises(NotFoundError, match="^A2AAgent solver not found$"):
        await DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="solver").execute(TaskStepContext())


@pytest.mark.asyncio
async def test_a_judge_deploying_an_unwritten_local_agent_names_the_run_that_writes_one():
    step = RubricsVerifierTaskStep(id="judge", version=1, criteria=[{"id": "done", "description": "done"}],
                                   prompt_id="p1", verifier_id="v", use_trajectory=False, judge_a2a_agent_id=AGENT)
    context = TaskStepContext(
        prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="do it")],
        metadata={"user_overrides": {"judge_litellm_api_key": "k", "judge_litellm_base_url": "http://litellm"}},
    )

    with pytest.raises(NotFoundError, match=WRITTEN_BY_A_RUN + r".*point judge_a2a_agent_id"):
        await step.execute(context)


class _Looked(Exception):
    """The store was asked for the agent an alias names."""


@pytest.fixture
def aliased_store(monkeypatch):
    """A store without an agent of the alias's id, whose lookup of the agent the alias names is recorded."""
    def get(self, id, version=None):
        if id == AGENT:
            raise _Looked(id)
        raise NotFoundError(f"A2AAgent {id} not found")

    monkeypatch.setattr(A2AAgentStore, "get", get)


@pytest.mark.asyncio
async def test_a_step_naming_the_shipped_alias_deploys_the_agent_it_names(aliased_store):
    with pytest.raises(_Looked):
        await DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="claude-code").execute(TaskStepContext())


def test_installing_the_shipped_alias_checks_the_agent_it_names(aliased_store):
    with pytest.raises(_Looked):
        InstallAgentTaskStep(id="install", version=None, sandbox_name="box", a2a_agent_id="claude-code").preflight()


@pytest.mark.asyncio
async def test_a_judge_named_by_the_shipped_alias_is_the_agent_it_names(aliased_store):
    step = RubricsVerifierTaskStep(id="judge", version=1, criteria=[{"id": "done", "description": "done"}],
                                   prompt_id="p1", verifier_id="v", use_trajectory=False, judge_a2a_agent_id="claude-code")
    context = TaskStepContext(
        prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="do it")],
        metadata={"user_overrides": {"judge_litellm_api_key": "k", "judge_litellm_base_url": "http://litellm"}},
    )

    with pytest.raises(_Looked):
        await step.execute(context)


@pytest.mark.asyncio
async def test_a_validator_step_naming_the_shipped_alias_checks_the_agent_it_names(aliased_store):
    deployed = DeployedAgent(agent_name="agent", api_url="http://agent", a2a_card={"name": "claude-code"})

    with pytest.raises(_Looked):
        await VerifyA2AAgentCardStep(id="card", version=None, a2a_agent_id="claude-code").execute(
            TaskStepContext(deployed_agents=[deployed]))
