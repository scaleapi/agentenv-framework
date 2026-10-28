"""``add_skills`` ``computer_env_ids``: each computer env's skill, written by its driver under ``computer-<its environment
name>``, installed on the agent and recorded in the run; refused for an env the run didn't deploy or bind to the agent,
or whose driver serves no skill."""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentenv_protocol.computer import COMPUTER

from agent_env.a2a_agent import A2AAgent
from agent_env.env import env as env_module
from agent_env.env.env import DeployedSandboxEnv
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.add_skills import COMPUTER_SKILLS_KEY, AddSkillsTaskStep

_SKILL_METHOD = {"uri": COMPUTER, "params": {"endpoint": "/agentenv/ext/computer", "methods": {
    "skill": {"method": "GET", "endpoint": "/agentenv/ext/computer/skill"}}}}


def _desk(env_id: str = "desk", name: str = "my-desk", extensions: tuple = (_SKILL_METHOD,)) -> DeployedSandboxEnv:
    return DeployedSandboxEnv(env_id=env_id, env_version=1, environment_card_url=f"http://{env_id}/.well-known/agent-env.json",
                              environment_card={"name": name, "capabilities": {"extensions": list(extensions)}},
                              sandbox_id=f"sb-{env_id}", sandbox_type="local")


@pytest.fixture
def agent(monkeypatch):
    """The agent's skill registrations and the driver calls the step made."""
    made = SimpleNamespace(calls=[], register=AsyncMock(return_value={"ok": True}))

    async def invoke(base_url, card, uri, params=None, timeout=30, verify=True, *, method=None):
        made.calls.append((base_url, uri, method, params))
        return {"name": params["name"], "description": "How to use this computer", "skill_md": f"---\nname: {params['name']}\n"}

    monkeypatch.setattr(env_module.protocol_v1, "invoke_extension", invoke)
    monkeypatch.setattr(A2AAgent, "register_skill", made.register)
    monkeypatch.setattr("agent_env.a2a_agent.store.get_a2a_agent_instance_store",
                        lambda: SimpleNamespace(get=lambda instance_id: "deployed-agent"))
    return made


def _context(*envs, bound=("desk",)) -> TaskStepContext:
    return TaskStepContext(deployed_envs=list(envs), deployed_agents=[DeployedAgent(
        agent_name="solver", api_url="http://a", instance_id="inst-1", env_ids=list(bound))])


def _step(*env_ids) -> AddSkillsTaskStep:
    return AddSkillsTaskStep(id="skills", version=None, agent_name="solver", computer_env_ids=list(env_ids))


@pytest.mark.asyncio
async def test_each_computer_envs_skill_is_installed_under_its_environment_name_and_recorded(agent):
    context = await _step("desk").execute(_context(_desk()))

    assert agent.calls == [("http://desk", COMPUTER, "skill", {"name": "computer-my-desk"})]
    agent.register.assert_awaited_once_with("deployed-agent", name="computer-my-desk",
                                            description="How to use this computer",
                                            skill_md="---\nname: computer-my-desk\n")
    assert context.metadata[COMPUTER_SKILLS_KEY] == {"solver": {"desk": {
        "name": "computer-my-desk", "skill_md": "---\nname: computer-my-desk\n"}}}


@pytest.mark.asyncio
async def test_envs_whose_names_come_out_alike_get_skills_of_their_own(agent):
    await _step("desk", "other").execute(_context(_desk(name="my-desk"), _desk("other", name="My_Desk"),
                                                  bound=("desk", "other")))

    digest = hashlib.sha256(b"other").hexdigest()[:12]
    assert [params["name"] for *_, params in agent.calls] == ["computer-my-desk", f"computer-my-desk-{digest}"]


@pytest.mark.asyncio
@pytest.mark.parametrize("envs, bound, problem", [
    ((), ("desk",), "computer env 'desk' isn't deployed in this run"),
    ((_desk(),), (), "computer env 'desk' isn't bound to agent 'solver': name it in the env_ids of its deploy_agent"),
    ((_desk(extensions=()),), ("desk",), f"env 'desk' serves no computer skill: its card has no {COMPUTER} skill method"),
], ids=["not-deployed", "not-bound", "no-skill"])
async def test_an_env_the_skill_cant_come_from_fails_the_step_naming_it(agent, envs, bound, problem):
    with pytest.raises(RuntimeError, match=problem.replace(".", "[.]")):
        await _step("desk").execute(_context(*envs, bound=bound))
    agent.register.assert_not_awaited()


def test_the_step_round_trips_and_a_step_without_computer_envs_serializes_as_before():
    assert AddSkillsTaskStep.from_dict(_step("desk").to_dict()).computer_env_ids == ["desk"]
    assert "computer_env_ids" not in AddSkillsTaskStep(id="s", version=None).to_dict()
