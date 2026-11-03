"""``add_skills`` ``env_ids``: each env's skill and each of its child envs' (a gateway's servers, a computer's apps) its
name reaches, written under that card's environment name, installed on the agent and recorded in the run, a skill with
files beside its SKILL.md sent as a stored bundle; refused for an env the run didn't deploy or bind to the agent, or
that serves no skill."""

import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentenv_protocol import SKILL_EXTENSION_URI

from agent_env.a2a_agent import A2AAgent
from agent_env.env import env as env_module
from agent_env.env.env import DeployedSandboxEnv
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.add_skills import ENV_SKILLS_KEY, AddSkillsTaskStep

_SKILL = {"uri": SKILL_EXTENSION_URI, "params": {"methods": {
    "get": {"method": "GET", "endpoint": "/agentenv/ext/skill/get"}}}}


def _env(env_id: str = "shop", name: str = "my-shop", extensions: tuple = (_SKILL,), children: tuple = ()
         ) -> DeployedSandboxEnv:
    card = {"name": name, "capabilities": {"extensions": list(extensions)}}
    if children:
        card["children_environments"] = [{"name": child, "capabilities": {"extensions": list(served)}}
                                         for child, served in children]
    return DeployedSandboxEnv(env_id=env_id, env_version=1, environment_card_url=f"http://{env_id}/.well-known/agent-env.json",
                              environment_card=card, sandbox_id=f"sb-{env_id}", sandbox_type="local")


@pytest.fixture
def agent(monkeypatch):
    """The agent's skill registrations and the env calls the step made."""
    made = SimpleNamespace(calls=[], register=AsyncMock(return_value={"ok": True}))

    async def invoke(base_url, card, uri, params=None, timeout=30, verify=True, *, method=None, headers=None):
        made.calls.append((base_url, card["name"], uri, method, params))
        return {"name": params["name"], "description": "How to use this env", "skill_md": f"---\nname: {params['name']}\n"}

    monkeypatch.setattr(env_module.protocol_v1, "invoke_extension", invoke)
    monkeypatch.setattr(A2AAgent, "register_skill", made.register)
    monkeypatch.setattr("agent_env.a2a_agent.store.get_a2a_agent_instance_store",
                        lambda: SimpleNamespace(get=lambda instance_id: "deployed-agent"))
    return made


def _context(*envs, bound=("shop",)) -> TaskStepContext:
    return TaskStepContext(deployed_envs=list(envs), deployed_agents=[DeployedAgent(
        agent_name="solver", api_url="http://a", instance_id="inst-1", env_ids=list(bound))])


def _step(*env_ids) -> AddSkillsTaskStep:
    return AddSkillsTaskStep(id="skills", version=None, agent_name="solver", env_ids=list(env_ids))


@pytest.mark.asyncio
async def test_each_envs_skill_is_installed_under_its_environment_name_and_recorded(agent):
    context = await _step("shop").execute(_context(_env()))

    assert agent.calls == [("http://shop", "my-shop", SKILL_EXTENSION_URI, "get", {"name": "my-shop"})]
    agent.register.assert_awaited_once_with("deployed-agent", name="my-shop", description="How to use this env",
                                            skill_md="---\nname: my-shop\n")
    assert context.metadata[ENV_SKILLS_KEY] == {"solver": {"shop": {"my-shop": "---\nname: my-shop\n"}}}


@pytest.mark.asyncio
async def test_a_skill_with_files_is_stored_once_and_installed_as_a_bundle(agent, local_stores, monkeypatch):
    files = {"LINUX.md": "# Linux\n", "WORKFLOW.md": "# Workflow\n"}

    async def invoke(base_url, card, uri, params=None, timeout=30, verify=True, *, method=None, headers=None):
        return {"name": params["name"], "description": "Drive the desktop", "skill_md": "# Skill\n", "files": files}

    monkeypatch.setattr(env_module.protocol_v1, "invoke_extension", invoke)
    await _step("shop").execute(_context(_env()))
    await _step("shop").execute(_context(_env()))

    (first, again) = [call.kwargs for call in agent.register.await_args_list]
    store = local_stores.get_object_store()
    assert first == again and set(first) == {"name", "description", "object_url"}
    assert sorted(store.get_object_key(url).rpartition("/")[2] for url in store.list_at(first["object_url"])) == [
        "LINUX.md", "SKILL.md", "WORKFLOW.md"]


@pytest.mark.asyncio
async def test_a_skill_file_whose_path_leaves_its_folder_is_refused_before_anything_is_stored(agent, local_stores,
                                                                                             monkeypatch):
    async def invoke(base_url, card, uri, params=None, timeout=30, verify=True, *, method=None, headers=None):
        return {"name": params["name"], "description": "d", "skill_md": "# Skill\n", "files": {"../../victim": "x"}}

    monkeypatch.setattr(env_module.protocol_v1, "invoke_extension", invoke)
    with pytest.raises(ValueError, match="must stay inside its folder, not \\['../../victim'\\]"):
        await _step("shop").execute(_context(_env()))
    assert local_stores.get_object_store().list("") == [] and not agent.register.await_args_list


@pytest.mark.asyncio
async def test_an_env_behind_a_gateway_installs_each_child_envs_skill_under_the_childs_name(agent):
    gateway = _env("office", name="office", extensions=(), children=(("mail", (_SKILL,)), ("Chat Room", (_SKILL,)),
                                                                     ("calendar", ())))

    context = await _step("office").execute(_context(gateway, bound=("office",)))

    digest = hashlib.sha256(b"office/Chat Room").hexdigest()[:12]
    assert [(card, params["name"]) for _, card, *_, params in agent.calls] == [
        ("mail", "mail"), ("Chat Room", f"chat-room-{digest}")]
    assert set(context.metadata[ENV_SKILLS_KEY]["solver"]["office"]) == {"mail", f"chat-room-{digest}"}


@pytest.mark.asyncio
async def test_a_computer_installs_its_own_skill_then_each_apps_and_an_app_alone_the_computers_its_name_reaches(agent):
    desk = _env("desk", name="desktop", children=(("mail", (_SKILL,)), ("notes", ())))
    alone = _env("mail-alone", name="mail", children=(("mail", (_SKILL,)),))
    for computer in (desk, alone):
        computer.environment_card["capabilities"]["operations"] = ["data/add", "data/get"]

    context = await _step("desk", "mail-alone").execute(_context(desk, alone, bound=("desk", "mail-alone")))

    assert [(base_url, card) for base_url, card, *_ in agent.calls] == [
        ("http://desk", "desktop"), ("http://desk", "mail"), ("http://mail-alone", "mail")]
    assert set(context.metadata[ENV_SKILLS_KEY]["solver"]["mail-alone"]) == {"mail"}


@pytest.mark.asyncio
async def test_envs_whose_names_come_out_alike_get_skills_of_their_own(agent):
    await _step("shop", "other").execute(_context(_env(name="my-shop"), _env("other", name="My_Shop"),
                                                  bound=("shop", "other")))

    digest = hashlib.sha256(b"other").hexdigest()[:12]
    assert [params["name"] for *_, params in agent.calls] == ["my-shop", f"my-shop-{digest}"]


@pytest.mark.asyncio
@pytest.mark.parametrize("envs, bound, problem", [
    ((), ("shop",), "env 'shop' isn't deployed in this run"),
    ((_env(),), (), "env 'shop' isn't bound to agent 'solver': name it in the env_ids of its deploy_agent"),
    ((_env(extensions=(), children=(("mail", ()),)),), ("shop",),
     f"env 'shop' serves no skill: neither its card nor a child env's declares {SKILL_EXTENSION_URI}"),
], ids=["not-deployed", "not-bound", "no-skill"])
async def test_an_env_the_skill_cant_come_from_fails_the_step_naming_it(agent, envs, bound, problem):
    with pytest.raises(RuntimeError, match=problem.replace(".", "[.]")):
        await _step("shop").execute(_context(*envs, bound=bound))
    agent.register.assert_not_awaited()


def test_the_step_round_trips_and_a_step_without_envs_serializes_as_before():
    assert AddSkillsTaskStep.from_dict(_step("shop").to_dict()).env_ids == ["shop"]
    assert "env_ids" not in AddSkillsTaskStep(id="s", version=None).to_dict()
