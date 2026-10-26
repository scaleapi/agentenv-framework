"""What ``EpisodeSpec.from_task`` reads off a task (the agents a loop plays), what it refuses, and the copy of the task
it runs."""

import pytest

from agent_env.task import EpisodeError, EpisodeSpec, Task
from agent_env.task.episode.rules import CHECKS, UNAFFECTED
from agent_env.task.episode.steps import _EpisodeAgentStep
from agent_env.task.task import substitute_seed_in_steps
from agent_env.task_step.registry import _builtin_registry
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

ENV = {"id": "env", "type": "deploy_env", "env_id": "portsim"}
LOAD = {"id": "load", "type": "load_artifact", "env_id": "portsim", "artifact_id": "week-<week>"}
AGENT = {"id": "agent", "type": "deploy_agent", "env_ids": ["portsim"], "role": "planner@<port>",
         "system_prompt": "You plan <port>."}
PLAY = {"id": "play", "type": "prompt_agent", "prompt_id": "play", "prompt": "Plan week <week>.",
        "depends_on": ["load"], "fail_task_on_error": False}
VERIFY = {"id": "verify", "type": "env_outcome_verifier", "env_id": "portsim", "file_artifact_id": "verify-py",
          "verifier_id": "berth"}
BOB = {"id": "bob", "type": "deploy_agent", "agent_name": "bob", "a2a_agent_id": "echo"}
ASK_BOB = {"id": "ask-bob", "type": "prompt_agent", "agent_name": "bob", "prompt": "Check the plan."}
SEED = {"week": "3", "port": "Valencia"}


def _task(*steps):
    return Task.from_dict({"id": "dock", "version": 4, "steps": [dict(step) for step in steps]})


def _portsim(*extra, **replace):
    steps = {step["id"]: step for step in (ENV, LOAD, AGENT, PLAY, VERIFY)}
    for step_id, step in replace.items():
        steps[step_id] = step
    return _task(*(step for step in steps.values() if step is not None), *extra)


def _refusal(task, trainer_agents=None):
    with pytest.raises(EpisodeError) as raised:
        EpisodeSpec.from_task(task, trainer_agents=trainer_agents)
    return str(raised.value)


def test_a_task_with_one_prompted_agent_is_played_as_that_agent():
    assert EpisodeSpec.from_task(_portsim(), seed=SEED).trainer_agents == ("default-agent",)


def test_an_agent_named_by_the_seed_is_the_one_played_from_outside():
    spec = EpisodeSpec.from_task(
        _portsim(agent={**AGENT, "agent_name": "<who>"}, play={**PLAY, "agent_name": "<who>"}),
        seed={**SEED, "who": "planner"},
    )

    assert spec.trainer_agents == ("planner",)
    assert [step.type for step in spec._copy.steps] == [
        "deploy_env", "load_artifact", "episode_agent", "prompt_agent", "env_outcome_verifier"]


def test_each_turns_prompt_is_read_seeded_before_anything_runs():
    again = {**PLAY, "id": "replan", "prompt_id": "replan", "prompt": "Replan <port>.", "system_prompt": "Berths only.",
             "depends_on": ["play"]}

    assert EpisodeSpec.from_task(_portsim(again), seed=SEED).prompts == (
        ("Plan week 3.", "You plan Valencia."), ("Replan Valencia.", "Berths only."))


def test_a_task_prompting_several_agents_needs_the_ones_to_play_named():
    task = _portsim(BOB, ASK_BOB)

    assert _refusal(task) == (
        "task 'dock' prompts ['default-agent', 'bob']; name the ones to play with trainer_agents=")
    assert EpisodeSpec.from_task(task, trainer_agents=["bob"]).trainer_agents == ("bob",)
    assert EpisodeSpec.from_task(task, trainer_agents=["bob", "default-agent", "bob"]).trainer_agents == (
        "bob", "default-agent")


@pytest.mark.parametrize(("trainer_agents", "message"), [
    (["carol"], "task 'dock' doesn't prompt ['carol']; it prompts ['default-agent']"),
    ([], "trainer_agents= names no agent; task 'dock' prompts ['default-agent']"),
])
def test_only_a_prompted_agent_can_be_played(trainer_agents, message):
    assert _refusal(_portsim(), trainer_agents=trainer_agents) == message


def test_a_task_prompting_no_agent_has_nothing_to_play():
    assert _refusal(_task(ENV, VERIFY)) == "task 'dock' prompts no agent"


def test_the_copy_stands_in_for_the_outside_agents_steps_and_keeps_everything_else():
    task = _portsim(BOB, ASK_BOB)
    copy = EpisodeSpec.from_task(task, trainer_agents=["default-agent"])._copy

    assert (copy.id, copy.version) == ("dock", 4)
    assert [step.type for step in copy.steps] == [
        "deploy_env", "load_artifact", "episode_agent", "prompt_agent", "env_outcome_verifier", "deploy_agent",
        "prompt_agent",
    ]
    assert [copy.steps[i] is task.steps[i] for i in range(len(task.steps))] == [
        True, True, False, True, True, True, True]
    for original, played in zip(task.steps, copy.steps):
        assert (played.id, played.version, played.depends_on, played.fail_task_on_error) == (
            original.id, original.version, original.depends_on, original.fail_task_on_error)


def test_the_outside_agents_deploy_runs_with_a_proxy_unseeded_and_round_trips():
    task = _portsim()
    agent = EpisodeSpec.from_task(task, seed=SEED)._copy.steps[2]

    assert isinstance(agent, _EpisodeAgentStep)
    assert (agent.step, agent.sandboxed) == (task.steps[2], False)
    assert agent.step.role == "planner@<port>"
    assert _EpisodeAgentStep.from_dict(agent.to_dict()).to_dict() == agent.to_dict()


def test_without_a_turn_timeout_a_turns_deadline_is_its_prompts_own():
    task = _portsim(play={**PLAY, "timeout_seconds": 30})

    assert EpisodeSpec.from_task(task)._copy.steps[3] is task.steps[3]


@pytest.mark.parametrize(("turn_timeout", "deadline"), [(90, 90), (0.5, 1)])
def test_a_turn_timeout_replaces_every_outside_prompts_own_in_whole_seconds(turn_timeout, deadline):
    again = {**PLAY, "id": "replan", "prompt_id": "replan", "timeout_seconds": 60}
    task = _portsim(again, play={**PLAY, "timeout_seconds": 7200})
    copy = EpisodeSpec.from_task(task, turn_timeout=turn_timeout)._copy

    assert [step.timeout_seconds for step in copy.steps if step.type == "prompt_agent"] == [deadline, deadline]
    assert copy.steps[3].to_dict() == {**task.steps[3].to_dict(), "timeout_seconds": deadline}


def test_an_inside_agent_keeps_its_own_steps():
    copy = EpisodeSpec.from_task(_portsim(BOB, ASK_BOB), trainer_agents=["bob"])._copy

    assert type(copy.steps[2]) is DeployAgentTaskStep and type(copy.steps[3]) is PromptAgentTaskStep
    assert [step.type for step in copy.steps[5:]] == ["episode_agent", "prompt_agent"]


INSTALL = {"id": "agent", "type": "install_agent", "sandbox_name": "box", "a2a_agent_id": "codex"}
IN_THE_SANDBOX = {
    "an install": ({}, {"agent": INSTALL}),
    "skills on the deploy": ({}, {"agent": {**AGENT, "skills": [{"name": "s", "skill_md": "Plan berths first."}]}}),
    "add_skills": (({"id": "s", "type": "add_skills", "skills": [{"name": "s", "description": "d", "body": "x"}]},),
                   {}),
    "a load into the agent": (({"id": "s", "type": "load_artifact", "artifact_id": "a",
                                "agent_name": "default-agent"},), {}),
    "a collect from the agent": (({"id": "s", "type": "collect_artifacts", "artifact_paths": ["/out"]},), {}),
    "run_code in the agent": (({"id": "s", "type": "run_code", "script_artifact_id": "s", "entrypoint": "main.py"},),
                              {}),
    "verify_sandbox": (({"id": "s", "type": "verify_sandbox", "criteria": []},), {}),
}
OUTSIDE_THE_SANDBOX = {
    "env tools alone": ((), {}),
    "a load into the env": (({**LOAD, "id": "load-2"},), {}),
    "a collect from the env": (({"id": "s", "type": "collect_artifacts", "env_id": "portsim",
                                 "artifact_paths": ["/out"]},), {}),
    "run_code in the env": (({"id": "s", "type": "run_code", "script_artifact_id": "s", "entrypoint": "main.py",
                              "env_id": "portsim"},), {}),
    "verify_sandbox on a sandbox": (({"id": "s", "type": "verify_sandbox", "sandbox_name": "box", "criteria": []},),
                                    {}),
}


@pytest.mark.parametrize(("extra", "replace"), IN_THE_SANDBOX.values(), ids=IN_THE_SANDBOX.keys())
def test_a_task_using_the_outside_agents_sandbox_deploys_its_proxy_there(extra, replace):
    assert EpisodeSpec.from_task(_portsim(*extra, **replace))._copy.steps[2].sandboxed


@pytest.mark.parametrize(("extra", "replace"), OUTSIDE_THE_SANDBOX.values(), ids=OUTSIDE_THE_SANDBOX.keys())
def test_a_task_using_only_the_outside_agents_envs_serves_its_proxy_here(extra, replace):
    assert not EpisodeSpec.from_task(_portsim(*extra, **replace))._copy.steps[2].sandboxed


AT_THE_AGENT = {"agent_name": "default-agent"}
REFUSED = {
    "no deploy_agent": ({}, {"agent": None}, "agent 'default-agent' is played from outside, so it needs exactly one "
                                             "deploy_agent or install_agent step, not 0"),
    "two deploy_agents": (({**AGENT, "id": "agent-2"},), {},
                          "needs exactly one deploy_agent or install_agent step, not 2"),
    "a deploy and an install": (({**INSTALL, "id": "install"},), {},
                                "needs exactly one deploy_agent or install_agent step, not 2"),
    "deploy_human_agent": (({"id": "human", "type": "deploy_human_agent", **AT_THE_AGENT},), {"agent": None},
                           "step 'human': deploy_human_agent deploys 'default-agent', which is played from outside"),
    "parts": ({}, {"play": {**PLAY, "prompt": None, "parts": [{"kind": "text", "text": "Plan."}]}},
              "step 'play': an outside agent's prompt can't have parts, only a text prompt"),
    "output_format": ({}, {"play": {**PLAY, "output_format": {"type": "json_schema", "schema": {}}}},
                      "step 'play': an outside agent's prompt can't have output_format"),
    "agentenv_tools": ({}, {"play": {**PLAY, "agentenv_tools": ["complete_step"]}},
                       "step 'play': an outside agent's prompt can't have agentenv_tools"),
    "snapshot_config": ({}, {"play": {**PLAY, "snapshot_config": {"env_id": "portsim"}}},
                        "step 'play': an outside agent's prompt can't have snapshot_config"),
    "snapshot files": ({}, {"agent": {**AGENT, "agent_snapshot_files_artifact_id": "files"}},
                       "step 'agent': an outside agent can't have agent_snapshot_files_artifact_id"),
    "changelog url": ({}, {"agent": {**AGENT, "agent_changelog_object_url": "file:///changelog"}},
                      "step 'agent': an outside agent can't have agent_changelog_object_url"),
    "changelog": ({}, {"agent": {**AGENT, "enable_agent_changelog": True}},
                  "step 'agent': an outside agent can't have enable_agent_changelog"),
    "register_agent_triggers": (({"id": "s", "type": "register_agent_triggers", "triggers": [], **AT_THE_AGENT},), {},
                                "step 's': register_agent_triggers needs 'default-agent' as a real agent"),
    "snapshot_agent_state": (({"id": "s", "type": "snapshot_agent_state", "artifact_id": "a", "prompt_id": "play"},),
                             {}, "step 's': snapshot_agent_state needs 'default-agent' as a real agent"),
    "collect a manifest": (({"id": "s", "type": "collect_artifacts", "env_id": "portsim", "manifest_step_id": "play"},),
                           {}, "step 's': its manifest is outside prompt 'play'"),
    "trigger executor": (({"id": "s", "type": "register_env_triggers", "env_id": "portsim", "triggers": [],
                           "executor_agent_name": "default-agent"},), {},
                         "step 's': its executor 'default-agent' is played from outside"),
    "peered": ((BOB, {"id": "s", "type": "peer_agents", "peerings": [
                   {"source_agent_name": "bob", "peer_agent_names": ["default-agent"]}]}), {},
               "step 's': peers 'default-agent', which is played from outside"),
    "rubric judged in the agent": (({"id": "s", "type": "rubrics_verifier", "prompt_id": "other", "criteria": [],
                                     "use_trajectory": False, **AT_THE_AGENT},), {},
                                   "step 's': judges in 'default-agent''s sandbox, and it is played from outside"),
    "retry_config": ({}, {"verify": {**VERIFY, "retry_config": {"retry_from_step_id": "verify"}}},
                     "step 'verify': retry_config can replay an outside turn"),
    "an A2A validator": (({"id": "s", "type": "verify_a2a_agent_card", "a2a_agent_id": "x"},), {},
                         "step 's': validates the task's deployed A2A agent, and an episode plays one from outside"),
}


@pytest.mark.parametrize(("extra", "replace", "message"), REFUSED.values(), ids=REFUSED.keys())
def test_a_step_an_outside_player_cant_stand_behind_is_refused_naming_it(extra, replace, message):
    refusal = _refusal(_portsim(*extra, **replace))

    assert refusal.startswith("task 'dock' can't run as an episode: ")
    assert message in refusal


def test_a_user_simulator_played_from_outside_is_refused():
    ask = {**ASK_BOB, "max_conversation_turns": 3, "user_agent_name": "default-agent"}

    assert "step 'ask-bob': its user simulator 'default-agent' is played from outside" in _refusal(
        _portsim(BOB, ask), trainer_agents=["default-agent"])


def test_every_refusal_is_reported_at_once():
    refusal = _refusal(_portsim(play={**PLAY, "agentenv_tools": ["complete_step"]},
                                agent={**AGENT, "enable_agent_changelog": True}))

    assert refusal == ("task 'dock' can't run as an episode: step 'agent': an outside agent can't have "
                       "enable_agent_changelog; step 'play': an outside agent's prompt can't have agentenv_tools")


ALLOWED = {
    "a reply verifier on an outside prompt": (
        {"verify": {"id": "graded", "type": "agent_prompt_response_verifier", "prompt_id": "play",
                    "criteria": [{"type": "response_contains", "needles": ["berths"]}]}},),
    "a rubric of the reply alone": (
        {"verify": {"id": "rubric", "type": "rubrics_verifier", "prompt_id": "play", "criteria": [],
                    "use_trajectory": False, "use_agent_judge": False}},),
    "collecting from an env": ({"collect": {"id": "collect", "type": "collect_artifacts", "env_id": "portsim",
                                            "artifact_paths": ["/out"]}},),
    "a rubric of the trajectory": (
        {"verify": {"id": "rubric", "type": "rubrics_verifier", "prompt_id": "play", "criteria": [],
                    "use_agent_judge": False}},),
    "installing the outside agent on a VM host": ({"agent": INSTALL},),
    "installing the outside agent into a container": ({"agent": {**INSTALL, "container_name": "task"}},),
    **{name: ({"agent": replace["agent"]} if replace else {"extra": extra[0]},)
       for name, (extra, replace) in IN_THE_SANDBOX.items() if name != "an install"},
    "tearing down an outside agent": ({"down": {"id": "down", "type": "teardown_sandboxes",
                                                "agent_names": ["default-agent"]}},),
}


@pytest.mark.parametrize("replace", [case[0] for case in ALLOWED.values()], ids=ALLOWED.keys())
def test_steps_that_need_only_what_an_outside_player_leaves_are_allowed(replace):
    EpisodeSpec.from_task(_portsim(**replace))

def test_every_built_in_step_type_is_checked_or_unaffected():
    registered, checked = set(_builtin_registry()), set(CHECKS)

    assert sorted(registered - checked - UNAFFECTED) == []
    assert sorted((checked | UNAFFECTED) - registered) == []
    assert sorted(checked & UNAFFECTED) == []


def test_the_seed_pass_rebuilds_each_step_as_its_own_class():
    class _Unregistered(PromptAgentTaskStep):
        pass

    steps = [*_portsim().steps, _Unregistered.from_dict({**PLAY, "id": "extra", "prompt": "Berth at <port>."})]

    seeded = substitute_seed_in_steps(steps, SEED)

    assert [type(step) for step in seeded] == [type(step) for step in steps]
    assert (seeded[2].role, seeded[3].prompt, seeded[-1].prompt) == (
        "planner@Valencia", "Plan week 3.", "Berth at Valencia.")
