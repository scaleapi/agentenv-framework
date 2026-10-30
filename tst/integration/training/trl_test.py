"""``AgentEnv`` on bundles served by ``AgentEnv.local`` on local Docker, used as TRL's trainers use it: rollouts on one
environment score as the task's verifier scores them, a second turn's prompt is handed over, a task whose agent works
in its sandbox takes a toolset with ``bash`` of its own, and nothing is left running once the env is closed.

Requires a docker daemon; reuses the items env, verifiers and bundle writer of the OpenEnv server tests."""

import inspect
import json
import uuid

import pytest

from agent_env.artifact import FileArtifact
from agent_env.training._openenv import ENDED
from agent_env.training.trl import AgentEnv
from tst.integration.openenv.build_app_test import _left

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]


def _tools(environment):
    return {name for name, _ in inspect.getmembers(environment, inspect.isfunction) if not name.startswith("_")}


def test_rollouts_on_one_environment_score_as_the_tasks_verifier_scores_them(items):
    env = AgentEnv.local(items.bundle("dock", {"crate": items(), "twice": items(turns=2)}), max_concurrent_envs=2)
    try:
        crate, twice = env.dataset.to_list()
        environment = env.factory()
        assert {"items_add_item", "items_whoami", "end_turn"} <= _tools(env.factory)

        environment.reset(**crate)
        assert json.loads(environment.items_add_item(item="crate")) == {"count": 1}
        assert environment.end_turn(response="Added a crate.") == ENDED
        assert environment.get_reward() == 1.0
        environment.reset(**crate)
        assert environment.get_reward() == 0.0
        environment.reset(**twice)
        environment.items_add_item(item="crate")
        assert environment.end_turn(response="Added a crate.") == "Add another."
        assert environment.end_turn(response="Added another.") == ENDED
        assert environment.get_reward() == 1.0
    finally:
        env.close()

    assert _left(items.sandboxes) == []


def test_a_task_whose_agent_works_in_its_sandbox_takes_a_toolset_with_bash(items):
    uid = uuid.uuid4().hex[:8]
    greeting = FileArtifact.put_bytes(id=f"trl-greeting-{uid}", description="trl e2e", filename="greeting.txt",
                                      content=b"hello", content_type="text/plain")
    env_step, policy, play, _ = items()
    copy = [env_step, policy, {"id": "load", "type": "load_artifact", "agent_name": "policy", "artifact_id": greeting.id,
                               "destination_path": "/app/inputs"}, play,
            {"id": "copied", "type": "verify_sandbox", "agent_name": "policy", "verifier_id": "copied", "criteria": [
                {"type": "probe_file_contains", "criterion": "the greeting is copied", "paths": ["outputs/answer.txt"],
                 "expected": "^hello$"}]}]

    env = AgentEnv.local(items.bundle("desk", {"copy": copy, "crate": items()}))
    try:
        plain, sandboxed = env_step["env_id"], env_step["env_id"] + "+bash"
        assert {name: "bash" in _tools(each) for name, each in env.factory.items()} == {sandboxed: True, plain: False}
        row = next(row for row in env.dataset if row["environment"] == sandboxed)
        environment = env.factory[sandboxed]()
        environment.reset(**row)
        assert environment.bash(command="cat inputs/greeting.txt") == "hello"
        environment.bash(command="mkdir -p outputs && cp inputs/greeting.txt outputs/answer.txt")
        environment.end_turn(response="Done.")
        assert environment.get_reward() == 1.0
    finally:
        env.close()

    assert _left(items.sandboxes) == []
