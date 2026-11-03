"""What the trainer helpers' tests share: the fake bundle they serve, the Episode API's fake steps as an OpenEnv env on
loopback, whose fixture takes the ``teardowns`` and ``tools`` fixtures from the conftest that imports it, and the tasks
and model replies they play it with."""

import pytest

from agent_env.openenv.app import _app
from agent_env.openenv.reward import _score_at
from agent_env.openenv.environment import Served, ServedTask
from agent_env.task import EpisodeSpec, Task
from agent_env.training._openenv import Server
from tst.unit.task.episode.run_test import ALICE, ITEMS, REPLIES, TURN

#: A task the model plays by adding a crate and saying so, which its verifier scores 1.
PLAYED = [ITEMS, ALICE, TURN, REPLIES]
#: The same task with an env that fails once the model's turn has ended.
FAILS_AFTER_THE_TURN = [ITEMS, ALICE, TURN,
                        {"id": "crane", "type": "episode_test_env", "env_id": "crane", "fail": True}, REPLIES]
#: The model's call that adds the crate, in Qwen's tool-call format.
ADDS_A_CRATE = '<tool_call>\n{"name": "add_item", "arguments": {"item": "crate"}}\n</tool_call>'


@pytest.fixture
def serve(teardowns, tools):
    """Serves ``tasks`` by name, every one in the split ``all`` and those ``splits`` name in theirs, as ``build_app``
    serves a bundle's under the env name ``name``, each rewarded by ``reward`` of its scores, the ``replies`` step's by
    default, asking for ``token`` when given; each server is stopped after the test, if the test hasn't stopped it to
    wait out its episodes' teardowns."""
    servers = []

    def served(tasks, splits=None, max_concurrent_envs=8, name="dock", reward=_score_at("replies"), token=None):
        by_name = {}
        for task_name, steps in tasks.items():
            task = Task.from_dict({"id": task_name, "version": 1, "steps": steps})
            by_name[task_name] = ServedTask(task, task_name, EpisodeSpec.from_task(task), reward)
        named = {split: tuple(by_name[each] for each in names) for split, names in (splits or {}).items()}
        servers.append(Server(_app(Served(splits={"all": tuple(by_name.values()), **named}, sandbox=None), name,
                                   max_concurrent_envs, token)))
        return servers[-1]

    yield served
    for server in servers:
        server.stop()
