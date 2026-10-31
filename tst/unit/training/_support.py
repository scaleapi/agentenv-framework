"""The fake bundle both trainer helpers' tests serve: the Episode API's fake steps as an OpenEnv env on loopback. Its
fixture takes the ``teardowns`` and ``tools`` fixtures from the conftest that imports it."""

import pytest

from agent_env.openenv.app import _app, _score_at
from agent_env.openenv.environment import Served, ServedTask
from agent_env.task import EpisodeSpec, Task
from agent_env.training._openenv import Server


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
