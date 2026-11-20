"""What the trainer helpers' tests share: the fake bundle they serve, the Episode API's fake steps as an OpenEnv env on
loopback, whose fixture takes the ``teardowns`` and ``tools`` fixtures from the conftest that imports it, and the tasks
and model replies they play it with."""

import pytest

from agent_env.config.runtime import Config
from agent_env.openenv.app import _app
from agent_env.openenv.reward import _score_at
from agent_env.openenv.environment import Served, ServedTask
from agent_env.task import EpisodeSpec, Task
from agent_env.task_step.task_step import TaskStep
from agent_env.training._openenv import Server
from tst.unit.task.episode.run_test import ALICE, ITEMS, REPLIES, TURN, _Env

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


class _Config(TaskStep):
    """Stands in for ``apply_server_config`` on ``env_id``, whose reply reveals the tools ``reveal``."""

    type = "apply_server_config"
    entity_refs = ()
    revealed: list[str] = []
    applied = 0

    def __init__(self, id, version, env_id="items", reveal=(), depends_on=None, fail_task_on_error=True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.env_id, self.reveal = env_id, list(reveal)

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), env_id=data.get("env_id", "items"), reveal=data.get("reveal", ()))

    async def execute(self, context):
        _Config.revealed[:] = self.reveal
        _Config.applied += 1
        return context


class _Offered:
    def __iter__(self):
        return iter(["add_item", *_Config.revealed])


@pytest.fixture
def reveals(monkeypatch, tools):
    """A config step whose reply reveals ``reveal``, given ``args``; each episode starts with nothing revealed, and
    ``reveals.applied()`` counts the config steps run."""
    registry = Config().task_step_registry()
    monkeypatch.setattr(Config, "task_step_registry", lambda self: {**registry, _Config.type: _Config})
    deploy = _Env.execute

    async def deploying(self, context):
        _Config.revealed.clear()
        return await deploy(self, context)

    monkeypatch.setattr(_Env, "execute", deploying)
    monkeypatch.setattr(_Config, "revealed", [])
    monkeypatch.setattr(_Config, "applied", 0)
    tools().offered = _Offered()

    def config(*reveal, **args):
        return {"id": "cfg", "type": _Config.type, "env_id": "items", "reveal": list(reveal),
                "directives": [{"service": "items", "uri": "urn:example:set-errors/v1", "args": args}]}

    config.applied = lambda: _Config.applied
    return config


def revealing(config, *reveal, **args):
    return [ITEMS, ALICE, config(*reveal, **args), TURN, REPLIES]
