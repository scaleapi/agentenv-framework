"""Every step runs under a grant of each deployed env's control token: a protocol-client call the step makes to an env's
address carries that env's token unless the call names its own headers, an env without one is sent none, an env an
earlier step deployed is granted to the steps after it, and the grant ends with the step."""

import pytest
from agentenv_protocol import WELL_KNOWN_PATH
from agentenv_protocol import client as protocol_v1

from agent_env.config.runtime import Config
from agent_env.env.env import DeployedEnv
from agent_env.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep

DESK = "http://desk:18765"
MAIL = "http://mail:8000"


def _deployed(env_id: str, url: str, token: str | None = None) -> DeployedEnv:
    return DeployedEnv(env_id=env_id, env_version=1, mcp_url=f"{url}/mcp", environment_card_url=f"{url}{WELL_KNOWN_PATH}",
                       control_token=token)


class _Calls(TaskStep):
    """Records what its calls to the desk, to the mail env and elsewhere send, and one to the desk with headers of its
    own."""

    type = "control_grant_test_calls"
    entity_refs = ()
    seen: list = []

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        urls = (f"{DESK}/agentenv/ext/computer-exec/run", f"{MAIL}/agentenv", "http://elsewhere/agentenv")
        _Calls.seen.append([protocol_v1.request_headers(url) for url in urls]
                           + [protocol_v1.request_headers(f"{DESK}/agentenv", {"X-Own": "1"})])
        return context


class _DeployDesk(TaskStep):
    """Deploys the desk, with its token."""

    type = "control_grant_test_deploy"
    entity_refs = ()

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data))

    async def execute(self, context):
        context.deployed_envs.append(_deployed("desk", DESK, "s3cret"))
        return context


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Calls.type: _Calls, _DeployDesk.type: _DeployDesk}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    monkeypatch.setattr(_Calls, "seen", [])


@pytest.mark.asyncio
async def test_a_step_calls_each_deployed_env_with_its_own_token_and_the_grant_ends_with_the_step(local_stores):
    task = Task(id="t", version=None, steps=[_Calls(id="before", version=None), _DeployDesk(id="deploy", version=None),
                                              _Calls(id="after", version=None)])

    await task.run(context=TaskStepContext(deployed_envs=[_deployed("mail", MAIL)]))

    assert _Calls.seen == [[{}, {}, {}, {"X-Own": "1"}],
                           [{"Authorization": "Bearer s3cret"}, {}, {}, {"X-Own": "1"}]]
    assert protocol_v1.request_headers(f"{DESK}/agentenv") == {}
