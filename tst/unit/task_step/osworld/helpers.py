"""A computer env as the ``osworld_*`` steps see one."""
from __future__ import annotations

from agentenv_protocol.types import WELL_KNOWN_PATH

from agent_env.env.env import DeployedEnv
from agent_env.task_step.task_steps.computer.osworld import GUEST_EXTENSION

ENV_URL = "https://computer.example"
TOKEN = "s3cret-control-token"


def card(endpoint: str = "/svc/osworld-server/osworld", *, uri: str = GUEST_EXTENSION) -> dict:
    """A computer's card with one app, which advertises ``uri`` at ``endpoint``."""
    app = {"name": "osworld-server", "url": "/svc/osworld-server/agentenv",
           "capabilities": {"extensions": [{"uri": uri, "params": {"endpoint": endpoint,
                                                                   "methods": {"guest": {"method": "GET"}}}}]}}
    return {"name": "desk", "capabilities": {"extensions": []}, "children_environments": [app]}


def computer(card_: dict | None = None, **metadata) -> DeployedEnv:
    return DeployedEnv(env_id="desk", env_version=1, environment_card_url=f"{ENV_URL}{WELL_KNOWN_PATH}",
                       environment_card=card() if card_ is None else card_, control_token=TOKEN,
                       metadata=dict(metadata))

