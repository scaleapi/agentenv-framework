"""Writes NeMo Gym's resources server for a bundle's tasks into a Gym root: the root ``NEMO_GYM_EXTRA_ROOTS`` names, or
a Gym checkout's own, as NeMo RL's is."""

from __future__ import annotations

import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from agent_env.openenv import build_app
from agent_env.training._openenv import Server, key, messages, tasks_and_tools

#: The routes a resources server serves itself, which a tool's route can't take.
GYM_ROUTES = frozenset({"seed_session", "verify", "close_session", "aggregate_metrics", "reverify_mode"})
APP = """from agent_env.training.nemo_gym.server import AgentEnvResourcesServer

if __name__ == "__main__":
    AgentEnvResourcesServer.run_webserver()
"""
#: The server's project. Gym installs it into the server's venv, with nemo-gym from the checkout at ``../..`` when the
#: root is one; it is a project rather than a requirements file because uv applies the ``[tool.uv]`` settings of the
#: nearest ``pyproject.toml``, and a Gym checkout's own exclude packages the framework imports from the venvs of the
#: servers under it.
PYPROJECT = """[project]
name = "{name}-resources-server"
version = "0.0.0"
dependencies = ["nemo-gym", "agentenv-framework[nemo-gym]=={version}"]

[build-system]
build-backend = "setuptools.build_meta"
requires = ["setuptools>=61"]

[tool.setuptools]
packages = []

[tool.uv.sources]
nemo-gym = {{ path = "../..", editable = true }}
"""

REWARD_AND_COMPONENTS = ("reward_components makes the reward the sum of the task's scores, as NeMo RL requires, so "
                         "it takes no reward: weight the scores in GDPO's reward_weights instead")
_ROUTE = re.compile(r"[A-Za-z0-9._~-]+")


def write_server(source: str | Path, root: str | Path, *, name: str | None = None, sandbox: str | None = None,
                 reward: str | None = None, reward_components: bool = False, max_concurrent_envs: int = 8,
                 reset_timeout: float = 600.0) -> Path:
    """``source``'s tasks as NeMo Gym's resources server ``name``, the bundle's env name by default, under the Gym root
    ``root``, returning the server's folder ``resources_servers/<name>``. It holds the server's ``app.py`` and
    ``pyproject.toml``; ``configs/<name>.yaml``, which runs the server, ``simple_agent`` to play its rows and the
    ``legacy_agent`` environment server NeMo RL dispatches through; and ``data/<split>.jsonl``, a row per task of each
    split.

    ``source`` is the URL of an OpenEnv server serving the bundle, which the config names, or a bundle, a folder or an
    installed bundle's name, which the resources server serves itself, deploying where ``sandbox`` names, at most
    ``max_concurrent_envs`` episodes at once. ``reward_components`` has the server answer verify with each of a task's
    scores, for NeMo RL's GDPO, their sum the reward, so it takes no ``reward``. A row's tools are every tool its
    task's turns offer, found by playing one task of each toolset to its last turn, unscored, so writing deploys
    those."""
    if reward is not None and reward_components:
        raise ValueError(REWARD_AND_COMPONENTS)
    url = str(source) if str(source).startswith(("http://", "https://")) else None
    served = None if url else Server(build_app(source, reward=summed if reward_components else reward, sandbox=sandbox,
                                               max_concurrent_envs=1))
    try:
        env_name, tasks, probed = tasks_and_tools(url or served.url, local="name the bundle's folder instead of its "
                                                  "server", timeout=reset_timeout)
        tools = {each: [_function(tool) for tool in found] for each, found in probed.items()}
    finally:
        if served is not None:
            served.stop()
    name = name or re.sub(r"\W+", "_", env_name).strip("_").lower()
    if url:
        where: dict[str, Any] = {"url": url}
    else:
        folder = Path(source)
        where = {"bundle": str(folder.resolve()) if folder.exists() else str(source),
                 "max_concurrent_envs": max_concurrent_envs,
                 **({"sandbox": sandbox} if sandbox else {}), **({"reward": reward} if reward else {})}
    if reward_components:
        where["reward_components"] = True
    server = Path(root) / "resources_servers" / name
    (server / "configs").mkdir(parents=True, exist_ok=True)
    (server / "data").mkdir(exist_ok=True)
    (server / "app.py").write_text(APP)
    (server / "pyproject.toml").write_text(PYPROJECT.format(name=name.replace("_", "-"),
                                                            version=version("agentenv-framework")))
    (server / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(_config(name, env_name, where), sort_keys=False))
    agent = {"type": "responses_api_agents", "name": f"{name}_simple_agent"}
    for split, described in tasks.items():
        (server / "data" / f"{split}.jsonl").write_text("".join(json.dumps({
            "responses_create_params": {"input": messages(task["system_prompt"], task["prompt"]),
                                        "tools": tools[key(task)]},
            "agent_ref": agent, "split": split, "index": task["index"], "task_id": task["id"],
        }) + "\n" for task in described))
    return server


def summed(scores: dict[str, float]) -> float:
    """The reward of a task whose scores are its components: their sum, as NeMo RL requires."""
    return sum(scores.values())


def _function(tool: dict[str, Any]) -> dict[str, Any]:
    """An MCP tool as a function tool of the Responses API, which simple_agent calls at the route of its name."""
    name = tool["name"]
    if name in GYM_ROUTES or not _ROUTE.fullmatch(name):
        raise ValueError(f"the env's tool {name!r} can't be a NeMo Gym tool: Gym calls a tool at the route of its name, "
                         f"so a tool's name must be one path segment other than {', '.join(sorted(GYM_ROUTES))}. "
                         "Rename the tool in the env")
    return {"type": "function", "name": name, "description": tool.get("description") or "",
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}}, "strict": False}


def _config(name: str, env_name: str, where: dict[str, Any]) -> dict[str, Any]:
    agent = f"{name}_simple_agent"
    return {
        name: {"resources_servers": {name: {
            "entrypoint": "app.py", "domain": "agent", "description": f"{env_name}'s tasks, served by AgentEnv",
            **where,
        }}},
        agent: {"responses_api_agents": {"simple_agent": {
            "entrypoint": "app.py", "resources_server": {"type": "resources_servers", "name": name},
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
        }}},
        f"{name}_environment_server": {"environment_servers": {"legacy_agent": {
            "entrypoint": "app.py", "agent_server": {"type": "responses_api_agents", "name": agent},
        }}},
    }
