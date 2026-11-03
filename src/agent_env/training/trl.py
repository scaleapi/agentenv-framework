"""TRL's trainers on a bundle served as an OpenEnv env (``agent_env.openenv.build_app``, or the package ``openenv import``
writes): ``AgentEnv`` reads the server's tasks into a dataset, and makes the ``environment_factory`` whose environments
each play one episode of a task over OpenEnv's WebSocket, offer its turn's tools as methods and own its reward.
``GRPOTrainer`` takes both as they are, verified with TRL 1.16 and 1.17, and AsyncGRPO the same ``environment_factory``;
either needs TRL 1.15 or later, for an environment that owns its reward. Unsloth's ``GRPOTrainer``, TRL's rewritten when
``unsloth`` is imported first, takes them too, verified with Unsloth 2026.10.11 on TRL 1.15. Every session runs on one
event-loop thread of this module's own, since TRL calls an environment synchronously, from its own threads."""

from __future__ import annotations

import asyncio
import copyreg
import json
import logging
import threading
import weakref
from collections.abc import Callable, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar

try:
    from datasets import Dataset
    from openenv.core.generic_client import GenericEnvClient  # noqa: F401
    from transformers.utils.chat_template_utils import render_jinja_template
except ModuleNotFoundError as exc:
    if exc.name not in ("datasets", "openenv", "transformers"):
        raise
    raise ImportError("agent_env.training.trl requires pip install 'agentenv-framework[trl]'") from exc

from agent_env.openenv import build_app
from agent_env.training._openenv import (
    CLOSE_SECONDS,
    CONNECT_SECONDS,
    END_TURN,
    GRACE_SECONDS,
    STOP_SECONDS,
    NamedToolset,
    Server,
    Session,
    Shared,
    key,
    messages,
    on_session_loop,
    probe,
    read_tasks,
    toolsets,
    why,
)
from agent_env.training._tools import SchemaTool, tool_method

logger = logging.getLogger(__name__)

#: How long the exit hook waits for the sessions to close.
EXIT_SECONDS = 30.0

_environments: weakref.WeakSet[_Environment] = weakref.WeakSet()
_servers: set[Server] = set()


class AgentEnv:
    """A server's tasks for TRL. ``dataset`` holds one row per task of ``split``, and ``factory`` is the
    ``environment_factory`` whose environments play them: each holds one session on the server, ``reset`` starts the
    row's task, its methods call the turn's tools, ``end_turn`` hands over the reply and answers with the next turn's
    prompt, and ``get_reward`` is the episode's reward, ``None`` when it ended in an error. TRL hands an environment
    no completion, so the model's reply reaches the task only through ``end_turn(response=...)``: a turn the model
    leaves open ends with an empty reply, which a verifier judging the reply scores as such. Tasks whose agents get
    different tools take TRL's multi-environment mode: ``factory`` is then a dict, and each row names its
    ``environment``.

    Args:
        url: The server's base URL.
        split: The split ``dataset`` holds.
        tool_timeout: One tool call's deadline in seconds. The server ends a call that misses it, and the episode with
            it, since what the call does may still be under way: the rollout is left unscored, and the session kept.
        reset_timeout: How long an episode may take to deploy and open its first turn, and each turn to be ended,
            in seconds. The server ends an episode that misses it, so the rollout is left unscored. A session dropped
            by a server restarting waits as long for it to come back.

    Example:

    ```python
    env = AgentEnv("http://localhost:8000")
    trainer = GRPOTrainer(
        model="Qwen/Qwen3-4B",
        args=GRPOConfig(max_tool_calling_iterations=12),
        train_dataset=env.dataset,
        environment_factory=env.factory,
    )
    ```
    """

    def __init__(self, url: str, split: str = "all", *, tool_timeout: float = 60.0, reset_timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.split = split
        self._tool_timeout, self._reset_timeout = tool_timeout, reset_timeout
        self._server: Server | None = None
        self._shared: list[Shared] = []
        self._name, self._tasks = read_tasks(self.url, local="train on AgentEnv.local(bundle)")
        self._toolsets = toolsets(self._name, self._tasks)
        self._split(split)

    @classmethod
    def local(cls, bundle: str | Path | Mapping[str, str | Path], split: str = "all", *,
              reward: str | Callable[[dict[str, float]], float] | None = None, sandbox: str | None = None,
              max_concurrent_envs: int = 8, tool_timeout: float = 60.0, reset_timeout: float = 600.0) -> AgentEnv:
        """``bundle`` served by ``build_app`` in this process on a loopback port, for a trainer on a machine that can
        deploy its envs. ``max_concurrent_envs`` must cover the rollouts the trainer generates at once. The server
        stops at ``close``, or when the process exits, ending its episodes."""
        server = Server(build_app(bundle, reward=reward, sandbox=sandbox, max_concurrent_envs=max_concurrent_envs))
        _servers.add(server)
        try:
            env = cls(server.url, split, tool_timeout=tool_timeout, reset_timeout=reset_timeout)
        except BaseException:
            _stop(server)
            raise
        env._server = server
        return env

    @cached_property
    def dataset(self) -> Dataset:
        return self.dataset_for(self.split)

    def dataset_for(self, split: str) -> Dataset:
        """``split``'s tasks as TRL's rows: the ``prompt`` as chat messages, the ``split`` and ``index`` that
        ``reset`` plays, the ``task_id`` it checks the episode against, and the ``environment`` in multi-environment
        mode. The prompt is the model's to change."""
        several = len(self._toolsets) > 1
        return Dataset.from_list([{
            "prompt": messages(task["system_prompt"], task["prompt"]), "split": split, "index": task["index"],
            "task_id": task["id"],
            **({"environment": self._toolsets[key(task)].name} if several else {}),
        } for task in self._split(split)])

    @cached_property
    def factory(self) -> type[_Environment] | dict[str, type[_Environment]]:
        """TRL's ``environment_factory``: an environment class per toolset of the server's tasks, or the one class when
        every task has the same. Building it walks one task of each toolset through its turns, in turn, ending each but
        the last empty and listing its tools, then ends the episode unscored, since TRL reads the tools from the
        class: its methods are every tool a turn offers, and one a turn doesn't offer answers so. Each call on a
        probe's session has the reset deadline."""
        classes = on_session_loop(self._environments(), None)
        return classes[0] if len(classes) == 1 else {each.__name__: each for each in classes}

    def close(self) -> None:
        """Ends the episodes this env's environments hold, closing their sessions, and stops the server ``local``
        started."""
        mine = [environment for environment in list(_environments) if environment._shared in self._shared]
        try:
            on_session_loop(_gathered(environment._close() for environment in mine), CLOSE_SECONDS)
        finally:
            if self._server is not None:
                _stop(self._server)

    async def _environments(self) -> list[type[_Environment]]:
        return [await self._environment(toolset) for toolset in self._toolsets.values()]

    async def _environment(self, toolset: NamedToolset) -> type[_Environment]:
        first, tools = await probe(self.url, toolset, self._reset_timeout)
        shared = Shared(url=self.url, offered=first, tool_timeout=self._tool_timeout, reset_timeout=self._reset_timeout,
                        waits_for_server=True)
        self._shared.append(shared)
        return _environment_class(toolset.name, tools, shared)

    def _split(self, split: str) -> list[dict[str, Any]]:
        if split not in self._tasks:
            raise ValueError(f"the server at {self.url} has no split {split!r}; it serves {', '.join(self._tasks)}")
        return self._tasks[split]


class _Environment(Session):
    """One session on the server, playing one episode at a time. Its public methods other than ``reset`` and
    ``get_reward`` are the tools, which ``AgentEnv.factory`` binds per toolset, and each but ``end_turn`` is set over
    with its ``SchemaTool``; every call on the session waits its turn, as AsyncGRPO can call two tools of a completion
    at once."""

    _shared: ClassVar[Shared]
    _tools: ClassVar[tuple[dict[str, Any], ...]]
    offered_by = "its toolset's first task does"
    opens_at_start = True

    def __init__(self) -> None:
        super().__init__()
        for tool in self._tools:
            if tool["name"] != END_TURN:
                setattr(self, tool["name"], SchemaTool(tool, getattr(self, tool["name"])))
        _environments.add(self)

    def reset(self, *, split: str, index: int, task_id: str | None = None, **_: Any) -> None:
        """Starts the row's episode and returns before it is deployed, since ``GRPOTrainer`` resets its environments
        one at a time: the first tool call or ``get_reward`` waits for it. Raises when the server refuses this
        environment's first session, as one at capacity does; a later session it refuses, after one was dropped,
        leaves only that rollout unscored, unless the server is restarting, which it waits out within the reset
        deadline."""
        on_session_loop(self._start(split, index, task_id), CONNECT_SECONDS + CLOSE_SECONDS)

    def get_reward(self) -> float | None:
        """The episode's reward, or ``None`` when it ended in an error. Raises when the episode's first turn offered a
        tool its toolset's first task doesn't, which no rollout fixes."""
        rollout, self._rollout = self._rollout, None
        if rollout is None:
            return None
        try:
            on_session_loop(self._finish(rollout), None)
        except Exception as e:
            rollout.failure = rollout.failure or why(e)
        if rollout.misgrouped:
            raise RuntimeError(f"{rollout.failure}. A task must offer no tool its toolset's first task doesn't, since TRL shows "
                               "the model a class's: their toolset misses what tells them apart")
        if rollout.failure is not None:
            logger.warning("rollout left unscored: %s", rollout.failure)
            return None
        return rollout.reward

    def _call(self, name: str, arguments: dict[str, Any]) -> str:
        """Waits at most for a restarting server, the deploy and the call, each to its deadline and a grace after, then
        the session's close: the first call of a rollout waits for its deploy, after its session reopens."""
        shared = self._shared
        reopen = max(shared.patience, shared.reset_timeout)
        waits = reopen + shared.reset_timeout + max(shared.reset_timeout, shared.tool_timeout)
        return on_session_loop(self._step(name, arguments), waits + 2 * GRACE_SECONDS + CLOSE_SECONDS)


class _EnvironmentClass(type):
    """The type of ``AgentEnv.factory``'s classes, which pickle as the toolset and server they're built from, since
    AsyncGRPO builds its environments in a process of its own."""


def _environment_class(name: str, tools: tuple[dict[str, Any], ...], shared: Shared) -> type[_Environment]:
    methods = {tool["name"]: tool_method(tool) for tool in tools}
    for method in methods.values():
        method.__qualname__ = f"{name}.{method.__name__}"
    _check_rendered([SchemaTool(tool, methods[tool["name"]]) for tool in tools if tool["name"] != END_TURN])
    return _EnvironmentClass(name, (_Environment,), {**methods, "_shared": shared, "_tools": tools})


def _check_rendered(tools: list[SchemaTool]) -> None:
    """Fails as the trainer sets up, not as the model meets the tools, when transformers doesn't render a tool given
    as its JSON Schema as it is."""
    [rendered], _ = render_jinja_template([[]], tools=tools, chat_template="{{ tools | tojson }}")
    if changed := [tool.__name__ for tool, shown in zip(tools, json.loads(rendered), strict=True) if shown != tool]:
        raise RuntimeError(f"this transformers renders the tools {changed} other than as their env's JSON Schema, so "
                           "TRL would show the model other tools than the env's")


def _reduce_environment_class(environment: _EnvironmentClass) -> tuple[Any, ...] | str:
    """A class ``_environment_class`` built pickles as what it's built from, and a subclass of one by its name, as any
    class does, so it keeps what it overrides."""
    if "_tools" not in vars(environment):
        return environment.__qualname__
    return _environment_class, (environment.__name__, environment._tools, environment._shared)


copyreg.pickle(_EnvironmentClass, _reduce_environment_class)


async def _gathered(coroutines: Any) -> list[Any]:
    return list(await asyncio.gather(*coroutines))


def _close_all() -> None:
    """A trainer that stops or crashes ends the episodes it left running, since TRL never closes its environments,
    then stops the servers ``AgentEnv.local`` started, which tear down their episodes, waiting at most
    ``EXIT_SECONDS`` for the sessions and ``STOP_SECONDS`` for each server. It runs before threading's shutdown, while
    the thread pools an episode's teardown uses still take work."""
    try:
        on_session_loop(_gathered(environment._close() for environment in list(_environments)), EXIT_SECONDS)
    except Exception as e:
        logger.warning("couldn't end the episodes left running: %s", why(e))
    for server in list(_servers):
        _stop(server)


def _stop(server: Server) -> None:
    _servers.discard(server)
    if not server.stop():
        logger.warning("the bundle's server at %s was still tearing down its episodes after %gs; their sandboxes "
                       "may be left running", server.url, STOP_SECONDS)


threading._register_atexit(_close_all)




__all__ = ["AgentEnv"]
