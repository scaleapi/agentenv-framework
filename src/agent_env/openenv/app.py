"""``build_app``: the OpenEnv server for a bundle's tasks."""

from __future__ import annotations

import functools
import hmac
import logging
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path

from fastapi import FastAPI
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation

from agent_env.bundle import BundleError, BundleKind, PreparedBundle, parse_bundle, prepare_bundle
from agent_env.bundle.installed import locate_bundle
from agent_env.store.routing import namespace_routing
from agent_env.task import EpisodeError, EpisodeSpec, Task

from .environment import AgentEnvEnvironment, Served, ServedTask

logger = logging.getLogger(__name__)

_Source = str | Path


def build_app(
    bundle: _Source | Mapping[str, _Source],
    *,
    reward: str | Callable[[dict[str, float]], float] | None = None,
    sandbox: str | None = None,
    max_concurrent_envs: int = 8,
    token: str | None = None,
) -> FastAPI:
    """The OpenEnv server for ``bundle``'s tasks: a folder or an installed bundle's name, as ``agent-env run BUNDLE``
    reads it, served as the split ``all`` and a split per eval; or a mapping of split name to bundle. Each session
    plays one episode of a task at a time, deploying where ``sandbox`` names, as ``agent-env run --sandbox`` does.

    ``reward`` makes the reward of a task's scores: a step id, or a function of the scores by step id; without it a
    task that records one score has that one. Each bundle is prepared as ``agent-env run`` prepares it before its
    first task, without the agent a client plays, so every task must pass its checks; one that can't run as an
    episode is then left out and logged. Namespace routing is on from the app's startup to its shutdown, which tears
    down every open episode. With ``token``, every request and WebSocket but ``/health`` must carry
    ``Authorization: Bearer <token>``, for a server others can reach: without it, anyone who reaches the server
    deploys and drives its envs. Raises BundleError when a bundle can't be prepared or serves no task, and ValueError
    when ``reward`` can't score every task."""
    with namespace_routing():
        prepared = {split: _prepared(source, sandbox) for split, source in _sources(bundle).items()}
        served = Served(splits=_served(_splits(bundle, prepared), reward), sandbox=sandbox)
    return _app(served, _env_name(prepared), max_concurrent_envs, token)


def _app(served: Served, env_name: str, max_concurrent_envs: int, token: str | None = None) -> FastAPI:
    """OpenEnv's server for what ``served`` serves, with namespace routing on from its startup to its shutdown, which
    tears down every open episode, and ``token`` required of every request but ``/health`` when given."""
    app = create_app(functools.partial(AgentEnvEnvironment, served), CallToolAction, CallToolObservation,
                     env_name=env_name, max_concurrent_envs=max_concurrent_envs, reset_observation_cls=Observation)
    if token:
        app.add_middleware(_Bearer, token=token)
    routing = ExitStack()

    async def route() -> None:
        routing.enter_context(namespace_routing())

    async def tear_down() -> None:
        try:
            await served.close("the server shut down")
        finally:
            routing.close()

    app.router.on_startup.append(route)
    app.router.on_shutdown.append(tear_down)
    return app


class _Bearer:
    """Refuses every request and WebSocket handshake but ``/health``'s that doesn't carry ``Authorization: Bearer
    <token>``: a request with 401, a WebSocket by closing it before it opens, which its client sees as HTTP 403."""

    def __init__(self, app: ASGIApp, token: str):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and scope["path"] != "/health":
            given = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(given, self.expected):
                refusal = (JSONResponse({"detail": "this server needs Authorization: Bearer <its token>"}, 401)
                           if scope["type"] == "http" else WebSocketClose(code=1008))
                await refusal(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _sources(bundle: _Source | Mapping[str, _Source]) -> dict[str | None, _Source]:
    return dict(bundle) if isinstance(bundle, Mapping) else {None: bundle}


def _prepared(source: _Source, sandbox: str | None) -> PreparedBundle:
    """``source`` prepared with every task and eval selected, so ``all`` holds every task, evals or not."""
    root, id_root = (source, None) if isinstance(source, Path) else locate_bundle(source)
    entries = parse_bundle(Path(root), id_root=id_root).entries
    return prepare_bundle(
        root, tasks=[entry.name for entry in entries if entry.kind is BundleKind.TASK],
        evals=[entry.name for entry in entries if entry.kind is BundleKind.EVAL], sandbox=sandbox,
        on_progress=logger.info, id_root=id_root, played=True)


def _splits(bundle: _Source | Mapping[str, _Source],
            prepared: dict[str | None, PreparedBundle]) -> dict[str, list[tuple[str, Task]]]:
    if isinstance(bundle, Mapping):
        return {split: [(entry.name, task) for entry, task in each.tasks] for split, each in prepared.items()}
    [only] = prepared.values()
    if any(entry.entry.name == "all" for entry in only.plan.evals):
        raise BundleError(["eval 'all' has the name of the split that holds every task; rename it, or serve a "
                           "mapping of split name to bundle"])
    tasks = [(entry.name, task) for entry, task in only.tasks]
    named = {task.id: (name, task) for name, task in tasks}
    return {"all": tasks,
            **{entry.entry.name: [named[ref.id] for ref in entry.references] for entry in only.plan.evals}}


def _env_name(prepared: dict[str | None, PreparedBundle]) -> str:
    """The bundles' names, with no ``.``: OpenEnv's web interface reads the env's name as a module path."""
    return "+".join(dict.fromkeys(each.plan.bundle.bundle.name for each in prepared.values())).replace(".", "-")


def _served(splits: dict[str, list[tuple[str, Task]]],
            reward: str | Callable[[dict[str, float]], float] | None) -> dict[str, tuple[ServedTask, ...]]:
    refused: dict[str, str] = {}
    unscorable: dict[str, str] = {}
    served: dict[str, tuple[ServedTask, ...]] = {}
    for split, tasks in splits.items():
        kept = []
        for name, task in tasks:
            try:
                spec = EpisodeSpec.from_task(task)
            except EpisodeError as e:
                refused[task.id] = str(e)
                continue
            reward_of, problem = _reward_of(task, reward)
            if problem:
                unscorable[task.id] = problem
                continue
            kept.append(ServedTask(task, name, spec, reward_of))
        served[split] = tuple(kept)
    for why in refused.values():
        logger.warning("left out: %s", why)
    if unscorable:
        raise ValueError("reward= can't make a reward of every task's scores:\n" + "\n".join(unscorable.values()))
    if not any(served.values()):
        raise BundleError([f"no task can be served as an OpenEnv env: {why}" for why in refused.values()]
                          or ["no task can be served as an OpenEnv env: the bundle has none"])
    return served


def _reward_of(task: Task, reward: str | Callable[[dict[str, float]], float] | None
               ) -> tuple[Callable[[dict[str, float]], float | None] | None, str | None]:
    """``task``'s reward as a function of its scores, or why ``reward`` can't make one."""
    if callable(reward):
        return reward, None
    scoring = [step.id for step in task.steps if getattr(step, "verifier_id", None)]
    if reward is None:
        if len(scoring) > 1:
            return None, (f"task {task.id!r} records a score at each of {', '.join(scoring)}; name the one that is "
                          "its reward with reward=<step id>")
        return (_score_at(scoring[0]) if scoring else _unscored), None
    if reward not in scoring:
        return None, f"task {task.id!r} records no score at step {reward!r}; it records them at {scoring}"
    return _score_at(reward), None


def _score_at(step: str) -> Callable[[dict[str, float]], float]:
    def score(scores: dict[str, float]) -> float:
        if step not in scores:
            raise LookupError(f"the run recorded no score at step {step!r}")
        return scores[step]

    return score


def _unscored(scores: dict[str, float]) -> None:
    return None
