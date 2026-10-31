"""What the trainer helpers share: a session on an OpenEnv server serving a bundle (``agent_env.openenv.build_app``, or
the package ``openenv import`` writes) that plays one episode of a task at a time, the server's tasks as its Task API
describes them, and a server run on a thread of this process."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx
import uvicorn
from openenv.core.generic_client import GenericEnvClient

from agent_env.openenv.environment import END_TURN as END_TURN_TOOL
from agent_env.openenv.environment import TIMEOUT_METADATA, Toolset
from agent_env.task.teardown import TERMINATE_TIMEOUT_SECONDS

FAILED = "The environment failed to answer, so this episode won't be scored."
ENDED = "The episode has ended."
END_TURN = END_TURN_TOOL.name
CONNECT_SECONDS = 60.0
CLOSE_SECONDS = 300.0
#: How much longer than a call's deadline the client waits for the server to end the call, before dropping the session.
GRACE_SECONDS = 30.0
#: How long a server run in this process is given to stop: its episodes' teardowns run at once, and each saves the
#: agents' changelogs, ends open recordings and terminates the sandboxes, every part within
#: ``TERMINATE_TIMEOUT_SECONDS``.
STOP_SECONDS = 3 * TERMINATE_TIMEOUT_SECONDS + 60.0
#: How long, and how far apart in seconds, a session tries to open while the server is at its limit, which it stays at
#: until it has seen a closed session go.
REOPEN_PATIENCE, REOPEN_SECONDS = 4.0, 1.0
#: What a session's opening fails with when the server is too busy to answer in time, or drops the connection.
_TRANSIENT = (TimeoutError, ConnectionResetError, ConnectionAbortedError, BrokenPipeError, EOFError,
              httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError)

_CAPACITY = re.compile(r"Server at capacity: \d+/(\d+)")
_FILE = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")

logger = logging.getLogger(__name__)
_waited: set[str] = set()


@dataclass(frozen=True)
class NamedToolset:
    name: str
    first: tuple[str, int, int]  # the split, index and turn count of its first task


@dataclass(frozen=True)
class Shared:
    """What a session needs of its server: where it is, the tools the model is shown, which an episode's first turn
    must offer no more than, the deadlines, how long to wait for room on the server, whether to wait as long for a
    server that refuses connections, as one restarting does, and the token it asks for. The key makes it equal to its
    pickled copies only."""

    url: str
    offered: frozenset[str]
    tool_timeout: float
    reset_timeout: float
    patience: float = REOPEN_PATIENCE
    waits_for_server: bool = False
    token: str | None = field(default=None, repr=False)
    key: str = field(default_factory=lambda: uuid.uuid4().hex)


class Failed(Exception):
    pass


class AtCapacity(RuntimeError):
    def __init__(self, url: str, limit: str):
        super().__init__(f"the server at {url} is at its limit of {limit} sessions at once, and the trainer holds one "
                         "per rollout it generates at once: serve the bundle with MAX_CONCURRENT_ENVS (or "
                         "max_concurrent_envs=) at least that many, or generate fewer rollouts at once")
        self.limit = limit


class Misgrouped(Failed):
    """An episode offering a tool the model isn't shown: a toolset that misses what tells tasks apart, which no rollout
    fixes."""


@dataclass
class Rollout:
    started: asyncio.Task[None] = field(init=False)
    system: str | None = None
    done: bool = False
    reward: float | None = None
    scores: dict[str, float] | None = None
    failure: str | None = None
    misgrouped: bool = False


class Session:
    """One session on the server, playing one episode at a time: ``_start`` starts the row's episode and returns before
    it is deployed, ``_step`` calls a tool once it is, and ``_finish`` ends the turns left open until it is scored.
    Every call on the session waits its turn, as a trainer can call two tools of a completion at once."""

    _shared: Shared
    #: What ``_shared.offered`` is, for the error an episode offering other tools raises.
    offered_by: ClassVar[str]
    #: Whether the session opens at its first ``_start``, raising when the server refuses it, rather than as the
    #: episode starts, which leaves the rollout unscored.
    opens_at_start: ClassVar[bool] = False

    def __init__(self) -> None:
        self._client: GenericEnvClient | None = None
        self._opened = False
        self._rollout: Rollout | None = None
        self._lock = asyncio.Lock()

    async def _start(self, split: str, index: int, expected: str | None) -> None:
        """Starts the row's episode in the background, ending the one under way, and returns before it is deployed."""
        async with self._lock:
            if (rollout := self._rollout) is not None:
                rollout.started.cancel()
                await asyncio.wait([rollout.started])
            self._rollout = None
            if self.opens_at_start and not self._opened:
                self._client, self._opened = await opened(self._shared.url, self._shared.reset_timeout,
                                                          token=self._shared.token), True
            rollout = Rollout()
            rollout.started = asyncio.ensure_future(self._begin(rollout, split, index, expected))
            self._rollout = rollout

    async def _begin(self, rollout: Rollout, split: str, index: int, expected: str | None) -> None:
        if self._client is None:
            self._client = await reopened(self._shared.url, self._shared.reset_timeout, dropped=self._opened,
                                          patience=self._shared.patience, waits_for_server=self._shared.waits_for_server,
                                          token=self._shared.token)
            self._opened = True
        reset = await self._client.reset(split=split, index=index, timeout_s=self._shared.reset_timeout)
        if reset.done:
            raise Failed(error_of(reset.observation))
        metadata = reset.observation.get("metadata") or {}
        rollout.system, played = metadata.get("system_prompt") or None, (metadata.get("task") or {}).get("id")
        if expected is not None and played != expected:
            raise Failed(f"the episode plays task {played!r}, not the row's {expected!r}")
        listed = await list_tools(self._client)
        if listed.done:
            raise Failed(error_of(listed.observation))
        if not (offered := frozenset(tool["name"] for tool in listed.observation["tools"])) <= self._shared.offered:
            raise Misgrouped(f"task {played!r} offers the tools {sorted(offered)}, not the "
                             f"{sorted(self._shared.offered)} {self.offered_by}")

    async def _step(self, name: str, arguments: dict[str, Any]) -> str:
        async with self._lock:
            rollout = self._rollout
            if rollout is None or rollout.failure is not None:
                return FAILED
            if rollout.done:
                return ENDED
            try:
                await rollout.started
                deadline = self._shared.reset_timeout if name == END_TURN else self._shared.tool_timeout
                result = await asyncio.wait_for(call_tool(self._client, name, arguments, deadline),
                                                deadline + GRACE_SECONDS)
            except Failed as e:
                rollout.misgrouped = isinstance(e, Misgrouped)
                await self._fail(rollout, str(e))
                return FAILED
            except TimeoutError:
                await self._fail(rollout, f"{name} didn't answer within {deadline:g}s, and the server didn't end "
                                          f"the call in the {GRACE_SECONDS:g}s after")
                return FAILED
            except Exception as e:
                await self._fail(rollout, f"{name}: {why(e)}")
                return FAILED
            return self._answer(rollout, name, result)

    async def _finish(self, rollout: Rollout, response: str = "") -> None:
        """Ends the turns the model left open, the first with ``response`` and any after it empty, until the episode
        is scored; each call on the session has the reset's deadline, as ``end_turn`` does, since the server answers it
        once the next turn opens or the verifiers have scored the episode."""
        async with self._lock:
            try:
                await rollout.started
                while not (rollout.done or rollout.failure):
                    result = await end_turn(self._client, response, self._shared.reset_timeout)
                    response = ""
                    if error := result.observation.get("error"):
                        rollout.failure = f"{END_TURN}: {error.get('message')}"
                    self._answer(rollout, END_TURN, result)
            except Exception as e:
                rollout.misgrouped = isinstance(e, Misgrouped)
                await self._fail(rollout, str(e) if isinstance(e, Failed) else why(e))

    def _answer(self, rollout: Rollout, name: str, result: Any) -> str:
        """What the model sees of a tool call: the tool's text or error, the next turn's prompt for ``end_turn``, after
        its system prompt when that changed, since a trainer can't change the conversation's own; or that the episode
        ended, which records its reward or why it has none."""
        observation = result.observation
        if error := observation.get("error"):
            return error.get("message") or json.dumps(error)
        if result.done:
            rollout.done = True
            if (observation.get("metadata") or {}).get("status") == "error":
                rollout.failure = error_of(observation)
                return FAILED
            rollout.reward, rollout.scores = result.reward, (observation.get("metadata") or {}).get("scores")
            return ENDED
        answer = observation.get("result") or {}
        if name == END_TURN:
            prompt, system = answer.get("prompt") or "", answer.get("system_prompt") or None
            if system is None or system == rollout.system:
                return prompt
            rollout.system = system
            return f"{system}\n\n{prompt}"
        return answer["data"] if isinstance(answer.get("data"), str) else json.dumps(answer)

    async def _fail(self, rollout: Rollout, why: str) -> None:
        """Records why the rollout won't be scored and drops the session, which may be out of step with the server; the
        server ends a dropped session's episode, and the next rollout opens another."""
        rollout.failure = rollout.failure or why
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()

    async def _close(self) -> None:
        rollout, self._rollout = self._rollout, None
        if rollout is not None:
            rollout.started.cancel()
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()


class Server:
    """An app under uvicorn on a thread of its own, its lifespan included, on a free loopback port."""

    def __init__(self, app: Any):
        self._uvicorn = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        self._thread = threading.Thread(target=self._uvicorn.run, name="agentenv-openenv-server", daemon=True)
        self._thread.start()
        while not self._uvicorn.started:
            if not self._thread.is_alive():
                raise RuntimeError("the bundle's server didn't start; its log says why")
            time.sleep(0.05)
        self.url = f"http://127.0.0.1:{self._uvicorn.servers[0].sockets[0].getsockname()[1]}"

    def stop(self, timeout: float = STOP_SECONDS) -> bool:
        """Whether the server stopped within ``timeout``, tearing down its episodes."""
        self._uvicorn.should_exit = True
        self._thread.join(timeout)
        return not self._thread.is_alive()


async def opened(url: str, timeout: float, *, token: str | None = None) -> GenericEnvClient:
    """A session on the server, opened now rather than at the first reset, so a server at capacity refuses it at once;
    each message waits ``GRACE_SECONDS`` longer than the ``timeout`` the server ends a call at. ``token`` goes with it
    as ``Authorization: Bearer``, for a server that asks for one."""
    client = GenericEnvClient(base_url=url, message_timeout_s=timeout + GRACE_SECONDS, headers=authorization(token))
    try:
        await client.state()
    except Exception as e:
        with contextlib.suppress(Exception):
            await client.close()
        if match := _CAPACITY.search(str(e)):
            raise AtCapacity(url, match[1]) from e
        raise ConnectionError(f"couldn't open a session on the server at {url}: {why(e)}") from e
    return client


async def reopened(url: str, timeout: float, *, dropped: bool, patience: float, waits_for_server: bool = False,
                   token: str | None = None) -> GenericEnvClient:
    """A session, tried again every ``REOPEN_SECONDS`` for ``patience`` seconds while the server is at its limit or too
    busy to answer, or with ``waits_for_server`` refuses the connection, as one restarting does, in place of a
    ``dropped`` one or for a rollout's own; raises Failed, which leaves the rollout unscored, when it can't open one, at
    once unless it was waiting."""
    lead = "its session was dropped, and " if dropped else ""
    deadline = time.monotonic() + patience
    while True:
        try:
            return await opened(url, timeout, token=token)
        except AtCapacity as e:
            if time.monotonic() + REOPEN_SECONDS > deadline:
                raise Failed(f"{lead}the server stayed at its limit of {e.limit} sessions for "
                             f"{patience:g}s{' after' if dropped else ''}") from e
            if url not in _waited:
                _waited.add(url)
                logger.warning("the server at %s is at its limit of %s sessions at once, so rollouts wait their turn: "
                               "serve the bundle with a higher MAX_CONCURRENT_ENVS (or max_concurrent_envs=) to run "
                               "more at once", url, e.limit)
            await asyncio.sleep(REOPEN_SECONDS)
        except ConnectionError as e:
            waits = _transient(e) or (waits_for_server and _caused_by(e, ConnectionRefusedError))
            if not waits or time.monotonic() + REOPEN_SECONDS > deadline:
                raise Failed(f"{lead}{e}") from e
            await asyncio.sleep(REOPEN_SECONDS)


def _transient(error: BaseException | None) -> bool:
    """Whether ``error`` came of a server too busy to answer in time or a connection it dropped, rather than of nothing
    listening at its address or an answer that isn't a session."""
    return _caused_by(error, _TRANSIENT)


def _caused_by(error: BaseException | None, causes: type[BaseException] | tuple[type[BaseException], ...]) -> bool:
    while error is not None:
        if isinstance(error, causes):
            return True
        error = error.__cause__ or error.__context__
    return False


async def probe(url: str, toolset: NamedToolset, timeout: float, *,
                token: str | None = None) -> tuple[frozenset[str], tuple[dict[str, Any], ...]]:
    """The tools ``toolset``'s first task offers at its first turn, and every tool any of its turns offers, found by
    walking it through its turns, ending each but the last empty and listing its tools, then ending the episode
    unscored. Each call on the probe's session has the deadline ``timeout``."""
    split, index, turns = toolset.first
    client = await opened(url, timeout, token=token)
    first, tools = None, {}
    try:
        turn = await client.reset(split=split, index=index, timeout_s=timeout)
        for number in range(turns):
            if turn.done:
                break
            listed = await list_tools(client)
            if listed.done:
                turn = listed
                break
            first = first or frozenset(tool["name"] for tool in listed.observation["tools"])
            tools |= {tool["name"]: tool for tool in listed.observation["tools"] if tool["name"] not in tools}
            if number + 1 < turns:
                turn = await end_turn(client, "", timeout)
                if error := turn.observation.get("error"):
                    raise RuntimeError(f"task {index} of split {split!r} refused an empty {END_TURN}: "
                                       f"{error.get('message')}")
    finally:
        with contextlib.suppress(Exception):
            await client.close()
    if first is None:
        raise RuntimeError(f"task {index} of split {split!r} ended before its tools could be listed: "
                           f"{error_of(turn.observation)}")
    return first, tuple(tools.values())


async def list_tools(client: GenericEnvClient) -> Any:
    return await client.step({"type": "list_tools"})


async def call_tool(client: GenericEnvClient, name: str, arguments: dict[str, Any],
                    deadline: float | None = None) -> Any:
    """The call, with its ``deadline`` in seconds for the server to end it at."""
    return await client.step({"type": "call_tool", "tool_name": name, "arguments": arguments,
                              **({"metadata": {TIMEOUT_METADATA: deadline}} if deadline is not None else {})})


async def end_turn(client: GenericEnvClient, response: str, deadline: float) -> Any:
    return await call_tool(client, END_TURN, {"response": response}, deadline)


def read_tasks(url: str, *, local: str, token: str | None = None) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """The server's env name, and each split's tasks as its Task API describes them. ``local`` says how to serve the
    bundle in this process instead, for when no server answers."""
    with httpx.Client(base_url=url, timeout=CONNECT_SECONDS, headers=authorization(token)) as http:
        try:
            [name] = http.get("/list_environments").raise_for_status().json()
        except httpx.TransportError as e:
            raise ConnectionError(f"no OpenEnv server answers at {url}: serve the bundle first (`uv run server` in the "
                                  f"package `openenv import` wrote), or {local}") from e
        splits = [split["name"] for split in http.get(f"/{name}/splits").raise_for_status().json()]
        tasks = {split: http.post(f"/{name}/tasks", json={"split": split}).raise_for_status().json()["tasks"]
                 for split in splits}
    if not all("toolset" in task and "turns" in task for described in tasks.values() for task in described):
        raise ValueError(f"the server at {url} doesn't describe its tasks' toolsets and turns, so it isn't a bundle "
                         "served by agent_env.openenv.build_app from this version of agentenv-framework or later")
    return name, tasks


def tasks_and_tools(url: str, *, local: str, timeout: float, token: str | None = None
                    ) -> tuple[str, dict[str, list[dict[str, Any]]], dict[str, tuple[dict[str, Any], ...]]]:
    """What a writer of a split's rows reads: the server's env name, each split's tasks, refusing a split whose name
    can't name a data file, and every tool each toolset's turns offer, by the toolset's key, found by playing its
    first task through (``probe``), so reading deploys those."""
    name, tasks = read_tasks(url, local=local, token=token)
    if unfit := [split for split in tasks if not _FILE.fullmatch(split)]:
        raise ValueError(f"the server's splits {unfit} can't name a data file: a split's name must be letters, "
                         "digits, '.', '_' and '-', not starting with '.'")
    return name, tasks, {each: asyncio.run(probe(url, toolset, timeout, token=token))[1]
                         for each, toolset in toolsets(name, tasks).items()}


def toolsets(name: str, tasks: dict[str, list[dict[str, Any]]]) -> dict[str, NamedToolset]:
    """Each toolset the tasks have, by its key, with the first task that has it, named after the server when it's the
    only one and otherwise after its envs, and ``bash`` when its turns offer it."""
    firsts: dict[str, tuple[str, int, int]] = {}
    for split, described in tasks.items():
        for task in described:
            firsts.setdefault(key(task), (split, task["index"], task["turns"]))
    if len(firsts) == 1:
        return {each: NamedToolset(name, first) for each, first in firsts.items()}
    named: dict[str, NamedToolset] = {}
    for each, first in firsts.items():
        toolset: Toolset = json.loads(each)
        base = "+".join([*(env.rsplit("/", 1)[-1] for env in toolset["envs"]), *(["bash"] if toolset["bash"] else [])])
        base = base or "agent"
        taken = {toolset.name for toolset in named.values()}
        name, n = base, 1
        while name in taken:
            n += 1
            name = f"{base}-{n}"
        named[each] = NamedToolset(name, first)
    return named


def authorization(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def key(task: Mapping[str, Any]) -> str:
    return json.dumps(task["toolset"], sort_keys=True)


def messages(system: str | None, prompt: str) -> list[dict[str, str]]:
    return [*([{"role": "system", "content": system}] if system else []), {"role": "user", "content": prompt}]


def error_of(observation: Mapping[str, Any]) -> str:
    return (observation.get("metadata") or {}).get("error") or "the episode ended in an error"


def why(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
