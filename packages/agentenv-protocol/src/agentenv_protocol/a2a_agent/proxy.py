"""The proxy: an agent with no model, whose turns a driver outside its sandbox answers.

Each message it gets is an open turn until the driver replies. Through ``urn:agentenv:proxy/v1`` the driver reads the
open turns, lists what a turn can call, has the proxy run each action where it works (a call to an env's MCP server
registered with it, or a ``bash`` command as its user, in its folder), reports the model's calls, and replies. The proxy
records the turn as it goes. Its first ``claim`` sets the token every later call carries. A turn its sender cancels
closes without a reply, stopping its running actions.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import math
import os
import shutil
import signal
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.exceptions import HTTPException

from .extensions import (
    MCP_CONFIG_V1,
    PROXY_V1,
    SKILL_CONFIG_V1,
    TRAJECTORY_V1,
    BundleSkillRequest,
    InlineSkillRequest,
    ProxyActionState,
    ProxyActRequest,
    ProxyBashAction,
    ProxyBashResult,
    ProxyClaimRequest,
    ProxyNextRequest,
    ProxyNextResponse,
    ProxyRecordRequest,
    ProxyReplyRequest,
    ProxyResultRequest,
    ProxySkill,
    ProxyTextPart,
    ProxyTool,
    ProxyToolAction,
    ProxyToolResult,
    ProxyToolsRequest,
    ProxyToolsResponse,
    ProxyTurn,
    enable,
    extension,
)
from .framework import AgentEnvAgent, AgentIdentity, a2a_agent
from .tasks.v1 import AgentConfig, TaskRequest, TaskResult, TextPart, TrajectoryLog, Usage
from ..tool_client import ToolResult, ToolSession, ToolSessionError
from ..transfers import download
from ..types import EnvironmentTool

LONG_POLL_SECONDS = 25.0
BASH_TIMEOUT_SECONDS = 3600
OUTPUT_GRACE_SECONDS = 1.0  # how long a done command's output may still arrive, from what it left running
EXIT_POLL_SECONDS = 0.05
TOOL_TIMEOUT_SECONDS = 600
SKILLS_DIR = Path.home() / "skills"
TRAJECTORY_FORMAT = "agentenv-proxy/v1"
BASH = "bash"
BASH_TOOL = ProxyTool(
    name=BASH,
    description="Run a bash command where the agent works, as its user, in its folder; returns its stdout, then its "
                "stderr.",
    input_schema={"type": "object", "properties": {"command": {"type": "string", "description": "The command."}},
                  "required": ["command"]},
)
BASH_TOOL_ON_SH = BASH_TOOL.model_copy(update={
    "description": "Run a command where the agent works, as its user, in its folder, with POSIX sh: this place has "
                   "no bash, so bash-only syntax such as [[ ]], arrays or <(...) fails. Returns its stdout, then its "
                   "stderr.",
})


class ProxyConfig(AgentConfig):
    system_prompt: str | None = None


@dataclass
class _Turn:
    task_id: str
    context_id: str
    parts: list[ProxyTextPart]
    system_prompt: str | None
    skills: list[ProxySkill]
    servers: dict[str, dict[str, Any]]
    trajectory: TrajectoryLog
    started: str = field(default_factory=lambda: _now())
    reply: asyncio.Future[ProxyReplyRequest] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    actions: dict[str, asyncio.Task[ProxyBashResult | ProxyToolResult]] = field(default_factory=dict)
    sessions: dict[str, ToolSession] = field(default_factory=dict)
    connecting: defaultdict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))
    closed: bool = False
    tools: dict[str, ProxyTool] | None = None
    listing: asyncio.Lock = field(default_factory=asyncio.Lock)
    recorded: bool = False
    pending: list[tuple[str, str]] = field(default_factory=list)  # (tool, its canonical arguments) not yet run
    last_event: str = field(default_factory=lambda: _now())
    model: str | None = None


@contextlib.asynccontextmanager
async def stopping_actions(app: Starlette) -> AsyncIterator[None]:
    """The proxy's lifespan: what its turns still run when the app stops is stopped with it. A subclass declaring its
    own definition passes it on."""
    try:
        yield
    finally:
        await app.state.agentenv_a2a.agent._stop()


IDENTITY = AgentIdentity(
    name="agentenv-proxy",
    description="An agent played from outside AgentEnv: a driver answers its turns.",
    version="1.0.0",
)
EXTENSIONS = (MCP_CONFIG_V1, enable(TRAJECTORY_V1, live=True), PROXY_V1)


@a2a_agent(identity=IDENTITY, config=ProxyConfig, extensions=EXTENSIONS, lifespan=stopping_actions)
class ProxyAgent(AgentEnvAgent):
    """``shell=False`` is a proxy for an agent that works in no sandbox: it offers no ``bash`` and keeps no skills.
    Where there is no ``bash``, its ``bash`` runs commands with ``sh``, as the tool's description says.
    ``long_poll_seconds`` bounds how long ``next``, ``act`` and ``result`` wait for something to report."""

    def __init__(self, *, shell: bool = True, long_poll_seconds: float = LONG_POLL_SECONDS):
        self._shell = shell
        self._program, self._bash_tool = ("bash", BASH_TOOL) if shutil.which("bash") else ("sh", BASH_TOOL_ON_SH)
        self._long_poll_seconds = long_poll_seconds
        self._token: bytes | None = None
        self._turns: dict[str, _Turn] = {}
        self._version = 0
        self._changed: asyncio.Future[None] | None = None
        self._actions: dict[str, tuple[_Turn, asyncio.Task[ProxyBashResult | ProxyToolResult]]] = {}
        self._draining: set[asyncio.Task[None]] = set()

    async def run(self, request: TaskRequest[ProxyConfig]) -> TaskResult:
        if any(not isinstance(part, TextPart) for part in request.parts):
            return TaskResult.failure("proxy.unsupported_part", "the proxy passes on text parts only")
        request.trajectory.set_format(TRAJECTORY_FORMAT)
        turn = _Turn(
            task_id=request.task_id,
            context_id=request.context_id,
            parts=[ProxyTextPart(text=part.text) for part in request.parts],
            system_prompt=request.config.system_prompt,
            skills=[ProxySkill(name=skill["name"], description=skill.get("description") or "",
                               path=str(SKILLS_DIR / skill["name"]))
                    for skill in request.skills],
            servers=dict(request.mcp_servers),
            trajectory=request.trajectory,
        )
        self._turns[turn.task_id] = turn
        self._bump()
        try:
            reply = await turn.reply
        finally:
            del self._turns[turn.task_id]
            self._bump()
            await self._close(turn)
        prompt = {"prompt": "\n".join(part.text for part in turn.parts), "system_prompt": turn.system_prompt}
        if reply.error is not None:
            turn.trajectory.append(_span("turn", "chain", turn.started, turn.model, prompt=prompt,
                                         completion={"error": reply.error}, failed=True))
            return TaskResult.failure("proxy.driver_error", reply.error)
        turn.trajectory.append(_span("turn", "chain", turn.started, turn.model, prompt=prompt, completion={
            "content": [{"type": "text", "text": part.text} for part in reply.parts]}))
        return TaskResult.success(parts=tuple(TextPart(text=part.text) for part in reply.parts),
                                  usage=Usage(tool_call_count=len(turn.actions)))

    @extension(PROXY_V1.claim)
    async def claim(self, request: ProxyClaimRequest) -> None:
        token = request.token.encode()
        if self._token is None:
            self._token = token
        elif not hmac.compare_digest(self._token, token):
            raise HTTPException(status_code=409, detail="this proxy is already claimed")

    @extension(PROXY_V1.next)
    async def next_turns(self, request: ProxyNextRequest) -> ProxyNextResponse:
        self._check(request.token)
        if self._version <= request.after:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(self._next_change()), self._long_poll_seconds)
        return ProxyNextResponse(version=self._version, turns=[
            ProxyTurn(task_id=turn.task_id, context_id=turn.context_id, parts=turn.parts,
                      system_prompt=turn.system_prompt, skills=turn.skills)
            for turn in self._turns.values() if not turn.reply.done()
        ])

    @extension(PROXY_V1.tools)
    async def list_tools(self, request: ProxyToolsRequest) -> ProxyToolsResponse:
        turn = self._open_turn(request.token, request.task_id)
        return ProxyToolsResponse(tools=list((await self._tools(turn)).values()))

    @extension(PROXY_V1.act)
    async def act(self, request: ProxyActRequest) -> ProxyActionState:
        self._check(request.token)
        if (known := self._actions.get(request.action_id)) is not None:
            if known[0].task_id != request.task_id:
                raise HTTPException(status_code=409, detail=f"action {request.action_id!r} is another turn's")
            return await self._state(request.action_id, known[1])
        turn = self._open_turn(request.token, request.task_id)
        action = request.action
        if isinstance(action, ProxyBashAction) and not self._shell:
            raise HTTPException(status_code=409, detail="this proxy works in no sandbox, so it runs no commands")
        if turn.recorded:
            call = (_call(action.name, action.arguments) if isinstance(action, ProxyToolAction)
                    else _call(BASH, {"command": action.command}, own_shell=True))
            if call not in turn.pending:
                raise HTTPException(status_code=409, detail=f"the last recorded model call didn't make this call to "
                                                            f"{call[0]!r}")
            turn.pending.remove(call)
        task = asyncio.create_task(self._perform(turn, action))
        turn.actions[request.action_id] = task
        self._actions[request.action_id] = (turn, task)
        return await self._state(request.action_id, task)

    @extension(PROXY_V1.result)
    async def result(self, request: ProxyResultRequest) -> ProxyActionState:
        self._check(request.token)
        if (known := self._actions.get(request.action_id)) is None:
            raise HTTPException(status_code=404, detail=f"no action {request.action_id!r}")
        return await self._state(request.action_id, known[1])

    @extension(PROXY_V1.record)
    async def record(self, request: ProxyRecordRequest) -> None:
        content, calls = _read_message(request.message)
        offered = await self._tools(self._open_turn(request.token, request.task_id))
        turn = self._open_turn(request.token, request.task_id)
        if turn.pending:
            raise HTTPException(status_code=409, detail=_unrun(turn))
        turn.pending = [_call(name, arguments, own_shell=offered[name].server is None)
                        for name, arguments in calls if name in offered]
        turn.recorded = True
        turn.model = request.model or turn.model
        span = _span("chat", "chat", turn.last_event, turn.model, completion={"content": content})
        usage = request.usage or {}
        for key, names in (("input_tokens", ("input_tokens", "prompt_tokens")),
                           ("output_tokens", ("output_tokens", "completion_tokens"))):
            if (count := next((usage[name] for name in names if name in usage), None)) is not None:
                span["attributes"][f"gen_ai.usage.{key}"] = count
        turn.trajectory.append(span)
        turn.last_event = span["end_time"]

    @extension(PROXY_V1.reply)
    async def reply(self, request: ProxyReplyRequest) -> None:
        turn = self._open_turn(request.token, request.task_id)
        if request.parts is not None and turn.pending:
            raise HTTPException(status_code=409, detail=_unrun(turn))
        turn.reply.set_result(request)

    @extension(SKILL_CONFIG_V1.add.inline)
    async def add_inline_skill(self, request: InlineSkillRequest) -> dict[str, str]:
        async with self._skill_folder(request.name) as folder:
            (folder / "SKILL.md").write_text(request.skill_md)
        return {"name": request.name}

    @extension(SKILL_CONFIG_V1.add.bundle)
    async def add_bundle_skill(self, request: BundleSkillRequest) -> dict[str, str]:
        async with self._skill_folder(request.name) as folder:
            for item in request.skill_bundle.files:
                await download(item.object, folder / item.path)
        return {"name": request.name}

    def _check(self, token: str) -> None:
        if self._token is None or not hmac.compare_digest(self._token, token.encode()):
            raise HTTPException(status_code=401, detail="the driver's token is missing or wrong")

    def _open_turn(self, token: str, task_id: str) -> _Turn:
        self._check(token)
        turn = self._turns.get(task_id)
        if turn is None or turn.reply.done():
            raise HTTPException(status_code=404, detail=f"no open turn {task_id!r}")
        return turn

    async def _tools(self, turn: _Turn) -> dict[str, ProxyTool]:
        """What the turn can call, listed once and all within ``TOOL_TIMEOUT_SECONDS``: each registered env server's
        tools, and ``bash``. Two offering one name are refused."""
        async with turn.listing:
            if turn.tools is None:
                offered = {BASH: self._bash_tool} if self._shell else {}
                loop = asyncio.get_running_loop()
                deadline = loop.time() + TOOL_TIMEOUT_SECONDS
                for server in turn.servers:
                    try:
                        listed = await asyncio.wait_for(self._listed(turn, server), deadline - loop.time())
                    except asyncio.TimeoutError:
                        raise HTTPException(status_code=502, detail=(
                            f"env server {server!r} didn't list its tools within {TOOL_TIMEOUT_SECONDS}s")) from None
                    except (ToolSessionError, httpx.HTTPError) as e:
                        raise HTTPException(status_code=502, detail=f"env server {server!r}: {e}") from None
                    for tool in listed:
                        if tool.name in offered:
                            raise HTTPException(status_code=409, detail=(
                                f"tool {tool.name!r} is offered by both {offered[tool.name].server or 'the proxy'!r} "
                                f"and {server!r}"))
                        offered[tool.name] = ProxyTool(name=tool.name, description=tool.description,
                                                       input_schema=tool.inputSchema, server=server)
                turn.tools = offered
            return turn.tools

    async def _listed(self, turn: _Turn, server: str) -> list[EnvironmentTool]:
        return await (await self._session(turn, server)).list_tools()

    async def _called(self, turn: _Turn, action: ProxyToolAction) -> ToolResult:
        return await (await self._session(turn, action.server)).call_tool(action.name, action.arguments)

    async def _session(self, turn: _Turn, server: str) -> ToolSession:
        """The turn's one connection to ``server``, which its first caller opens. It sets no deadline of its own: each
        caller bounds what it waits for. None is opened, or kept, once the turn has closed."""
        async with turn.connecting[server]:
            if turn.closed:
                raise ToolSessionError("the turn is over")
            if server not in turn.sessions:
                registration = turn.servers.get(server)
                if registration is None:
                    raise ToolSessionError(f"no env server {server!r} is registered")
                session = await ToolSession(mcp_url=registration["url"], headers=registration.get("headers") or None,
                                            timeout=math.inf).open()
                if turn.closed:
                    await session.close()
                    raise ToolSessionError("the turn is over")
                turn.sessions[server] = session
            return turn.sessions[server]

    async def _perform(self, turn: _Turn, action: ProxyToolAction | ProxyBashAction) -> ProxyBashResult | ProxyToolResult:
        started = _now()
        if isinstance(action, ProxyBashAction):
            name, arguments = BASH, {"command": action.command}
            result, completion, failed = await self._bash(action)
        else:
            name, arguments = action.name, action.arguments
            result, completion, failed = await self._tool(turn, action)
        span = _span(name, "execute_tool", started, None, prompt={"input": arguments}, completion=completion,
                     failed=failed)
        turn.trajectory.append(span)
        turn.last_event = span["end_time"]
        return result

    async def _bash(self, action: ProxyBashAction) -> tuple[ProxyBashResult, dict[str, Any], bool]:
        """Run a command; what it left reading output is drained until the proxy stops."""
        timeout = action.timeout_seconds or BASH_TIMEOUT_SECONDS
        result, draining = await _run(self._program, action.command, timeout)
        for drain in draining:
            self._draining.add(drain)
            drain.add_done_callback(self._draining.discard)
        output = f"error: the command timed out after {timeout:g}s" if result.timed_out else result.stdout + result.stderr
        return result, {"output": output, "exit_code": result.exit_code}, result.timed_out or result.exit_code != 0

    async def _tool(self, turn: _Turn, action: ProxyToolAction) -> tuple[ProxyToolResult, dict[str, Any], bool]:
        timeout = action.timeout_seconds or TOOL_TIMEOUT_SECONDS
        try:
            called = await asyncio.wait_for(self._called(turn, action), timeout)
            result = ProxyToolResult(content=called.content, is_error=called.isError)
        except asyncio.TimeoutError:
            result = _tool_error(f"the tool call timed out after {timeout:g}s")
        except (ToolSessionError, httpx.HTTPError) as e:
            result = _tool_error(f"{type(e).__name__}: {e}")
        text = "\n".join(block["text"] for block in result.content if block.get("type") == "text")
        return result, ({"error": text} if result.is_error else {"output": text}), result.is_error

    async def _state(self, action_id: str, task: asyncio.Task[ProxyBashResult | ProxyToolResult]) -> ProxyActionState:
        await asyncio.wait((task,), timeout=self._long_poll_seconds)
        if not task.done():
            return ProxyActionState(done=False)
        if task.cancelled():
            raise HTTPException(status_code=404, detail=f"action {action_id!r} was stopped: its turn is over")
        return ProxyActionState(done=True, result=task.result())

    async def _close(self, turn: _Turn) -> None:
        """Stop what the turn still runs and forget its actions, so what the driver never collected goes with it."""
        turn.closed = True
        for action_id, task in turn.actions.items():
            task.cancel()
            self._actions.pop(action_id, None)
        await asyncio.gather(*turn.actions.values(), return_exceptions=True)
        await asyncio.gather(*(session.close() for session in turn.sessions.values()), return_exceptions=True)

    async def _stop(self) -> None:
        await asyncio.gather(*(self._close(turn) for turn in list(self._turns.values())), return_exceptions=True)
        for drain in list(self._draining):
            drain.cancel()

    @contextlib.asynccontextmanager
    async def _skill_folder(self, name: str) -> AsyncIterator[Path]:
        """``name``'s folder, made for the skill and removed again if writing it fails."""
        if not self._shell:
            raise HTTPException(status_code=409, detail="this proxy works in no sandbox, so it keeps no skills")
        folder = SKILLS_DIR / name
        if folder.exists():
            raise HTTPException(status_code=409, detail=f"skill {name!r} is already added")
        folder.mkdir(parents=True)
        try:
            yield folder
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    def _next_change(self) -> asyncio.Future[None]:
        if self._changed is None:
            self._changed = asyncio.get_running_loop().create_future()
        return self._changed

    def _bump(self) -> None:
        self._version += 1
        if self._changed is not None:
            self._changed.set_result(None)
            self._changed = None


def _read_message(message: str | dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """A model call's content blocks (thinking, text, tool_use) and the tool calls it made, from an OpenAI chat message,
    an Anthropic message or a string. Raises a 400 for anything else."""
    if isinstance(message, str):
        return [{"type": "text", "text": message}], []
    if message.get("role", "assistant") != "assistant":
        raise HTTPException(status_code=400, detail="a recorded model call is the assistant's message")
    content = message.get("content")
    if isinstance(content, list) and not any(key in message for key in ("tool_calls", "reasoning_content",
                                                                         "reasoning")):
        return _anthropic(content)
    if content is None or isinstance(content, (str, list)):
        return _openai(message)
    raise HTTPException(status_code=400, detail="a recorded model call is an OpenAI chat message, an Anthropic "
                                                "message or a string")


def _openai(message: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    content: list[dict[str, Any]] = []
    if isinstance(reasoning := message.get("reasoning_content") or message.get("reasoning"), str) and reasoning:
        content.append({"type": "thinking", "thinking": reasoning})
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        content += [_text_block(part) for part in text]
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise HTTPException(status_code=400, detail="a tool call names no function")
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail=f"tool call {function['name']!r}: its arguments aren't "
                                                            "JSON") from None
        if not isinstance(arguments, dict):
            raise HTTPException(status_code=400, detail=f"tool call {function['name']!r}: its arguments aren't an "
                                                        "object")
        content.append({"type": "tool_use", "id": call.get("id"), "name": function["name"], "input": arguments})
        calls.append((function["name"], arguments))
    return content, calls


def _anthropic(blocks: list[Any]) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    content: list[dict[str, Any]] = []
    calls = []
    for block in blocks:
        kind = block.get("type") if isinstance(block, dict) else None
        if kind == "tool_use":
            if not isinstance(block.get("name"), str) or not isinstance(block.get("input", {}), dict):
                raise HTTPException(status_code=400, detail="a tool_use block needs a name and an input object")
            content.append({"type": "tool_use", "id": block.get("id"), "name": block["name"],
                            "input": block.get("input", {})})
            calls.append((block["name"], block.get("input", {})))
        elif kind == "thinking" and isinstance(block.get("thinking"), str):
            content.append({"type": "thinking", "thinking": block["thinking"]})
        elif kind == "redacted_thinking":
            content.append({"type": "redacted_thinking"})
        else:
            content.append(_text_block(block))
    return content, calls


def _text_block(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
        raise HTTPException(status_code=400, detail=f"unreadable content block: {block!r:.200}")
    return {"type": "text", "text": block["text"]}


def _unrun(turn: _Turn) -> str:
    return f"the last recorded model call's calls to {sorted({name for name, _ in turn.pending})} haven't run"


def _call(name: str, arguments: dict[str, Any], *, own_shell: bool = False) -> tuple[str, str]:
    """A call as recorded calls and actions are compared: the proxy's own ``bash`` by its command alone, since its
    action carries nothing else, and a whole float equal to the int a driver may have written for it."""
    if own_shell:
        arguments = {"command": arguments.get("command")}
    return name, json.dumps(_whole(arguments), sort_keys=True, separators=(",", ":"), default=str)


def _whole(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _whole(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_whole(item) for item in value]
    return value


def _tool_error(message: str) -> ProxyToolResult:
    return ProxyToolResult(content=[{"type": "text", "text": f"error: {message}"}], is_error=True)


def _span(name: str, operation: str, started: str, model: str | None, *, prompt: Any = None,
          completion: Any = None, failed: bool = False) -> dict[str, Any]:
    """An OTel GenAI span, the trajectory format rubrics and the explorer read."""
    attributes: dict[str, Any] = {"gen_ai.operation.name": operation}
    if model:
        attributes["gen_ai.request.model"] = model
    if prompt is not None:
        attributes["gen_ai.prompt"] = json.dumps(prompt, default=str)
    if completion is not None:
        attributes["gen_ai.completion"] = json.dumps(completion, default=str)
    return {"name": name, "start_time": started, "end_time": _now(),
            "status": {"status_code": "ERROR" if failed else "OK"}, "attributes": attributes}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run(program: str, command: str, timeout: float) -> tuple[ProxyBashResult, list[asyncio.Task[None]]]:
    """Run ``command`` in a process group of its own. It's done when its shell exits: what it started in the background
    keeps running, and output that arrives after ``OUTPUT_GRACE_SECONDS`` is read and dropped by the readers returned
    with the result, so a server it started never blocks on a full pipe. A command not done within ``timeout``, or
    whose call is cancelled, is stopped with whatever it started."""
    try:
        process = await asyncio.create_subprocess_exec(
            program, "-c", command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, start_new_session=True,
        )
    except OSError as e:
        # As a shell reports a command it can't execute: an argument list too long, say.
        return ProxyBashResult(exit_code=126, stdout="", stderr=f"error: the command couldn't start: {e}\n",
                               timed_out=False), []
    stdout, stderr = _Output(), _Output()
    readers = [asyncio.create_task(_read(process.stdout, stdout)), asyncio.create_task(_read(process.stderr, stderr))]
    try:
        await asyncio.wait_for(_exited(process), timeout)
        await asyncio.wait(readers, timeout=OUTPUT_GRACE_SECONDS)
    except BaseException as e:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await _exited(process)
        for reader in readers:
            reader.cancel()
        if isinstance(e, asyncio.TimeoutError):
            return ProxyBashResult(exit_code=None, stdout="", stderr="", timed_out=True), []
        raise
    result = ProxyBashResult(exit_code=process.returncode, stdout=stdout.take(), stderr=stderr.take(), timed_out=False)
    return result, [reader for reader in readers if not reader.done()]


async def _exited(process: asyncio.subprocess.Process) -> None:
    """Until the shell exits: ``Process.wait()`` also waits for everything still holding its output."""
    while process.returncode is None:
        await asyncio.sleep(EXIT_POLL_SECONDS)


class _Output:
    """What a command writes to one stream, kept until it's taken and dropped after."""

    def __init__(self) -> None:
        self._data: bytearray | None = bytearray()

    def add(self, chunk: bytes) -> None:
        if self._data is not None:
            self._data.extend(chunk)

    def take(self) -> str:
        data, self._data = self._data or bytearray(), None
        return data.decode(errors="replace")


async def _read(stream: asyncio.StreamReader, output: _Output) -> None:
    while chunk := await stream.read(65536):
        output.add(chunk)
