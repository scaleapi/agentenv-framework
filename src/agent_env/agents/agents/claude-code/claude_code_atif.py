"""Claude Code's ATIF adapter: the CLI's stream-json events (claude-code-stream-json/1) as ATIF, built live. A step is
one model response, keyed by its message id. Its time and usage come from the same message's record in the CLI's
session transcript, which the bridge feeds as aux records: stream-json carries no time and repeats a message's
starting usage on every record it splits the message into. A record waits for its session record, so a live reader
never sees a step change. One whose session record never comes is written without them once a later record of the
same agent has its own, or the task ends.

Sub-agents are embedded trajectories, each named by the call that spawned it in its task. Their final answer reaches
only their own transcript, so a message the session gives and stdout doesn't is written from the session record.
Records match across the two by uuid, the CLI's own id for each. A sub-agent the CLI continues in a later task is a new
trajectory there, under the call that continued it.

A compaction (a ``compact_boundary``, then the summary as a user message) is the step of the call that wrote the
summary, which no session record counts: it waits for the turn's own count, then takes what the turn reported beyond
its other steps, so a live reader never sees it change. The CLI's own messages, a command's output say, are system
steps."""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any

from agentenv_protocol.atif import (
    COMPACTION_TRIGGERS,
    Adapter,
    AtifError,
    Native,
    atif_adapter,
    aux,
    finish,
    is_token_count,
    native,
)

NATIVE_FORMAT = "claude-code-stream-json/1"
#: The CLI versions the recorded fixtures come from; a unit test holds the Dockerfile's pin to this list.
TESTED_WITH = ("2.1.322",)
#: The model a message the CLI wrote itself names, such as an API error's, which no model call produced.
_SYNTHETIC_MODEL = "<synthetic>"
#: A step's token counts as the builder takes them, and as the turn record and a session record give them.
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


@dataclass(frozen=True)
class _Boundary:
    """A ``compact_boundary``'s facts: what set the compaction off, and the history's tokens before and after it."""

    trigger: str | None = None
    tokens_before: int | None = None
    tokens_after: int | None = None


@dataclass(eq=False)
class _Entry:
    seq: int
    record: dict[str, Any]
    uuid: str | None
    agent: str | None
    from_stdout: bool
    ready: bool = False
    #: A compaction's summary: its boundary, held until the turn's count gives its call's usage.
    boundary: _Boundary | None = None
    held: bool = False
    usage: dict[str, int] | None = None
    model: str | None = None


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    message = record.get("message")
    content = message.get("content") if _is_mapping(message) else None
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [block for block in content or () if _is_mapping(block)]


def _function(tool: Any) -> dict[str, Any] | None:
    """A tool of an API request's shape as an OpenAI function entry."""
    if not _is_mapping(tool) or not isinstance(tool.get("name"), str) or not tool["name"]:
        return None
    function: dict[str, Any] = {"name": tool["name"]}
    if isinstance(tool.get("description"), str):
        function["description"] = tool["description"]
    if _is_mapping(tool.get("input_schema")):
        function["parameters"] = tool["input_schema"]
    return {"type": "function", "function": function}


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=False, tested_with=TESTED_WITH)
class ClaudeCodeAtif(Adapter):
    """One A2A context's Claude Code session as ATIF. Aux files the bridge feeds: ``session``, the transcript's message
    records (one already on stdout without its content); ``tools``, each request shape's tools, once per shape;
    ``attempt``, the error `result` of an attempt the bridge resumed; ``turn``, the turn's cost and token counts as the
    agent reports them."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._queue: list[_Entry] = []
        self._session: dict[str, dict[str, Any]] = {}
        self._written: set[str] = set()
        self._matched: dict[str | None, int] = {}
        self._spawner_of_uuid: dict[str, str] = {}
        self._agent_of_cli: dict[str, str] = {}
        self._cli_of_uuid: dict[str, str] = {}
        self._task = 0
        self._declared: dict[str, int] = {}
        self._spawn_names: dict[str, str] = {}
        self._stamped: set[str] = set()
        # Each CLI sub-agent's latest tools, for every trajectory it gets in this document, a continuation's included.
        self._tools_of_cli: dict[str, list[dict[str, Any]]] = {}
        self._unplaced: set[str] = set()
        self._inits = 0
        self._attempt_step: str | None = None
        #: The boundary whose summary is the next root user text on stdout.
        self._boundary: _Boundary | None = None
        self._task_counts: dict[str, int] = {}
        #: The model the task's latest root call named, as the API answered, and each model the turn's tally named as
        #: called (a gateway route prefixes it) by the name its calls answered as: a compaction is named as they are.
        self._task_model: str | None = None
        self._answered_as: dict[str, str] = {}
        #: The task's model calls written so far, by step key: those not in ``_stamped`` have no counts yet.
        self._task_calls: set[str] = set()

    # Records

    @native(type=("assistant", "user"))
    def message(self, record: dict[str, Any], turn: Any) -> None:
        uuid = record.get("uuid") if isinstance(record.get("uuid"), str) else None
        if uuid is not None and uuid in self._written:
            return
        agent = record.get("parent_tool_use_id") or None
        if uuid is not None and agent is not None:
            self._spawner_of_uuid[uuid] = agent
            self._link(uuid)
        entry = self._enqueue(record, uuid, agent, from_stdout=True)
        if self._boundary is not None and agent is None and record.get("type") == "user" and _texts(record):
            entry.boundary, entry.held, self._boundary = self._boundary, True, None
        if uuid is not None and uuid in self._session:
            self._match(entry)
        self._flush(turn)

    @native(type="system", where={"subtype": "compact_boundary"})
    def boundary(self, record: dict[str, Any], turn: Any) -> None:
        """The CLI's compaction: the next user message on stdout is its summary."""
        metadata = record.get("compact_metadata") if _is_mapping(record.get("compact_metadata")) else {}
        trigger, before, after = (metadata.get(key) for key in ("trigger", "pre_tokens", "post_tokens"))
        self._boundary = _Boundary(trigger=trigger if trigger in COMPACTION_TRIGGERS else None,
                                   tokens_before=before if is_token_count(before) else None,
                                   tokens_after=after if is_token_count(after) else None)

    @native(type="system", where={"subtype": "init"})
    def init(self, record: dict[str, Any], turn: Any) -> None:
        self._enqueue({"type": "init"}, None, None, from_stdout=False).ready = True
        self._flush(turn)

    @native(type="result")
    def result(self, record: dict[str, Any], turn: Any) -> None:
        """The run's last event: its answer is the task's final response, which the SDK writes."""

    @aux("session", type=("assistant", "user"))
    def session(self, record: dict[str, Any], turn: Any) -> None:
        uuid = record.get("uuid") if isinstance(record.get("uuid"), str) else None
        if uuid is None or uuid in self._written:
            return
        self._session[uuid] = record
        cli_agent = record.get("agentId") if isinstance(record.get("agentId"), str) else None
        if cli_agent is not None:
            self._cli_of_uuid[uuid] = cli_agent
            self._link(uuid)
        queued = next((entry for entry in self._queue if entry.uuid == uuid), None)
        if queued is not None:
            self._match(queued)
        elif record.get("type") == "assistant" and _blocks(record):
            agent = self._agent_of_cli.get(cli_agent) if cli_agent is not None else None
            entry = self._enqueue(record, uuid, agent, from_stdout=False)
            entry.ready = cli_agent is None or agent is not None
            if cli_agent is not None and agent is None:
                entry.agent = f"cli:{cli_agent}"
        self._flush(turn)

    @aux("tools", type="api-request-shape")
    def tools(self, record: dict[str, Any], turn: Any) -> None:
        definitions = [entry for entry in map(_function, record.get("tools") or ()) if entry is not None]
        cli_agent = record.get("agentId") if isinstance(record.get("agentId"), str) else None
        if cli_agent is None:
            self._define(turn, definitions, None)
        else:
            self._tools_of_cli[cli_agent] = definitions
            spawner = self._agent_of_cli.get(cli_agent)
            if spawner is not None and self._placed(spawner):
                self._define(turn, definitions, spawner)

    @aux("attempt", type="result")
    def attempt(self, record: dict[str, Any], turn: Any) -> None:
        self._enqueue({"type": "attempt", "result": record}, None, None, from_stdout=False).ready = True
        self._flush(turn)

    @aux("turn", type="turn")
    def cost(self, record: dict[str, Any], turn: Any) -> None:
        cost = record.get("cost_usd")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            turn.task_cost(float(cost), source="harness")
        held = [entry for entry in self._queue if entry.held]
        left = self._left_over(record) if len(held) == 1 else None
        called = record.get("model") if isinstance(record.get("model"), str) else None
        if called is not None and self._task_model is not None:
            self._answered_as[called] = self._task_model
        model = self._answered_as.get(called, called) if called is not None else self._task_model
        for entry in held:
            entry.usage, entry.held, entry.model = left, False, model
        self._flush(turn)

    @finish
    def end(self, turn: Any) -> None:
        self._flush(turn, everything=True)
        self._queue.clear()
        self._session.clear()
        self._matched.clear()
        self._spawner_of_uuid.clear()
        self._agent_of_cli.clear()
        self._cli_of_uuid.clear()
        self._inits = 0
        self._attempt_step = None
        self._boundary = None
        self._task_model = None
        self._task_counts = {}
        self._task_calls = set()
        self._task += 1

    # The queue: records in the order they came, each written once its session record is in

    def _enqueue(self, record: dict[str, Any], uuid: str | None, agent: str | None, *, from_stdout: bool) -> _Entry:
        self._seq += 1
        entry = _Entry(self._seq, record, uuid, agent, from_stdout)
        self._queue.append(entry)
        return entry

    def _match(self, entry: _Entry) -> None:
        entry.ready = True
        if entry.from_stdout:
            self._matched[entry.agent] = max(self._matched.get(entry.agent, 0), entry.seq)

    def _link(self, uuid: str) -> None:
        """Name the CLI's sub-agent by the call that spawned it, once one record shows both."""
        spawner, cli_agent = self._spawner_of_uuid.get(uuid), self._cli_of_uuid.get(uuid)
        if spawner is None or cli_agent is None or cli_agent in self._agent_of_cli:
            return
        self._agent_of_cli[cli_agent] = spawner
        for entry in self._queue:
            if entry.agent == f"cli:{cli_agent}":
                entry.agent, entry.ready = spawner, True

    def _flush(self, turn: Any, *, everything: bool = False) -> None:
        while self._queue:
            head = self._queue[0]
            waiting = head.held or (not head.ready and not (head.from_stdout and self._matched.get(head.agent, 0) >
                                                             head.seq))
            if waiting and not everything:
                return
            self._queue.pop(0)
            if head.agent is not None and head.agent.startswith("cli:"):
                continue
            self._write(head, turn)

    # Writing

    def _write(self, entry: _Entry, turn: Any) -> None:
        record = entry.record
        kind = record.get("type")
        if kind == "init":
            self._inits += 1
            self._attempt_step = None
            if self._inits > 1:
                turn.attempt(self._inits)
            return
        if kind == "attempt":
            self._attempt_failed(record["result"], turn)
            return
        if entry.uuid is not None:
            if entry.uuid in self._written:
                return
            self._written.add(entry.uuid)
        session = self._session.get(entry.uuid) if entry.uuid is not None else None
        if entry.agent is not None and entry.agent not in self._declared:
            self._declare(entry.agent, turn)
        if entry.agent is not None and not self._placed(entry.agent):
            return
        if kind == "assistant":
            self._assistant(record, session, entry.agent, turn)
        elif kind == "user":
            self._user(entry, session, turn)

    def _declare(self, agent: str, turn: Any) -> None:
        """Embed the sub-agent the call ``agent`` spawned. One whose call already has its result, as when its first
        record came after it, has nowhere to go, and its records are left to the native log."""
        self._declared[agent] = self._task
        try:
            turn.subagent(agent, spawned_by=agent, name=self._spawn_names.get(agent))
        except AtifError:
            self._unplaced.add(agent)
            return
        cli_agent = next((cli for cli, spawner in self._agent_of_cli.items() if spawner == agent), None)
        if cli_agent is not None and cli_agent in self._tools_of_cli:
            self._define(turn, self._tools_of_cli[cli_agent], agent)

    def _placed(self, agent: str) -> bool:
        """Whether the sub-agent's records can go in its trajectory: it was embedded, in this task."""
        return agent not in self._unplaced and self._declared.get(agent) == self._task

    def _define(self, turn: Any, definitions: list[dict[str, Any]], agent: str | None) -> None:
        """The tools on the trajectory's agent block, where the protocol has them (agentenv-framework-protocol from
        the release that adds ``tool_definitions``); an older one leaves the block without them."""
        write = getattr(turn, "tool_definitions", None)
        if definitions and write is not None:
            write(definitions, agent=agent)

    def _assistant(self, record: dict[str, Any], session: dict[str, Any] | None, agent: str | None,
                   turn: Any) -> None:
        message = record.get("message") if _is_mapping(record.get("message")) else {}
        key = _message_key(record)
        if key is None:
            return
        model = message.get("model") if isinstance(message.get("model"), str) else None
        synthetic = model == _SYNTHETIC_MODEL
        when = session.get("timestamp") if session is not None else None
        for block in _blocks(record):
            self._block(block, key, agent, when, turn)
        if synthetic:
            turn.step(key=key, llm_call_count=0, time=when, agent=agent)
            text = "".join(b.get("text", "") for b in _blocks(record) if b.get("type") == "text")
            turn.error(message=text or "the CLI reported an error", kind="infra_error", step=key, agent=agent)
        elif key not in self._stamped:
            self._task_calls.add(key)
            usage = self._usage(session)
            if usage is not None:
                self._stamped.add(key)
                for name, value in usage.items():
                    self._task_counts[name] = self._task_counts.get(name, 0) + value
            turn.step(key=key, model_name=model, time=when, agent=agent, **(usage or {}))
        if agent is None:
            self._attempt_step = key
            if model is not None and not synthetic:
                self._task_model = model

    def _block(self, block: dict[str, Any], key: str, agent: str | None, when: Any, turn: Any) -> None:
        kind = block.get("type")
        if kind == "text" and isinstance(block.get("text"), str) and block["text"]:
            turn.message(step=key, text=block["text"], time=when, agent=agent)
        elif kind == "thinking":
            text = block.get("thinking") if isinstance(block.get("thinking"), str) and block["thinking"] else None
            turn.reasoning(step=key, text=text, visibility="raw" if text is not None else "omitted",
                           signature=block.get("signature") or None, time=when, agent=agent)
        elif kind == "redacted_thinking":
            turn.reasoning(step=key, text=None, visibility="redacted", data=block.get("data") or None, time=when,
                           agent=agent)
        elif kind == "tool_use" and isinstance(block.get("id"), str) and isinstance(block.get("name"), str):
            arguments = block.get("input")
            turn.tool_call(step=key, tool_call_id=block["id"], function_name=block["name"], arguments=arguments,
                           time=when, agent=agent)
            if _is_mapping(arguments) and isinstance(arguments.get("subagent_type"), str):
                self._spawn_names[block["id"]] = arguments["subagent_type"]

    def _user(self, entry: _Entry, session: dict[str, Any] | None, turn: Any) -> None:
        """A user record's tool results; and on the root, its text: a compaction's summary, or else a message the CLI
        put in the conversation itself, as a system step. A sub-agent's prompt stays in its spawning call."""
        when = session.get("timestamp") if session is not None else None
        for block in _blocks(entry.record):
            if block.get("type") == "tool_result" and isinstance(block.get("tool_use_id"), str):
                turn.tool_result(source_call_id=block["tool_use_id"], content=self._result_content(block, turn),
                                 status="error" if block.get("is_error") else "ok", time=when)
        if entry.agent is not None:
            return
        texts = _texts(entry.record)
        if entry.boundary is not None:
            boundary, entry.boundary = entry.boundary, None
            key = f"compaction-{entry.uuid or entry.seq}"
            turn.message(step=key, text="\n".join(texts), time=when)
            turn.compaction(step=key, written_by="assistant", trigger=boundary.trigger,
                            tokens_before=boundary.tokens_before, tokens_after=boundary.tokens_after)
            turn.step(key=key, model_name=entry.model, time=when, **(entry.usage or {}))
            return
        for text in texts:
            turn.system_message(text=text, time=when)

    def _left_over(self, record: dict[str, Any]) -> dict[str, int] | None:
        """What the turn reported beyond its model calls written or still queued: the one call no session record
        counts, the summary's. None when the turn gave no counts, counted less than its calls (a lower bound), or one of
        its calls has no counts, whose tokens the remainder would hold too."""
        reported = {name: record.get(name) for name in _COUNTS}
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in reported.values()):
            return None
        if self._task_calls - self._stamped:
            return None
        counted, seen = dict(self._task_counts), set(self._stamped)
        for queued in self._queue:
            key = _message_key(queued.record) if queued.record.get("type") == "assistant" else None
            message = queued.record.get("message") if _is_mapping(queued.record.get("message")) else {}
            if key is None or key in seen or message.get("model") == _SYNTHETIC_MODEL:
                continue
            usage = self._usage(self._session.get(queued.uuid)) if queued.uuid is not None else None
            if usage is None:
                return None
            seen.add(key)
            for name, value in usage.items():
                counted[name] = counted.get(name, 0) + value
        left = {name: reported[name] - counted.get(name, 0) for name in _COUNTS}
        return left if all(value >= 0 for value in left.values()) else None

    def _result_content(self, block: dict[str, Any], turn: Any) -> list[Any]:
        content = block.get("content")
        if content is None:
            return [""]
        if isinstance(content, str):
            return [content]
        parts: list[Any] = []
        for item in content if isinstance(content, list) else [content]:
            if _is_mapping(item) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif _is_mapping(item) and item.get("type") == "image" and (media := self._image(item, turn)) is not None:
                parts.append(media)
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return parts or [""]

    def _image(self, item: dict[str, Any], turn: Any) -> Any:
        source = item.get("source")
        if not _is_mapping(source) or source.get("type") != "base64" or not isinstance(source.get("data"), str):
            return None
        try:
            data = base64.b64decode(source["data"], validate=True)
        except (binascii.Error, ValueError):
            return None
        return turn.media(data, source.get("media_type") or "application/octet-stream")

    @staticmethod
    def _usage(session: dict[str, Any] | None) -> dict[str, int] | None:
        message = session.get("message") if session is not None else None
        usage = message.get("usage") if _is_mapping(message) else None
        if not _is_mapping(usage):
            return None
        counts = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                  "cache_read_tokens": usage.get("cache_read_input_tokens"),
                  "cache_write_tokens": usage.get("cache_creation_input_tokens")}
        counts = {name: value for name, value in counts.items() if is_token_count(value)}
        return counts or None

    def _attempt_failed(self, result: dict[str, Any], turn: Any) -> None:
        """An attempt the bridge resumed: its error `result`, on the attempt's last step, or on a step of its own when
        the attempt wrote none."""
        key = self._attempt_step
        if key is None:
            key = f"attempt-{max(self._inits, 1)}"
            turn.step(key=key, llm_call_count=0)
        text = result.get("result") if isinstance(result.get("result"), str) else None
        code = result.get("api_error_status") or result.get("terminal_reason")
        turn.error(message=text or "the CLI's attempt failed", kind="infra_error",
                   code=str(code) if code is not None else None, step=key)

    # Back to stream-json

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each task's model responses and tool results as stream-json events (an `assistant` event per step, a `user`
        event per result), and a `result` event with the task's last agent message; best effort, no session files."""
        log: list[dict[str, Any]] = []
        last_text: str | None = None
        for step in document.get("steps") or ():
            if step.get("source") == "user":
                if last_text is not None:
                    log.append({"type": "result", "subtype": "success", "is_error": False, "result": last_text})
                last_text = None
                continue
            if step.get("source") != "agent":
                continue
            content = _reasoning(step) + _text(step.get("message")) + [
                {"type": "tool_use", "id": call["tool_call_id"], "name": call["function_name"],
                 "input": call.get("arguments") or {}} for call in step.get("tool_calls") or ()]
            log.append({"type": "assistant", "message": {"id": f"msg_{step.get('step_id')}", "role": "assistant",
                                                         "model": step.get("model_name"), "content": content}})
            for result in (step.get("observation") or {}).get("results") or ():
                log.append({"type": "user", "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": result.get("source_call_id"),
                     "content": _result_text(result.get("content"))}]}})
            text = _text(step.get("message"))
            if text:
                last_text = text[-1]["text"]
        if last_text is not None:
            log.append({"type": "result", "subtype": "success", "is_error": False, "result": last_text})
        return Native(log=log)


def _texts(record: dict[str, Any]) -> list[str]:
    """A user record's text blocks, in order, each with something in it."""
    return [block["text"] for block in _blocks(record)
            if block.get("type") == "text" and isinstance(block.get("text"), str) and block["text"].strip()]


def _message_key(record: dict[str, Any]) -> str | None:
    """An assistant record's step key: its message id, else its uuid."""
    message = record.get("message") if _is_mapping(record.get("message")) else {}
    key = message.get("id") if isinstance(message.get("id"), str) else record.get("uuid")
    return key if isinstance(key, str) else None


def _text(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, str):
        return [{"type": "text", "text": message}] if message else []
    return [{"type": "text", "text": part["text"]} for part in message or ()
            if _is_mapping(part) and part.get("type") == "text" and isinstance(part.get("text"), str)]


def _reasoning(step: dict[str, Any]) -> list[dict[str, Any]]:
    text = step.get("reasoning_content")
    return [{"type": "thinking", "thinking": text}] if isinstance(text, str) and text else []


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n\n".join(part.get("text", "") for part in content or ()
                       if _is_mapping(part) and part.get("type") == "text")
