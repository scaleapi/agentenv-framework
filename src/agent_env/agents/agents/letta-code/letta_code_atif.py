"""Letta Code's ATIF adapter: the CLI's stream-json events, the native log, and the model calls the turn's relay
forwarded, which the bridge feeds as aux records (``calls``, ``agentenv.model_call``), as ATIF. The local backend prints
each model response as a run: a line per streamed text or reasoning delta, then its usage and stop reason, and after the
run the calls the CLI made and their results. Every call the CLI and the processes it starts make goes through the
relay, its sub-agents', compaction summaries' and refused attempts' too, none of which print usage of their own.

A step is one run's response, keyed by its run id and written once the next run starts or the task ends, since its
calls follow the run: its reasoning, its text, its calls with their results, and the model and usage of the relay call
whose counts it printed, else its own counts. The relay's other calls are steps in their place, before the response
whose window they ended in (a task's last ones before its last response, so the answer stays its last step); one the
endpoint refused is a step with no model call holding the CLI's report of the failure, and a compaction's summary is
the step of the call that wrote it. A run whose stream broke after its output began is its response, holding the
failure, with the call the relay answered without usage. A harness retry, the run's input sent again in a new run,
starts an attempt.

The Agent tool starts a sub-agent in the background, a CLI of its own whose output the turn never prints, and its
result names the sub-agent, which is declared under the call. Once the turn's processes are stopped, the bridge feeds
what each sub-agent wrote to its transcript (``subagents``), a pi-ai message an entry: each response it holds is a step
of the sub-agent's embedded trajectory, with the relay call it made, and a sub-agent it started with its own Agent call
(a depth-one one may) is embedded under that call. Until then the calls only the relay saw are held, since any may be a
sub-agent's, and those that aren't are steps before the task's last response."""
from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agentenv_protocol.atif import Adapter, Native, atif_adapter, aux, finish, is_token_count, native

NATIVE_FORMAT = "letta-code-events/1"
#: The CLI versions the recorded fixtures come from; a unit test holds package.json's pin to this list.
TESTED_WITH = ("0.34.26",)
_STATUSES = {"success": "ok", "error": "error"}
_COUNTS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_MEDIA_ITEMS = ("image", "audio")
#: The provider the bridge connects the CLI to, which the CLI's model handle names before the model as called.
_HANDLE = "openai-compatible/"
_SPAWNS = ("Agent", "Task")
_SPAWNED = re.compile(r"^Agent ID: (\S+)$", re.MULTILINE)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _count(value: Any) -> int | None:
    return value if is_token_count(value) else None


def _when(record: dict[str, Any]) -> str | None:
    """When the CLI wrote a record: its backend chunk's date, else the line's own stamp."""
    return _string(record.get("date")) or _string(record.get("timestamp"))


def _iso(millis: Any) -> str | None:
    """A transcript message's time, milliseconds since the epoch, as the builder writes times."""
    if not isinstance(millis, int) or isinstance(millis, bool):
        return None
    seconds, rest = divmod(millis, 1000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{rest:03d}Z"


def _tokens(usage: Any, *names: str) -> dict[str, int]:
    """A usage the CLI recorded, under ``names`` (its prompt, output, cache read and cache write), as the relay counts
    a call: its input with the cache pi-ai counts apart; none when every count is 0, as the CLI records a response that
    carried no usage."""
    if not _is_mapping(usage):
        return {}
    prompt, output, read, write = (_count(usage.get(name)) for name in names)
    read, write = read or 0, write or 0
    if prompt is None or output is None or not (prompt or output or read or write):
        return {}
    return {"input_tokens": prompt + read + write, "output_tokens": output, "cache_read_tokens": read,
            "cache_write_tokens": write}


def _native_counts(usage: dict[str, Any] | None) -> dict[str, int]:
    """A run's printed usage."""
    return _tokens(usage, "prompt_tokens", "completion_tokens", "cached_input_tokens", "cache_write_tokens")


def _transcript_counts(usage: Any) -> dict[str, int]:
    """A sub-agent's response's usage, as its transcript records it."""
    return _tokens(usage, "input", "output", "cacheRead", "cacheWrite")


def _relay_counts(call: dict[str, Any]) -> dict[str, int]:
    usage = call.get("usage") if call.get("outcome") == "answered" else None
    return {name: usage[name] for name in _COUNTS if is_token_count(usage.get(name))} if _is_mapping(usage) else {}


def _made(call: dict[str, Any], counts: dict[str, int]) -> bool:
    """Whether the relay call is the one a response the CLI recorded with ``counts`` made: answered with that input,
    cache included, and output, or, for a response recorded without usage, answered without any."""
    if not counts:
        return call.get("outcome") == "unreported"
    relayed = _relay_counts(call)
    return bool(relayed) and all(relayed.get(name) == counts[name] for name in ("input_tokens", "output_tokens"))


@dataclass
class _Run:
    """One run of the local backend as the CLI printed it, held until it ends: its text and reasoning by segment, its
    usage, its compactions' summaries, each with how many relay calls were queued when it was printed, the failures it
    reported (each in-run retry's, then the run's error, each a call the endpoint refused), the failure that broke its
    stream once its output began and the CLI's final error, and the calls the CLI made after it with their results."""

    key: str
    first: str | None = None
    time: str | None = None
    text: dict[str, list[str]] = field(default_factory=dict)
    reasoning: dict[str, list[str]] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    summaries: list[tuple[dict[str, Any], int]] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    broken: str | None = None
    final_error: str | None = None
    stopped: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def responded(self) -> bool:
        return self.usage is not None or bool(self.text or self.reasoning or self.calls)


@atif_adapter(native_format=NATIVE_FORMAT, version="0.1.0", input_includes_cache=True, tested_with=TESTED_WITH)
class LettaCodeAtif(Adapter):
    """One A2A context's Letta Code conversation as ATIF, from the CLI's stream-json events, the relay's records of the
    model calls, which the bridge feeds as the ``calls`` aux file, and its sub-agents' transcripts (``subagents``)."""

    def __init__(self, turn: Any) -> None:
        self._seq = 0
        self._declared: set[str] = set()
        self._new_task()

    def _new_task(self) -> None:
        self._model: str | None = None
        self._run: _Run | None = None
        self._calls: list[dict[str, Any]] = []
        self._held: list[dict[str, Any]] = []
        self._spawned: list[str] = []
        self._subs: dict[str, list[dict[str, Any]]] = {}
        self._attempt = 1
        self._retried = False

    @native(type="system")
    def init(self, record: dict[str, Any], turn: Any) -> None:
        """The task's model, the fallback for a step the relay saw no call of, as called: the CLI's handle names the
        provider first."""
        model = _string(record.get("model"))
        self._model = model.removeprefix(_HANDLE) if model else self._model

    @native(type="message", where={"message_type": "assistant_message"})
    def assistant(self, record: dict[str, Any], turn: Any) -> None:
        run = self._content(record, turn)
        texts = [part["text"] for part in record.get("content") or () if _is_mapping(part)
                 and part.get("type") == "text" and isinstance(part.get("text"), str)]
        run.text.setdefault(record.get("id") or "", []).extend(texts)

    @native(type="message", where={"message_type": "reasoning_message"})
    def reasoning(self, record: dict[str, Any], turn: Any) -> None:
        run = self._content(record, turn)
        if isinstance(record.get("reasoning"), str):
            run.reasoning.setdefault(record.get("id") or "", []).append(record["reasoning"])

    @native(type="message", where={"message_type": "usage_statistics"})
    def usage(self, record: dict[str, Any], turn: Any) -> None:
        self._content(record, turn).usage = record

    @native(type="message", where={"message_type": "stop_reason"})
    def stop(self, record: dict[str, Any], turn: Any) -> None:
        self._at(record, turn).stopped = True

    @native(type="message", where={"message_type": "event_message"})
    def event(self, record: dict[str, Any], turn: Any) -> None:
        """An in-run retry, after a failed call and before any output, or a compaction's start."""
        run = self._at(record, turn)
        data = record.get("event_data") if _is_mapping(record.get("event_data")) else {}
        if record.get("event_type") == "retry":
            run.failures.append(_string(data.get("message")) or "the call failed and was retried")

    @native(type="message", where={"message_type": "summary_message"})
    def summary(self, record: dict[str, Any], turn: Any) -> None:
        self._at(record, turn).summaries.append((record, len(self._calls)))

    @native(type="message", where={"message_type": "tool_call_message"})
    def tool_call(self, record: dict[str, Any], turn: Any) -> None:
        self._content(record, turn).calls.append(record)

    @native(type="message", where={"message_type": "tool_return_message"})
    def tool_return(self, record: dict[str, Any], turn: Any) -> None:
        if _string(record.get("tool_call_id")):
            self._at(record, turn).results[record["tool_call_id"]] = record

    @native(type="error")
    def error(self, record: dict[str, Any], turn: Any) -> None:
        """A run's failed call, printed in place of its error chunk, or after the run's stop the CLI's final error. One
        after the run's text or reasoning began is its own call's stream breaking, which the endpoint answered."""
        run = self._at(record, turn)
        message = _string(record.get("message")) or "Letta Code failed"
        if run.stopped:
            run.final_error = message
        elif run.text or run.reasoning:
            run.broken = message
        else:
            run.failures.append(message)

    @native(type="retry")
    def retry(self, record: dict[str, Any], turn: Any) -> None:
        self._attempt += 1
        self._retried = True

    @native(type=("result", "recovery"))
    def known(self, record: dict[str, Any], turn: Any) -> None:
        """The task's result, whose usage is only its own process's, and an approval recovery: nothing to write."""

    @aux("calls", type="agentenv.model_call")
    def call(self, record: dict[str, Any], turn: Any) -> None:
        self._calls.append(record)

    @aux("subagents", type="message")
    def subagent(self, record: dict[str, Any], turn: Any) -> None:
        """A message a sub-agent wrote to its transcript, which the bridge reads once the turn's processes are
        stopped, so all of a task's arrive before it ends."""
        if _string(record.get("agent_id")) and _is_mapping(record.get("message")):
            self._subs.setdefault(record["agent_id"], []).append(record["message"])

    @finish
    def end(self, turn: Any) -> None:
        self._flush(self._run or _Run(key=self._key()), turn, last=True)
        self._new_task()

    def _at(self, record: dict[str, Any], turn: Any) -> _Run:
        """The run ``record`` is part of: the one being printed, or, for a record naming another run, a new one, the
        last written first. A harness retry's attempt starts with the run after it."""
        run_id = _string(record.get("run_id"))
        if self._run is None or run_id is not None and run_id != self._run.key:
            if self._run is not None:
                self._flush(self._run, turn)
            if self._retried:
                turn.attempt(self._attempt)
                self._retried = False
            self._run = _Run(key=run_id or self._key(), first=_when(record))
        return self._run

    def _content(self, record: dict[str, Any], turn: Any) -> _Run:
        run = self._at(record, turn)
        run.time = run.time or _when(record)
        return run

    def _key(self) -> str:
        self._seq += 1
        return f"call:{self._seq}"

    def _flush(self, run: _Run, turn: Any, *, last: bool = False) -> None:
        """Write a run and the relay calls that ended since the last one: those before its own, or, for a run that made
        none, through the refused one of its last failure and its summaries' calls; at the task's end, every call left.
        After the task spawned a sub-agent, the answered ones are held to its end, where its sub-agents' transcripts
        claim theirs."""
        index = self._own(run)
        writers = self._writers(run, index)
        cut = index if index is not None else max([self._through_failures(len(run.failures))] + [
            position + 1 for _, position in writers if position is not None])
        own = self._calls[index] if index is not None else None
        head, rest = self._calls[:cut], self._calls[cut + (own is not None):]
        offset = 0
        if last:
            offset = len(self._held)
            head, rest, self._held = self._held + head + rest, [], []
        summaries: dict[Any, Any] = {offset + position: summary for summary, position in writers
                                     if position is not None}
        self._calls = rest
        responds = own is not None or run.responded
        messages = list(run.failures)
        if run.final_error is not None and not responds:
            messages[-1:] = [run.final_error]
        messages.reverse()
        claimed: dict[int, tuple[str, int]] = {}
        if last:
            spawning = [spawn[0] for record in run.calls if (spawn := self._spawn(record, run)) is not None]
            claimed = self._claim(head, self._family(self._spawned + spawning), set(summaries))
        for position, call in enumerate(head):
            if position in claimed:
                continue
            if call.get("outcome") == "refused":
                self._refused(call, messages.pop() if messages else None, turn)
            elif self._spawned and not last and position not in summaries:
                self._held.append(call)
            else:
                self._side(call, summaries.get(position), turn)
        for summary, position in writers:
            if position is None:
                self._compaction(self._key(), summary, None, run.first, turn)
        while messages:
            key = self._key()
            turn.step(key=key, llm_call_count=0, time=run.first)
            turn.error(message=messages.pop(), kind="infra_error", step=key)
        if responds:
            self._response(run, own, turn)
            if run.final_error is not None:
                turn.error(message=run.final_error, step=run.key)
        made = {where: head[position] for position, where in claimed.items()}
        for agent in self._spawned if last else ():
            self._embed(agent, made, turn)

    def _own(self, run: _Run) -> int | None:
        """Which queued call made the run's response: the first answered one with the counts it printed, or for a
        response printed without usage or whose stream broke, the first the relay saw answered without after the calls
        its failures were refused and those queued by its summaries' printing, all before its call began; None when
        the relay saw neither."""
        if run.usage is None and run.broken is None:
            return None
        counts = _native_counts(run.usage)
        start = 0 if counts else max([self._through_failures(len(run.failures))] + [
            printed for _, printed in run.summaries])
        return next((index for index, call in enumerate(self._calls) if index >= start and _made(call, counts)), None)

    def _writers(self, run: _Run, own: int | None) -> list[tuple[dict[str, Any], int | None]]:
        """Each of the run's compaction summaries with the place of the queued call that wrote it, among the answered
        ones before its own: the CLI prints a summary once its call has ended, so the last queued when it was printed,
        else the first after; None when the relay saw none."""
        taken: set[int] = set()
        writers = []
        for summary, printed in run.summaries:
            answered = [position for position in range(len(self._calls) if own is None else own)
                        if self._calls[position].get("outcome") != "refused" and position not in taken]
            before = [position for position in answered if position < printed]
            position = before[-1] if before else next(iter(answered), None)
            if position is not None:
                taken.add(position)
            writers.append((summary, position))
        return writers

    def _claim(self, head: list[dict[str, Any]], agents: list[str], taken: set[int]) -> dict[int, tuple[str, int]]:
        """Which queued calls the task's sub-agents made, by their place, each naming the response in its sub-agent's
        transcript that it made: for each response, in order, the first call not yet claimed nor ``taken`` that made
        it."""
        claimed: dict[int, tuple[str, int]] = {}
        for agent in agents:
            for number, message in enumerate(self._subs.get(agent, ())):
                if message.get("role") != "assistant":
                    continue
                counts = _transcript_counts(message.get("usage"))
                position = next((position for position, call in enumerate(head)
                                 if position not in claimed and position not in taken and _made(call, counts)), None)
                if position is not None:
                    claimed[position] = (agent, number)
        return claimed

    def _through_failures(self, count: int) -> int:
        refused = 0
        for index, call in enumerate(self._calls):
            if refused == count:
                return index
            refused += call.get("outcome") == "refused"
        return len(self._calls)

    def _side(self, call: dict[str, Any], summary: dict[str, Any] | None, turn: Any) -> None:
        """A call whose response the CLI didn't print, a sub-agent's, a summary's or a side call's: its usage alone."""
        key, when = self._key(), _string(call.get("started")) or _string(call.get("timestamp"))
        turn.step(key=key, model_name=_string(call.get("model")) or self._model, time=when, **_relay_counts(call))
        if summary is not None:
            self._compaction(key, summary, call, when, turn)

    def _compaction(self, key: str, summary: dict[str, Any], call: dict[str, Any] | None, when: str | None,
                    turn: Any) -> None:
        if call is None:
            turn.step(key=key, model_name=self._model, time=when)
        stats = summary.get("compaction_stats") if _is_mapping(summary.get("compaction_stats")) else {}
        turn.message(step=key, text=summary.get("summary") if isinstance(summary.get("summary"), str) else "",
                     time=_when(summary))
        turn.compaction(step=key, written_by="assistant", trigger="auto",
                        tokens_before=_count(stats.get("context_tokens_before")),
                        tokens_after=_count(stats.get("context_tokens_after")))

    def _refused(self, call: dict[str, Any], message: str | None, turn: Any) -> None:
        key, status = self._key(), call.get("status")
        turn.step(key=key, llm_call_count=0, model_name=_string(call.get("model")) or self._model,
                  time=_string(call.get("started")) or _string(call.get("timestamp")))
        turn.error(message=message or (f"the endpoint refused the call (HTTP {status})" if status else
                                       "the call reached no endpoint"), kind="infra_error", step=key)

    def _response(self, run: _Run, call: dict[str, Any] | None, turn: Any) -> None:
        key, when = run.key, run.time or run.first
        counts = _relay_counts(call) if call is not None else _native_counts(run.usage)
        model = (_string(call.get("model")) if call is not None else None) or self._model
        turn.step(key=key, model_name=model, time=when, **counts)
        for parts in run.reasoning.values():
            if text := "".join(parts):
                turn.reasoning(step=key, text=text, visibility="raw", time=when)
        if not any("".join(parts) for parts in run.reasoning.values()) and _count(
                (run.usage or {}).get("reasoning_tokens")):
            turn.reasoning(step=key, text=None, visibility="omitted", time=when)
        for parts in run.text.values():
            if text := "".join(parts):
                turn.message(step=key, text=text, time=when)
        if run.broken is not None:
            turn.error(message=run.broken, kind="infra_error", step=key)
        for record in run.calls:
            tool = record.get("tool_call") if _is_mapping(record.get("tool_call")) else {}
            call_id, name = tool.get("tool_call_id"), tool.get("name")
            turn.tool_call(step=key, tool_call_id=call_id, function_name=name, arguments=_arguments(tool),
                           time=_when(record))
            if (result := run.results.get(call_id)) is not None:
                if (spawn := self._spawn(record, run)) is not None:
                    turn.subagent(spawn[0], spawned_by=call_id, name=spawn[1])
                    self._declared.add(spawn[0])
                    self._spawned.append(spawn[0])
                content = _result_content(name, result.get("tool_return"), turn)
                turn.tool_result(source_call_id=call_id, content=content,
                                 status=_STATUSES.get(result.get("status"), "ok"), time=_when(result))

    def _spawn(self, record: dict[str, Any], run: _Run) -> tuple[str, str | None] | None:
        """The sub-agent an Agent call started, as its result names it, with the type the call asked for; None for
        another call, a failed one or one naming a sub-agent declared already, as a continued one is."""
        tool = record.get("tool_call") if _is_mapping(record.get("tool_call")) else {}
        result = run.results.get(tool.get("tool_call_id")) or {}
        text = result.get("tool_return")
        if tool.get("name") not in _SPAWNS or result.get("status") != "success" or not isinstance(text, str):
            return None
        found = _SPAWNED.search(text)
        if found is None or found[1] in self._declared:
            return None
        arguments = _arguments(tool)
        return found[1], _string(arguments.get("subagent_type")) if _is_mapping(arguments) else None

    def _embed(self, agent: str, made: dict[tuple[str, int], dict[str, Any]], turn: Any) -> None:
        """A sub-agent's steps, from its transcript: a step per response, with the model and usage of the relay call it
        made (``made``, by its place there), else the counts it recorded, and each of its calls' results; then the
        sub-agents it started, each declared under its call before the call's result."""
        pending: set[str] = set()
        spawns = {call: (child, name) for child, call, name in self._children(agent)}
        started = []
        for number, message in enumerate(self._subs.get(agent, ())):
            if message.get("role") == "assistant":
                pending |= self._sub_response(agent, number, message, made.get((agent, number)), turn)
            elif message.get("role") == "toolResult" and message.get("toolCallId") in pending:
                call_id = message["toolCallId"]
                pending.discard(call_id)
                if (spawn := spawns.get(call_id)) is not None and spawn[0] not in self._declared:
                    turn.subagent(spawn[0], spawned_by=call_id, name=spawn[1])
                    self._declared.add(spawn[0])
                    started.append(spawn[0])
                turn.tool_result(source_call_id=call_id, content=_transcript_content(message.get("content"), turn),
                                 status="error" if message.get("isError") else "ok",
                                 time=_iso(message.get("timestamp")))
        for child in started:
            self._embed(child, made, turn)

    def _children(self, agent: str) -> list[tuple[str, str, str | None]]:
        """The sub-agents ``agent`` started, as its transcript holds them: each Agent or Task call whose result names
        a sub-agent not declared already, with the call and the type it asked for."""
        calls: dict[str, dict[str, Any]] = {}
        found = []
        for message in self._subs.get(agent, ()):
            if message.get("role") == "assistant":
                calls |= {block["id"]: block for block in message.get("content") or () if _is_mapping(block)
                          and block.get("type") == "toolCall" and block.get("name") in _SPAWNS
                          and _string(block.get("id"))}
            elif message.get("role") == "toolResult" and not message.get("isError") and (
                    block := calls.get(message.get("toolCallId"))) is not None:
                text = "\n".join(part["text"] for part in message.get("content") or () if _is_mapping(part)
                                 and part.get("type") == "text" and isinstance(part.get("text"), str))
                child = _SPAWNED.search(text)
                if child is not None and child[1] != agent and child[1] not in self._declared:
                    arguments = block.get("arguments")
                    found.append((child[1], block["id"],
                                  _string(arguments.get("subagent_type")) if _is_mapping(arguments) else None))
        return found

    def _family(self, agents: list[str]) -> list[str]:
        """``agents`` and the sub-agents each started, theirs included."""
        family: list[str] = []
        pending = list(agents)
        while pending:
            agent = pending.pop(0)
            if agent not in family:
                family.append(agent)
                pending += [child for child, _, _ in self._children(agent)]
        return family

    def _sub_response(self, agent: str, number: int, message: dict[str, Any], call: dict[str, Any] | None,
                      turn: Any) -> set[str]:
        """One response of a sub-agent as a step of its trajectory: its thinking, its text and its calls, whose ids it
        gives back."""
        key, when = f"{agent}:{_string(message.get('id')) or number}", _iso(message.get("timestamp"))
        counts = _relay_counts(call) if call is not None else _transcript_counts(message.get("usage"))
        model = (_string(call.get("model")) if call is not None else None) or _string(message.get("model"))
        turn.step(key=key, model_name=model or self._model, time=when, agent=agent, **counts)
        blocks = [block for block in message.get("content") or () if _is_mapping(block)]
        thoughts = [block["thinking"] for block in blocks if block.get("type") == "thinking"
                    and _string(block.get("thinking"))]
        for text in thoughts:
            turn.reasoning(step=key, text=text, visibility="raw", time=when, agent=agent)
        if not thoughts and _count((message.get("usage") or {}).get("reasoning")):
            turn.reasoning(step=key, text=None, visibility="omitted", time=when, agent=agent)
        if text := "".join(block["text"] for block in blocks if block.get("type") == "text"
                           and isinstance(block.get("text"), str)):
            turn.message(step=key, text=text, time=when, agent=agent)
        calls = [block for block in blocks if block.get("type") == "toolCall" and _string(block.get("id"))
                 and _string(block.get("name"))]
        for block in calls:
            turn.tool_call(step=key, tool_call_id=block["id"], function_name=block["name"],
                           arguments=block.get("arguments"), time=when, agent=agent)
        return {block["id"] for block in calls}

    @classmethod
    def from_atif(cls, document: dict[str, Any]) -> Native:
        """Each task as the CLI prints it: an init, a run per agent step (its reasoning, its text, its usage and stop
        reason, then its calls and their results) and a result with the task's answer; best effort, no session."""
        log: list[dict[str, Any]] = []
        model = (document.get("agent") or {}).get("model_name")
        state: dict[str, Any] = {"runs": 0, "answer": ""}

        def end_task() -> None:
            log.append({"type": "result", "subtype": "success", "result": state["answer"],
                        "run_ids": [f"local-run-{n}" for n in range(1, state["runs"] + 1)]})
            state.update(runs=0, answer="")

        for step in document.get("steps") or ():
            if step.get("source") == "user":
                if log:
                    end_task()
                log.append({"type": "system", "subtype": "init", "model": model or ""})
                continue
            if step.get("source") != "agent" or step.get("llm_call_count") == 0:
                continue
            state["runs"] += 1
            run = {"type": "message", "run_id": f"local-run-{state['runs']}"}
            if _string(step.get("reasoning_content")):
                log.append({**run, "message_type": "reasoning_message", "reasoning": step["reasoning_content"]})
            if text := _result_text(step.get("message")):
                log.append({**run, "message_type": "assistant_message", "content": [{"type": "text", "text": text}]})
                state["answer"] = text
            metrics = step.get("metrics") if _is_mapping(step.get("metrics")) else {}
            cached = _count(metrics.get("cached_tokens")) or 0
            prompt = _count(metrics.get("prompt_tokens"))
            log.append({**run, "message_type": "usage_statistics", "step_count": 1,
                        "prompt_tokens": prompt - cached if prompt is not None else 0, "cached_input_tokens": cached,
                        "completion_tokens": _count(metrics.get("completion_tokens")) or 0})
            calls = step.get("tool_calls") or ()
            log.append({**run, "message_type": "stop_reason",
                        "stop_reason": "requires_approval" if calls else "end_turn"})
            results = {result.get("source_call_id"): result
                       for result in (step.get("observation") or {}).get("results") or ()}
            for call in calls:
                call_id = call.get("tool_call_id")
                log.append({"type": "message", "message_type": "tool_call_message", "tool_call": {
                    "name": call.get("function_name"), "tool_call_id": call_id,
                    "arguments": json.dumps(call.get("arguments") or {})}})
                if (result := results.get(call_id)) is not None:
                    log.append({"type": "message", "message_type": "tool_return_message", "tool_call_id": call_id,
                                "status": "success", "tool_return": _result_text(result.get("content"))})
        if log:
            end_task()
        return Native(log=log)


def _arguments(tool: dict[str, Any]) -> Any:
    """A call's arguments as the model gave them: the JSON its string holds, else the string itself."""
    arguments = tool.get("arguments")
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def _result_content(name: Any, text: Any, turn: Any) -> list[Any]:
    """A result as content. An MCP tool's is its whole CallToolResult as JSON text, as the bridge's mod hands it to the
    model; one holding an image or audio is its text parts and each such item as a media file."""
    text = text if isinstance(text, str) else ""
    if not (isinstance(name, str) and name.startswith("mcp__")):
        return [text]
    try:
        result = json.loads(text)
    except ValueError:
        return [text]
    items = result.get("content") if _is_mapping(result) else None
    if not isinstance(items, list) or not any(_is_mapping(item) and item.get("type") in _MEDIA_ITEMS for item in items):
        return [text]
    parts: list[Any] = []
    for item in items:
        if _is_mapping(item) and item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif _is_mapping(item) and item.get("type") in _MEDIA_ITEMS and (media := _media(item, turn)) is not None:
            parts.append(media)
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    return parts


def _transcript_content(blocks: Any, turn: Any) -> list[Any]:
    """A sub-agent's tool result as its transcript keeps it, as pi-ai sends it to the model: its text blocks joined by
    lines, then each image as a media file."""
    blocks = [block for block in blocks or () if _is_mapping(block)]
    parts: list[Any] = []
    if text := "\n".join(block["text"] for block in blocks if block.get("type") == "text"
                         and isinstance(block.get("text"), str)):
        parts.append(text)
    parts += [media for block in blocks if block.get("type") == "image" and (media := _media(block, turn)) is not None]
    return parts or [""]


def _media(item: dict[str, Any], turn: Any) -> Any:
    media_type, data = _string(item.get("mimeType")), _string(item.get("data"))
    if media_type is None or data is None or "/" not in media_type:
        return None
    try:
        return turn.media(base64.b64decode(data, validate=True), media_type)
    except (binascii.Error, ValueError):
        return None


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if _is_mapping(part) and part.get("type") == "text")
    return ""
