"""An always_on run as files an agent judge reads with search and offset reads (``stage_run``), built from the run's
stored timeline and each wake's trajectory: one file per wake, one event per line, so no line is a whole turn.

- ``wakes/wake_NNN.jsonl``: what woke the assistant (``woke``, the user's messages in full ``user_message``, the full
  text it was sent ``wake_input``), then its turn's tool calls with their inputs and results (``tool_call``) and its
  own text (``assistant_text``), then the rest of the wake in time order (``app_change`` for every change,
  ``turn_end`` with its closing reply, ``message_to_user``, a ``nudge`` and its calls, the user's reads). A turn's
  trajectory is matched to its wake by the timeline's ``trajectory_url``; one that's missing is a line saying so.
- ``wakes/results/``: what is too long for a line to be read whole (a tool's input or result, a long message), named
  by wake and call (or line); the line keeps a preview.
- ``events.jsonl``: every line of every wake, and the run's start and end. ``triggers.jsonl``: every trigger firing.
- ``index.md``: a block per wake with its time, what woke it, the user's message, its write actions (tool and key
  arguments) and messages with the ``file:line`` they're at, and its errors.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from agent_env.task_step.task_steps.always_on import BULKY_EVENT_FIELDS, woke_by_text
from agent_env.task_step.task_steps.notification_delivery import label_time
from agent_env.task_step.task_steps.verifiers.judge_utils.turn_events import UNKNOWN, TurnEvents
from agent_env.task_step.task_steps.verifiers.run_context import Firing

WAKES_DIR = "wakes"
RESULTS_DIR = "wakes/results"
INDEX_FILE = "index.md"
EVENTS_FILE = "events.jsonl"
TRIGGERS_FILE = "triggers.jsonl"
# A tool call's line stays under this, so a reader shows it whole: past it, its input or result (the longer first) goes
# to its own file under RESULTS_DIR and the line keeps a preview.
LINE_CHARS = 1800
PREVIEW_CHARS = 300
# What index.md shows of a message (the wake file has it whole), of one argument, and of one action line.
INDEX_TEXT_CHARS = 400
ARG_CHARS = 40
ARGS_SHOWN = 6
ACTION_CHARS = 240
_DROPPED = frozenset({*BULKY_EVENT_FIELDS, "real_time", "real_elapsed_s", "user_sim_trajectory_urls"})
# A tool whose name has one of these words reads; index.md lists every other call as a write action.
_READS = re.compile(r"(?:^|_)(get|list|search|read|find|whoami|snapshot|view|query|lookup|fetch|navigate|wait|hover|"
                    r"resize|tabs|screenshot|console|network|time)(?:_|$)")
_NOT_ACTIONS = frozenset({"message_user", "stay_quiet"})
# A line's fields that say what and where it is; ``fit`` moves only the others.
_IDENTITY = frozenset({"event", "wake", "at", "call", "turn", "tool", "kind", "state", "outcome", "phase", "by"})


@dataclass(frozen=True)
class StagedRun:
    """``files`` by path relative to the staging directory; ``record`` is what the verification entry keeps."""

    files: dict[str, str]
    record: dict


@dataclass
class _Wake:
    n: int
    at: Optional[str] = None
    woke_by: list[str] = field(default_factory=list)
    head: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    body: list[dict] = field(default_factory=list)
    call_count: int = 0

    def lines(self) -> list[dict]:
        return self.head + self.calls + self.body


def stage_run(timeline: Sequence[dict], turns: Mapping[str, Optional[TurnEvents]], *,
              firings: Sequence[Firing] = (), unavailable: Optional[Mapping[str, str]] = None) -> StagedRun:
    """The run's files: ``turns`` holds each trajectory the timeline names, by its url (None for one that couldn't be
    read); ``firings`` and ``unavailable`` are the run context's trigger firings and the logs it couldn't read."""
    stager = _Stager(turns, max((e.get("n") or 0 for e in timeline if e.get("event") == "wake"), default=0))
    for event in timeline:
        stager.add(event)
    return stager.finish(firings, unavailable or {})


def wake_file(n: int, width: int = 3) -> str:
    return f"{WAKES_DIR}/wake_{n:0{width}d}.jsonl"


def _line(event: str, wake: Optional[int], **fields: Any) -> dict:
    return {"event": event, **({"wake": wake} if wake is not None else {}),
            **{k: v for k, v in fields.items() if v is not None and v is not False}}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _cut(text: Any, chars: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= chars else text[:chars] + "…"


def _short_tool(tool: str) -> str:
    """A tool's own name, without an MCP server prefix (``mcp__server__tool``)."""
    return tool.rsplit("__", 1)[-1]


def is_write_action(tool: str) -> bool:
    name = _short_tool(tool)
    return name not in _NOT_ACTIONS and not _READS.search(name.lower())


def _key_args(tool_input: Any) -> str:
    if not isinstance(tool_input, dict):
        return _cut(_dumps(tool_input), ARG_CHARS) if tool_input not in (None, "") else ""
    shown = [f"{k}={_cut(v if isinstance(v, str) else _dumps(v), ARG_CHARS)}"
             for k, v in list(tool_input.items())[:ARGS_SHOWN]]
    more = len(tool_input) - ARGS_SHOWN
    return " ".join(shown) + (f" (+{more} more)" if more > 0 else "")


class _Stager:
    def __init__(self, turns: Mapping[str, Optional[TurnEvents]], last_wake: int):
        self.turns = turns
        self.width = max(3, len(str(last_wake)))
        self.wakes: dict[int, _Wake] = {}
        self.run_lines: list[dict] = []
        self.current = 0
        self.pending_user: list[dict] = []
        self.files: dict[str, str] = {}
        self.formats: Counter = Counter()
        self.raw_bytes = self.staged_bytes = 0
        self.missing: list[int] = []
        self.unreadable: list[int] = []
        self.not_compacted: list[dict] = []
        self.unrecorded_changes = 0
        self.by_changes = False  # whether the run recorded app changes, so write actions are read from them

    def wake(self, n: Optional[int]) -> _Wake:
        n = n or self.current
        return self.wakes.setdefault(n, _Wake(n))

    def add(self, event: dict) -> None:
        kind = event.get("event")
        at = event.get("virtual_time")
        if kind == "wake":
            self.current = event["n"]
            w = self.wake(event["n"])
            w.at, w.woke_by = at, list(event.get("woke_by") or [])
            w.head.append(_line("woke", w.n, at=at, woke_by=w.woke_by, notifications=event.get("notifications") or None,
                                last=event.get("last")))
            w.head += [_line(m.pop("event"), w.n, **m) for m in self.pending_user]
            self.pending_user = []
            w.head.append(_line("wake_input", w.n, text=event.get("parts_text") or ""))
        elif kind in ("user_reply_delivered", "scripted_user_line"):
            self.pending_user.append(_line("user_message", None, at=at, text=event.get("text"),
                                           kind="user_reply" if kind == "user_reply_delivered" else event.get("kind"),
                                           done=event.get("done")))
        elif kind == "app_changes":
            w = self.wake(event.get("wake"))
            self.by_changes = True
            for change in event.get("changes") or []:
                # The changelog's own timestamp is wall-clock time, not the run's pretend time.
                w.body.append(_line("app_change", w.n, **{k: v for k, v in change.items() if k not in ("wake", "at")},
                                    recorded_at=change.get("at")))
            if event.get("not_kept"):
                self.unrecorded_changes += event["not_kept"]
                w.body.append(_line("app_changes_not_recorded", w.n, count=event["not_kept"], phase=event.get("phase")))
        elif kind == "agent_turn_end":
            w = self.wake(event.get("n"))
            w.calls += self.turn(w, event.get("trajectory_url"), "turn")
            w.body.append(_line("turn_end", w.n, at=at, state=event.get("state"), closing_reply=event.get("plain_reply"),
                                tool_calls=event.get("tool_calls"), messages_to_user=event.get("messages_to_user"),
                                stay_quiet_calls=event.get("stay_quiet_calls") or None,
                                virtual_seconds=event.get("virtual_seconds")))
        elif kind == "nudge":
            w = self.wake(event.get("wake"))
            w.body.append(_line("nudge", w.n, at=at, closing_text=event.get("closing_text"),
                                outcome=event.get("outcome"), state=event.get("state"),
                                reply_after=event.get("plain_reply_after")))
            w.body += self.turn(w, event.get("trajectory_url"), "nudge")
        elif kind == "message_to_user":
            self.wake(event.get("wake")).body.append(_line(
                "message_to_user", event.get("wake") or self.current, at=at, text=event.get("text"),
                after_nudge=event.get("after_nudge"), leftover=event.get("leftover")))
        else:
            fields = {k: v for k, v in event.items() if k not in _DROPPED and k not in ("event", "virtual_time", "wake")}
            if kind in ("start", "end") or not (event.get("wake") or self.current):
                self.run_lines.append(_line(kind, None, at=at, **fields))
            else:
                w = self.wake(event.get("wake"))
                w.body.append(_line(kind, w.n, at=at, **fields))

    def turn(self, w: _Wake, url: Optional[str], which: str) -> list[dict]:
        """A turn's events as lines of wake ``w``; its long inputs and results go to their own files."""
        if not url:
            self.missing.append(w.n)
            return [_line("trajectory_missing", w.n, turn=which)]
        events = self.turns.get(url)
        if events is None:
            self.unreadable.append(w.n)
            return [_line("trajectory_unreadable", w.n, turn=which)]
        self.formats[events.format] += 1
        lines, side = [], 0
        for e in events.events:
            kind = e["type"]
            if kind == "tool_call":
                w.call_count += 1
                line, written = self.call(w, e, which)
                lines.append(line)
                side += written
            elif kind == "text":
                lines.append(_line("assistant_text", w.n, turn=which, text=e["text"], subagent=e.get("subagent")))
            elif kind == "note":
                lines.append(_line("harness_note", w.n, turn=which, text=e["text"]))
            elif kind == "raw":
                lines.append(_line("trajectory_item", w.n, turn=which, item=e["event"]))
        staged = sum(len(_dumps(line).encode()) for line in lines) + side
        self.raw_bytes += events.raw_bytes
        self.staged_bytes += staged
        if events.format == UNKNOWN or staged >= events.raw_bytes:
            self.not_compacted.append({"wake": w.n, "turn": which, "format": events.format,
                                       "raw_bytes": events.raw_bytes, "staged_bytes": staged})
        return lines

    def call(self, w: _Wake, e: dict, which: str) -> tuple[dict, int]:
        """The call's line; while it is longer than LINE_CHARS its longer of input and result goes to a file."""
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", _short_tool(e["tool"]))[:60]
        stem = f"{RESULTS_DIR}/wake_{w.n:0{self.width}d}_call_{w.call_count:03d}_{name}"
        line = _line("tool_call", w.n, call=w.call_count, turn=which, tool=e["tool"], subagent=e.get("subagent"),
                     input=e.get("input"), error=e.get("error"))
        if "result" in e:
            line["result"] = e["result"]
        else:
            line["no_result"] = True
        return self.fit(line, stem, ("input", "result"))

    def fit(self, line: dict, stem: str, movable: Sequence[str]) -> tuple[dict, int]:
        """``line`` shortened to LINE_CHARS, so a reader shows it whole: its longest ``movable`` field at a time goes to
        a file named from ``stem`` (``<stem>.json``, ``<stem>.txt``, or with ``_<field>`` for a field other than a
        tool call's result) and leaves a preview. Returns the bytes it wrote."""
        written = 0
        while len(_dumps(line)) > LINE_CHARS and (inline := [k for k in movable if line.get(k) not in (None, "")]):
            key = max(inline, key=lambda k: len(_dumps(line[k])))
            value = line.pop(key)
            path = f"{stem}{'' if key == 'result' else '_' + key}.{'txt' if isinstance(value, str) else 'json'}"
            body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1, default=str)
            self.files[path] = body
            written += len(body.encode())
            line[f"{key}_file"] = path
            line[f"{key}_preview"] = _cut(value if isinstance(value, str) else _dumps(value), PREVIEW_CHARS)
        return line, written

    def fit_all(self, lines: list[dict], prefix: str) -> list[dict]:
        """Every line but a tool call's (``call`` fits those) shortened by ``fit``, its files named by line number."""
        for i, line in enumerate(lines, 1):
            if line["event"] != "tool_call":
                self.fit(line, f"{RESULTS_DIR}/{prefix}_line_{i:03d}_{line['event']}",
                         [k for k in line if k not in _IDENTITY])
        return lines

    def finish(self, firings: Sequence[Firing], unavailable: Mapping[str, str]) -> StagedRun:
        if self.pending_user:
            self.run_lines += self.pending_user
        self.fit_all(self.run_lines, "run")
        start = [line for line in self.run_lines if line["event"] != "end"]
        end = [line for line in self.run_lines if line["event"] == "end"]
        everything = list(start)
        index_blocks = []
        for n in sorted(self.wakes):
            w = self.wakes[n]
            lines = self.fit_all(w.lines(), f"wake_{n:0{self.width}d}")
            path = wake_file(n, self.width)
            self.files[path] = "".join(_dumps(line) + "\n" for line in lines)
            everything += lines
            index_blocks.append(_index_block(w, path, lines, self.by_changes))
        everything += end
        self.files[EVENTS_FILE] = "".join(_dumps(line) + "\n" for line in everything)
        self.files[TRIGGERS_FILE] = _triggers(firings, unavailable, self.wakes)
        self.files[INDEX_FILE] = _index(self, index_blocks, start, end, firings)
        record: dict = {"wakes": len(self.wakes), "files": len(self.files),
                        "side_files": sum(1 for p in self.files if p.startswith(RESULTS_DIR + "/")),
                        "trajectories": sum(self.formats.values()), "formats": dict(self.formats),
                        "raw_bytes": self.raw_bytes, "staged_bytes": self.staged_bytes}
        for key, value in (("missing_trajectories", self.missing), ("unreadable_trajectories", self.unreadable),
                           ("not_compacted", self.not_compacted),
                           ("app_changes_not_recorded", self.unrecorded_changes)):
            if value:
                record[key] = value
        return StagedRun(self.files, record)


def _text(line: dict, key: str) -> Any:
    """A field of a line, or its preview when ``fit`` moved it to a file."""
    return line.get(key) if line.get(key) is not None else line.get(f"{key}_preview")


def _changing_tools(lines: list[dict]) -> set[str]:
    """The tools whose calls made the assistant's app changes in a wake, as the changes' actors name them."""
    return {_short_tool(a["tool"]) for line in lines if line["event"] == "app_change" and line.get("by") == "agent"
            for a in line.get("actors") or [] if a.get("tool")}


def _index_block(w: _Wake, path: str, lines: list[dict], by_changes: bool) -> str:
    """The wake's block. With ``by_changes`` its write actions are the calls of a tool that made one of the wake's app
    changes; else (a run whose apps keep no change log) calls whose tool name has no read word (``is_write_action``)."""
    calls = sum(1 for line in lines if line["event"] == "tool_call")
    changing = _changing_tools(lines)
    out = [f"## Wake {w.n} · {label_time(w.at)} · woken by {woke_by_text(w.woke_by)} · {path} "
           f"({len(lines)} lines, {calls} tool calls)"]
    users = [(i, line) for i, line in enumerate(lines, 1) if line["event"] == "user_message"]
    for i, line in users:
        out.append(f"- the user ({label_time(line.get('at'))}): \"{_cut(_text(line, 'text'), INDEX_TEXT_CHARS)}\" "
                   f"({path}:{i})")
    wake_input = next(((i, line) for i, line in enumerate(lines, 1) if line["event"] == "wake_input"), None)
    if not users and wake_input:
        i, line = wake_input
        out.append(f"- input: \"{_cut(_text(line, 'text'), INDEX_TEXT_CHARS)}\" ({path}:{i})")
    changes = Counter()
    for i, line in enumerate(lines, 1):
        event = line["event"]
        writes = _short_tool(line.get("tool") or "") in changing if by_changes else is_write_action(line.get("tool") or "")
        if event == "tool_call" and writes:
            action = _cut(f"{_short_tool(line['tool'])} {_key_args(line.get('input', line.get('input_preview')))}",
                          ACTION_CHARS)
            out.append(f"- {'FAILED ' if line.get('error') else ''}action: {action} ({path}:{i})")
        elif event == "tool_call" and line.get("error"):
            out.append(f"- error: {_short_tool(line['tool'])} failed ({path}:{i})")
        elif event == "message_to_user":
            out.append(f"- sent the user ({label_time(line.get('at'))}): \"{_cut(_text(line, 'text'), INDEX_TEXT_CHARS)}\" "
                       f"({path}:{i})")
        elif event == "turn_end":
            if line.get("stay_quiet_calls"):
                out.append(f"- stayed quiet ({path}:{i})")
            if line.get("state") not in (None, "completed", "TaskState.completed"):
                out.append(f"- error: its turn ended {line.get('state')} ({path}:{i})")
        elif event in ("trajectory_missing", "trajectory_unreadable"):
            out.append(f"- error: the {line.get('turn')}'s trajectory is "
                       f"{'missing' if event == 'trajectory_missing' else 'unreadable'}: its tool calls aren't here "
                       f"({path}:{i})")
        elif event == "nudge":
            out.append(f"- nudged after closing text; then {line.get('outcome')} ({path}:{i})")
        elif event == "app_change":
            changes["by the assistant" if line.get("by") == "agent" else "by others" if line.get("by")
                    else "with no known actor"] += 1
        elif event == "app_changes_not_recorded":
            out.append(f"- {line.get('count')} app changes weren't recorded ({path}:{i})")
    if changes:
        out.append("- app changes: " + ", ".join(f"{count} {who}" for who, count in changes.items())
                   + f" (\"event\": \"app_change\" in {path})")
    untold = sum(1 for line in lines if line["event"] == "app_change" and line.get("by") == "agent"
                 and not any(a.get("tool") for a in line.get("actors") or []))
    if untold:
        out.append(f"- {untold} of the assistant's app changes name no tool (made through a website, say)")
    return "\n".join(out)


def _index(stager: _Stager, blocks: list[str], start: list[dict], end: list[dict],
           firings: Sequence[Firing]) -> str:
    first = next((line for line in start if line["event"] == "start"), {})
    last = end[0] if end else {}
    width = stager.width
    lines = ["# The run, wake by wake", "",
             f"{len(stager.wakes)} wakes, from {label_time(first.get('at'))} to {label_time(last.get('at'))}. "
             f"End: {last.get('note') or '(not recorded)'} (end reason: {last.get('reason')}).",
             "",
             f"Each wake's events are in {wake_file(1, width)} … {wake_file(max(stager.wakes or [1]), width)}, one JSON "
             "object a line; `file:line` below points into them. Long tool results and inputs are in "
             f"{RESULTS_DIR}/ (the tool_call line names its file). {EVENTS_FILE} has every wake's lines in one file; "
             f"{TRIGGERS_FILE} has every trigger firing. A line too long to read whole keeps a preview of its longest "
             f"fields and names the file in {RESULTS_DIR}/ that holds them (\"<field>_file\").",
             "",
             "Actions below are the assistant's tool calls of a tool that changed the apps' data in that wake (from the "
             "apps' change logs)." if stager.by_changes else
             "Actions below are the assistant's tool calls whose tool name has no read word (get, list, search, …): "
             "this run's apps kept no change log, so this is a guess from the name."]
    gaps = sorted(set(stager.missing) | set(stager.unreadable))
    if gaps:
        lines += ["", "Wakes whose tool calls are not here (their trajectory is missing or unreadable): "
                  + ", ".join(str(n) for n in gaps) + "."]
    if firings:
        per: dict[str, list[Firing]] = {}
        for f in firings:
            per.setdefault(f"{f.trigger_id} ({f.source})", []).append(f)
        lines += ["", "## Triggers"]
        for name, fs in per.items():
            failed = sum(1 for f in fs if not f.ok)
            times = [f.virtual_time for f in fs if f.virtual_time]
            span = f", {label_time(min(times).isoformat())} to {label_time(max(times).isoformat())}" if times else ""
            lines.append(f"- {name}: fired {len(fs) - failed}, failed {failed}{span}")
    return "\n".join(lines + [""] + [block + "\n" for block in blocks])


def _triggers(firings: Sequence[Firing], unavailable: Mapping[str, str], wakes: Mapping[int, _Wake]) -> str:
    """One line per firing, oldest first, with the last wake at or before its pretend time (``after_wake``, 0 before
    the first); then one per firing log that couldn't be read."""
    starts = sorted((w.at, n) for n, w in wakes.items() if w.at)
    out = []
    for f in sorted(firings, key=lambda f: f.at):
        vt = f.virtual_time.isoformat().replace("+00:00", "Z") if f.virtual_time else None
        after = max((n for at, n in starts if vt and at <= vt), default=0)
        out.append(_dumps({"trigger_id": f.trigger_id, "source": f.source, "fired": f.ok, "at": vt,
                           "recorded_at": f.at.isoformat(), "after_wake": after, "provoked_by": f.provoked_by}))
    out += [_dumps({"unavailable": source, "why": why}) for source, why in unavailable.items() if source != "conversation"]
    return "".join(line + "\n" for line in out)


_WAKE_NAME = re.compile(r"wake_(\d+)")
# A shell loop over wake numbers (``for w in 080 075; do … wake_$w.jsonl``): the numbers it names.
_WAKE_LOOP = re.compile(r"\bfor\s+(\w+)\s+in\s+([\d\s]+?)\s*;")
# A glob or directory walk that takes every wake file at once.
_ALL_WAKES = re.compile(r"wake_\*|wakes/\*|os\.walk|rglob|listdir")
_WAKE_FIELD = re.compile(r'"wake":\s*(\d+)')
_HELPER_TOOLS = frozenset({"Agent", "Task", "Workflow"})


def judge_reads(judge: TurnEvents) -> dict:
    """What a judge's trajectory shows it read of the staged run, whichever tool it read with (Read, Grep, or Bash
    running grep, sed or python): the wakes its calls named (a file, a result file, a shell loop's numbers) or whose
    events came back in a result (``"wake": N``; a result of reading index.md, which names every wake, doesn't count),
    whether one call took every wake file at once (``read_all_wakes``), whether it read index.md or searched
    events.jsonl / triggers.jsonl, and any helper agent it started."""
    calls = [e for e in judge.events if e["type"] == "tool_call"]
    wakes: set[int] = set()
    every = False
    for call in calls:
        named = _dumps(call.get("input"))
        wakes |= {int(n) for n in _WAKE_NAME.findall(named)}
        for var, numbers in _WAKE_LOOP.findall(named):
            if f"wake_${var}" in named or f"wake_${{{var}}}" in named:
                wakes |= {int(n) for n in numbers.split()}
        every = every or bool(_ALL_WAKES.search(named))
        if INDEX_FILE not in named:
            wakes |= {int(n) for n in _WAKE_FIELD.findall(_dumps(call.get("result")))}
    text = " ".join(_dumps(c.get("input")) for c in calls)
    return {"tool_calls": len(calls), "wakes_opened": sorted(wakes), "read_all_wakes": every,
            "read_index": INDEX_FILE in text, "searched_events": EVENTS_FILE in text,
            "read_triggers": TRIGGERS_FILE in text,
            "helpers_started": sum(1 for c in calls if _short_tool(c["tool"]) in _HELPER_TOOLS)}
