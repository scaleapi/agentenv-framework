"""The seven canonical cases, built through the adapter framework from native-shaped records of a toy format, so the
checked-in fixtures are what a real adapter would write. ``python tests/atif_cases.py`` rewrites the fixtures; a test
checks they still match."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from agentenv_protocol.atif import Adapter, AdapterRun, TurnContext, atif_adapter, finish, native, strip_profile

FIXTURES = Path(str(files("agentenv_protocol.atif").joinpath("fixtures")))
CONTEXT = "ctx-7f3a"


@atif_adapter(native_format="toy-events/1", version="0.1.0", input_includes_cache=False,
              tested_with=("toy 1.0",))
class Toy(Adapter):
    """A toy CLI whose records map one to one onto builder calls."""

    def __init__(self, turn: Any) -> None:
        self.tasks = 0

    @native(type="say")
    def say(self, r: dict, turn: Any) -> None:
        turn.message(step=r["step"], text=r["text"], time=r.get("at"), agent=r.get("agent"))

    @native(type="think")
    def think(self, r: dict, turn: Any) -> None:
        turn.reasoning(step=r["step"], text=r.get("text"), visibility=r.get("visibility", "raw"),
                       signature=r.get("signature"), data=r.get("data"), time=r.get("at"))

    @native(type="call")
    def call(self, r: dict, turn: Any) -> None:
        turn.tool_call(step=r["step"], tool_call_id=r["id"], function_name=r["name"], arguments=r.get("args"),
                       time=r.get("at"), agent=r.get("agent"))

    @native(type="result")
    def result(self, r: dict, turn: Any) -> None:
        content = None
        if "png" in r:
            content = [turn.media(r["png"].encode(), "image/png", width=1280, height=800)]
        turn.tool_result(source_call_id=r["id"], text=r.get("out") if content is None else None, content=content,
                         status=r.get("status", "ok"), exit_code=r.get("exit"), time=r.get("at"))

    @native(type="usage")
    def usage(self, r: dict, turn: Any) -> None:
        turn.step(key=r["step"], model_name=r.get("model", "claude-opus-5-5"), input_tokens=r.get("in"),
                  cache_read_tokens=r.get("read"), cache_write_tokens=r.get("write"), output_tokens=r.get("out"),
                  cost_usd=r.get("cost"), cost_source=r.get("cost_source"), reasoning_effort=r.get("effort"),
                  time=r.get("at"), agent=r.get("agent"))

    @native(type="dispatch")
    def dispatch(self, r: dict, turn: Any) -> None:
        turn.message(step=r["step"], text=r["text"])
        turn.step(key=r["step"], llm_call_count=0)

    @native(type="spawn")
    def spawn(self, r: dict, turn: Any) -> None:
        turn.subagent(r["agent"], spawned_by=r["id"], name=r["name"], model_name=r.get("model"))

    @native(type="error")
    def error(self, r: dict, turn: Any) -> None:
        turn.error(message=r["message"], kind=r.get("kind", "agent_error"), code=r.get("code"), step=r.get("step"))

    @native(type="retry")
    def retry(self, r: dict, turn: Any) -> None:
        turn.attempt(r["n"])

    @native(type="total")
    def total(self, r: dict, turn: Any) -> None:
        turn.task_cost(r["cost"], source="harness")

    @finish
    def close(self, turn: Any) -> None:
        self.tasks += 1


@dataclass
class Task:
    context: dict[str, Any]
    records: list[dict[str, Any]]
    final: list[Any] | None = None
    state: str = "completed"
    ended_at: str | None = None
    inbound_media: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)


def _ctx(index: int, task_id: str, message_id: str, started: str, context: str = CONTEXT, **kw: Any) -> dict[str, Any]:
    return {"index": index, "context_id": context, "task_id": task_id, "message_id": message_id,
            "agent_name": "toy", "agent_version": "1.0", "agent_model": "claude-opus-5-5", "started_at": started, **kw}


def _u(step: str, inp: int, read: int, write: int, out: int, **kw: Any) -> dict[str, Any]:
    return {"type": "usage", "step": step, "in": inp, "read": read, "write": write, "out": out, **kw}


LONG_READ = "".join(f"2026-01-{d:02d}T03:04:05Z -> 2026-01-{d:02d} 03:04:05+00:00\n" for d in range(1, 29))

CASES: dict[str, list[Task]] = {
    "coding": [Task(
        _ctx(0, "task-91c2", "msg-a1", "2026-11-04T17:02:11.204Z",
             inbound=["tests/test_parse.py::test_utc fails. Fix parse_date without changing the test."]),
        [{"type": "think", "step": "msg_01A", "text": "Run the failing test first to see the error.",
          "signature": "EqQBCkgIBhABGAIiQKm3"},
         {"type": "call", "step": "msg_01A", "id": "toolu_01", "name": "Bash", "args": {"command": "pytest tests/test_parse.py -q"}},
         _u("msg_01A", 140, 16890, 1402, 212),
         {"type": "result", "id": "toolu_01", "status": "error", "exit": 1, "at": "2026-11-04T17:02:19.877Z",
          "out": "FAILED tests/test_parse.py::test_utc - ValueError: Invalid isoformat string: '2026-01-02T03:04:05Z'"},
         {"type": "call", "step": "msg_01B", "id": "toolu_02", "name": "Read", "args": {"file_path": "tests/fixtures/dates.txt"}},
         _u("msg_01B", 140, 18292, 469, 96),
         {"type": "result", "id": "toolu_02", "out": LONG_READ, "at": "2026-11-04T17:02:24.310Z"},
         {"type": "call", "step": "msg_01C", "id": "toolu_03", "name": "Edit",
          "args": {"file_path": "src/parse.py", "old_string": "datetime.fromisoformat(s)",
                   "new_string": "datetime.fromisoformat(s.replace('Z', '+00:00'))"}},
         _u("msg_01C", 140, 18761, 12476, 188),
         {"type": "result", "id": "toolu_03", "out": "The file src/parse.py has been updated.", "at": "2026-11-04T17:02:41.002Z"},
         {"type": "call", "step": "msg_01D", "id": "toolu_04", "name": "Bash", "args": {"command": "pytest tests/test_parse.py -q"}},
         _u("msg_01D", 140, 31237, 325, 74),
         {"type": "result", "id": "toolu_04", "out": "3 passed in 0.18s", "at": "2026-11-04T17:02:55.419Z"},
         {"type": "say", "step": "msg_01E", "text": "parse_date now accepts a trailing Z; all 3 tests in tests/test_parse.py pass."},
         _u("msg_01E", 140, 31562, 188, 41),
         {"type": "total", "cost": 0.2117},
         {"type": "system", "subtype": "hook_started"}],
        final=["parse_date now accepts a trailing Z; all 3 tests in tests/test_parse.py pass."],
        ended_at="2026-11-04T17:03:02.551Z")],
    "computer_use": [Task(
        _ctx(0, "task-c1", "msg-c1", "2026-11-04T19:10:01.020Z",
             inbound=["In the open spreadsheet, set B2 to 42 and save the file."]),
        [{"type": "call", "step": "msg_c1", "id": "toolu_c1", "name": "computer", "args": {"action": "screenshot"}},
         _u("msg_c1", 66, 5120, 1024, 58),
         {"type": "result", "id": "toolu_c1", "png": "screenshot-1", "at": "2026-11-04T19:10:04.118Z"},
         {"type": "think", "step": "msg_c2", "visibility": "omitted", "signature": "EpwBCkYIBRgCIkB2"},
         {"type": "call", "step": "msg_c2", "id": "toolu_c2", "name": "computer",
          "args": {"action": "left_click", "coordinate": [212, 318]}},
         _u("msg_c2", 0, 6144, 1658, 121),
         {"type": "result", "id": "toolu_c2", "png": "screenshot-2", "at": "2026-11-04T19:10:07.412Z"},
         {"type": "call", "step": "msg_c3", "id": "toolu_c3", "name": "computer", "args": {"action": "type", "text": "42"}},
         _u("msg_c3", 0, 7680, 2125, 47),
         {"type": "result", "id": "toolu_c3", "png": "screenshot-3", "at": "2026-11-04T19:10:10.906Z"},
         {"type": "say", "step": "msg_c4", "text": "B2 is now 42 and the file is saved."},
         _u("msg_c4", 0, 10969, 1160, 23)],
        final=["B2 is now 42 and the file is saved."], ended_at="2026-11-04T19:10:16.004Z")],
    "late_result": [Task(
        _ctx(0, "task-l1", "msg-l1", "2026-11-04T17:00:00.000Z",
             inbound=["Run the slow suite and check src/app.py meanwhile."]),
        [{"type": "call", "step": "msg_A", "id": "toolu_07", "name": "Task",
          "args": {"description": "run the slow suite", "run_in_background": True}},
         _u("msg_A", 100, 800, 100, 50),
         {"type": "call", "step": "msg_B", "id": "toolu_08", "name": "Read", "args": {"file_path": "src/app.py"}},
         _u("msg_B", 100, 900, 100, 40),
         {"type": "result", "id": "toolu_08", "out": "def main(): ...", "at": "2026-11-04T17:08:31.220Z"},
         {"type": "result", "id": "toolu_07", "out": "312 passed", "at": "2026-11-04T17:09:12.004Z"},
         {"type": "say", "step": "msg_C", "text": "The suite passes and app.py looks fine."},
         _u("msg_C", 100, 1000, 100, 30)],
        final=["The suite passes and app.py looks fine."], ended_at="2026-11-04T17:09:30.000Z")],
    "multi_agent": [Task(
        _ctx(1, "task-b2", "msg-b2", "2026-11-04T18:01:40.530Z", context="ctx-1",
             inbound=["[Email from Priya Shah] Board review moved to 3:00pm today.",
                      "Ugh. Can you make sure my 1:1 with Sam doesn't clash?"]),
        [{"type": "think", "step": "resp-1a", "text": "Board is now 3-4pm; the 1:1 is at 3pm. Move the 1:1 to 4pm.",
          "signature": "ErUBCkYIBRgCKkAx", "at": "2026-11-04T18:01:46.902Z"},
         {"type": "call", "step": "resp-1a", "id": "toolu_9a", "name": "calendar_update_event",
          "args": {"event_id": "evt_1on1_sam", "start": "2026-03-12T16:00:00-05:00", "end": "2026-03-12T16:30:00-05:00"}},
         _u("resp-1a", 9120, 0, 0, 144, at="2026-11-04T18:01:46.902Z"),
         {"type": "result", "id": "toolu_9a", "out": "Updated evt_1on1_sam: 4:00pm-4:30pm", "at": "2026-11-04T18:01:47.611Z"},
         {"type": "say", "step": "resp-1b", "text": "Moved your 1:1 with Sam to 4:00pm, right after the board review."},
         _u("resp-1b", 9402, 0, 0, 31, at="2026-11-04T18:01:52.812Z")],
        final=["Moved your 1:1 with Sam to 4:00pm, right after the board review."],
        ended_at="2026-11-04T18:01:52.850Z")],
    "odd_parts": [Task(
        _ctx(0, "task-o1", "msg-o1", "2026-11-04T17:00:00.000Z", inbound=["Summarize the attached report and the clip."]),
        [{"type": "think", "step": "msg_A", "visibility": "summary", "text": "Read the report first."},
         {"type": "think", "step": "msg_A", "visibility": "redacted", "data": "RVJyZWRhY3RlZA=="},
         {"type": "say", "step": "msg_A", "text": "Patching the summary file."},
         {"type": "call", "step": "msg_A", "id": "toolu_p1", "name": "apply_patch", "args": "*** Begin Patch\n*** End Patch"},
         {"type": "say", "step": "msg_A", "text": "Patch sent."},
         _u("msg_A", 1000, 800, 100, 50, effort="high"),
         {"type": "result", "id": "toolu_p1", "status": "error", "exit": 2, "out": "patch failed: no such file"},
         {"type": "error", "message": "sandbox disk full", "kind": "infra_error", "step": "msg_A"},
         {"type": "rate_limit_event"},
         {"type": "retry", "n": 2},
         {"type": "dispatch", "step": "dispatch-1", "text": "Retrying after a 529."},
         {"type": "say", "step": "msg_B", "text": "Done: the summary is in SUMMARY.md."},
         _u("msg_B", 2000, 1500, 0, 40, cost=0.01, cost_source="provider")],
        final=["Done: the summary is in SUMMARY.md.", {"type": "text", "text": "{\"files\": [\"SUMMARY.md\"]}"}],
        ended_at="2026-11-04T17:09:30.000Z",
        inbound_media=[("report", "application/pdf", {"pages": 12}), ("clip", "video/mp4", {"duration_ms": 31000}),
                       ("photo", "image/heic", {"width": 640, "height": 480})])],
    "subagent": [Task(
        _ctx(0, "task-s1", "msg-s1", "2026-11-04T17:00:00.000Z", inbound=["Find the config files."]),
        [{"type": "call", "step": "msg_A", "id": "toolu_s1", "name": "Task", "args": {"description": "find config files"}},
         _u("msg_A", 100, 800, 100, 50),
         {"type": "spawn", "id": "toolu_s1", "agent": "sub1", "name": "explore", "model": "claude-haiku-5-5"},
         {"type": "call", "step": "msg_S1", "id": "toolu_s2", "name": "Bash", "args": {"command": "ls config"}, "agent": "sub1"},
         _u("msg_S1", 50, 400, 50, 20, agent="sub1", model="claude-haiku-5-5"),
         {"type": "result", "id": "toolu_s2", "out": "a.toml b.toml c.toml", "at": "2026-11-04T17:01:05.000Z"},
         {"type": "say", "step": "msg_S2", "text": "Found a.toml, b.toml and c.toml.", "agent": "sub1"},
         _u("msg_S2", 60, 450, 0, 15, agent="sub1", model="claude-haiku-5-5"),
         {"type": "result", "id": "toolu_s1", "out": "Found a.toml, b.toml and c.toml.", "at": "2026-11-04T17:01:09.000Z"},
         {"type": "say", "step": "msg_B", "text": "There are three: a.toml, b.toml and c.toml."},
         _u("msg_B", 100, 900, 50, 30)],
        final=["There are three: a.toml, b.toml and c.toml."], ended_at="2026-11-04T17:01:20.000Z")],
}
CASES["two_turns"] = CASES["coding"] + [Task(
    _ctx(1, "task-91c3", "msg-a2", "2026-11-04T17:05:00.000Z", inbound=["Also cover a +05:30 offset."]),
    [{"type": "call", "step": "msg_02A", "id": "toolu_05", "name": "Edit",
      "args": {"file_path": "tests/test_parse.py", "old_string": "", "new_string": "def test_offset(): ..."}},
     _u("msg_02A", 140, 31700, 400, 90),
     {"type": "result", "id": "toolu_05", "out": "The file tests/test_parse.py has been updated.", "at": "2026-11-04T17:05:02.000Z"},
     {"type": "say", "step": "msg_02B", "text": "Added test_offset."},
     _u("msg_02B", 140, 32100, 120, 12),
     {"type": "total", "cost": 0.0441}],
    final=["Added test_offset."], ended_at="2026-11-04T17:05:09.000Z")]


def build(name: str) -> tuple[dict[str, Any], list[dict[str, Any] | None], dict[str, bytes]]:
    """A case's document, the live page after every call, and its media files."""
    run = AdapterRun(Toy)
    pages: list[dict[str, Any] | None] = []
    for task in CASES[name]:
        context = dict(task.context)
        inbound = list(context.pop("inbound", []))
        inbound += [run.media(label.encode(), media_type, **info) for label, media_type, info in task.inbound_media]
        if task.inbound_media and name == "odd_parts":
            inbound.append({"type": "text", "text": json.dumps({"ticket": 4411, "priority": "high"})})
        pages.append(run.start_task(TurnContext(inbound=inbound, **context)))
        for record in task.records:
            pages.append(run.native(record))
        pages.append(run.end_task(state=task.state, final=task.final, ended_at=task.ended_at))
    document = run.document()
    assert document is not None
    return document, pages, run.media_files


def fixtures() -> dict[str, Any]:
    """Every fixture file by name: each case's document, the coding case stripped of its profile, and the media."""
    out: dict[str, Any] = {}
    media: dict[str, bytes] = {}
    for name in CASES:
        document, _, files = build(name)
        out[f"{name}.json"] = document
        media.update(files)
    out["coding.stripped.json"] = strip_profile(out["coding.json"])
    for path, data in media.items():
        out[path] = data
    return out


def _text(value: Any) -> bytes:
    return value if isinstance(value, bytes) else (json.dumps(value, indent=1, ensure_ascii=False) + "\n").encode()


if __name__ == "__main__":
    for rel, value in fixtures().items():
        path = FIXTURES / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_text(value))
    print(f"wrote {len(fixtures())} files to {FIXTURES}", file=sys.stderr)
