"""transcript_from_raw on coding-agent CLI trajectories (stream-json rows, OpenCode events) and the action log's cuts."""
from __future__ import annotations

import json

from agent_env.task_step.task_steps.verifiers.judge_utils import frame_selection as FS


def _assistant(*blocks: dict) -> dict:
    return {"type": "assistant", "message": {"content": list(blocks)}}


def _user(*blocks: dict) -> dict:
    return {"type": "user", "message": {"content": list(blocks)}}


def _use(tid: str, name: str, args: dict) -> dict:
    return {"type": "tool_use", "id": tid, "name": name, "input": args}


def _res(tid: str, content: object, is_error: bool = False) -> dict:
    return {"type": "tool_result", "tool_use_id": tid, "content": content, "is_error": is_error}


def _stream_run() -> str:
    return json.dumps([
        {"type": "system", "subtype": "init", "tools": ["Bash"]},
        _assistant({"type": "text", "text": "Looking up the issue first."},
                   _use("t1", "mcp__gw__linear_get_issue", {"id": "ENG-1"})),
        _user(_res("t1", [{"type": "text", "text": '{"state": "Todo"}'}, {"type": "image", "source": {}}])),
        _assistant(_use("t2", "mcp__gw__linear_update_issue", {"id": "ENG-1", "state": "Done"})),
        _user(_res("t2", "permission denied", is_error=True)),
        _assistant(_use("t3", "mcp__gw__gmail_send", {"to": "a@b.c"})),
        _user(_res("t9", "an orphan result with no call")),
        {"type": "result", "result": "Updated ENG-1."},
    ])


def test_stream_json_pairs_each_call_with_its_result_by_id_in_run_order():
    t = FS.transcript_from_raw(_stream_run())
    assert [(a.index, a.tool, a.args) for a in t.actions] == [
        (1, "mcp__gw__linear_get_issue", {"id": "ENG-1"}),
        (2, "mcp__gw__linear_update_issue", {"id": "ENG-1", "state": "Done"}),
        (3, "mcp__gw__gmail_send", {"to": "a@b.c"})]
    assert t.actions[0].result == '{"state": "Todo"}\n<image>'
    assert t.actions[1].result == {"error": "permission denied"}
    assert t.actions[2].result is None and all(a.screenshot is None for a in t.actions)
    assert t.messages == [FS.AgentMessage(0, "Looking up the issue first.")]


def test_stream_json_action_log_shows_calls_results_and_the_agent_text():
    t = FS.transcript_from_raw(_stream_run())
    lines = FS.action_log(t.actions, messages=t.messages).splitlines()
    assert lines[0] == 'agent said: "Looking up the issue first."'
    assert lines[1].startswith('1. mcp__gw__linear_get_issue {"id": "ENG-1"} -> ')
    assert lines[2] == ('2. mcp__gw__linear_update_issue {"id": "ENG-1", "state": "Done"}'
                        ' -> {"error": "permission denied"}')
    assert lines[3] == '3. mcp__gw__gmail_send {"to": "a@b.c"}' and "Updated ENG-1" not in "\n".join(lines)


def _part(pid: str, tool: str, state: dict) -> dict:
    return {"type": "tool_use", "part": {"id": pid, "type": "tool", "tool": tool, "state": state}}


def test_opencode_tool_parts_become_actions_with_output_or_error_and_drop_the_hex_prefix():
    raw = json.dumps([
        {"part": {"type": "text", "text": "Checking the sheet."}},
        _part("p1", "mcp_3fa9c0_sheets_read", {"status": "running", "input": {"range": "A1:B2"}}),
        _part("p1", "mcp_3fa9c0_sheets_read", {"status": "completed", "input": {"range": "A1:B2"}, "output": "2 rows"}),
        _part("p2", "mcp_3fa9c0_slack_post", {"status": "error", "input": {"channel": "#ops"}, "error": "not_in_channel"}),
        {"part": {"type": "step-finish", "reason": "stop"}},
    ])
    t = FS.transcript_from_raw(raw)
    assert [(a.index, a.tool, a.args, a.result) for a in t.actions] == [
        (1, "sheets_read", {"range": "A1:B2"}, "2 rows"),
        (2, "slack_post", {"channel": "#ops"}, {"error": "not_in_channel"})]
    assert t.messages == [FS.AgentMessage(0, "Checking the sheet.")]


def test_an_opencode_error_with_its_reason_in_output_keeps_the_reason():
    raw = json.dumps([_part("p1", "crm_search", {"status": "error", "input": {"name": "x"},
                                                 "output": "Error: company not found"})])
    assert FS.transcript_from_raw(raw).actions[0].result == {"error": "Error: company not found"}


def test_an_empty_stream_json_result_is_kept_apart_from_no_result():
    raw = json.dumps([_assistant(_use("t1", "search", {"q": "x"}), _use("t2", "lookup", {"id": 1})),
                      _user(_res("t1", "[]"), _res("t2", ""))])
    first, second = FS.transcript_from_raw(raw).actions
    assert first.result == [] and second.result is None
    assert "search" in FS.action_log([first]) and "-> []" in FS.action_log([first])


def test_unrecognised_lists_still_give_an_empty_transcript():
    for raw in (json.dumps([{"foo": 1}, 3, "x"]), json.dumps([]), json.dumps([{"type": "other"}])):
        assert FS.transcript_from_raw(raw) == FS.Transcript([], [])


def test_the_transcript_names_its_source_format_and_none_when_unrecognised():
    assert FS.transcript_from_raw(json.dumps([{"name": "chat", "attributes": {}}])).source == "otel"
    assert FS.transcript_from_raw(json.dumps([{"type": "system", "subtype": "init"}])).source == "stream-json"
    assert FS.transcript_from_raw(json.dumps([{"part": {"type": "step-finish"}}])).source == "opencode"
    assert FS.transcript_from_raw(json.dumps([])).source == "empty"
    assert FS.transcript_from_raw("not json").source == "malformed"
    for raw in (json.dumps({"a": 1}), json.dumps([{"foo": 1}, 3]), json.dumps([{"type": "other"}])):
        assert FS.transcript_from_raw(raw).source is None


def test_the_action_log_cuts_args_and_results_at_the_given_widths_and_keeps_the_defaults_otherwise():
    acts = [FS.TrajectoryAction(index=1, tool="docs_write", args={"body": "a" * 500}, screenshot=None,
                                result="r" * 500)]
    args_part, result_part = FS.action_log(acts).split(" -> ")
    assert len(args_part) == len("1. docs_write ") + FS.MAX_ARGS_CHARS + 1 and len(result_part) == FS.MAX_RESULT_CHARS + 1
    args_part, result_part = FS.action_log(acts, max_args_chars=300, max_result_chars=400).split(" -> ")
    assert len(args_part) == len("1. docs_write ") + 300 + 1 and len(result_part) == 400 + 1
    assert FS.rendered_action(acts[0]) == "docs_write " + FS._render_json(acts[0].args, FS.MAX_ARGS_CHARS)


def test_json_wrapped_in_json_text_is_decoded_so_the_log_is_not_mostly_escapes():
    inner = json.dumps({"id": "usr_1", "name": "alice", "teams": [{"id": "team_a"}]})
    raw = json.dumps([_assistant(_use("t1", "linear_get_user", {"query": "me"})),
                      _user(_res("t1", json.dumps({"result": inner}))),
                      _assistant(_use("t2", "notes", {})),
                      _user(_res("t2", "{not json at all}"))])
    t = FS.transcript_from_raw(raw)
    assert t.actions[0].result == {"result": {"id": "usr_1", "name": "alice", "teams": [{"id": "team_a"}]}}
    assert t.actions[1].result == "{not json at all}"
    assert "\\" not in FS.action_log(t.actions, max_result_chars=500)
