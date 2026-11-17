"""Reading a yatf/1 stream: JSON array or JSONL, (turn, seq) order, ATIF-style steps and paired tool calls."""
import json

import pytest

from agentenv_protocol import yatf


def coding():
    return yatf.load_example("coding")


def test_a_json_array_and_jsonl_read_the_same():
    events = coding()
    as_jsonl = yatf.dump_events(events)
    as_array = yatf.dump_events(events, jsonl=False)
    assert len(as_jsonl.splitlines()) == len(events) and as_array.startswith("[")
    for text in (as_jsonl, as_array, as_jsonl.encode(), "﻿" + as_jsonl.replace("\n", "\n\n")):
        assert [yatf.dump_event(e) for e in yatf.parse_events(text)] == [yatf.dump_event(e) for e in events]


def test_a_stream_that_is_not_json_says_where():
    lines = yatf.dump_events(coding()).splitlines()
    lines[3] = lines[3][:-5]
    with pytest.raises(yatf.YatfError, match="line 4"):
        yatf.parse_events("\n".join(lines))
    with pytest.raises(yatf.YatfError, match="event 1"):
        yatf.parse_events([yatf.dump_event(coding()[0]), {"type": "note"}])
    with pytest.raises(yatf.YatfError):
        yatf.parse_events('{"not": "an array"}\n[1')
    with pytest.raises(yatf.YatfError, match="not UTF-8"):
        yatf.parse_events(b'{"text": "\xff"}\n')
    with pytest.raises(yatf.YatfError, match="NaN isn't JSON"):
        yatf.parse_events(lines[0].replace('"turn":0', '"turn":NaN'))
    with pytest.raises(yatf.YatfError, match="event 0: .*isn't a JSON number"):
        yatf.parse_events(lines[0].replace('"turn":0', '"turn":0,"later":{"n":1e400}'))


def test_any_text_an_event_holds_dumps_to_encodable_jsonl():
    note = {"seq": 0, "id": "0.0", "type": "note", "turn": 0, "actor": "agent", "observer": "agent",
            "time": {"at": None}, "text": "\ud800 é 漢 🤖 \x00", "later": {"\ude3a": ["\udfff"]}}
    dumped = yatf.dump_events(yatf.parse_events([note]))
    assert dumped.isascii() and dumped.encode("utf-8")
    assert yatf.dump_event(yatf.parse_events(dumped)[0]) == note


def test_order_is_turn_then_seq_never_time():
    events = coding()
    shuffled = list(reversed(events))
    later_turn = yatf.parse_event({**yatf.dump_event(events[0]), "turn": 1, "time": {"at": "2000-01-01T00:00:00Z"}})
    assert yatf.ordered(shuffled + [later_turn]) == events + [later_turn]


def test_tool_calls_pair_with_their_results_across_the_context():
    events = coding()
    pairs = yatf.pair_tool_calls(events)
    assert [pair.call.tool_call_id for pair in pairs] == ["toolu_01", "toolu_02", "toolu_03", "toolu_04"]
    assert [pair.results[0].status for pair in pairs] == ["error", "ok", "ok", "ok"]
    orphan = yatf.parse_event({**yatf.dump_event(events[6]), "seq": 99, "id": "0.99", "source_call_id": "gone"})
    unanswered = yatf.parse_event({**yatf.dump_event(events[4]), "seq": 98, "id": "0.98", "tool_call_id": "toolu_05"})
    pairs = yatf.pair_tool_calls(events + [orphan, unanswered])
    assert pairs[-1].call.tool_call_id == "toolu_05" and pairs[-1].results == ()
    assert all(result.source_call_id != "gone" for pair in pairs for result in pair.results)


def test_events_fold_into_steps_with_their_results_regrouped():
    steps = yatf.fold_steps(coding())
    assert [step.key for step in steps] == ["in-0", "msg_01A", "msg_01B", "msg_01C", "msg_01D", "msg_01E"]
    inbound, first = steps[0], steps[1]
    assert inbound.step is None and inbound.actor == "caller"
    assert [type(event) for event in first.events] == [yatf.Reasoning, yatf.ToolCall]
    assert first.step.metrics.prompt_tokens == 18432
    assert [result.source_call_id for result in first.results] == ["toolu_01"]
    assert steps[-1].results == () and isinstance(steps[-1].events[0], yatf.Message)


def test_a_dump_round_trips_unknown_fields():
    raw = json.loads(yatf.dump_events(coding(), jsonl=False))
    raw[2]["later_field"] = [1]
    assert json.loads(yatf.dump_events(yatf.parse_events(raw), jsonl=False)) == raw
