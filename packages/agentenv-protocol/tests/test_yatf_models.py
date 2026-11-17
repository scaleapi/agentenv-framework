"""yatf/1's models: each type parses as its own model, and a reader keeps what this version doesn't define."""
import subprocess
import sys

import pytest

from agentenv_protocol import yatf


def event(kind: str, **body):
    return {"seq": 0, "id": "0.0", "type": kind, "turn": 0, "actor": "agent", "observer": "agent",
            "time": {"at": None}, **body}


def test_every_core_type_parses_as_its_model():
    bodies = {
        "turn.start": {"spec_version": "1.0", "context_id": "c", "task_id": "t", "native_format": "toy/1",
                       "adapter": {"name": "toy-yatf", "version": "0.1.0"}, "usage_input_includes_cache": None,
                       "actors": [{"id": "agent", "kind": "agent"}]},
        "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        "final_response": {"content": [{"type": "data", "data": {"tool_call_count": 0}}]},
        "turn.end": {"state": "completed"},
        "step": {"source": "agent", "metrics": {"prompt_tokens": 3, "cost_usd": None, "cost_status": "unpriced"}},
        "reasoning": {"visibility": "omitted", "text": None, "opaque": [{"kind": "signature", "value": "sig"}]},
        "tool_call": {"tool_call_id": "c1", "function_name": "Bash", "arguments": None},
        "tool_result": {"source_call_id": "c1", "status": "ok", "content": []},
        "tools": {"mode": "replace", "tools": [{"name": "Bash", "kind": "builtin"}]},
        "actor": {"declared": {"id": "sub", "kind": "subagent", "parent": "agent", "spawned_by": "c1"}},
        "error": {"error_type": "infra_error", "message": "upstream 529"},
        "attempt": {"number": 2, "reason": "api error"},
        "note": {"text": "compacted"},
    }
    assert set(bodies) == yatf.CORE_TYPES
    for kind, body in bodies.items():
        parsed = yatf.parse_event(event(kind, **body))
        assert type(parsed) is yatf.EVENT_MODELS[kind]
        assert yatf.dump_event(parsed) == event(kind, **body)


def test_a_reader_keeps_fields_this_version_does_not_define():
    raw = event("message", role="assistant", step="s1", later_field={"a": 1},
                content=[{"type": "text", "text": "hi", "tone": "dry"}])
    raw["time"]["leap"] = True
    parsed = yatf.parse_event(raw)
    assert parsed.model_extra == {"later_field": {"a": 1}}
    assert yatf.dump_event(parsed) == raw


def test_unknown_and_reserved_types_are_kept_and_namespaced_ones_are_extensions():
    later = yatf.parse_event(event("delegation", to="peer"))
    vendor = yatf.parse_event(event("claude-code/raw", record={"type": "system"}))
    assert type(later) is yatf.UnknownEvent and later.model_extra == {"to": "peer"}
    assert type(vendor) is yatf.ExtensionEvent and vendor.model_extra == {"record": {"type": "system"}}
    assert type(yatf.parse_event(event("yatf/thing"))) is yatf.UnknownEvent
    assert "delegation" in yatf.RESERVED_TYPES


def test_a_reader_keeps_part_types_and_vocabulary_values_this_version_lacks():
    raw = event("message", role="narrator", observer="robot", content=[{"type": "hologram", "frames": 3}])
    parsed = yatf.parse_event(raw)
    assert parsed.observer == "robot" and parsed.role == "narrator"
    assert type(parsed.content[0]) is yatf.UnknownPart
    assert yatf.dump_event(parsed) == raw


def test_time_at_is_always_present_and_may_be_null():
    assert yatf.parse_event(event("note", text="x")).time.at is None
    missing = event("note", text="x")
    missing["time"] = {}
    with pytest.raises(yatf.YatfError):
        yatf.parse_event(missing)


def test_a_dump_writes_only_the_fields_the_event_was_given():
    dumped = yatf.dump_event(yatf.parse_event(event("note", text="x")))
    assert "attempt" not in dumped and "derived_from" not in dumped and dumped["time"] == {"at": None}


def test_a_lone_surrogate_may_be_in_a_value_but_not_a_field_name():
    kept = event("note", text="cut \ud83d", later={"\ud800": ["\udfff"]})
    assert yatf.dump_event(yatf.parse_event(kept)) == kept
    with pytest.raises(yatf.YatfError):
        yatf.parse_event({**event("note", text="x"), "later\ud800": 1})


@pytest.mark.parametrize("value", [{"type": "note"}, [], "note", {**event("note", text="x"), "seq": "zero"}])
def test_a_non_event_is_refused(value):
    with pytest.raises(yatf.YatfError):
        yatf.parse_event(value)


@pytest.mark.parametrize("field,value", [("seq", "5"), ("seq", 5.0), ("seq", True), ("attempt", False),
                                         ("text", 5), ("is_copied_context", "yes")])
def test_values_are_taken_as_given_never_coerced(field, value):
    with pytest.raises(yatf.YatfError):
        yatf.parse_event({**event("note", text="x"), field: value})


def test_a_number_dumps_as_it_was_written_and_nan_is_refused():
    raw = event("step", source="agent", metrics={"cost_usd": 0, "cost_status": "reported", "prompt_tokens": 3})
    assert yatf.dump_event(yatf.parse_event(raw))["metrics"]["cost_usd"] == 0
    assert type(yatf.dump_event(yatf.parse_event(raw))["metrics"]["cost_usd"]) is int
    raw["metrics"]["cost_usd"] = float("nan")
    with pytest.raises(yatf.YatfError):
        yatf.parse_event(raw)


@pytest.mark.parametrize("later", [float("inf"), [float("-inf")], {"deep": [float("nan")]}])
def test_a_number_json_cannot_write_is_refused_anywhere_in_an_event(later):
    with pytest.raises(yatf.YatfError, match="isn't a JSON number"):
        yatf.parse_event({**event("note", text="x"), "later": later})


@pytest.mark.parametrize("kind,ok", [("claude-code/raw", True), ("codex/item.v2", True), ("yatf/raw", False),
                                     ("delegation", False), ("Claude/raw", False), ("a/b/c", False)])
def test_extension_types_are_namespaced_outside_yatf(kind, ok):
    assert yatf.is_extension_type(kind) is ok


def test_the_package_imports_without_the_agent_sdk():
    script = ("import sys, agentenv_protocol.yatf; "
              "sys.exit(any(name == 'a2a' or name.startswith('a2a.') for name in sys.modules))")
    assert subprocess.run([sys.executable, "-c", script]).returncode == 0
