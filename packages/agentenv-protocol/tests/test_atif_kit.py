"""The adapter kit: the toy adapter passes against its own fixtures, and the kit names what's wrong with an adapter that
writes a duplicate tool_call_id, a document other than the expected one, or a from_atif that gives no native log."""
import json
from typing import Any

import pytest

from agentenv_protocol.atif import (
    AdapterRun,
    Fed,
    Native,
    RecordedTask,
    TurnContext,
    atif_adapter,
    check_adapter,
    native,
    strip_profile,
)
from atif_cases import CASES, FIXTURES, Toy

DECLARED = {"native_format": "toy-events/1", "version": "0.1.0", "input_includes_cache": False,
            "tested_with": ("toy 1.0",)}


@atif_adapter(**DECLARED)
class Reversible(Toy):
    @classmethod
    def from_atif(cls, document: Any) -> Native:
        return Native(log=[{"type": "say", "step": f"s{step['step_id']}", "text": step["message"]}
                           for step in document["steps"] if step["source"] == "agent"
                           and isinstance(step.get("message"), str)])


@atif_adapter(**DECLARED)
class Duplicating(Reversible):
    @native(type="call")
    def call(self, r: dict, turn: Any) -> None:
        turn.tool_call(step=r["step"], tool_call_id="toolu_01", function_name=r["name"], arguments=r.get("args"))


@atif_adapter(**DECLARED)
class NoLog(Toy):
    @classmethod
    def from_atif(cls, document: Any) -> Any:
        return {"log": []}


def recorded(name: str) -> tuple[list[RecordedTask], dict[str, bytes]]:
    """A case's tasks as the SDK records them, and the files of the media their inbound content attaches."""
    store, out = AdapterRun(Toy), []
    for task in CASES[name]:
        context = dict(task.context)
        inbound = list(context.pop("inbound", []))
        inbound += [store.media(label.encode(), media_type, **info) for label, media_type, info in task.inbound_media]
        if task.inbound_media and name == "odd_parts":
            inbound.append({"type": "text", "text": json.dumps({"ticket": 4411, "priority": "high"})})
        out.append(RecordedTask(context=TurnContext(inbound=inbound, **context), records=[Fed(r) for r in task.records],
                                state=task.state, final=task.final, ended_at=task.ended_at))
    return out, store.media_files


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_kit_passes_the_toy_adapter_against_its_fixture(name):
    stripped = json.loads((FIXTURES / "coding.stripped.json").read_text(encoding="utf-8"))
    tasks, media = recorded(name)
    result = check_adapter(Reversible, tasks, expected=FIXTURES / f"{name}.json",
                           from_atif=[stripped, strip_profile(_fixture(name))], media=media)
    assert result.errors == []
    assert result.document == _fixture(name)
    assert len(result.pages) == sum(len(task.records) + 2 for task in CASES[name])
    unmatched = "unmatched" in _fixture(name)["extra"]["agentenv"]
    assert {f.rule for f in result.findings} == ({"kit-unmatched"} if unmatched else set())


def test_the_kit_warns_of_native_records_no_handler_takes():
    (finding,) = check_adapter(Reversible, recorded("coding")[0]).findings
    assert (finding.rule, finding.severity, finding.message) == (
        "kit-unmatched", "warning", "no handler takes these native records: system (1)")


def test_the_kit_fails_an_adapter_that_writes_a_duplicate_tool_call_id():
    result = check_adapter(Duplicating, recorded("coding")[0])
    (finding,) = result.errors
    assert (finding.rule, finding.where) == ("kit-run", "task 0, record 4")
    assert "tool_call_id 'toolu_01' is already in this document" in finding.message
    with pytest.raises(AssertionError, match="kit-run"):
        result.raise_for_errors()


def test_the_kit_names_where_the_document_differs_from_the_expected_one():
    expected = _fixture("coding")
    expected["steps"][5]["message"] = "something else"
    (finding,) = check_adapter(Reversible, recorded("coding")[0], expected=expected).errors
    assert (finding.rule, finding.where) == ("kit-expected", "steps[5].message")


def test_the_kit_fails_a_from_atif_that_gives_no_native_log_and_warns_when_there_is_none():
    (failed,) = check_adapter(NoLog, recorded("late_result")[0]).errors
    assert (failed.rule, failed.message) == ("kit-from-atif", "returned dict, not a Native holding a list log")
    result = check_adapter(Toy, recorded("late_result")[0])
    assert result.errors == []
    assert [(f.rule, f.severity) for f in result.findings] == [("kit-from-atif", "warning")]
