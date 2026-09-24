"""The tool name of an action comes from the ``gen_ai.prompt`` attribute when it carries one; the span name
is the fallback. iOS bridges name every action span ``execute_tool`` and put the tool in the prompt, so
without this every action read as "execute_tool" and the tool-name patterns (``always_show_actions``,
``evidence_exclude_pattern``) never matched."""
import json

from agent_env.task_step.task_steps.verifiers.judge_utils.frame_selection import transcript_from_raw


def _span(name, prompt, screenshot="aGk="):
    return {"name": name, "attributes": {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.prompt": json.dumps(prompt),
        "gen_ai.completion": json.dumps({"screenshot": screenshot, "ok": True}),
    }}


def test_tool_name_comes_from_the_prompt_attribute_when_present():
    raw = json.dumps([
        _span("execute_tool", {"tool": "ios_type", "input": {"text": "Starbucks"}}),
        _span("execute_tool", {"tool": "ios_tap", "model_input": {"x": 1, "y": 2}, "input": {"x": 3, "y": 4}}),
    ])
    actions = transcript_from_raw(raw).actions
    assert [a.tool for a in actions] == ["ios_type", "ios_tap"]
    assert actions[0].args == {"text": "Starbucks"} and actions[1].args == {"x": 3, "y": 4}


def test_span_name_is_the_fallback_when_the_prompt_names_no_tool():
    raw = json.dumps([_span("click", {"x": 1, "y": 2}), _span("type_text", {"input": {"text": "hi"}})])
    assert [a.tool for a in transcript_from_raw(raw).actions] == ["click", "type_text"]
