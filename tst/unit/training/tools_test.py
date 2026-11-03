"""The tool methods TRL finds an env's tools by: their signatures and docstrings from each tool's JSON Schema, the
schema transformers' ``get_json_schema`` reads back from them, and the tools refused; and the ``SchemaTool`` that shows
a model the tool as its env publishes it, rendered by Qwen3's chat template and parsed back from its tool call."""

import inspect
import json
from pathlib import Path
from typing import Any, Literal

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
from transformers.utils import DocstringParsingException, get_json_schema
from transformers.utils.chat_parsing import parse_response

from agent_env.training._tools import SchemaTool, tool_method

QWEN3 = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
    chat_template=(Path(__file__).parents[2] / "data" / "training" / "qwen3_chat_template.jinja").read_text())
#: Qwen3's response template as TRL sets it on the tokenizer, which ``parse_response`` reads a completion with.
QWEN3_RESPONSE = {"defaults": {"role": "assistant"}, "start_anchor": "<|im_start|>assistant\n", "fields": {
    "reasoning_content": {"open": "<think>", "close_pattern": r"</think>\s*", "content": "text"},
    "tool_calls": {"open": "<tool_call>", "close_pattern": r"</tool_call>\s*", "repeats": True, "content": "json",
                   "transform": {"type": "function", "function": "{content}"}},
    "content": {"close_pattern": r"<\|im_end\|>\s*", "content": "text"},
}}

RESTOCK = {"name": "restock", "description": "Restock a shelf.", "input_schema": {
    "type": "object", "required": ["shelf", "count"],
    "properties": {
        "shelf": {"type": "string", "enum": ["top", "bottom"], "description": "which shelf"},
        "count": {"type": "integer"},
        "labels": {"type": "array", "items": {"type": "string"}, "title": "Labels"},
        "note": {"type": ["string", "null"], "default": None, "description": "for the night shift"},
        "price": {"type": "number", "default": 1.5},
        "options": {"type": "object", "properties": {"rush": {"type": "boolean"}}, "required": ["rush"]},
    },
}}
PLAN = {"name": "check_plan", "description": "Check a plan.", "input_schema": {
    "type": "object", "required": ["plan"],
    "properties": {"plan": {"description": "one entry per ship", "anyOf": [
        {"type": "array", "items": {"type": "object", "required": ["ship"], "properties": {
            "ship": {"type": "integer", "description": "the ship's id"},
            "cranes": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}},
        {"type": "string"}]}},
}}


def _shown(tool: dict[str, Any]) -> dict[str, Any]:
    environment = type("Env", (), {tool["name"]: tool_method(tool)})()
    return get_json_schema(getattr(environment, tool["name"]))["function"]


def test_a_tools_arguments_are_typed_and_documented_from_its_schema():
    method = tool_method(RESTOCK)

    parameters = inspect.signature(method).parameters
    assert {name: (parameter.annotation, parameter.default) for name, parameter in parameters.items()} == {
        "self": (inspect.Parameter.empty, inspect.Parameter.empty),
        "shelf": (Literal["top", "bottom"], inspect.Parameter.empty),
        "count": (int, inspect.Parameter.empty),
        "labels": (list[str] | None, None),
        "note": (str | None, None),
        "price": (float | None, 1.5),
        "options": (dict | None, None),
    }
    assert method.__doc__ == (
        "Restock a shelf.\n\nArgs:\n    shelf: which shelf\n    count: count\n    labels: Labels\n"
        "    note: for the night shift\n    price: price\n    options: options Fields: rush (boolean, required).")
    assert tool_method({"name": "ping"}).__doc__ == "Call the ping tool."


def test_transformers_reads_back_each_tool_as_the_env_describes_it():
    assert _shown(RESTOCK) == {"name": "restock", "description": "Restock a shelf.", "parameters": {
        "type": "object", "required": ["shelf", "count"], "properties": {
            "shelf": {"type": "string", "enum": ["top", "bottom"], "description": "which shelf"},
            "count": {"type": "integer", "description": "count"},
            "labels": {"type": "array", "items": {"type": "string"}, "nullable": True, "description": "Labels"},
            "note": {"type": "string", "nullable": True, "description": "for the night shift"},
            "price": {"type": "number", "nullable": True, "description": "price"},
            "options": {"type": "object", "nullable": True, "description": "options Fields: rush (boolean, required)."},
        }}, "return": {"type": "string"}}
    assert _shown(PLAN)["parameters"] == {"type": "object", "required": ["plan"], "properties": {"plan": {
        "anyOf": [{"type": "array", "items": {"type": "object"}}, {"type": "string"}],
        "description": "one entry per ship Each entry: ship (integer, required): the ship's id; cranes (integer or "
                       "null)."}}}
    assert _shown({"name": "ping"}) == {"name": "ping", "description": "Call the ping tool.",
                                        "parameters": {"type": "object", "properties": {}}, "return": {"type": "string"}}


@pytest.mark.parametrize(("tool", "message"), [
    ({"name": "get-situation"}, "can't be a TRL tool"),
    ({"name": "reset"}, "can't be a TRL tool"),
    ({"name": "_inspect"}, "can't be a TRL tool"),
    ({"name": "plan", "input_schema": {"properties": {"ship-id": {"type": "integer"}}}}, "argument 'ship-id'"),
    ({"name": "upload", "input_schema": {"properties": {"file": {"type": "file"}}}}, r"upload\.file: JSON Schema"),
], ids=lambda value: value["name"] if isinstance(value, dict) else "")
def test_a_tool_trl_cant_show_the_model_is_refused(tool: dict[str, Any], message):
    with pytest.raises(ValueError, match=message):
        tool_method(tool)


def test_a_schema_tool_shows_the_model_its_envs_schema_nested_types_included_and_calls_through():
    called = []
    tool = SchemaTool(PLAN, lambda **arguments: called.append(arguments) or "planned")

    assert shown_to_qwen3([tool]) == [{"type": "function", "function": {
        "name": "check_plan", "description": "Check a plan.", "parameters": PLAN["input_schema"]}}]
    [call] = called_by_qwen3([tool], {"name": "check_plan", "arguments": {"plan": json.dumps([{"ship": 3}])}})
    assert call == {"type": "function", "function": {"name": "check_plan", "arguments": {"plan": [{"ship": 3}]}}}
    assert tool(**call["function"]["arguments"]) == "planned"
    assert called == [{"plan": [{"ship": 3}]}]
    with pytest.raises(DocstringParsingException, match="for check_plan because it has no docstring"):
        get_json_schema(tool)


def shown_to_qwen3(tools: list[Any]) -> list[dict[str, Any]]:
    """The tools Qwen3's chat template shows the model, as TRL renders a prompt."""
    return tools_in(QWEN3.apply_chat_template([{"role": "user", "content": "Plan."}], tools=tools, tokenize=False))


def tools_in(prompt: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in prompt.split("<tools>\n")[1].split("\n</tools>")[0].splitlines()]


def called_by_qwen3(tools: list[Any], *calls: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool calls ``parse_response`` reads from Qwen3's completion making ``calls``, typed by ``tools``' schemas as
    TRL parses a completion."""
    prompt = QWEN3.apply_chat_template([{"role": "user", "content": "Plan."}], tools=tools, tokenize=False,
                                       add_generation_prompt=True)
    completion = "<think>\n\n</think>\n\n" + "".join(f"<tool_call>\n{json.dumps(call)}\n</tool_call>" for call in calls)
    return parse_response(completion + "<|im_end|>", QWEN3_RESPONSE, prefix=prompt, tools=tools)["tool_calls"]
