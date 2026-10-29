"""The tool methods TRL calls an env's tools through: their signatures and docstrings from each tool's JSON Schema, the
schema transformers' ``get_json_schema`` reads back from them, which is what TRL shows a model, and the tools refused."""

import inspect
from typing import Any, Literal

import pytest
from transformers.utils import get_json_schema

from agent_env.training._tools import tool_method

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
