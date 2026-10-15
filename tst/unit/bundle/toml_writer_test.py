"""The TOML a bundle import writes reads back as what was written."""

import math
import tomllib

import pytest

from agent_env.bundle._toml import dumps

DOCUMENT = {
    "tasks": ["a", "b c", 'q"\\x\n\x01\x7fé'],
    "one": ["x"],
    "none": [],
    "n": 3,
    "f": 1.5,
    "b": True,
    "image": {"artifact": "img/1", "version": 2},
    "mcp_server_envs": ["e", {"env": "crm", "version": 2}],
    "nested": {"l": [1, 2, {"k": [3, 4]}]},
    "weird key": "v",
    "metadata": {"default_model": "m", "min_disk_size_gb": 20, "tags": ["x", "y"]},
}


def test_what_it_writes_reads_back_the_same():
    assert tomllib.loads(dumps(DOCUMENT)) == DOCUMENT


def test_a_list_of_several_spans_lines_only_at_the_top():
    assert dumps({"tasks": ["a", "b"], "t": {"l": ["a", "b"]}}) == (
        'tasks = [\n  "a",\n  "b",\n]\nt = { l = ["a", "b"] }\n')


def test_an_empty_document_is_empty():
    assert dumps({}) == ""


@pytest.mark.parametrize("value", [None, math.nan, math.inf, object()])
def test_a_value_toml_cant_hold_is_refused(value):
    with pytest.raises(ValueError):
        dumps({"k": value})
