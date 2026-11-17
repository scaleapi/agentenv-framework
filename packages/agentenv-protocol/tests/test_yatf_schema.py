"""yatf/1's strict JSON schema: generated from the models, checked in, and stricter than a reader."""
import copy
import subprocess
import sys
from importlib.resources import files

import jsonschema
import pytest

from agentenv_protocol import yatf
from agentenv_protocol.yatf.schema import SCHEMA_FILE, schema_text

VALIDATOR = jsonschema.Draft202012Validator(yatf.load_schema())


def example_events():
    return [yatf.dump_event(event) for name in yatf.EXAMPLES for event in yatf.load_example(name)]


def test_the_checked_in_schema_is_the_generated_one():
    assert files("agentenv_protocol.yatf").joinpath(SCHEMA_FILE).read_text(encoding="utf-8") == schema_text()


def test_python_m_prints_the_schema():
    out = subprocess.run([sys.executable, "-m", "agentenv_protocol.yatf", "schema"], capture_output=True, text=True,
                         check=True)
    assert out.stdout == schema_text() and out.stderr == ""


def test_every_example_event_is_valid():
    for raw in example_events():
        VALIDATOR.validate(raw)


@pytest.mark.parametrize("change", [
    lambda e: e.update(later_field=1),
    lambda e: e["time"].update(leap=True),
    lambda e: e.update(observer="robot"),
    lambda e: e.update(type="delegation"),
    lambda e: e.update(type="yatf/thing"),
    lambda e: e.pop("time"),
])
def test_what_a_reader_keeps_a_producer_may_not_write(change):
    raw = copy.deepcopy(next(e for e in example_events() if e["type"] == "message"))
    change(raw)
    assert not VALIDATOR.is_valid(raw)
    if "time" in raw:
        yatf.parse_event(raw)


def test_extension_events_and_origins_are_open():
    vendor = {"seq": 0, "id": "0.0", "type": "claude-code/raw", "turn": 0, "actor": "agent", "observer": "agent",
              "time": {"at": None}, "record": {"anything": [1, 2]}}
    VALIDATOR.validate(vendor)
    inbound = next(e for e in example_events() if e["type"] == "message" and e["actor"] == "user_sim")
    assert "channel" in inbound["content"][0]["origin"]
    VALIDATOR.validate(inbound)
