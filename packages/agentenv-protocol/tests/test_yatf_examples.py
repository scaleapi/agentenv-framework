"""yatf/1's worked examples, the spec's canonical fixtures: each is one turn that reads back exactly as written and
agrees with itself."""
import json
from importlib.resources import files

import pytest

from agentenv_protocol import yatf


@pytest.mark.parametrize("name", yatf.EXAMPLES)
def test_an_example_reads_back_exactly_as_written(name):
    text = files("agentenv_protocol.yatf").joinpath("examples", f"{name}.json").read_text(encoding="utf-8")
    assert [yatf.dump_event(event) for event in yatf.load_example(name)] == json.loads(text)


@pytest.mark.parametrize("name", yatf.EXAMPLES)
def test_an_example_is_one_whole_turn(name):
    events = yatf.load_example(name)
    assert [type(events[0]), type(events[1]), type(events[-2]), type(events[-1])] == [
        yatf.TurnStart, yatf.Message, yatf.FinalResponse, yatf.TurnEnd]
    assert [event.seq for event in events] == list(range(len(events)))
    assert all(event.id == f"{event.turn}.{event.seq}" for event in events)
    start = events[0]
    assert start.spec_version == yatf.SPEC_VERSION and start.usage_input_includes_cache is not None
    declared = {actor.id for actor in start.actors}
    assert {event.actor for event in events} <= declared


@pytest.mark.parametrize("name", yatf.EXAMPLES)
def test_an_examples_totals_add_up(name):
    events = yatf.load_example(name)
    steps = [event for event in events if isinstance(event, yatf.StepEvent)]
    end = events[-1]
    totals = end.final_metrics
    assert totals.total_steps == len(steps)
    assert end.tool_call_count == sum(isinstance(event, yatf.ToolCall) for event in events)
    for field in ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens"):
        assert sum(getattr(step.metrics, field) for step in steps) == getattr(totals, f"total_{field}")
    for step in steps:
        assert step.metrics.cached_tokens + step.metrics.cache_write_tokens <= step.metrics.prompt_tokens


def test_every_call_in_the_examples_has_its_result():
    for name in yatf.EXAMPLES:
        pairs = yatf.pair_tool_calls(yatf.load_example(name))
        assert pairs and all(len(pair.results) == 1 for pair in pairs)


def test_load_example_names_the_examples_it_has():
    with pytest.raises(yatf.YatfError, match="coding"):
        yatf.load_example("chess")
