"""The shared model relay keeps each answered call's model, as its request named it, so a turn's usage gives its
counts by model (``Usage.models``); a call that named none counts as the turn's configured model's."""

from agentenv_protocol.a2a_agent import ModelUsage, Usage

from agent_env.utils.model_relay import model_relay


def _call(inputs: int, output: int) -> Usage:
    return Usage(input_tokens=inputs, output_tokens=output, input_includes_cache=True)


def test_a_turns_calls_are_given_by_the_model_each_named_and_an_unnamed_one_by_the_configured_model():
    calls = model_relay.Calls()
    calls.add(200, _call(1000, 10), "gpt-4.1-mini")
    calls.add(200, _call(300, 5), "gpt-4o-mini")
    calls.add(200, _call(200, 2), None)
    calls.add(500, None, "gpt-4o-mini")

    usage = calls.usage(1, "gpt-4.1-mini")

    assert usage.models == (
        ModelUsage(model="gpt-4.1-mini", calls=2, input_tokens=1200, output_tokens=12, input_includes_cache=True),
        ModelUsage(model="gpt-4o-mini", calls=1, input_tokens=300, output_tokens=5, input_includes_cache=True))
    assert (usage.input_tokens, usage.output_tokens) == (1500, 17)


def test_a_turn_with_a_call_that_named_no_model_and_none_configured_gives_no_rows():
    calls = model_relay.Calls()
    calls.add(200, _call(1000, 10), "gpt-4.1-mini")
    calls.add(200, _call(200, 2), None)

    assert calls.usage(0).models == () and calls.usage(0).input_tokens == 1200


def test_a_requests_model_is_read_from_its_json_body():
    assert model_relay._request_model(b'{"model": "gpt-4.1-mini", "messages": []}') == "gpt-4.1-mini"
    assert model_relay._request_model(b"not json") is None
    assert model_relay._request_model(b'["a list"]') is None
    assert model_relay._request_model(b'{"model": ""}') is None
