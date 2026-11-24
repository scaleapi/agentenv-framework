"""``Usage.models``, a turn's counts by model: each ``ModelUsage`` validated like ``Usage``'s own counts, written by
``to_dict`` only when there are some, read back by ``from_dict`` from mappings, and ignored by an older reader."""

import pytest

from agentenv_protocol.a2a_agent import ModelUsage, Usage

SONNET = ModelUsage(model="claude-sonnet-4-6", calls=2, input_tokens=9000, output_tokens=400, cache_read_tokens=8000,
                    cache_write_tokens=500, input_includes_cache=True, cost_usd=0.012)
HAIKU = ModelUsage(model="claude-haiku-4-5", calls=1, input_tokens=3000, output_tokens=90, input_includes_cache=True)
TURN = Usage(tool_call_count=3, input_tokens=12000, output_tokens=490, cache_read_tokens=8000, cache_write_tokens=500,
             input_includes_cache=True, models=(SONNET, HAIKU))


def test_a_turns_models_round_trip_and_are_written_only_when_there_are_some():
    assert Usage.from_dict(TURN.to_dict()) == TURN
    assert TURN.to_dict()["models"] == [
        {"model": "claude-sonnet-4-6", "calls": 2, "input_tokens": 9000, "output_tokens": 400,
         "cache_read_tokens": 8000, "cache_write_tokens": 500, "input_includes_cache": True, "cost_usd": 0.012},
        {"model": "claude-haiku-4-5", "calls": 1, "input_tokens": 3000, "output_tokens": 90,
         "input_includes_cache": True}]
    assert "models" not in Usage(input_tokens=5).to_dict()
    assert not Usage(models=()).to_dict() and Usage(models=()).is_empty
    assert not Usage(models=(HAIKU,)).is_empty


def test_models_given_as_mappings_are_read_as_each_models_usage_and_an_unknown_key_is_ignored():
    usage = Usage(models=[{"model": "gpt-4.1-mini", "input_tokens": 10, "output_tokens": 2, "reasoning_tokens": 1}])
    assert usage.models == (ModelUsage(model="gpt-4.1-mini", input_tokens=10, output_tokens=2),)
    assert Usage.from_dict({"input_tokens": 10, "models": None}) == Usage(input_tokens=10)


@pytest.mark.parametrize("field", ["calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"])
@pytest.mark.parametrize("value", [-1, 1.5, "3", True])
def test_a_models_count_that_isnt_a_non_negative_integer_is_refused(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        ModelUsage(model="m", **{field: value})
    with pytest.raises(ValueError, match=field):
        Usage.from_dict({"models": [{"model": "m", field: value}]})


@pytest.mark.parametrize("row", [{"input_tokens": 1}, {"model": ""}, {"model": 7}])
def test_a_models_usage_must_name_its_model(row):
    with pytest.raises((ValueError, TypeError)):
        Usage(models=[row])


@pytest.mark.parametrize("models", [{"model": "m"}, "m", 3])
def test_models_must_be_a_list(models):
    with pytest.raises(TypeError, match="models must be a list"):
        Usage(models=models)


def test_a_models_own_cost_and_flag_are_checked_as_usages_are():
    with pytest.raises(ValueError, match="cost_usd cannot be negative"):
        ModelUsage(model="m", cost_usd=-1)
    with pytest.raises(ValueError, match="input_includes_cache must be true, false or null"):
        ModelUsage(model="m", input_includes_cache="yes")


def test_a_turn_that_gives_only_its_rows_takes_their_sums_as_its_counts():
    usage = Usage(models=(SONNET, HAIKU))
    assert (usage.input_tokens, usage.output_tokens, usage.input_includes_cache) == (12000, 490, True)
    assert (usage.cache_read_tokens, usage.cache_write_tokens) == (None, None)
    assert Usage(input_tokens=5, models=(HAIKU,)).input_tokens == 5


@pytest.mark.parametrize("models", [{}, "", 0, False])
def test_from_dict_refuses_a_models_value_that_isnt_a_list_even_when_empty(models):
    with pytest.raises(TypeError, match="models must be a list"):
        Usage.from_dict({"input_tokens": 1, "models": models})


def test_a_models_row_of_its_calls_sums_what_they_gave_and_counts_each_call():
    row = ModelUsage.of("m", [Usage(input_tokens=100, output_tokens=10, cache_read_tokens=80, input_includes_cache=True),
                              Usage(input_tokens=50, output_tokens=5, input_includes_cache=True), Usage()])
    assert row == ModelUsage(model="m", calls=3, input_tokens=150, output_tokens=15, cache_read_tokens=80,
                             input_includes_cache=True)


def test_a_models_row_takes_its_calls_cost_only_when_every_call_reported_one():
    assert ModelUsage.of("m", [Usage(input_tokens=1, cost_usd=0.25), Usage(input_tokens=1, cost_usd=0.5)]).cost_usd == 0.75
    assert ModelUsage.of("m", [Usage(input_tokens=1, cost_usd=0.25), Usage(input_tokens=1)]).cost_usd is None


def test_a_models_row_takes_the_flag_its_counted_calls_agree_on():
    assert ModelUsage.of("m", [Usage(input_tokens=1, input_includes_cache=True), Usage()]).input_includes_cache is True
    assert ModelUsage.of("m", [Usage(input_tokens=1, input_includes_cache=True),
                               Usage(input_tokens=1, input_includes_cache=False)]).input_includes_cache is None
