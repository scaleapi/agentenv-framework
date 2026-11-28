"""A turn whose calls used several models (``Usage.models``) is priced model by model, each at its own rate from the
price map, its step's entry keeping a row per model; a turn without them is priced at the role's model, as before."""

from dataclasses import replace

import pytest
from agentenv_protocol.a2a_agent import ModelUsage, Usage

from agent_env.a2a_agent.protocol import TerminalResponse, read_usage
from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import ModelPrice, PriceTable
from agent_env.task_step import usage as usage_module
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.usage import StepUsage

SONNET, HAIKU = "claude-sonnet-4-6", "claude-haiku-4-5"
PRICES = PriceTable({SONNET: ModelPrice(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75),
                     HAIKU: ModelPrice(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)}, "litellm@test")
# A Sonnet solver's turn whose subagent ran on Haiku: Sonnet's 500 uncached input, 400 output, 8,000 cache reads and
# 500 cache writes cost $0.011775; Haiku's 3,000 input and 90 output, $0.00345.
SONNET_SHARE = ModelUsage(model=SONNET, calls=2, input_tokens=9000, output_tokens=400, cache_read_tokens=8000,
                          cache_write_tokens=500, input_includes_cache=True)
HAIKU_SHARE = ModelUsage(model=HAIKU, calls=1, input_tokens=3000, output_tokens=90, input_includes_cache=True)
TURN = Usage(input_tokens=12000, output_tokens=490, total_tokens=12490, cache_read_tokens=8000, cache_write_tokens=500,
             input_includes_cache=True, models=(SONNET_SHARE, HAIKU_SHARE))


def _entry(monkeypatch, *turns: Usage, prices: PriceTable = PRICES) -> dict:
    monkeypatch.setattr(usage_module, "litellm_prices", lambda: prices)
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, SONNET, None)
    for turn in turns:
        usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=turn))
    [entry] = context.metadata["step_usage"]["solve"]
    return entry


def test_a_turns_models_are_each_priced_at_their_own_rate(monkeypatch):
    entry = _entry(monkeypatch, TURN)

    sonnet, haiku = entry["models"]
    assert (sonnet["model"], sonnet["calls"], sonnet["cost_status"]) == (SONNET, 2, "priced")
    assert sonnet["cost_usd"] == pytest.approx(0.011775) and sonnet["uncached_input_tokens"] == 500
    assert (haiku["model"], haiku["cost_usd"], haiku["unit_prices"]["input"]) == (HAIKU, pytest.approx(0.00345), 1.0)
    assert (entry["cost_status"], entry["cost_source"]) == ("priced", "litellm@test")
    assert entry["cost_usd"] == pytest.approx(0.015225)
    assert "unit_prices" not in entry and "uncached_input_tokens" not in entry
    assert (entry["model"], entry["input_tokens"], entry["output_tokens"]) == (SONNET, 12000, 490)


def test_a_turn_without_models_beside_one_with_them_counts_as_the_roles_model(monkeypatch):
    plain = Usage(input_tokens=1000, output_tokens=10, input_includes_cache=True)

    entry = _entry(monkeypatch, TURN, plain)

    sonnet, haiku = entry["models"]
    assert (sonnet["model"], sonnet["calls"], sonnet["input_tokens"], sonnet["output_tokens"]) == (SONNET, 2, 10000, 410)
    assert sonnet["cost_usd"] == pytest.approx(0.011775 + (1000 * 3.0 + 10 * 15.0) / 1e6)
    assert entry["cost_usd"] == pytest.approx(sonnet["cost_usd"] + haiku["cost_usd"])


def test_a_model_the_map_lacks_leaves_the_entry_unpriced_and_each_models_cost_is_kept(monkeypatch):
    turn = Usage(input_tokens=9100, output_tokens=410, input_includes_cache=True, models=(
        SONNET_SHARE, ModelUsage(model="mystery-1", input_tokens=100, output_tokens=10, input_includes_cache=True)))

    entry = _entry(monkeypatch, turn)

    assert (entry["cost_usd"], entry["cost_status"], entry["unpriced_reason"]) == (None, "unpriced", "no_price")
    sonnet, mystery = entry["models"]
    assert sonnet["cost_status"] == "priced" and sonnet["cost_usd"] == pytest.approx(0.011775)
    assert (mystery["cost_usd"], mystery["unpriced_reason"]) == (None, "no_price")


def test_a_models_own_cost_stands_in_where_the_map_lacks_the_model(monkeypatch):
    turn = Usage(input_tokens=9100, output_tokens=410, input_includes_cache=True, cost_usd=0.02, models=(
        SONNET_SHARE, ModelUsage(model="mystery-1", input_tokens=100, output_tokens=10, input_includes_cache=True,
                                 cost_usd=0.002)))

    entry = _entry(monkeypatch, turn)

    assert (entry["cost_status"], entry["cost_source"]) == ("reported", "harness")
    assert entry["cost_usd"] == pytest.approx(0.011775 + 0.002)
    assert entry["reported_cost_usd"] == 0.02


def test_an_entry_whose_turns_have_no_models_keeps_its_shape(monkeypatch):
    entry = _entry(monkeypatch, Usage(input_tokens=1000, output_tokens=10, input_includes_cache=True))

    assert "models" not in entry and entry["unit_prices"]["input"] == 3.0
    assert entry["cost_usd"] == pytest.approx((1000 * 3.0 + 10 * 15.0) / 1e6)


def test_the_host_reads_a_turns_models_and_keeps_a_bad_one_aside():
    usage, unparsed = read_usage({"input_tokens": 3000, "output_tokens": 90,
                                  "models": [{"model": HAIKU, "input_tokens": 3000, "output_tokens": 90}]})
    assert usage.models == (ModelUsage(model=HAIKU, input_tokens=3000, output_tokens=90),) and unparsed is None

    usage, unparsed = read_usage({"input_tokens": 3000, "models": [{"input_tokens": 3000}]})
    assert usage.models == () and usage.input_tokens == 3000 and "models" in unparsed


def test_a_turn_that_gives_only_its_rows_is_priced_from_them(monkeypatch):
    entry = _entry(monkeypatch, Usage(models=(SONNET_SHARE, HAIKU_SHARE)))

    assert entry["cost_usd"] == pytest.approx(0.015225) and entry["input_tokens"] == 12000


def test_an_agents_figure_for_a_model_the_map_lacks_stands_in_only_when_every_turn_gave_one(monkeypatch):
    def mystery(cost):
        return Usage(models=(ModelUsage(model="mystery-1", input_tokens=100, output_tokens=10,
                                        input_includes_cache=True, cost_usd=cost),))

    entry = _entry(monkeypatch, mystery(0.002), mystery(None))
    [row] = entry["models"]
    assert (row["cost_usd"], row["cost_status"]) == (None, "unpriced")
    assert (entry["cost_usd"], entry["cost_status"]) == (None, "unpriced")

    entry = _entry(monkeypatch, mystery(0.002), mystery(0.003))
    assert (entry["cost_status"], entry["cost_usd"]) == ("reported", pytest.approx(0.005))


def test_a_role_on_one_model_the_map_lacks_takes_the_agents_figure_only_when_every_turn_gave_one(monkeypatch):
    monkeypatch.setattr(usage_module, "litellm_prices", lambda: PRICES)
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "mystery-1", None)
    for cost in (0.002, None):
        usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=Usage(
            input_tokens=100, output_tokens=10, input_includes_cache=True, cost_usd=cost)))
    [entry] = context.metadata["step_usage"]["solve"]
    assert (entry["cost_usd"], entry["cost_status"], entry["reported_cost_usd"]) == (None, "unpriced", 0.002)


def test_a_cut_off_turn_beside_a_reported_one_keeps_the_agents_figure_beside_the_maps_price(monkeypatch):
    monkeypatch.setattr(usage_module, "litellm_prices", lambda: PRICES)
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, SONNET, None)
    usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=Usage(
        input_tokens=1000, output_tokens=10, input_includes_cache=True, cost_usd=0.004)))
    usage.cut_off(ModelRole.SOLVER)
    [entry] = context.metadata["step_usage"]["solve"]
    assert (entry["cost_status"], entry["reported_cost_usd"], entry["partial"]) == ("priced", 0.004, True)
    assert entry["cost_usd"] == pytest.approx((1000 * 3.0 + 10 * 15.0) / 1e6)


def test_a_turn_whose_rows_cant_all_be_priced_leaves_the_entry_unpriced_rather_than_dropping(monkeypatch):
    rows = (ModelUsage(model=SONNET, input_tokens=1000, output_tokens=10, input_includes_cache=True),
            ModelUsage(model=HAIKU, output_tokens=5))
    priced = Usage(input_tokens=1000, output_tokens=10, input_includes_cache=True)

    entry = _entry(monkeypatch, priced, Usage(models=rows))

    sonnet, haiku = entry["models"]
    assert (haiku["cost_usd"], haiku["unpriced_reason"]) == (None, "no_tokens")
    assert (entry["cost_usd"], entry["cost_status"]) == (None, "unpriced") and entry["turns"] == 2


# The same models with a rate for cache writes kept an hour, as Claude's are.
HOURLY = PriceTable({SONNET: ModelPrice(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75,
                                        cache_write_ttl_rates={3600: 6.0}),
                     HAIKU: ModelPrice(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25,
                                       cache_write_ttl_rates={3600: 2.0})},
                    "litellm@test")


def _split(hour: int) -> dict:
    return {"cache_creation": {"ephemeral_5m_input_tokens": 500 - hour, "ephemeral_1h_input_tokens": hour}}


def test_a_one_model_turn_gives_its_row_the_turns_cache_write_split(monkeypatch):
    turn = Usage(input_tokens=9000, output_tokens=400, cache_read_tokens=8000, cache_write_tokens=500,
                 input_includes_cache=True, models=(SONNET_SHARE,), provider_details=_split(300))

    entry = _entry(monkeypatch, turn, prices=HOURLY)

    [row] = entry["models"]
    assert row["cost_usd"] == pytest.approx((500 * 3.0 + 400 * 15.0 + 8000 * 0.3 + 200 * 3.75 + 300 * 6.0) / 1e6)
    assert (row["cache_write_ttl_tokens"], row["unit_prices"]["cache_write_1h"]) == ({"1h": 300}, 6.0)
    assert "cost_assumed" not in row and "cost_assumed" not in entry
    assert entry["cache_write_ttl_tokens"] == {"1h": 300}


def test_a_turn_on_several_models_with_writes_kept_an_hour_marks_its_rows_as_possibly_low(monkeypatch):
    entry = _entry(monkeypatch, replace(TURN, provider_details=_split(300)), prices=HOURLY)

    sonnet, haiku = entry["models"]
    assert sonnet["cost_usd"] == pytest.approx(0.011775) and sonnet["cost_assumed"] == ["default_cache_ttl"]
    assert "cost_assumed" not in haiku and entry["cost_assumed"] == ["default_cache_ttl"]


def test_a_turn_on_several_models_whose_split_misses_some_writes_marks_its_rows(monkeypatch):
    partial = {"cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 0}}

    entry = _entry(monkeypatch, replace(TURN, provider_details=partial), prices=HOURLY)

    sonnet, _ = entry["models"]
    assert sonnet["cost_assumed"] == ["default_cache_ttl"] and entry["cost_assumed"] == ["default_cache_ttl"]


def test_a_turn_on_several_models_whose_rows_say_how_long_their_writes_were_kept_prices_each_exactly(monkeypatch):
    sonnet = replace(SONNET_SHARE, cache_write_ttl_tokens={"5m": 200, "1h": 300})

    entry = _entry(monkeypatch, replace(TURN, models=(sonnet, HAIKU_SHARE)), prices=HOURLY)

    sonnet_row, haiku_row = entry["models"]
    assert sonnet_row["cost_usd"] == pytest.approx(0.011775 + 300 * (6.0 - 3.75) / 1e6)
    assert sonnet_row["cache_write_ttl_tokens"] == {"1h": 300} and "cost_assumed" not in entry
    assert "cache_write_ttl_tokens" not in haiku_row and entry["cache_write_ttl_tokens"] == {"1h": 300}


def test_a_row_whose_writes_were_kept_twelve_hours_is_charged_that_times_rate(monkeypatch):
    prices = PriceTable({SONNET: ModelPrice(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75,
                                            cache_write_ttl_rates={43200: 7.5}),
                         HAIKU: ModelPrice(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)}, "litellm@test")
    sonnet = replace(SONNET_SHARE, cache_write_ttl_tokens={"12h": 500})

    entry = _entry(monkeypatch, replace(TURN, models=(sonnet, HAIKU_SHARE)), prices=prices)

    sonnet_row, _ = entry["models"]
    assert sonnet_row["cost_usd"] == pytest.approx(0.011775 + 500 * (7.5 - 3.75) / 1e6)
    assert sonnet_row["cache_write_ttl_tokens"] == {"12h": 500} and "cost_assumed" not in entry


def test_rows_that_give_no_times_take_the_one_time_their_turn_kept_every_write(monkeypatch):
    prices = PriceTable({SONNET: ModelPrice(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75,
                                            cache_write_ttl_rates={43200: 7.5}),
                         HAIKU: ModelPrice(input=1.0, output=5.0, cache_read=0.1, cache_write=1.25)}, "litellm@test")
    twelve = {"cache_write_tokens_details": {"ttl_12h_tokens": 500}}

    entry = _entry(monkeypatch, replace(TURN, provider_details=twelve), prices=prices)

    sonnet_row, _ = entry["models"]
    assert sonnet_row["cost_usd"] == pytest.approx(0.011775 + 500 * (7.5 - 3.75) / 1e6)
    assert sonnet_row["cache_write_ttl_tokens"] == {"12h": 500} and "cost_assumed" not in entry


def test_a_rows_own_times_come_before_its_turns(monkeypatch):
    sonnet = replace(SONNET_SHARE, cache_write_ttl_tokens={"5m": 500})
    turn = replace(TURN, models=(sonnet, HAIKU_SHARE), provider_details=_split(500))

    entry = _entry(monkeypatch, turn, prices=HOURLY)

    sonnet_row, _ = entry["models"]
    assert sonnet_row["cost_usd"] == pytest.approx(0.011775) and "cache_write_ttl_tokens" not in sonnet_row
