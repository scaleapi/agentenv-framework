import importlib.metadata
import logging
import subprocess
import sys

import litellm
import pytest
from agentenv_protocol.a2a_agent import Usage
from litellm.types.utils import CacheCreationTokenDetails, PromptTokensDetailsWrapper
from litellm.types.utils import Usage as LiteLLMUsage

from agent_env.config import model_pricing as prices_module
from agent_env.config.model import JUDGE_FALLBACK_MODEL
from agent_env.config.model_pricing import (
    PER_MILLION,
    Assumed,
    ContextTier,
    CostStatus,
    ModelPrice,
    PriceTable,
    ReportedBy,
    Tokens,
    Unpriced,
    litellm_prices,
    price,
)

# Entries as LiteLLM's map writes them, in USD per token, beside ones it can't price by the token.
COST_MAP = {
    "sample_spec": {"input_cost_per_token": "the cost per input token", "output_cost_per_token": 0},
    "gpt-4o-mini": {"input_cost_per_token": 1.5e-07, "output_cost_per_token": 6e-07,
                    "cache_read_input_token_cost": 7.5e-08, "cache_creation_input_token_cost": 3e-07},
    "my-finetune": {"input_cost_per_token": 1e-06, "output_cost_per_token": 4e-06, "cache_read_input_token_cost": 1e-07},
    "meta-llama/Llama-3-70b": {"input_cost_per_token": 5e-07, "output_cost_per_token": 5e-07},
    "free-local": {"input_cost_per_token": 0, "output_cost_per_token": 0},
    "dall-e-3": {"output_cost_per_pixel": 1e-07},
    "broken": {"input_cost_per_token": -1e-06, "output_cost_per_token": 1e-06},
    "half": {"input_cost_per_token": 1e-06},
}
TABLE = PriceTable.from_cost_map(COST_MAP, "litellm@test")
USAGE = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_tokens": None,
         "cache_write_tokens": None, "input_includes_cache": None, "cost_usd": None}


def _agent(usage, model="gpt-4o-mini", *, table=TABLE):
    return price(Tokens.of(usage), usage["cost_usd"], model=model, prices=table, reported_by=ReportedBy.HARNESS)


def _judge(usage, model="gpt-4o-mini"):
    return price(Tokens.of(usage), usage["cost_usd"], model=model, prices=TABLE, reported_by=ReportedBy.PROVIDER)


def test_the_map_is_read_per_million_tokens_keeping_only_entries_with_an_input_and_an_output_price():
    assert TABLE.entries == {
        "gpt-4o-mini": ModelPrice(input=0.15, output=0.6, cache_read=0.075, cache_write=0.3),
        "my-finetune": ModelPrice(input=1.0, output=4.0, cache_read=0.1),
        "meta-llama/Llama-3-70b": ModelPrice(input=0.5, output=0.5),
        "free-local": ModelPrice(input=0.0, output=0.0),
    }
    assert TABLE.source == "litellm@test"


def test_the_installed_litellms_map_is_read_without_importing_litellm():
    probe = ("import sys; from agent_env.config.model_pricing import litellm_prices; t = litellm_prices(); "
             "print(t.source, t.lookup('gpt-4.1-mini'), 'litellm' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()

    version = importlib.metadata.version("litellm")
    assert out == (f"litellm@{version} ModelPrice(input=0.4, output=1.6, cache_read=0.1, cache_write=None, "
                   "cache_write_ttl_rates={}, tiers=()) False")
    assert len(litellm_prices().entries) > 1000


def test_the_judge_a_run_falls_back_to_is_priced_by_the_installed_map():
    assert litellm_prices().lookup(JUDGE_FALLBACK_MODEL) is not None


def test_a_map_that_cant_be_read_prices_nothing_and_says_why(monkeypatch, caplog):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(prices_module.importlib.metadata, "distribution", missing)
    litellm_prices.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            table = litellm_prices()
    finally:
        litellm_prices.cache_clear()

    assert table.entries == {} and _agent(USAGE, table=table).unpriced is Unpriced.NO_PRICE
    assert "Usage stays unpriced, since LiteLLM's price map can't be read" in caplog.text


def test_a_routing_prefix_is_stripped_to_find_a_model():
    assert TABLE.lookup("openai/gpt-4o-mini") is TABLE.lookup("gpt-4o-mini") is not None
    assert TABLE.lookup("meta-llama/Llama-3-70b") is TABLE.lookup("hosted_vllm/meta-llama/Llama-3-70b") is not None


@pytest.mark.parametrize("model", ["gpt-4o", "openai/gpt-4o", "Llama-3-70b", "dall-e-3", "broken", "half", ""])
def test_a_model_the_map_cant_price_has_no_price_never_a_zero_one(model):
    assert TABLE.lookup(model) is None
    cost = _agent(USAGE, model or None)
    assert (cost.usd, cost.status) == (None, CostStatus.UNPRICED)


def test_a_model_the_map_lists_as_free_costs_nothing():
    cost = _agent(USAGE, "free-local")
    assert (cost.usd, cost.status) == (0.0, CostStatus.PRICED)


AGENT_BRANCHES = {
    # name: (model, reported) -> (status, source, usd)
    "priced": (("gpt-4o-mini", None), (CostStatus.PRICED, "litellm@test", 0.15 + 0.06)),
    "priced-keeps-reported-beside": (("gpt-4o-mini", 9.0), (CostStatus.PRICED, "litellm@test", 0.21)),
    "reported-without-a-price": (("gpt-4o", 9.0), (CostStatus.REPORTED, "harness", 9.0)),
    "unpriced": (("gpt-4o", None), (CostStatus.UNPRICED, None, None)),
    "no-model": ((None, None), (CostStatus.UNPRICED, None, None)),
}


@pytest.mark.parametrize("given, expected", AGENT_BRANCHES.values(), ids=AGENT_BRANCHES.keys())
def test_an_agents_cost(given, expected):
    model, reported = given
    cost = _agent({**USAGE, "cost_usd": reported}, model)
    assert (cost.status, cost.source) == expected[:2]
    assert cost.usd == pytest.approx(expected[2]) if expected[2] is not None else cost.usd is None
    assert cost.reported_usd == reported


JUDGE_BRANCHES = {
    "proxy-cost-first": (("gpt-4o-mini", 0.5), (CostStatus.REPORTED, "provider", 0.5)),
    "priced-without-a-proxy-cost": (("gpt-4o-mini", None), (CostStatus.PRICED, "litellm@test", 0.21)),
    "unpriced": (("claude-sonnet-4-6", None), (CostStatus.UNPRICED, None, None)),
}


@pytest.mark.parametrize("given, expected", JUDGE_BRANCHES.values(), ids=JUDGE_BRANCHES.keys())
def test_a_judges_cost(given, expected):
    model, proxy_cost = given
    cost = _judge({**USAGE, "cost_usd": proxy_cost}, model)
    assert (cost.status, cost.source) == expected[:2]
    assert cost.usd == pytest.approx(expected[2]) if expected[2] is not None else cost.usd is None


@pytest.mark.parametrize("usage, model, why", [
    (USAGE, None, Unpriced.NO_MODEL),
    (USAGE, "gpt-4o", Unpriced.NO_PRICE),
    ({**USAGE, "input_tokens": None}, "gpt-4o-mini", Unpriced.NO_TOKENS),
    ({**USAGE, "output_tokens": None}, "gpt-4o-mini", Unpriced.NO_TOKENS),
])
def test_an_unpriced_cost_says_why(usage, model, why):
    cost = _agent(usage, model)
    assert cost.fields() == {"cost_usd": None, "cost_status": "unpriced", "cost_source": None,
                             "reported_cost_usd": None, "unpriced_reason": why.value}


def test_cached_input_is_charged_once_whichever_way_input_counts_it():
    # 1,000 uncached input, 600 cache reads, 200 cache writes, 50 output
    counted = _agent({**USAGE, "input_tokens": 1800, "output_tokens": 50, "cache_read_tokens": 600,
                      "cache_write_tokens": 200, "input_includes_cache": True})
    apart = _agent({**USAGE, "input_tokens": 1000, "output_tokens": 50, "cache_read_tokens": 600,
                    "cache_write_tokens": 200, "input_includes_cache": False})
    expected = (1000 * 0.15 + 50 * 0.6 + 600 * 0.075 + 200 * 0.3) / PER_MILLION
    assert counted.usd == pytest.approx(expected) and apart.usd == pytest.approx(expected)
    assert counted.assumed == apart.assumed == ()


def test_an_unknown_includes_cache_flag_is_priced_as_an_upper_bound_and_marked():
    unknown = _agent({**USAGE, "input_tokens": 1800, "output_tokens": 50, "cache_read_tokens": 600,
                      "cache_write_tokens": 200, "input_includes_cache": None})
    as_if_apart = (1800 * 0.15 + 50 * 0.6 + 600 * 0.075 + 200 * 0.3) / PER_MILLION
    assert unknown.usd == pytest.approx(as_if_apart)
    assert unknown.assumed == (Assumed.INPUT_EXCLUDES_CACHE,)
    assert unknown.fields()["cost_assumed"] == ["input_excludes_cache"]


def test_counts_that_prove_input_leaves_out_the_cache_are_priced_so_unmarked():
    proven = _agent({**USAGE, "input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 600,
                     "cache_write_tokens": 200, "input_includes_cache": None})
    assert proven.usd == pytest.approx((100 * 0.15 + 50 * 0.6 + 600 * 0.075 + 200 * 0.3) / PER_MILLION)
    assert proven.assumed == ()


def test_records_that_disagree_on_whether_input_counts_the_cache_add_up_to_their_own_uncached_input():
    counted = Tokens.of({"input_tokens": 100, "output_tokens": 1, "cache_read_tokens": 80, "input_includes_cache": True})
    apart = Tokens.of({"input_tokens": 20, "output_tokens": 1, "cache_read_tokens": 80, "input_includes_cache": False})
    unknown = Tokens.of({"input_tokens": 50, "output_tokens": 1, "cache_read_tokens": 30, "input_includes_cache": None})

    assert counted + apart == Tokens(uncached_input=40, output=2, cache_read=160, prompt=100)
    assert counted + apart + unknown == Tokens(uncached_input=90, output=3, cache_read=190, assumed_uncached=True,
                                               prompt=100)
    cost = price(counted + apart, None, model="gpt-4o-mini", prices=TABLE, reported_by=ReportedBy.HARNESS)
    assert cost.usd == pytest.approx((40 * 0.15 + 2 * 0.6 + 160 * 0.075) / PER_MILLION) and cost.assumed == ()
    assert cost.fields()["uncached_input_tokens"] == 40


def test_cache_tokens_with_no_cache_rate_take_the_input_rate_and_are_marked():
    cost = _agent({**USAGE, "input_tokens": 1000, "output_tokens": 0, "cache_read_tokens": 400,
                   "cache_write_tokens": 100, "input_includes_cache": False}, "my-finetune")
    assert cost.usd == pytest.approx((1000 * 1.0 + 400 * 0.1 + 100 * 1.0) / PER_MILLION)
    assert cost.assumed == (Assumed.CACHE_WRITE_RATE,)
    assert cost.unit_prices == {"input": 1.0, "output": 4.0, "cache_read": 0.1, "cache_write": 1.0}
    no_cache = _agent({**USAGE, "input_tokens": 1000, "output_tokens": 0}, "meta-llama/Llama-3-70b")
    assert no_cache.assumed == ()


def test_a_cost_keeps_what_it_was_priced_at_so_a_newer_map_changes_no_old_record():
    usage = {**USAGE, "input_tokens": 3000, "output_tokens": 700, "cache_read_tokens": 900,
             "cache_write_tokens": 50, "input_includes_cache": True, "cost_usd": 0.002}
    record = _agent(usage).fields()
    newer = PriceTable.from_cost_map({"gpt-4o-mini": {"input_cost_per_token": 5e-06, "output_cost_per_token": 9e-06}},
                                     "litellm@newer")

    rates = record["unit_prices"]
    assert record["uncached_input_tokens"] == 3000 - 950
    again = (record["uncached_input_tokens"] * rates["input"] + 700 * rates["output"] + 900 * rates["cache_read"]
             + 50 * rates["cache_write"]) / PER_MILLION
    assert record["cost_usd"] == pytest.approx(again) and record["cost_source"] == "litellm@test"
    assert _agent(usage, table=newer).usd != pytest.approx(record["cost_usd"])
    assert record["reported_cost_usd"] == 0.002 and record["cost_status"] == "priced"


# A long-context model with two tiers, its first named in thousands and its second in tokens, beside service-tier and
# batch rates a price never takes; and Claude-style models, one with a rate for cache writes kept an hour.
TIERED_MAP = {
    "long-1": {"input_cost_per_token": 2e-06, "output_cost_per_token": 1e-05, "cache_read_input_token_cost": 2e-07,
               "input_cost_per_token_above_272k_tokens": 4e-06, "output_cost_per_token_above_272k_tokens": 1.5e-05,
               "input_cost_per_token_above_1000000_tokens": 8e-06,
               "input_cost_per_token_priority_above_272k_tokens": 9e-06,
               "input_cost_per_token_above_272k_tokens_batches": 1e-06},
    "claude-x": {"input_cost_per_token": 1e-06, "output_cost_per_token": 5e-06, "cache_read_input_token_cost": 1e-07,
                 "cache_creation_input_token_cost": 1.25e-06, "cache_creation_input_token_cost_above_1hr": 2e-06,
                 "cache_creation_input_token_cost_above_1hr_above_100k_tokens": 4e-06},
    "claude-flat": {"input_cost_per_token": 1e-06, "output_cost_per_token": 5e-06,
                    "cache_read_input_token_cost": 1e-07, "cache_creation_input_token_cost": 1.25e-06},
}
TIERED = PriceTable.from_cost_map(TIERED_MAP, "litellm@test")


def _call(prompt: int, output: int = 1000, **more) -> dict:
    return {"input_tokens": prompt, "output_tokens": output, "input_includes_cache": True, **more}


def _reported(prompt: int, **more) -> dict:
    """An agent's usage as the protocol reads it, so a provider's split in ``provider_details`` is read into
    ``cache_write_ttl_tokens`` as an agent's turn is."""
    return Usage.from_dict(_call(prompt, **more)).to_dict()


def _priced(tokens, model="long-1"):
    return price(tokens, None, model=model, prices=TIERED, reported_by=ReportedBy.HARNESS)


def test_a_maps_tiers_are_read_in_either_spelling_lowest_first_without_service_tier_or_batch_rates():
    assert TIERED.lookup("long-1").tiers == (ContextTier(272_000, input=4.0, output=15.0),
                                              ContextTier(1_000_000, input=8.0))
    assert (TIERED.lookup("claude-x").cache_write_ttl_rates, TIERED.lookup("claude-x").tiers) == (
        {3600: 2.0}, (ContextTier(100_000, cache_write_ttl_rates={3600: 4.0}),))


def test_records_each_under_the_lowest_tier_are_priced_at_the_base_rate_even_when_they_sum_past_it():
    cost = _priced(Tokens.of(_call(200_000)) + Tokens.of(_call(200_000)))

    assert cost.usd == pytest.approx((400_000 * 2.0 + 2000 * 10.0) / PER_MILLION) and cost.assumed == ()


def test_one_call_past_a_tier_takes_each_tier_it_passes_a_rate_a_tier_lacks_staying_at_the_one_below():
    over_one = _priced(Tokens.of(_call(300_000), one_call=True))
    over_both = _priced(Tokens.of(_call(1_200_000), one_call=True))

    assert over_one.usd == pytest.approx((300_000 * 4.0 + 1000 * 15.0) / PER_MILLION) and over_one.assumed == ()
    assert over_both.usd == pytest.approx((1_200_000 * 8.0 + 1000 * 15.0) / PER_MILLION) and over_both.assumed == ()
    assert over_both.unit_prices == {"input": 8.0, "output": 15.0, "cache_read": 0.2, "cache_write": 8.0}


def test_a_sum_with_a_record_past_a_tier_is_priced_at_the_base_rate_and_marked_as_possibly_low():
    cost = _priced(Tokens.of(_call(300_000)) + Tokens.of(_call(10_000)))
    one_turn = _priced(Tokens.of(_call(300_000)))

    assert cost.usd == pytest.approx((310_000 * 2.0 + 2000 * 10.0) / PER_MILLION)
    assert cost.assumed == (Assumed.BELOW_CONTEXT_TIER,) == one_turn.assumed
    assert Assumed.BELOW_CONTEXT_TIER.errs_low and not Assumed.CACHE_READ_RATE.errs_low


@pytest.mark.parametrize("usage", [
    _reported(10_000, cache_write_tokens=4000,
              provider_details={"cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 3000}}),
    Usage.from_model_response({"prompt_tokens": 10_000, "completion_tokens": 1000, "prompt_tokens_details": {
        "cached_tokens": 0, "cache_write_tokens": 4000, "cache_creation_token_details": {
            "ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 3000}}}).to_dict(),
], ids=["anthropic", "litellm"])
def test_writes_kept_an_hour_are_charged_the_hours_rate_and_the_split_is_kept(usage):
    cost = _priced(Tokens.of(usage), model="claude-x")

    assert cost.usd == pytest.approx((6000 * 1.0 + 1000 * 5.0 + 1000 * 1.25 + 3000 * 2.0) / PER_MILLION)
    assert cost.assumed == () and cost.unit_prices["cache_write_1h"] == 2.0
    assert cost.fields()["cache_write_ttl_tokens"] == {"1h": 3000}


def test_writes_a_split_doesnt_cover_are_charged_the_five_minute_rate_and_marked():
    cost = _priced(Tokens.of(_reported(10_000, cache_write_tokens=1000, provider_details={
        "cache_creation": {"ephemeral_5m_input_tokens": 100}})), model="claude-x")

    assert cost.usd == pytest.approx((9000 * 1.0 + 1000 * 5.0 + 1000 * 1.25) / PER_MILLION)
    assert cost.assumed == (Assumed.DEFAULT_CACHE_TTL,)


@pytest.mark.parametrize("hour", [-3000, True, "3000", 2.5], ids=["negative", "bool", "string", "float"])
def test_a_split_count_that_isnt_a_non_negative_integer_is_unknown(hour):
    from_provider = _reported(10_000, cache_write_tokens=4000, provider_details={
        "cache_creation": {"ephemeral_1h_input_tokens": hour}})
    as_recorded = {**_call(10_000, cache_write_tokens=4000), "cache_write_ttl_tokens": {"1h": hour}}

    for usage in (from_provider, as_recorded):
        cost = _priced(Tokens.of(usage), model="claude-x")
        assert cost.usd == pytest.approx((6000 * 1.0 + 1000 * 5.0 + 4000 * 1.25) / PER_MILLION)
        assert cost.assumed == (Assumed.DEFAULT_CACHE_TTL,) and cost.cache_write_ttl_tokens is None


def test_writes_with_no_split_are_charged_the_five_minute_rate_marked_only_when_the_hour_costs_more():
    marked = _priced(Tokens.of(_call(10_000, cache_write_tokens=4000)), model="claude-x")
    no_hour_rate = _priced(Tokens.of(_call(10_000, cache_write_tokens=4000)), model="claude-flat")
    no_writes = _priced(Tokens.of(_call(10_000)), model="claude-x")

    assert marked.usd == pytest.approx((6000 * 1.0 + 1000 * 5.0 + 4000 * 1.25) / PER_MILLION)
    assert marked.assumed == (Assumed.DEFAULT_CACHE_TTL,) and Assumed.DEFAULT_CACHE_TTL.errs_low
    assert no_hour_rate.assumed == () == no_writes.assumed
    assert "cache_write_1h" not in marked.unit_prices and "cache_write_ttl_tokens" not in marked.fields()


# A model that charges more for writes kept twelve hours and ninety minutes, as OpenAI's twelve-hour rate is written.
KEPT = PriceTable.from_cost_map({"gpt-x": {
    "input_cost_per_token": 2e-06, "output_cost_per_token": 1e-05, "cache_read_input_token_cost": 2e-07,
    "cache_creation_input_token_cost": 2.5e-06, "cache_creation_input_token_cost_above_12hr": 5e-06,
    "cache_creation_input_token_cost_above_90min": 3e-06}}, "litellm@test")


def _kept(usage):
    return price(Tokens.of(usage), None, model="gpt-x", prices=KEPT, reported_by=ReportedBy.HARNESS)


def test_a_maps_rates_for_writes_kept_a_while_are_read_for_any_time():
    assert KEPT.lookup("gpt-x").cache_write_ttl_rates == {43200: 5.0, 5400: 3.0}


def test_writes_split_by_any_time_are_charged_the_rate_of_the_longest_time_at_most_theirs():
    cost = _kept(_reported(10_000, cache_write_tokens=6000, provider_details={"cache_write_tokens_details": {
        "ttl_30m_tokens": 1000, "ttl_2h_tokens": 2000, "ttl_12h_tokens": 3000}}))

    assert cost.usd == pytest.approx((4000 * 2.0 + 1000 * 10.0 + 1000 * 2.5 + 2000 * 3.0 + 3000 * 5.0) / PER_MILLION)
    assert cost.fields()["cache_write_ttl_tokens"] == {"2h": 2000, "12h": 3000} and cost.assumed == ()
    assert (cost.unit_prices["cache_write_2h"], cost.unit_prices["cache_write_12h"]) == (3.0, 5.0)
    assert "cache_write_30m" not in cost.unit_prices


def test_writes_with_no_split_on_a_model_that_charges_more_for_a_longer_time_are_marked():
    cost = _kept(_call(10_000, cache_write_tokens=6000))

    assert cost.usd == pytest.approx((4000 * 2.0 + 1000 * 10.0 + 6000 * 2.5) / PER_MILLION)
    assert cost.assumed == (Assumed.DEFAULT_CACHE_TTL,)


def _litellm_cost(model: str, prompt: int, read: int, write: int, hour: int, output: int) -> tuple[float, float]:
    """One call's cost by LiteLLM's own cost function, and by ``price`` from the same usage as a response carries it."""
    usage = LiteLLMUsage(
        prompt_tokens=prompt, completion_tokens=output, total_tokens=prompt + output,
        cache_read_input_tokens=read, cache_creation_input_tokens=write,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=read, cache_creation_tokens=write, cache_creation_token_details=CacheCreationTokenDetails(
                ephemeral_5m_input_tokens=write - hour, ephemeral_1h_input_tokens=hour)))
    theirs = sum(litellm.cost_per_token(model=model, usage_object=usage))
    ours = price(Tokens.of(Usage.from_model_response(usage.model_dump()).to_dict(), one_call=True), None, model=model,
                 prices=litellm_prices(), reported_by=ReportedBy.HARNESS)
    assert ours.assumed == ()
    return theirs, ours.usd


def _tiered(prefix: str, *, hour: bool) -> str:
    """The first model of the installed map named ``prefix``... without a provider, with one long-context tier and, or
    without, a rate for writes kept an hour; LiteLLM's maps differ in which models they tier."""
    return next(name for name, rates in sorted(litellm_prices().entries.items())
                if name.startswith(prefix) and "/" not in name and len(rates.tiers) == 1
                and bool(rates.cache_write_ttl_rates) == hour)


@pytest.mark.parametrize(("prefix", "hour"), [("claude-", 5000), ("gpt-", 0)], ids=["claude-hour-writes", "gpt"])
@pytest.mark.parametrize("past_tier", [True, False], ids=["past", "under"])
def test_one_calls_price_matches_litellms_own_cost_past_and_under_its_tier(prefix, hour, past_tier):
    model = _tiered(prefix, hour=bool(hour))
    [tier] = litellm_prices().lookup(model).tiers
    prompt = tier.threshold + 50_000 if past_tier else tier.threshold - 10_000

    theirs, ours = _litellm_cost(model, prompt, read=40_000, write=20_000 if hour else 0, hour=hour, output=2000)

    assert ours == pytest.approx(theirs)
