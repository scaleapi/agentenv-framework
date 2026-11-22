import importlib.metadata
import logging
import subprocess
import sys

import pytest

from agent_env.config import model_pricing as prices_module
from agent_env.config.model_pricing import (
    PER_MILLION,
    Assumed,
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
    assert out == (f"litellm@{version} ModelPrice(input=0.4, output=1.6, cache_read=0.1, cache_write=None) False")
    assert len(litellm_prices().entries) > 1000


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

    assert counted + apart == Tokens(uncached_input=40, output=2, cache_read=160)
    assert counted + apart + unknown == Tokens(uncached_input=90, output=3, cache_read=190, assumed_uncached=True)
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
