import tomllib

import pytest

from agent_env.config.errors import ConfigError
from agent_env.config.model import ModelConfig
from agent_env.config.prices import (
    PER_MILLION,
    Assumed,
    CostStatus,
    ModelPrice,
    PlannedModel,
    PriceTable,
    ReportedBy,
    Unpriced,
    price,
    unpriced,
)


def _table(toml: str) -> PriceTable:
    return ModelConfig.from_section(tomllib.loads(toml)["model"], secret_resolver=lambda ref: None).prices


TABLE = _table("""
[model.prices]
"gpt-4o-mini" = { input = 0.15, output = 0.6, cache_read = 0.075, cache_write = 0.3 }
"my-finetune" = { input = 1.0, output = 4.0, cache_read = 0.1 }
"meta-llama/Llama-3-70b" = { input = 0.5, output = 0.5 }
""")
USAGE = {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_tokens": None,
         "cache_write_tokens": None, "input_includes_cache": None, "cost_usd": None}


def _agent(usage, model="gpt-4o-mini", *, table=TABLE, on_endpoint=True, prefer=False):
    return price(usage, model=model, prices=table, on_endpoint=on_endpoint, reported_first=prefer,
                 reported_by=ReportedBy.HARNESS)


def _judge(usage, model="gpt-4o-mini", *, on_endpoint=True):
    return price(usage, model=model, prices=TABLE, on_endpoint=on_endpoint, reported_first=True,
                 reported_by=ReportedBy.PROVIDER)


def test_an_entry_parses_and_a_routing_prefix_is_stripped_to_find_it():
    assert TABLE.lookup("my-finetune") == ModelPrice(input=1.0, output=4.0, cache_read=0.1)
    assert TABLE.lookup("openai/gpt-4o-mini") is TABLE.lookup("gpt-4o-mini") is not None
    assert TABLE.lookup("meta-llama/Llama-3-70b") is TABLE.lookup("hosted_vllm/meta-llama/Llama-3-70b") is not None


@pytest.mark.parametrize("model", ["gpt-4o", "openai/gpt-4o", "Llama-3-70b", ""])
def test_a_model_the_table_lacks_has_no_price_never_a_zero_one(model):
    assert TABLE.lookup(model) is None
    cost = _agent(USAGE, model or None)
    assert (cost.usd, cost.status) == (None, CostStatus.UNPRICED)


@pytest.mark.parametrize("entry, named", [
    ('"m" = { input = -1, output = 1 }', "'m' input is -1"),
    ('"m" = { input = "1", output = 1 }', "'m' input is '1'"),
    ('"m" = { input = true, output = 1 }', "'m' input is True"),
    ('"m" = { input = inf, output = 1 }', "'m' input is inf"),
    ('"m" = { input = nan, output = 1 }', "'m' input is nan"),
    ('"m" = { input = 1 }', "'m' needs output"),
    ('"m" = { output = 1 }', "'m' needs input"),
    ('"m" = { input = 1, output = 1, cached = 0.1 }', "'m' has unknown keys ['cached']"),
    ('"m" = 1.0', "'m' is a table"),
    ('"" = { input = 1, output = 1 }', "keys are model names"),
])
def test_a_bad_entry_is_refused_naming_it(entry, named):
    with pytest.raises(ConfigError, match="\\[model.prices\\]") as refused:
        _table(f"[model.prices]\n{entry}\n")
    assert named in str(refused.value)


def test_prefer_reported_cost_is_a_bool_and_prices_take_no_references():
    config = ModelConfig.from_section({"prefer_reported_cost": True}, secret_resolver=lambda ref: "1")
    assert config.prefer_reported_cost is True and ModelConfig().prefer_reported_cost is False
    with pytest.raises(ConfigError, match="prefer_reported_cost is true or false"):
        ModelConfig.from_section({"prefer_reported_cost": "yes"}, secret_resolver=lambda ref: None)
    with pytest.raises(ConfigError, match="'m' input is 'env:PRICE'"):
        ModelConfig.from_section({"prices": {"m": {"input": "env:PRICE", "output": 1}}},
                                 secret_resolver=lambda ref: "1")


def test_the_version_changes_with_any_price_and_not_with_spelling_or_order():
    one = _table('[model.prices]\n"a" = { input = 1, output = 2 }\n"b" = { input = 3.0, output = 4.0 }\n')
    same = _table('[model.prices]\n"b" = { output = 4, input = 3 }\n"a" = { input = 1.0, output = 2.0 }\n')
    assert one.version == same.version and len(one.version) == 12 and one.source == f"agentenv-prices@{one.version}"
    for changed in ('"a" = { input = 1, output = 2.5 }\n"b" = { input = 3, output = 4 }',
                    '"a" = { input = 1, output = 2, cache_read = 0.5 }\n"b" = { input = 3, output = 4 }',
                    '"a" = { input = 1, output = 2 }',
                    '"a" = { input = 1, output = 2 }\n"b" = { input = 3, output = 4 }\n"c" = { input = 0, output = 0 }'):
        assert _table(f"[model.prices]\n{changed}\n").version != one.version
    assert PriceTable().version != one.version


AGENT_BRANCHES = {
    # name: (model, on_endpoint, reported, prefer) -> (status, source, usd)
    "priced": (("gpt-4o-mini", True, None, False), (CostStatus.PRICED, TABLE.source, 0.15 + 0.06)),
    "priced-keeps-reported-beside": (("gpt-4o-mini", True, 9.0, False), (CostStatus.PRICED, TABLE.source, 0.21)),
    "reported-without-a-price": (("gpt-4o", True, 9.0, False), (CostStatus.REPORTED, "harness", 9.0)),
    "unpriced": (("gpt-4o", True, None, False), (CostStatus.UNPRICED, None, None)),
    "prefer-reported-flips": (("gpt-4o-mini", True, 9.0, True), (CostStatus.REPORTED, "harness", 9.0)),
    "prefer-reported-without-one-prices": (("gpt-4o-mini", True, None, True), (CostStatus.PRICED, TABLE.source, 0.21)),
    "other-endpoint-reported": (("gpt-4o-mini", False, 9.0, False), (CostStatus.REPORTED, "harness", 9.0)),
    "other-endpoint-unpriced": (("gpt-4o-mini", False, None, False), (CostStatus.UNPRICED, None, None)),
    "no-model": ((None, True, None, False), (CostStatus.UNPRICED, None, None)),
}


@pytest.mark.parametrize("given, expected", AGENT_BRANCHES.values(), ids=AGENT_BRANCHES.keys())
def test_an_agents_cost(given, expected):
    model, on_endpoint, reported, prefer = given
    cost = _agent({**USAGE, "cost_usd": reported}, model, on_endpoint=on_endpoint, prefer=prefer)
    assert (cost.status, cost.source) == expected[:2]
    assert cost.usd == pytest.approx(expected[2]) if expected[2] is not None else cost.usd is None
    assert cost.reported_usd == reported


JUDGE_BRANCHES = {
    "proxy-cost-first": (("gpt-4o-mini", 0.5), (CostStatus.REPORTED, "provider", 0.5)),
    "priced-without-a-proxy-cost": (("gpt-4o-mini", None), (CostStatus.PRICED, TABLE.source, 0.21)),
    "unpriced": (("claude-sonnet-4-6", None), (CostStatus.UNPRICED, None, None)),
}


@pytest.mark.parametrize("given, expected", JUDGE_BRANCHES.values(), ids=JUDGE_BRANCHES.keys())
def test_a_judges_cost(given, expected):
    model, proxy_cost = given
    cost = _judge({**USAGE, "cost_usd": proxy_cost}, model)
    assert (cost.status, cost.source) == expected[:2]
    assert cost.usd == pytest.approx(expected[2]) if expected[2] is not None else cost.usd is None


@pytest.mark.parametrize("usage, model, on_endpoint, why", [
    (USAGE, None, True, Unpriced.NO_MODEL),
    (USAGE, "gpt-4o-mini", False, Unpriced.OTHER_ENDPOINT),
    (USAGE, "gpt-4o", True, Unpriced.NO_PRICE),
    ({**USAGE, "input_tokens": None}, "gpt-4o-mini", True, Unpriced.NO_TOKENS),
    ({**USAGE, "output_tokens": None}, "gpt-4o-mini", True, Unpriced.NO_TOKENS),
])
def test_an_unpriced_cost_says_why(usage, model, on_endpoint, why):
    cost = _agent(usage, model, on_endpoint=on_endpoint)
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


def test_cache_tokens_with_no_cache_rate_take_the_input_rate_and_are_marked():
    cost = _agent({**USAGE, "input_tokens": 1000, "output_tokens": 0, "cache_read_tokens": 400,
                   "cache_write_tokens": 100, "input_includes_cache": False}, "my-finetune")
    assert cost.usd == pytest.approx((1000 * 1.0 + 400 * 0.1 + 100 * 1.0) / PER_MILLION)
    assert cost.assumed == (Assumed.CACHE_WRITE_RATE,)
    assert cost.unit_prices == {"input": 1.0, "output": 4.0, "cache_read": 0.1, "cache_write": 1.0}
    no_cache = _agent({**USAGE, "input_tokens": 1000, "output_tokens": 0}, "meta-llama/Llama-3-70b")
    assert no_cache.assumed == ()


def test_a_cost_keeps_what_it_was_priced_at_so_editing_the_table_changes_no_old_record():
    usage = {**USAGE, "input_tokens": 3000, "output_tokens": 700, "cache_read_tokens": 900,
             "cache_write_tokens": 50, "input_includes_cache": True, "cost_usd": 0.002}
    record = _agent(usage).fields()
    repriced = _table('[model.prices]\n"gpt-4o-mini" = { input = 5, output = 9, cache_read = 1, cache_write = 2 }\n')

    rates = record["unit_prices"]
    again = ((3000 - 950) * rates["input"] + 700 * rates["output"] + 900 * rates["cache_read"]
             + 50 * rates["cache_write"]) / PER_MILLION
    assert record["cost_usd"] == pytest.approx(again)
    assert record["cost_source"] == TABLE.source != repriced.source
    assert _agent(usage, table=repriced).usd != pytest.approx(record["cost_usd"])
    assert record["reported_cost_usd"] == 0.002 and record["cost_status"] == "priced"


def test_unpriced_names_each_role_the_table_cant_price_and_prints_lines_that_price_them():
    plan = [PlannedModel("solver", "openai/gpt-4o-mini"), PlannedModel("judge", "claude-sonnet-4-6"),
            PlannedModel("user_simulator", "gpt-4o-mini", on_endpoint=False),
            PlannedModel("reviewer", None), PlannedModel("judge", "claude-sonnet-4-6")]
    found = unpriced(plan, TABLE)

    assert [(planned.role, why) for planned, why in found.missing] == [
        ("judge", Unpriced.NO_PRICE), ("user_simulator", Unpriced.OTHER_ENDPOINT), ("reviewer", Unpriced.NO_MODEL),
        ("judge", Unpriced.NO_PRICE)]
    assert found.toml() == '"claude-sonnet-4-6" = { input = 0.0, output = 0.0 }  # USD per million tokens'
    assert "judge: claude-sonnet-4-6 has no price in [model.prices]" in found.message()

    fixed = _table('[model.prices]\n"gpt-4o-mini" = { input = 0.15, output = 0.6, cache_read = 0.075, '
                   f'cache_write = 0.3 }}\n{found.toml()}\n')
    assert [why for _, why in unpriced(plan, fixed).missing] == [Unpriced.OTHER_ENDPOINT, Unpriced.NO_MODEL]
    assert unpriced([PlannedModel("judge", "claude-sonnet-4-6")], fixed).missing == ()


def test_unpriced_warns_about_a_priced_model_without_cache_rates():
    found = unpriced([PlannedModel("solver", "my-finetune"), PlannedModel("judge", "gpt-4o-mini")], TABLE)
    assert found.missing == () and found.no_cache_rates == ("my-finetune",)
    assert "my-finetune has no cache_read or cache_write rate" in found.message()


@pytest.mark.parametrize("configured, url, applies", [
    ("http://localhost:4000/v1", "http://localhost:4000", True),
    ("http://localhost:4000/v1", "HTTP://LOCALHOST:4000/v1/", True),
    ("http://localhost:4000/v1", "http://127.0.0.1:4000/v1", False),
    ("http://localhost:4000/v1", None, False),
    (None, None, True),
    (None, "http://localhost:4000", False),
])
def test_prices_apply_only_to_the_configured_endpoint_or_a_native_route(monkeypatch, configured, url, applies):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    assert ModelConfig(base_url=configured).prices_apply(url) is applies
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1")
    assert ModelConfig(base_url=configured).prices_apply("http://127.0.0.1:4000") is True
