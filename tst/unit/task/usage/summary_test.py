"""A run's ledger summed for people: by role, then by model, an entry's per-model rows split along their models,
unpriced usage counted apart and never as $0, and a partial or running entry making the total a lower bound; its
bound word and JSON shape; and the lines the run summaries print from it."""

import json

import pytest

from agent_env.cli.usage_output import cost_line, sources_line
from agent_env.task.usage.summary import Bound, Share, summarize


def _entry(role, model, input_tokens, output_tokens, cost, attempt=0, **more):
    return {"role": role, "model": model, "attempt": attempt, "step_id": "ask", "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost_usd": cost, "cost_status": "priced" if cost is not None else "unpriced",
            "cost_source": "litellm@1.80.0" if cost is not None else None, **more}


def _shares(summary):
    return [(s.role, s.model, s.input_tokens, s.output_tokens, s.cost_usd, s.unpriced, s.partial) for s in summary.shares]


def _bounds(summary):
    return summary.bound, [share.bound for share in summary.shares]


def test_a_ledger_is_split_by_role_in_order_then_by_model_and_summed_across_steps():
    summary = summarize([
        _entry("judge", "claude-sonnet-4-6", 2900, 120, 0.0104),
        _entry("solver", "gpt-4.1-mini", 100_000, 1000, 0.04),
        _entry("user_simulator", "gpt-4.1-nano", 16_000, 210, 0.0016),
        _entry("solver", "gpt-4.1-mini", 15_000, 200, 0.0012, step_id="follow"),
    ])

    assert _shares(summary) == [
        ("solver", "gpt-4.1-mini", 115_000, 1200, pytest.approx(0.0412), False, False),
        ("user_simulator", "gpt-4.1-nano", 16_000, 210, 0.0016, False, False),
        ("judge", "claude-sonnet-4-6", 2900, 120, 0.0104, False, False)]
    assert summary.cost_usd == pytest.approx(0.0532) and not summary.lower_bound
    assert _bounds(summary) == (Bound.EXACT, [Bound.EXACT] * 3)
    assert (summary.input_tokens, summary.output_tokens) == (133_900, 1530)
    assert (summary.attempts, summary.sources) == (1, ("litellm@1.80.0",))
    assert cost_line(summary) == ("cost: $0.0532 (solver gpt-4.1-mini $0.0412, 115k in, 1.2k out; user simulator "
                                  "gpt-4.1-nano $0.0016, 16k in, 210 out; judge claude-sonnet-4-6 $0.0104, 2.9k in, "
                                  "120 out)")


def test_an_entry_with_per_model_rows_splits_along_them():
    rows = [{"model": "claude-sonnet-4-6", "input_tokens": 9000, "output_tokens": 400, "cost_usd": 0.0118,
             "cost_source": "litellm@1.80.0"},
            {"model": "claude-haiku-4-5", "input_tokens": 3000, "output_tokens": 90, "cost_usd": 0.0035,
             "cost_source": "litellm@1.80.0"}]
    summary = summarize([_entry("solver", "claude-sonnet-4-6", 12_000, 490, 0.0153, models=rows)])

    assert [(s.model, s.input_tokens, s.cost_usd) for s in summary.shares] == [
        ("claude-haiku-4-5", 3000, 0.0035), ("claude-sonnet-4-6", 9000, 0.0118)]
    assert summary.cost_usd == pytest.approx(0.0153)
    assert _bounds(summary) == (Bound.EXACT, [Bound.EXACT, Bound.EXACT])


def test_unpriced_usage_is_counted_apart_never_as_nothing_and_makes_the_total_a_lower_bound():
    summary = summarize([_entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004),
                         _entry("judge", "mystery-1", 3100, 90, None, unpriced_reason="no_price")])

    assert summary.cost_usd == 0.0004 and summary.lower_bound and summary.sources == ("litellm@1.80.0",)
    assert _bounds(summary) == (Bound.AT_LEAST, [Bound.EXACT, Bound.UNPRICED])
    assert _bounds(summarize([_entry("judge", "mystery-1", 3100, 90, None)])) == (Bound.UNPRICED, [Bound.UNPRICED])
    assert cost_line(summary) == ("cost: at least $0.0004 (solver gpt-4.1-mini $0.0004, 1.0k in, 10 out; judge "
                                  "mystery-1 unpriced, 3.1k in, 90 out)")
    assert cost_line(summarize([_entry("judge", "mystery-1", 3100, 90, None)])) == (
        "cost: unpriced (judge mystery-1 unpriced, 3.1k in, 90 out)")


def test_a_cost_priced_on_an_assumed_rate_is_at_most_and_with_uncounted_usage_an_estimate():
    assumed = _entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004, cost_assumed=["cache_read_rate"])
    summary = summarize([assumed, _entry("judge", "j", 100, 1, 0.0001)])

    assert summary.upper_bound and not summary.lower_bound
    assert _bounds(summary) == (Bound.AT_MOST, [Bound.AT_MOST, Bound.EXACT])
    assert _bounds(summarize([assumed, _entry("solver", "gpt-4.1-mini", 100, 1, None)])) == (
        Bound.ESTIMATED, [Bound.ESTIMATED])
    assert cost_line(summary) == ("cost: at most $0.0005 (solver gpt-4.1-mini at most $0.0004, 1.0k in, 10 out; judge "
                                  "j $0.0001, 100 in, 1 out)")
    assert cost_line(summarize([assumed, _entry("judge", "mystery-1", 100, 1, None)])).startswith(
        "cost: estimated $0.0004 (solver gpt-4.1-mini at most $0.0004")
    assert cost_line(summarize([assumed, _entry("solver", "gpt-4.1-mini", 100, 1, None)])) == (
        "cost: estimated $0.0004 (solver gpt-4.1-mini estimated $0.0004, 1.1k in, 11 out)")
    assert cost_line(summarize([assumed, _entry("solver", "gpt-4.1-mini", 100, 1, 0.0001, running=True)])) == (
        "cost: estimated $0.0005 (solver gpt-4.1-mini estimated $0.0005, 1.1k in, 11 out, partial)")


def test_a_per_model_row_priced_on_an_assumed_rate_bounds_its_own_share():
    rows = [{"model": "a", "input_tokens": 9, "output_tokens": 1, "cost_usd": 0.2,
             "cost_assumed": ["cache_write_rate"]},
            {"model": "b", "input_tokens": 3, "output_tokens": 1, "cost_usd": 0.1}]
    summary = summarize([_entry("solver", "a", 12, 2, 0.3, cost_assumed=["cache_write_rate"], models=rows)])

    assert [(share.model, share.assumed) for share in summary.shares] == [("a", True), ("b", False)]
    assert _bounds(summary) == (Bound.AT_MOST, [Bound.AT_MOST, Bound.EXACT])
    assert cost_line(summary) == ("cost: at most $0.3000 (solver a at most $0.2000, 9 in, 1 out; solver b $0.1000, 3 in, "
                                  "1 out)")


def test_a_partial_or_running_entry_makes_its_share_partial_and_the_total_a_lower_bound():
    for mark in ({"partial": True}, {"running": True}):
        summary = summarize([_entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004, **mark)])
        assert summary.lower_bound and summary.shares[0].partial
        assert _bounds(summary) == (Bound.AT_LEAST, [Bound.AT_LEAST])
        assert cost_line(summary) == ("cost: at least $0.0004 (solver gpt-4.1-mini at least $0.0004, 1.0k in, 10 out, "
                                      "partial)")


def test_every_attempt_counts_and_is_named_when_there_were_more_than_one():
    summary = summarize([_entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004, attempt=0),
                         _entry("solver", "gpt-4.1-mini", 2000, 20, 0.0008, attempt=1)])

    assert (summary.attempts, summary.shares[0].input_tokens, summary.cost_usd) == (2, 3000, pytest.approx(0.0012))
    assert _bounds(summary) == (Bound.EXACT, [Bound.EXACT])
    assert cost_line(summary).startswith("cost: $0.0012 (2 attempts; solver gpt-4.1-mini $0.0012, 3.0k in, 30 out")
    assert "attempts" not in cost_line(summary, "cost over 1 run", attempts=False)


def test_runs_ledgers_together_are_an_evals_total_by_role_and_model():
    first = [_entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004)]
    second = [_entry("solver", "gpt-4.1-mini", 2000, 20, 0.0008), _entry("judge", "mystery-1", 500, 5, None)]

    total = summarize([*first, *second])

    assert [(s.role, s.model, s.input_tokens, s.cost_usd, s.unpriced) for s in total.shares] == [
        ("solver", "gpt-4.1-mini", 3000, pytest.approx(0.0012), False), ("judge", "mystery-1", 500, None, True)]
    assert total.cost_usd == pytest.approx(0.0012) and total.lower_bound
    assert _bounds(total) == (Bound.AT_LEAST, [Bound.EXACT, Bound.UNPRICED])
    assert summarize([]) is None and cost_line(None) is None


def test_counts_and_costs_read_at_a_glance():
    summary = summarize([_entry("solver", "m", 2_500_000, 999, 12.5), _entry("judge", "j", None, None, 0.00002)])

    assert cost_line(summary) == "cost: $12.5000 (solver m $12.5000, 2.5M in, 999 out; judge j under $0.0001)"
    assert _bounds(summary) == (Bound.EXACT, [Bound.EXACT, Bound.EXACT])
    assert (summary.input_tokens, summary.output_tokens) == (2_500_000, 999)


def test_a_summary_is_one_json_shape_with_the_bound_word_on_its_total_and_each_share():
    summary = summarize([
        _entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004, cost_assumed=["cache_read_rate"]),
        _entry("solver", "gpt-4.1-mini", 200, 2, 0.0001, running=True, attempt=1),
        _entry("judge", "mystery-1", 300, 3, None, unpriced_reason="no_price"),
        {**_entry("user_simulator", "gpt-4.1-nano", None, None, 0.0002), "cost_source": "harness"},
    ])

    shape = summary.to_dict()

    assert shape == {
        "cost_usd": pytest.approx(0.0007), "bound": "estimated", "input_tokens": 1500, "output_tokens": 15,
        "attempts": 2, "sources": ["harness", "litellm@1.80.0"],
        "shares": [
            {"role": "solver", "model": "gpt-4.1-mini", "cost_usd": pytest.approx(0.0005), "bound": "estimated",
             "input_tokens": 1200, "output_tokens": 12, "cache_read_tokens": None, "cache_write_tokens": None,
             "unpriced": False, "unpriced_reasons": [], "assumed": True, "cost_assumed": ["cache_read_rate"],
             "partial": True},
            {"role": "user_simulator", "model": "gpt-4.1-nano", "cost_usd": 0.0002, "bound": "exact",
             "input_tokens": None, "output_tokens": None, "cache_read_tokens": None, "cache_write_tokens": None,
             "unpriced": False, "unpriced_reasons": [], "assumed": False, "cost_assumed": [], "partial": False},
            {"role": "judge", "model": "mystery-1", "cost_usd": None, "bound": "unpriced", "input_tokens": 300,
             "output_tokens": 3, "cache_read_tokens": None, "cache_write_tokens": None, "unpriced": True,
             "unpriced_reasons": ["no_price"], "assumed": False, "cost_assumed": [], "partial": False}]}
    assert json.loads(json.dumps(shape)) == shape
    assert summary.to_dict(attempts=False)["attempts"] is None


def test_each_share_says_why_it_is_unpriced_and_which_rates_it_assumed_with_its_cache_counts():
    rows = [{"model": "a", "input_tokens": 9, "output_tokens": 1, "cache_read_tokens": 6, "cost_usd": 0.2,
             "cost_assumed": ["cache_write_rate"]},
            {"model": "a", "input_tokens": 3, "output_tokens": 1, "cost_usd": None, "unpriced_reason": "no_price"}]
    summary = summarize([
        _entry("solver", "a", 12, 2, 0.2, models=rows),
        _entry("solver", "a", 100, 5, 0.01, cache_read_tokens=40, cache_write_tokens=8,
               cost_assumed=["cache_read_rate", "cache_write_rate"]),
        _entry("judge", "j", None, None, None, unpriced_reason="no_tokens"),
        _entry("judge", "j", 5, 1, None, unpriced_reason="no_price"),
        _entry("judge", "j", 5, 1, None),
    ])

    solver, judge = summary.to_dict()["shares"]
    assert (solver["cache_read_tokens"], solver["cache_write_tokens"]) == (46, 8)
    assert solver["unpriced_reasons"] == ["no_price"]
    assert solver["cost_assumed"] == ["cache_read_rate", "cache_write_rate"]
    assert judge["unpriced_reasons"] == ["no_price", "no_tokens"] and judge["cost_assumed"] == []
    assert (judge["cache_read_tokens"], judge["cache_write_tokens"]) == (None, None)


def test_a_share_built_as_plugins_built_it_before_the_reasons_and_cache_counts_still_is():
    share = Share("solver", "m", 1, 2, 0.5, False, True, False)
    share.assumed = True

    assert (share.cost_usd, share.unpriced, share.assumed, share.partial) == (0.5, False, True, False)
    assert Share(role="judge", model=None, assumed=True).bound is Bound.UNPRICED


def test_the_closing_line_names_each_source_of_the_costs():
    priced = [_entry("solver", "m", 1, 1, 0.1)]
    reported = [{**_entry("judge", "j", 1, 1, 0.2), "cost_source": "harness"},
                {**_entry("user_simulator", "u", 1, 1, 0.3), "cost_source": "provider"}]

    assert sources_line(summarize(priced)) == "Costs priced from LiteLLM's price map, litellm@1.80.0."
    assert sources_line(summarize([*priced, *reported])) == (
        "Costs priced from LiteLLM's price map, litellm@1.80.0; the endpoint's bills; agents' own figures.")
    assert sources_line(summarize([_entry("judge", "x", 1, 1, None)])) is None and sources_line(None) is None
