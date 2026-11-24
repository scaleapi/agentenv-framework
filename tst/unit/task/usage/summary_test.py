"""A run's ledger summed for people: by role, then by model, an entry's per-model rows split along their models,
unpriced usage counted apart and never as $0, and a partial or running entry making the total a lower bound; and the
lines the run summaries print from it."""

import pytest

from agent_env.cli.usage_output import cost_line, sources_line
from agent_env.task.usage.summary import combine, summarize


def _entry(role, model, input_tokens, output_tokens, cost, attempt=0, **more):
    return {"role": role, "model": model, "attempt": attempt, "step_id": "ask", "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost_usd": cost, "cost_status": "priced" if cost is not None else "unpriced",
            "cost_source": "litellm@1.80.0" if cost is not None else None, **more}


def _shares(summary):
    return [(s.role, s.model, s.input_tokens, s.output_tokens, s.cost_usd, s.unpriced, s.partial) for s in summary.shares]


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


def test_unpriced_usage_is_counted_apart_never_as_nothing_and_makes_the_total_a_lower_bound():
    summary = summarize([_entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004),
                         _entry("judge", "mystery-1", 3100, 90, None, unpriced_reason="no_price")])

    assert summary.cost_usd == 0.0004 and summary.lower_bound and summary.sources == ("litellm@1.80.0",)
    assert cost_line(summary) == ("cost: at least $0.0004 (solver gpt-4.1-mini $0.0004, 1.0k in, 10 out; judge "
                                  "mystery-1 unpriced, 3.1k in, 90 out)")
    assert cost_line(summarize([_entry("judge", "mystery-1", 3100, 90, None)])) == (
        "cost: unpriced (judge mystery-1 unpriced, 3.1k in, 90 out)")


def test_a_partial_or_running_entry_makes_its_share_partial_and_the_total_a_lower_bound():
    for mark in ({"partial": True}, {"running": True}):
        summary = summarize([_entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004, **mark)])
        assert summary.lower_bound and summary.shares[0].partial
        assert cost_line(summary) == "cost: at least $0.0004 (solver gpt-4.1-mini $0.0004, 1.0k in, 10 out, partial)"


def test_every_attempt_counts_and_is_named_when_there_were_more_than_one():
    summary = summarize([_entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004, attempt=0),
                         _entry("solver", "gpt-4.1-mini", 2000, 20, 0.0008, attempt=1)])

    assert (summary.attempts, summary.shares[0].input_tokens, summary.cost_usd) == (2, 3000, pytest.approx(0.0012))
    assert cost_line(summary).startswith("cost: $0.0012 (2 attempts; solver gpt-4.1-mini $0.0012, 3.0k in, 30 out")
    assert "attempts" not in cost_line(summary, "cost over 1 run", attempts=False)


def test_runs_combine_into_an_evals_total_by_role_and_model():
    first = summarize([_entry("solver", "gpt-4.1-mini", 1000, 10, 0.0004)])
    second = summarize([_entry("solver", "gpt-4.1-mini", 2000, 20, 0.0008),
                        _entry("judge", "mystery-1", 500, 5, None)])

    total = combine([first, None, second])

    assert [(s.role, s.model, s.input_tokens, s.cost_usd, s.unpriced) for s in total.shares] == [
        ("solver", "gpt-4.1-mini", 3000, pytest.approx(0.0012), False), ("judge", "mystery-1", 500, None, True)]
    assert total.cost_usd == pytest.approx(0.0012) and total.lower_bound
    assert combine([None]) is None and summarize([]) is None and cost_line(None) is None


def test_counts_and_costs_read_at_a_glance():
    summary = summarize([_entry("solver", "m", 2_500_000, 999, 12.5), _entry("judge", "j", None, None, 0.00002)])

    assert cost_line(summary) == "cost: $12.5000 (solver m $12.5000, 2.5M in, 999 out; judge j under $0.0001)"


def test_the_closing_line_names_each_source_of_the_costs():
    priced = summarize([_entry("solver", "m", 1, 1, 0.1)])
    reported = summarize([{**_entry("judge", "j", 1, 1, 0.2), "cost_source": "harness"},
                          {**_entry("user_simulator", "u", 1, 1, 0.3), "cost_source": "provider"}])

    assert sources_line([priced]) == "Costs priced from LiteLLM's price map, litellm@1.80.0."
    assert sources_line([priced, reported]) == ("Costs priced from LiteLLM's price map, litellm@1.80.0; the endpoint's "
                                                "bills; agents' own figures.")
    assert sources_line([summarize([_entry("judge", "x", 1, 1, None)]), None]) is None
