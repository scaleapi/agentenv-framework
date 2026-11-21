"""A role's step usage entry: the sums, what they leave out, how they're priced, and where the entry is kept."""

import asyncio

import pytest
from agentenv_protocol.a2a_agent import Usage

from agent_env.a2a_agent.protocol import TerminalResponse
from agent_env.config.model import ModelRole
from agent_env.config.model_pricing import ModelPrice, PriceTable
from agent_env.task_step.context import ATTEMPT_EPOCH, DeployedAgent, TaskStepContext
from agent_env.task_step import usage as usage_module
from agent_env.task_step.usage import RoleUsage, StepUsage


def test_turns_that_disagree_on_whether_input_counts_the_cache_leave_the_flag_unknown():
    role = RoleUsage("solver")
    role.add(Usage(input_tokens=10, output_tokens=1, input_includes_cache=True))
    role.add(Usage(input_tokens=20, output_tokens=2))

    entry = role.entry()

    assert (entry["input_tokens"], entry["input_includes_cache"], entry["partial"]) == (30, None, False)


def test_a_proxy_turn_short_of_some_calls_counts_is_partial():
    role = RoleUsage("solver")
    role.add(Usage(input_tokens=10, output_tokens=1, provider_details={"unreported": {"cost_usd": 2}}))

    entry = role.entry()

    assert entry["partial"] is True and "unreported_calls" not in entry


def test_the_entry_carries_the_attempt_its_step_ran_under():
    token = ATTEMPT_EPOCH.set(3)
    try:
        role = RoleUsage("user_simulator", "claude-haiku-4-5", "proxy.example")
    finally:
        ATTEMPT_EPOCH.reset(token)
    role.add(None)

    entry = role.entry()

    assert (entry["attempt"], entry["model"], entry["turns"], entry["tool_call_count"]) == (3, "claude-haiku-4-5", 1, None)


def test_a_step_whose_roles_ran_no_turns_records_nothing():
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "m", None)

    assert usage.entries() is None
    assert "step_usage" not in context.metadata


def test_a_step_records_each_started_role_as_each_turn_ends_and_skips_one_never_started():
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "m", DeployedAgent(agent_name="a", api_url="u", model_endpoint_host="gw:443"))

    usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=Usage(input_tokens=3, output_tokens=1)))
    usage.turn(ModelRole.USER_SIMULATOR, TerminalResponse(response_text="", usage=Usage(input_tokens=99)))

    [entry] = context.metadata["step_usage"]["solve"]
    assert (entry["role"], entry["model"], entry["endpoint_host"], entry["input_tokens"]) == ("solver", "m", "gw:443", 3)
    assert context.metadata["model_endpoints"] == {"solver": {"solve": {"model": "m", "endpoint_host": "gw:443"}}}


@pytest.mark.parametrize("cut", [TimeoutError("no answer"), asyncio.CancelledError()])
def test_a_turn_cut_off_before_its_reply_is_a_turn_of_unknown_usage(cut):
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "m", None)
    usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=Usage(input_tokens=3, output_tokens=1)))

    with pytest.raises(type(cut)), usage.awaiting(ModelRole.SOLVER):
        raise cut
    with pytest.raises(TimeoutError), usage.awaiting(ModelRole.USER_SIMULATOR):
        raise TimeoutError("a human peer's turn")
    with pytest.raises(ConnectionError), usage.awaiting(ModelRole.SOLVER):
        raise ConnectionError("refused")

    [entry] = context.metadata["step_usage"]["solve"]
    assert (entry["turns"], entry["input_tokens"], entry["partial"]) == (2, 3, True)


def test_a_role_whose_only_turn_was_cut_off_has_no_counts_and_is_partial():
    context = TaskStepContext()
    usage = StepUsage(context, "rate")
    usage.start(ModelRole.USER_SIMULATOR, "m", None)

    with pytest.raises(TimeoutError), usage.awaiting(ModelRole.USER_SIMULATOR):
        raise TimeoutError

    [entry] = context.metadata["step_usage"]["rate"]
    assert (entry["turns"], entry["input_tokens"], entry["cost_usd"], entry["partial"]) == (1, None, None, True)


def test_a_turn_that_reported_any_token_count_makes_a_turn_that_reported_none_partial():
    role = RoleUsage("solver")
    role.add(Usage(total_tokens=100))
    role.add(None)

    entry = role.entry()

    assert (entry["total_tokens"], entry["input_tokens"], entry["partial"]) == (100, None, True)


def test_an_entry_keeps_the_first_turns_unreadable_values_and_counts_them_all():
    role = RoleUsage("solver")
    for n in range(8):
        role.add(None, {"input_tokens": -n})

    entry = role.entry()

    assert entry["unparsed"] == [{"input_tokens": -n} for n in range(5)] and entry["unparsed_turns"] == 8


PRICES = PriceTable({"gpt-4o-mini": ModelPrice(input=0.15, output=0.6, cache_read=0.075)}, "litellm@test")


def _step_entry(monkeypatch) -> dict:
    """The entry a step records for one solver turn on openai/gpt-4o-mini that reported a million input tokens, a
    hundred thousand output tokens and its own cost of $0.50."""
    monkeypatch.setattr(usage_module, "litellm_prices", lambda: PRICES)
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "openai/gpt-4o-mini",
                DeployedAgent(agent_name="a", api_url="u"))
    usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=Usage(
        input_tokens=1_000_000, output_tokens=100_000, input_includes_cache=True, cost_usd=0.5)))
    [entry] = context.metadata["step_usage"]["solve"]
    return entry


def test_usage_on_the_configured_endpoint_is_priced_with_the_agents_own_cost_kept_beside_it(monkeypatch):
    entry = _step_entry(monkeypatch)

    assert entry["cost_usd"] == pytest.approx(0.15 + 0.06)
    assert (entry["cost_status"], entry["cost_source"], entry["reported_cost_usd"]) == (
        "priced", PRICES.source, 0.5)
    assert entry["unit_prices"] == {"input": 0.15, "output": 0.6, "cache_read": 0.075, "cache_write": 0.15}
    assert "cost_assumed" not in entry and "unpriced_reason" not in entry


def test_turns_that_disagree_on_whether_input_counts_the_cache_are_each_charged_for_their_cached_input_once(
        monkeypatch):
    monkeypatch.setattr(usage_module, "litellm_prices", lambda: PRICES)
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "gpt-4o-mini", DeployedAgent(agent_name="a", api_url="u"))
    for turn in (Usage(input_tokens=100, output_tokens=1, cache_read_tokens=80, input_includes_cache=True),
                 Usage(input_tokens=20, output_tokens=1, cache_read_tokens=80, input_includes_cache=False)):
        usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=turn))

    [entry] = context.metadata["step_usage"]["solve"]
    assert (entry["input_tokens"], entry["cache_read_tokens"], entry["input_includes_cache"]) == (120, 160, None)
    assert entry["uncached_input_tokens"] == 40 and "cost_assumed" not in entry
    assert entry["cost_usd"] == pytest.approx((40 * 0.15 + 2 * 0.6 + 160 * 0.075) / 1_000_000)


@pytest.mark.parametrize("second, why", [
    (Usage(output_tokens=20), "no_tokens"),
    (Usage(input_tokens=20), "no_tokens"),
    (Usage(total_tokens=20), "no_tokens"),
], ids=["output-only", "input-only", "total-only"])
def test_a_turn_without_input_or_output_leaves_the_entry_unpriced_rather_than_priced_short(
        monkeypatch, second, why):
    monkeypatch.setattr(usage_module, "litellm_prices", lambda: PRICES)
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "gpt-4o-mini", DeployedAgent(agent_name="a", api_url="u"))
    for turn in (Usage(input_tokens=100, output_tokens=10), second):
        usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=turn))

    [entry] = context.metadata["step_usage"]["solve"]
    assert (entry["cost_usd"], entry["cost_status"], entry["unpriced_reason"]) == (None, "unpriced", why)


def test_a_turn_that_reported_no_tokens_beside_priced_ones_leaves_a_partial_lower_bound(monkeypatch):
    monkeypatch.setattr(usage_module, "litellm_prices", lambda: PRICES)
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "gpt-4o-mini", DeployedAgent(agent_name="a", api_url="u"))
    for turn in (Usage(input_tokens=1_000_000, output_tokens=0), Usage(tool_call_count=2), None):
        usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=turn))

    [entry] = context.metadata["step_usage"]["solve"]
    assert (entry["cost_status"], entry["partial"], entry["uncached_input_tokens"]) == ("priced", True, 1_000_000)
    assert entry["cost_usd"] == pytest.approx(0.15)
