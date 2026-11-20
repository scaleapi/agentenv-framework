"""A role's step usage entry: the sums, what they leave out, how they're priced, and where the entry is kept."""

import logging

import pytest
from agentenv_protocol.a2a_agent import Usage

from agent_env.a2a_agent.protocol import TerminalResponse
from agent_env.config.errors import ConfigError
from agent_env.config.model import ModelConfig, ModelRole
from agent_env.config.prices import ModelPrice, PriceTable
from agent_env.task_step.context import ATTEMPT_EPOCH, DeployedAgent, TaskStepContext
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


PRICES = PriceTable({"gpt-4o-mini": ModelPrice(input=0.15, output=0.6, cache_read=0.075)})


def _step_entry(monkeypatch, local_stores, *, prefer=False, on_endpoint=True) -> dict:
    """The entry a step records for one solver turn on openai/gpt-4o-mini that reported a million input tokens, a
    hundred thousand output tokens and its own cost of $0.50."""
    monkeypatch.setattr(local_stores, "_model_cfg", ModelConfig(prices=PRICES, prefer_reported_cost=prefer))
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "openai/gpt-4o-mini",
                DeployedAgent(agent_name="a", api_url="u", on_configured_endpoint=on_endpoint))
    usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=Usage(
        input_tokens=1_000_000, output_tokens=100_000, input_includes_cache=True, cost_usd=0.5)))
    [entry] = context.metadata["step_usage"]["solve"]
    return entry


def test_usage_on_the_configured_endpoint_is_priced_with_the_agents_own_cost_kept_beside_it(monkeypatch, local_stores):
    entry = _step_entry(monkeypatch, local_stores)

    assert entry["cost_usd"] == pytest.approx(0.15 + 0.06)
    assert (entry["cost_status"], entry["cost_source"], entry["reported_cost_usd"]) == (
        "priced", PRICES.source, 0.5)
    assert entry["unit_prices"] == {"input": 0.15, "output": 0.6, "cache_read": 0.075, "cache_write": 0.15}
    assert "cost_assumed" not in entry and "unpriced_reason" not in entry


def test_prefer_reported_cost_puts_the_agents_own_cost_first(monkeypatch, local_stores):
    entry = _step_entry(monkeypatch, local_stores, prefer=True)

    assert (entry["cost_usd"], entry["cost_status"], entry["cost_source"]) == (0.5, "reported", "harness")
    assert "unit_prices" not in entry


@pytest.mark.parametrize("on_endpoint", [False, None], ids=["other-endpoint", "recorded-before-prices"])
def test_an_agent_not_known_to_call_the_configured_endpoint_is_never_priced_from_the_table(
        monkeypatch, local_stores, on_endpoint):
    entry = _step_entry(monkeypatch, local_stores, on_endpoint=on_endpoint)

    assert (entry["cost_usd"], entry["cost_status"], entry["cost_source"]) == (0.5, "reported", "harness")


def test_a_model_config_that_cant_be_read_leaves_usage_unpriced_without_failing_the_step(
        monkeypatch, local_stores, caplog):
    def unreadable():
        raise ConfigError("[model] base_url names env:UNSET, which is not set")

    monkeypatch.setattr(local_stores, "_model_config", unreadable)
    context = TaskStepContext()
    with caplog.at_level(logging.WARNING):
        usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "gpt-4o-mini", DeployedAgent(agent_name="a", api_url="u",
                                                               on_configured_endpoint=True))
    usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=Usage(input_tokens=1, output_tokens=1)))

    [entry] = context.metadata["step_usage"]["solve"]
    assert (entry["cost_status"], entry["unpriced_reason"]) == ("unpriced", "no_price")
    assert "usage stays unpriced, since [model] can't be read: [model] base_url names env:UNSET" in caplog.text


def test_turns_that_disagree_on_whether_input_counts_the_cache_are_each_charged_for_their_cached_input_once(
        monkeypatch, local_stores):
    monkeypatch.setattr(local_stores, "_model_cfg", ModelConfig(prices=PRICES))
    context = TaskStepContext()
    usage = StepUsage(context, "solve")
    usage.start(ModelRole.SOLVER, "gpt-4o-mini", DeployedAgent(agent_name="a", api_url="u", on_configured_endpoint=True))
    for turn in (Usage(input_tokens=100, output_tokens=1, cache_read_tokens=80, input_includes_cache=True),
                 Usage(input_tokens=20, output_tokens=1, cache_read_tokens=80, input_includes_cache=False)):
        usage.turn(ModelRole.SOLVER, TerminalResponse(response_text="", usage=turn))

    [entry] = context.metadata["step_usage"]["solve"]
    assert (entry["input_tokens"], entry["cache_read_tokens"], entry["input_includes_cache"]) == (120, 160, None)
    assert entry["uncached_input_tokens"] == 40 and "cost_assumed" not in entry
    assert entry["cost_usd"] == pytest.approx((40 * 0.15 + 2 * 0.6 + 160 * 0.075) / 1_000_000)
