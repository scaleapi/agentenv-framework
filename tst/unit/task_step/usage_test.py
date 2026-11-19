"""A role's step usage entry: the sums, what they leave out, and where the entry is kept."""

from agentenv_protocol.a2a_agent import Usage

from agent_env.task_step.context import ATTEMPT_EPOCH, TaskStepContext
from agent_env.task_step.usage import RoleUsage, record_step_usage


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

    assert record_step_usage(context, "solve", (RoleUsage("solver"), None)) is None
    assert "step_usage" not in context.metadata


def test_a_turn_that_reported_any_token_count_makes_a_turn_that_reported_none_partial():
    role = RoleUsage("solver")
    role.add(Usage(total_tokens=100))
    role.add(None)

    entry = role.entry()

    assert (entry["total_tokens"], entry["input_tokens"], entry["partial"]) == (100, None, True)
