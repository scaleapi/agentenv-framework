"""The gateway and an env serving tool access itself (agentenv_protocol.tool_access) keep separate code, since the
gateway's image installs no protocol; the same rule cases run against both so a task's tool-access steps mean the
same with or without a gateway in front."""
from __future__ import annotations

import random

import pytest
from agentenv_protocol.tool_access import ToolAccess

from agent_env.env.gateway.gateway import Gateway

ROLES = ("default", "grader", "sim")
TOOLS = ("h", "n")
HIDDEN = {"h"}


def _both() -> tuple[Gateway, ToolAccess]:
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[])
    gateway._hidden_by_default = set(HIDDEN)
    return gateway, ToolAccess(hidden_by_default=HIDDEN)


def _assert_same(gateway: Gateway, access: ToolAccess, steps: list) -> None:
    assert access.rules == gateway._role_rules, steps
    for role in ROLES + ("*", "never-mentioned"):
        for tool in TOOLS + ("unlisted",):
            assert access.is_disabled(role, tool) is gateway._is_disabled(role, tool), (steps, role, tool)


def test_random_step_sequences_leave_both_with_the_same_rules_and_answers():
    rng = random.Random(20261026)
    for _ in range(5000):
        gateway, access = _both()
        steps = [(rng.choice(ROLES + ("*",)), rng.choice(TOOLS + ("*",)), rng.random() < 0.5)
                 for _ in range(rng.randint(0, 10))]
        for step in steps:
            gateway._apply_rule(*step)
            access.apply(*step)
        _assert_same(gateway, access, steps)


@pytest.mark.parametrize("steps", [
    [],
    [("*", "*", False)],
    [("*", "*", True), ("grader", "n", False)],
    [("sim", "*", True), ("sim", "h", False)],
    [("sim", "h", False), ("sim", "*", True)],
    [("*", "h", False), ("default", "h", True)],
    [("default", "n", True), ("*", "n", False)],
    [("grader", "*", True), ("*", "*", False), ("grader", "h", False)],
])
def test_the_steps_a_task_writes(steps):
    gateway, access = _both()
    for step in steps:
        gateway._apply_rule(*step)
        access.apply(*step)
    _assert_same(gateway, access, steps)


@pytest.mark.parametrize("action, body", [
    ("disable", {"role": "sim", "tools": ["n", "h"]}),
    ("enable", {"role": "sim", "tools": "*"}),
    ("disable", {"role": "*", "tools": ["n"]}),
])
def test_a_change_answers_with_the_gateways_body(action, body):
    gateway, access = _both()
    for tool in body["tools"] if isinstance(body["tools"], list) else ["*"]:
        gateway._apply_rule(body["role"], tool, action == "disable")
    expected = {"role": body["role"], **gateway._format_role_rules(gateway._role_rules.get(body["role"], {}))}
    assert access.change(action, body) == expected
