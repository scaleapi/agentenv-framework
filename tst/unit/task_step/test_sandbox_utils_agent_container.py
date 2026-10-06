"""``find_agent_container``: which container on a VM sandbox is the agent's."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_env.task_step.task_steps.sandbox_utils.sandbox_utils import find_agent_container


def _sandbox(running: str, *, owns_container: bool = False, exit_code: int = 0):
    async def exec_with_output(*args):
        assert args == ("sudo", "docker", "ps", "--format", "{{.Names}}")
        return exit_code, running, "boom"

    return SimpleNamespace(container_name="agent-local-1", owns_container=owns_container, exec_with_output=exec_with_output)


def test_the_sandboxs_own_container_wins():
    assert asyncio.run(find_agent_container(_sandbox("a2a-agent-x\nagent-local-1\n"))) == "agent-local-1"


def test_a_sandbox_that_owns_its_container_never_takes_another():
    with pytest.raises(RuntimeError, match="'agent-local-1' is not running"):
        asyncio.run(find_agent_container(_sandbox("a2a-agent-x\n", owns_container=True)))


def test_a_vm_the_run_does_not_own_falls_back_to_an_a2a_agent_container():
    assert asyncio.run(find_agent_container(_sandbox("db\na2a-agent-x\na2a-agent-y\n"))) == "a2a-agent-x"


@pytest.mark.parametrize("running, exit_code, message", [
    ("db\n", 0, "No agent container found"),
    ("", 1, "Failed to list containers: boom"),
])
def test_no_agent_container_is_an_error(running, exit_code, message):
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(find_agent_container(_sandbox(running, exit_code=exit_code)))
