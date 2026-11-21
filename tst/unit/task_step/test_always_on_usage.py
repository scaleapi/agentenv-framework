"""An always_on step keeps its agents' usage as prompt_agent does: one entry for the agent's wakes and nudges and one
for the user sim agent's reads, each summed, with the endpoint each called."""

from __future__ import annotations

import pytest

from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")


class _AskingPlugin:
    @staticmethod
    def setup(version, person):
        return {"system_prompt": "you are Linda", "output_format": None}

    @staticmethod
    async def answer(version, person, history, read, ask):
        got = await ask(f"read at {read['time']}")
        return {**(got["structured"] or {}), "raw": got["text"]}


def _agent(n, texts, rig):
    rig.message("blue or green?") if n == 1 else rig.quiet()
    return ""


@pytest.mark.asyncio
async def test_each_role_s_turns_are_summed_with_the_endpoint_it_called(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent, plugin=_AskingPlugin,
              user_texts=['{"outcome": "reply", "message": "blue"}'])
    rig.solver_usage = {"tool_call_count": 1, "input_tokens": 100, "output_tokens": 10, "input_includes_cache": True}
    rig.user_usage = {"input_tokens": 30, "output_tokens": 3, "input_includes_cache": True, "cost_usd": 0.001}
    ctx = rig.context()
    ctx.deployed_agents[0].model_endpoint_host = "proxy.example"
    ctx.deployed_agents[1].model_endpoint_host = "sim-proxy.example"
    await rig.arm()

    await rig.run(rig.step(), ctx)

    solver, user = ctx.prompt_responses[0].usage
    wakes, reads = len(rig.sent), len(rig.user_calls)
    assert (solver["role"], solver["endpoint_host"], solver["turns"], solver["input_tokens"]) == (
        "solver", "proxy.example", wakes, 100 * wakes)
    assert (user["role"], user["model"], user["endpoint_host"], user["turns"], user["cost_usd"]) == (
        "user_simulator", None, "sim-proxy.example", reads, pytest.approx(0.001 * reads))
    assert ctx.metadata["step_usage"]["week"] == [solver, user]
    assert ctx.metadata["model_endpoints"]["user_simulator"]["week"]["endpoint_host"] == "sim-proxy.example"



def _times_out_on_its_second_wake(n, texts, rig):
    if n == 2:
        raise TimeoutError("the agent's turn ran out")
    rig.message("blue or green?")
    return ""


@pytest.mark.asyncio
async def test_a_wake_that_times_out_leaves_the_agent_s_entry_a_lower_bound(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_times_out_on_its_second_wake, plugin=_AskingPlugin,
              user_texts=['{"outcome": "reply", "message": "blue"}'])
    rig.solver_usage = {"tool_call_count": 1, "input_tokens": 100, "output_tokens": 10}
    ctx = rig.context()
    await rig.arm()

    await rig.run(rig.step(), ctx)

    solver = ctx.metadata["step_usage"]["week"][0]
    assert len(rig.sent) == 2
    assert (solver["role"], solver["turns"], solver["input_tokens"], solver["partial"]) == ("solver", 2, 100, True)


@pytest.mark.asyncio
async def test_a_read_that_times_out_leaves_the_user_sim_s_entry_a_lower_bound(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent, plugin=_AskingPlugin,
              user_texts=[TimeoutError("the user sim took too long")])
    rig.solver_usage = {"tool_call_count": 1, "input_tokens": 100, "output_tokens": 10}
    ctx = rig.context()
    await rig.arm()

    await rig.run(rig.step(), ctx)

    solver, user = ctx.metadata["step_usage"]["week"]
    assert len(rig.user_calls) == 1 and solver["partial"] is False
    assert (user["role"], user["turns"], user["input_tokens"], user["partial"]) == ("user_simulator", 1, None, True)
