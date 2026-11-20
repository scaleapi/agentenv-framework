"""always_on prescribes the run's trajectory formats to the agent it prompts, once, before its first wake, so the agent
builds atif/1 for the run's conversation only when the run prescribes it."""

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import SOLVER, Rig

pytestmark = pytest.mark.usefixtures("local_stores")


@pytest.mark.asyncio
@pytest.mark.parametrize("formats", [["atif/1"], None])
async def test_always_on_prescribes_the_runs_formats_to_its_agent_before_its_first_wake(monkeypatch, tmp_path, formats):
    happened: list[tuple] = []

    async def prescribe(agent, a2a_url, wanted):
        happened.append(("prescribe", agent.agent_name, a2a_url, wanted))

    def agent(n, texts, rig):
        happened.append(("wake", n))
        rig.quiet()
        return ""

    monkeypatch.setattr(ao, "prescribe_formats", prescribe)
    rig = Rig(monkeypatch, tmp_path, agent=agent)
    await rig.arm()
    context = rig.context()
    if formats is not None:
        context.metadata["user_overrides"] = {"trajectory_formats": formats}

    await rig.run(rig.step(), context)

    assert happened[0] == ("prescribe", "solver", SOLVER, tuple(formats or ()))
    assert [event for event in happened if event[0] == "prescribe"] == [happened[0]]
    assert happened[1][0] == "wake"
