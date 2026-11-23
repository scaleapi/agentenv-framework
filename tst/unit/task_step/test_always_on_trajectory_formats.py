"""always_on prescribes no format beside native to the agent it prompts, once, before its first wake, whatever the run
prescribes: until always_on reads atif/1 copies, its conversations build native only, even on an agent an earlier step
prescribed atif/1."""

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import SOLVER, Rig

pytestmark = pytest.mark.usefixtures("local_stores")


@pytest.mark.asyncio
@pytest.mark.parametrize("formats", [["atif/1"], None])
async def test_always_on_prescribes_native_only_to_its_agent_before_its_first_wake(monkeypatch, tmp_path, formats):
    happened: list[tuple] = []

    async def prescribe(context, step_id, agent, a2a_url, *, formats=None):
        happened.append(("prescribe", step_id, agent.agent_name, a2a_url, formats))

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

    assert happened[0] == ("prescribe", "week", "solver", SOLVER, ())
    assert [event for event in happened if event[0] == "prescribe"] == [happened[0]]
    assert happened[1][0] == "wake"
