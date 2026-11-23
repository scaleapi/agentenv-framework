"""always_on prescribes the run's formats to the agent it prompts, once, before its first wake, and reads each turn's
trajectory as prompt_agent does: a run that prescribes atif/1 to an agent that offers it has each wake's and nudge's
atif/1 copy as its trajectory, read off the turn's path and filled in when the run ends, and every turn records what its
url holds."""

import asyncio

import pytest
from agentenv_protocol.a2a_agent import TRAJECTORY_V1

from agent_env.task_step.context import AtifRecord
from agent_env.task_step.snapshot_utils import atif_copies
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.prompt_agent import StoredTrajectory
from tst.unit.task_step.always_on_harness import SOLVER, Rig

pytestmark = pytest.mark.usefixtures("local_stores")

DOCUMENT = "file:///store/atif/instance_id=ti-1/solver/conversation.json"
TRAJECTORY = {"uri": TRAJECTORY_V1.uri, "params": {
    "formats": ["shots/1", "atif/1"], "native": "shots/1", "default": "shots/1",
    "methods": {"get": {"endpoint": "/ext/trajectory"}}}}
NATIVE_ONLY = {"uri": TRAJECTORY_V1.uri, "params": {
    "formats": ["shots/1"], "native": "shots/1", "default": "shots/1", "methods": {"get": {"endpoint": "/ext/trajectory"}}}}


def _quiet(n, texts, rig):
    rig.quiet()
    return ""


def _nudged(n, texts, rig):
    """The wake closes with text and neither messages nor stays quiet, so a nudge turn follows, in which it is quiet."""
    if n == 1:
        return "All set."
    rig.quiet()
    return ""


def _offering(rig: Rig, prescribed: bool, extension: dict = TRAJECTORY):
    """A run whose solver's card offers ``extension``'s formats (atif/1 beside its native one, by default), prescribing
    atif/1 or not."""
    context = rig.context()
    solver = next(agent for agent in context.deployed_agents if agent.agent_name == "solver")
    solver.a2a_card = {"capabilities": {"extensions": [extension]}}
    if prescribed:
        context.metadata["user_overrides"] = {"trajectory_formats": ["atif/1"]}
    return context


def _reads(monkeypatch, *, copy_seconds: float = 0.0, gate: asyncio.Event | None = None) -> dict[str, list]:
    """Native reads and atif/1 copies, by task: each native read stored under its task, each copy a valid record of
    the one conversation document, taking ``copy_seconds`` once ``gate`` is set."""
    seen: dict[str, list] = {"native": [], "copies": [], "cancelled": []}

    async def native(a2a_url, extension, task_id, prefix, name, **kwargs):
        seen["native"].append(task_id)
        return StoredTrajectory(f"file:///store/native-{task_id}.json", "shots/1")

    async def stored(self, task_id, name):
        try:
            if gate is not None:
                await gate.wait()
            await asyncio.sleep(copy_seconds)
        except asyncio.CancelledError:
            seen["cancelled"].append(task_id)
            raise
        seen["copies"].append(task_id)
        return AtifRecord(url=DOCUMENT, task_id=task_id, through_step=len(seen["copies"]) * 3, status="valid", rules=[],
                          reason=None, validator_version="v", adapter_version="a", aux_url=None, turn_context_url=None)

    monkeypatch.setattr(ao, "fetch_native_trajectory", native)
    monkeypatch.setattr(atif_copies.AtifCopies, "_stored", stored)
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("formats", [["atif/1"], None])
async def test_always_on_prescribes_the_runs_formats_to_its_agent_before_its_first_wake(monkeypatch, tmp_path, formats):
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

    assert happened[0] == ("prescribe", "week", "solver", SOLVER, None)
    assert [event for event in happened if event[0] == "prescribe"] == [happened[0]]
    assert happened[1][0] == "wake"


@pytest.mark.asyncio
async def test_a_prescribed_runs_turns_are_their_atif_copies_each_named_in_the_timeline(monkeypatch, tmp_path):
    seen = _reads(monkeypatch)
    rig = Rig(monkeypatch, tmp_path, agent=_nudged)
    await rig.arm()
    context = await rig.run(rig.step(max_wakes=2), _offering(rig, prescribed=True))

    [response] = context.prompt_responses
    records = response.target_agent_per_turn_trajectories
    assert seen["native"] == [] and len(seen["copies"]) == len(records) >= 2
    assert response.target_agent_per_turn_trajectory_object_urls == [DOCUMENT] * len(records)
    assert [(r["format"], r["atif"]["status"], r["atif"]["task_id"]) for r in records] == [
        ("atif/1", "valid", task_id) for task_id in seen["copies"]]
    ends = [e for e in rig.stored(context)["timeline"] if e["event"] in ("agent_turn_end", "nudge") and "state" in e]
    assert [e["event"] for e in ends] == ["agent_turn_end", "nudge"]
    assert [(e["turn"], e["trajectory_format"], e["trajectory_url"]) for e in ends] == [
        (n, "atif/1", DOCUMENT) for n in range(len(records))]


@pytest.mark.asyncio
async def test_a_wake_goes_on_while_its_copy_is_read_and_each_copy_fills_its_turn_when_the_run_ends(monkeypatch,
                                                                                                    tmp_path):
    gate = asyncio.Event()
    seen = _reads(monkeypatch, gate=gate)
    monkeypatch.setattr(ao, "TRAJECTORY_FETCH_SECONDS", 5)

    def agent(n, texts, rig):
        """The wake closes with text, so a nudge follows; the wake's copy is held until the nudge runs."""
        if n == 1:
            return "All set."
        gate.set()
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    await rig.arm()
    context = await rig.run(rig.step(max_wakes=2), _offering(rig, prescribed=True))

    [response] = context.prompt_responses
    records = response.target_agent_per_turn_trajectories
    assert len(seen["copies"]) == len(records) == 2
    assert [r["atif"]["status"] for r in records] == ["valid", "valid"]
    assert response.target_agent_per_turn_trajectory_object_urls == [DOCUMENT, DOCUMENT]
    ends = [e for e in rig.stored(context)["timeline"] if e["event"] in ("agent_turn_end", "nudge") and "state" in e]
    assert [(e["turn"], e["trajectory_format"], e["trajectory_url"]) for e in ends] == [
        (0, "atif/1", DOCUMENT), (1, "atif/1", DOCUMENT)]


@pytest.mark.asyncio
async def test_a_step_whose_end_fails_before_it_joins_its_copies_cancels_them(monkeypatch, tmp_path):
    seen = _reads(monkeypatch, gate=asyncio.Event())
    finish = ao._Run.finish

    async def failing(run):
        """The run's end, once its copy is held reading, failing at its first clock read."""
        async def clock_read_fails():
            raise RuntimeError("the clock read failed")
        await asyncio.sleep(0.05)
        run.now = clock_read_fails
        await finish(run)

    monkeypatch.setattr(ao._Run, "finish", failing)
    rig = Rig(monkeypatch, tmp_path, agent=_quiet)
    await rig.arm()
    with pytest.raises(RuntimeError, match="clock read failed"):
        await rig.run(rig.step(max_wakes=1), _offering(rig, prescribed=True))
    await asyncio.sleep(0)
    assert len(seen["cancelled"]) == 1 and seen["copies"] == []


@pytest.mark.asyncio
async def test_a_copy_still_going_a_fetch_cap_after_the_run_ends_is_unavailable_with_no_url(monkeypatch, tmp_path):
    _reads(monkeypatch, copy_seconds=5)
    monkeypatch.setattr(ao, "TRAJECTORY_FETCH_SECONDS", 0.05)
    rig = Rig(monkeypatch, tmp_path, agent=_quiet)
    await rig.arm()
    context = await rig.run(rig.step(max_wakes=1), _offering(rig, prescribed=True))

    [response] = context.prompt_responses
    assert response.target_agent_per_turn_trajectory_object_urls == [None]
    [record] = response.target_agent_per_turn_trajectories
    assert (record["format"], record["atif"]["status"], record["atif"]["reason"]) == (None, "unavailable", "timeout")


@pytest.mark.asyncio
async def test_a_run_prescribing_atif_to_an_agent_offering_only_native_reads_native_and_names_no_turn(monkeypatch,
                                                                                                      tmp_path):
    seen = _reads(monkeypatch)
    rig = Rig(monkeypatch, tmp_path, agent=_nudged)
    await rig.arm()
    context = await rig.run(rig.step(max_wakes=2), _offering(rig, prescribed=True, extension=NATIVE_ONLY))

    [response] = context.prompt_responses
    assert response.error_type is None
    assert seen["copies"] == [] and len(seen["native"]) == 2
    assert response.target_agent_per_turn_trajectory_object_urls == [
        f"file:///store/native-{task_id}.json" for task_id in seen["native"]]
    assert [(r["format"], r["atif"]["status"], r["atif"]["reason"])
            for r in response.target_agent_per_turn_trajectories] == [
        ("shots/1", "unavailable", "format_not_offered")] * 2
    ends = [e for e in rig.stored(context)["timeline"] if e["event"] in ("agent_turn_end", "nudge")]
    assert ends and all("turn" not in e and "trajectory_format" not in e for e in ends)


@pytest.mark.asyncio
async def test_a_run_that_prescribes_nothing_reads_native_and_names_no_turn_in_the_timeline(monkeypatch, tmp_path):
    seen = _reads(monkeypatch)
    rig = Rig(monkeypatch, tmp_path, agent=_nudged)
    await rig.arm()
    context = await rig.run(rig.step(max_wakes=2), _offering(rig, prescribed=False))

    [response] = context.prompt_responses
    assert seen["copies"] == [] and seen["native"]
    assert response.target_agent_per_turn_trajectory_object_urls == [
        f"file:///store/native-{task_id}.json" for task_id in seen["native"]]
    assert response.target_agent_per_turn_trajectories == [{"format": "shots/1", "atif": None}] * len(seen["native"])
    ends = [e for e in rig.stored(context)["timeline"] if e["event"] in ("agent_turn_end", "nudge")]
    assert ends and all("turn" not in e and "trajectory_format" not in e for e in ends)
