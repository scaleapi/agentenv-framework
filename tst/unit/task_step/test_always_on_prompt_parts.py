"""A task prompt's file (and other non-text) parts reach the agent on the opening wake, as prompt_agent sends them, and
are in the user's opening turn of the graded conversation; its text is the user's labelled line."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

ITINERARY = {"kind": "file", "file": {"uri": "s3://trips/<trip>/itinerary.pdf", "name": "itinerary.pdf",
                                      "mimeType": "application/pdf"}}


@pytest.mark.asyncio
async def test_the_prompt_s_file_reaches_the_agent_on_the_opening_wake_in_its_place(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    step = ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt_id="ask", agent_name="solver",
                               parts=[{"kind": "text", "text": "sort the cdmx weekend"}, ITINERARY,
                                      {"kind": "text", "text": "flights are in the pdf"}])
    ctx = rig.context()
    ctx.metadata["seed"] = {"trip": "cdmx"}
    ctx = await rig.run(step, ctx)

    opening = rig.parts[0]
    files = [p for p in opening if p.get("kind") == "file"]
    assert files == [{"kind": "file", "file": {**ITINERARY["file"], "uri": "s3://trips/cdmx/itinerary.pdf"}}]
    texts = [p["text"] for p in opening if p.get("kind") == "text"]
    line = "[Mon Nov 9, 2:00 PM UTC] Linda: sort the cdmx weekend\nflights are in the pdf"
    assert texts.count(line) == 1
    assert opening.index(files[0]) == next(i for i, p in enumerate(opening) if p.get("text") == line) + 1
    [user_turn, *_] = [m for m in rig.chat(ctx) if m["role"] == "user"]
    assert [p.get("kind") for p in user_turn["parts"]] == ["text", "file"]
    assert user_turn["parts"][1]["file"]["uri"] == "s3://trips/cdmx/itinerary.pdf"
