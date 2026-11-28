"""Claude Code's ATIF adapter on real runs, recorded through the agent's image on the pinned CLI with atif/1 prescribed
(``tst/data/claude_code_atif``): each scenario's turns as the agent served them, its native events, its aux log and
its turn context, replayed through the adapter kit as the SDK ran it live. Each request shape's tools are kept once,
under ``tools/``, and named where they were fed.

The scenarios: ``coding`` (two turns of one conversation), ``subagent`` (an Agent call), ``browser`` (screenshots from a
browser's MCP tools), ``thinking`` (a thinking model), ``compacted`` (a conversation compacted between its turns),
``api-error`` (the endpoint refusing every call) and ``retried`` (a stalled call that the bridge resumed). Regenerate
the expected documents after an intended change with: python
tst/unit/agents/claude_code/test_claude_code_atif_recorded.py. Record the runs again, as a bump of the pinned CLI asks,
with ``tst/manual/claude_code_atif/record.py``, whose docstring has the steps."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import check_adapter, check_media, task_document, tasks  # noqa: E402

from agent_env.task_step.task_steps.verifiers.judge_utils.atif_turns import (  # noqa: E402
    compact_atif_trajectory,
    is_compaction,
    transcript_from_atif,
)
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import TrajectoryFilter  # noqa: E402
from agent_env.task_step.task_steps.verifiers.judge_utils.turn_events import (  # noqa: E402
    TrajectoryFormatError,
    atif_turn_events,
    turn_events,
)
from tst.unit.agents.claude_code.agent import claude_code_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    trajectories,
)

FIXTURES = REPO / "tst" / "data" / "claude_code_atif"
ClaudeCodeAtif = claude_code_atif.ClaudeCodeAtif


def _tool_files() -> dict[str, list]:
    return {f"tools/{path.name}": [claude_code_atif._function(tool) for tool in json.loads(path.read_text())]
            for path in sorted((FIXTURES / "tools").glob("*.json"))}


def _fed(entry: dict) -> dict:
    """An aux entry as the agent fed it, a request shape's tools read back from the shared file it names."""
    if entry["name"] != "tools":
        return entry
    tools = json.loads((FIXTURES / entry["record"]["tools"]["shared"]).read_text())
    return {**entry, "record": {**entry["record"], "tools": tools}}


def _read_back(document: dict) -> dict:
    """The checked-in document, each agent block's tool definitions read back from the shared file it names."""
    files = _tool_files()
    for trajectory in trajectories(document):
        named = trajectory["agent"].get("tool_definitions")
        if isinstance(named, dict):
            trajectory["agent"]["tool_definitions"] = files[named["shared"]]
    return document


def _named(document: dict) -> dict:
    """A replay's document as it's checked in, each agent block's tool definitions named by their shared file."""
    files = _tool_files()
    for trajectory in trajectories(document):
        definitions = trajectory["agent"].get("tool_definitions")
        named = next((name for name, converted in files.items() if converted == definitions), None)
        if named is not None:
            trajectory["agent"]["tool_definitions"] = {"shared": named}
    return document


RECORDINGS = Recordings(FIXTURES, ClaudeCodeAtif, aux=_fed, expected=_read_back, saved=_named)
SCENARIOS = RECORDINGS.scenarios
recorded, replay = RECORDINGS.recorded, RECORDINGS.replay


def compacted_turns(scenario: str) -> set[int]:
    """The turns whose native stdout compacts the conversation (a ``compact_boundary``)."""
    return {index for index, turn in enumerate(RECORDINGS.data(scenario)["turns"])
            if any(item.get("subtype") == "compact_boundary" for item in turn["native"] if isinstance(item, dict))}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """The copy's totals are the turns' reported usage, a turn that compacts the conversation's included: the step of
    the call that wrote its summary carries what the turn reported beyond its other steps."""
    assert_steps_sum_to_reported_usage(replay(scenario).document, recorded(scenario)[1])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_transcripts_time_and_usage(scenario):
    assert_every_model_call_has_its_time_and_usage(replay(scenario).document)


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: replay(scenario).document for scenario in SCENARIOS}
    assert documents["subagent"]["subagent_trajectories"][0]["agent"]["tool_definitions"]
    assert any(part.get("type") == "image" for step in documents["browser"]["steps"]
               for result in (step.get("observation") or {}).get("results") or ()
               for part in result.get("content") if isinstance(part, dict))
    retried = [step for step in documents["retried"]["steps"] if step["source"] == "agent"]
    assert any(step.get("extra", {}).get("agentenv", {}).get("attempt") == 2 for step in retried)
    assert any("error" in step.get("extra", {}).get("agentenv", {}) for step in documents["api-error"]["steps"])
    assert any(step.get("extra", {}).get("agentenv", {}).get("reasoning") for step in documents["thinking"]["steps"])
    compaction = [step for step in documents["compacted"]["steps"] if "compaction" in step.get("extra", {}).get(
        "agentenv", {})]
    assert compacted_turns("compacted") == {1} and len(compaction) == 1 and compaction[0]["metrics"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_judge_reads_each_recorded_turn_with_its_calls_results_answer_and_screenshots(scenario):
    """What the judges take from each turn's copy, cut out as its own task: every tool call and result, a sub-agent's
    included, the final answer, and each screenshot as a frame whose bytes match its name."""
    kit = replay(scenario)
    for task in tasks(kit.document):
        view = task_document(kit.document, [task.task_id])
        steps = [step for trajectory in trajectories(view) for step in trajectory["steps"] if step["source"] == "agent"]
        results = [result for step in steps for result in (step.get("observation") or {}).get("results") or ()]
        images = [part for result in results for part in result.get("content") or ()
                  if isinstance(part, dict) and part.get("type") == "image"]
        events, _ = compact_atif_trajectory(view, TrajectoryFilter())
        kinds = [event["type"] for event in events]
        assert kinds.count("tool_call") == sum(len(step.get("tool_calls") or ()) for step in steps)
        assert kinds.count("tool_result") == len(results)
        assert ("final_response" in kinds) == any(isinstance(step.get("message"), str) and step["message"].strip()
                                                 for step in view["steps"]
                                                 if step["source"] == "agent" and not is_compaction(step))
        transcript, frames = transcript_from_atif(view, lambda path: check_media(path, kit.media[path]) or kit.media[path])
        assert len(transcript.actions) == kinds.count("tool_call")
        assert len(frames) == len(images)
    if scenario == "browser":
        assert images and frames
    if scenario == "subagent":
        assert [event["tool"] for event in events if event["type"] == "tool_call" and event.get("subagent")] == ["Bash"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_recorded_turns_copy_gives_a_judge_the_events_its_native_trajectory_does(scenario):
    """always_on's judge stages each wake from its events. A copy gives the native turn's, but for a sub-agent's prompt,
    which only native stdout carries (as a note), and its closing text, which only its transcript, so the copy, has. A
    turn that compacts the conversation gives its summary as a compaction note with its facts, and the CLI's
    "Compacted" as a note, from both."""
    data = json.loads((FIXTURES / f"{scenario}.json").read_text())
    document = replay(scenario).document
    for turn in data["turns"]:
        native = turn_events(json.dumps(turn["native"])).events
        copied = atif_turn_events(task_document(document, [turn["turn"]["task_id"]])).events
        if scenario == "subagent":
            assert [e["type"] for e in native if e.get("subagent")] == ["note", "tool_call"]
            assert [e["type"] for e in copied if e.get("subagent")] == ["tool_call", "text"]
            native = [e for e in native if not (e["type"] == "note" and e.get("subagent"))]
            copied = [e for e in copied if not (e["type"] == "text" and e.get("subagent"))]
        assert copied == native


@pytest.mark.parametrize(("scenario", "turn", "facts", "said"), [
    ("compacted", 1, {"written_by": "assistant", "trigger": "manual", "tokens_before": 28634, "tokens_after": 2839},
     "it was set off on request, such as by a command (manual); the history went from 28,634 tokens to 2,839"),
    ("autocompacted", 0, {"written_by": "assistant", "trigger": "auto", "tokens_before": 58741, "tokens_after": 32484},
     "the harness set it off on its own (auto); the history went from 58,741 tokens to 32,484")])
def test_a_recorded_compaction_reaches_a_judge_as_one_note_with_its_facts_natively_and_from_its_copy(scenario, turn,
                                                                                                     facts, said):
    """/compact between turns, and an auto-compaction partway through one as the history filled."""
    recorded = json.loads((FIXTURES / f"{scenario}.json").read_text())["turns"][turn]
    native = turn_events(json.dumps(recorded["native"])).events
    copied = atif_turn_events(task_document(replay(scenario).document, [recorded["turn"]["task_id"]])).events
    [compaction] = [event for event in native if event.get("compaction")]
    assert compaction["compaction"] == facts
    assert compaction["text"].startswith(
        "Compaction summary (written by the assistant's own model, in a summary turn the harness asked for; it replaces "
        f"the older history the assistant sees; {said}):\nThis session is being continued from a previous conversation")
    assert [event for event in copied if event.get("compaction")] == [compaction]


def test_a_copys_compaction_that_doesnt_say_who_wrote_its_summary_is_refused_as_native_is():
    data = json.loads((FIXTURES / "compacted.json").read_text())
    document = replay("compacted").document
    step = next(step for step in document["steps"] if is_compaction(step))
    del step["extra"]["agentenv"]["compaction"]["written_by"]
    with pytest.raises(TrajectoryFormatError, match="who wrote the compaction summary isn't known"):
        atif_turn_events(task_document(document, [data["turns"][1]["turn"]["task_id"]]))


def test_from_atif_reads_harbors_own_conversion_of_a_recorded_session():
    harbor = json.loads((FIXTURES / "harbor" / "subagent.harbor.json").read_text())
    tasks, _ = recorded("subagent")
    kit = check_adapter(ClaudeCodeAtif, tasks, from_atif=[harbor])
    assert not [finding for finding in kit.findings if finding.rule == "kit-from-atif"]
    assert ClaudeCodeAtif.from_atif(harbor).log


if __name__ == "__main__":
    RECORDINGS.regenerate()
