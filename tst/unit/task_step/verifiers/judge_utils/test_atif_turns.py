from __future__ import annotations

import base64
import json
import logging

import pytest
from agentenv_protocol.atif import AtifError, errors_in, strip_profile, task_document, validate

from agent_env.config import get_config
from agent_env.task_step.context import PromptResponse
from agent_env.task_step.task_steps.verifiers.judge_utils import atif_turns as T
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import TrajectoryFilter
from tst.util import atif_runs

pytestmark = pytest.mark.usefixtures("local_stores")

SHOT = "media/82e60066a2de261b6653bcdda90d1cb9a7af2f474cecd82e8d371e8384d72315.png"


def _response(urls, records) -> PromptResponse:
    return PromptResponse(prompt_id="p", response="", target_agent_per_turn_trajectory_object_urls=urls,
                          target_agent_per_turn_trajectories=records)


def test_each_turn_is_read_by_its_label():
    url = atif_runs.stored("two_turns")
    response = _response(["file:///native.json", None, url],
                         [atif_runs.native("task-91c2"), atif_runs.native("task-x"),
                          atif_runs.record(url, "task-91c3", 9)])
    assert T.turns(response) == [T.TurnSource("file:///native.json"), None, T.TurnSource(url, "task-91c3", 9)]
    assert T.atif_turns(response) == [T.TurnSource("file:///native.json"), T.TurnSource(url, "task-91c3", 9)]


@pytest.mark.parametrize("records", [None, [atif_runs.native("task-1")], [{"format": "x", "atif": None}],
                                     [{"format": "atif/1", "atif": None}]])
def test_a_response_with_no_atif_turn_is_left_to_the_native_readers(records):
    response = _response(["file:///a.json"], records)
    assert T.atif_turns(response) is None
    assert T.turns(response) == [T.TurnSource("file:///a.json")]


def test_a_view_is_the_turns_own_task_and_each_document_is_read_once(monkeypatch):
    url = atif_runs.stored("two_turns")
    store = get_config().get_object_store()
    reads: list[str] = []
    get = store.get
    monkeypatch.setattr(type(store), "get", lambda self, u: reads.append(u) or get(u))
    documents = T.AtifDocuments()
    first, second = (documents.view(T.TurnSource(url, task)) for task in ("task-91c2", "task-91c3"))
    whole = atif_runs.fixture("two_turns")
    assert first == task_document(whole, ["task-91c2"]) and second == task_document(whole, ["task-91c3"])
    assert [step["message"] for step in second["steps"]][0] == "Also cover a +05:30 offset."
    assert "final_metrics" not in second
    assert reads == [url]
    assert json.loads(documents.text(T.TurnSource(url, "task-91c3"))) == second


def test_a_copy_without_the_profile_finds_its_turn_by_the_step_it_reached():
    """Every profile field is optional, so a copy may carry no task ids: its turn is the task ending at its step."""
    url = atif_runs.stored("two_turns", document=strip_profile(atif_runs.fixture("two_turns")))
    view = T.AtifDocuments().view(T.TurnSource(url, "task-91c3", 9))
    assert [step["step_id"] for step in view["steps"]] == [1, 2, 3] and errors_in(validate(view)) == []


def test_a_copy_the_turn_cant_be_found_in_is_read_as_stored(caplog):
    """The requested trajectory is best effort: a judge grades what the copy holds rather than nothing."""
    url = atif_runs.stored("two_turns")
    caplog.set_level(logging.WARNING)
    assert T.AtifDocuments().view(T.TurnSource(url, "task-gone", 99)) == atif_runs.fixture("two_turns")
    broken = atif_runs.stored("two_turns", document={**atif_runs.fixture("two_turns"), "steps": [None]}, media=False)
    assert T.AtifDocuments().view(T.TurnSource(broken, "task-91c2", 6))["steps"] == [None]
    assert sum("reads the copy as stored" in record.message for record in caplog.records) == 2


def test_staging_hashes_each_file_once_and_keeps_none_of_its_bytes(monkeypatch):
    url = atif_runs.stored("computer_use")
    store = get_config().get_object_store()
    opened: list[str] = []
    real_open = type(store).open
    monkeypatch.setattr(type(store), "open", lambda self, u: opened.append(u) or real_open(self, u))
    documents = T.AtifDocuments()
    source = T.TurnSource(url, "task-c1", 5)
    staged = documents.staged_media(source)
    assert len(staged) == 3 and documents.staged_media(source) == staged
    assert len(opened) == 3 and all(isinstance(held, bool) for held in documents._stored.values())


def test_a_screenshot_is_typed_by_its_bytes_not_the_type_its_part_declares():
    jpeg = b"\xff\xd8\xff\xe0" + b"j" * 16
    document = atif_runs.fixture("computer_use")
    _, frames = T.transcript_from_atif(document, lambda path: jpeg)
    assert {frame.media_type for frame in frames} == {"image/jpeg"}
    _, frames = T.transcript_from_atif(document, lambda path: b"not an image")
    assert {frame.media_type for frame in frames} == {"image/png"}


def test_a_sub_agents_screenshots_are_actions_and_frames_after_the_call_that_started_it():
    parent = atif_runs.fixture("subagent")
    child = atif_runs.fixture("computer_use")
    parent["subagent_trajectories"][0]["steps"] = child["steps"]
    transcript, frames = T.transcript_from_atif(parent, lambda path: b"\x89PNG\r\n\x1a\n" + path.encode())
    assert len(frames) == 3 and sum(action.screenshot is not None for action in transcript.actions) == 3


def test_media_resolve_in_the_documents_folder_checked_against_their_names(caplog):
    url = atif_runs.stored("computer_use")
    documents = T.AtifDocuments()
    assert documents.media(url, SHOT) == b"screenshot-1"
    assert T.media_url(url, SHOT) == f"{url.rsplit('/', 1)[0]}/{SHOT}"
    with pytest.raises(AtifError):
        T.media_url(url, "../secrets.json")
    caplog.set_level(logging.WARNING)
    assert documents.media(url, "media/../../x.png") is None
    assert documents.media(url, "media/" + "0" * 64 + ".png") is None
    store = get_config().get_object_store()
    store.put(f"{atif_runs.folder()}{SHOT}", b"another file", allow_overwrite=True)
    assert T.AtifDocuments().media(url, SHOT) is None
    assert sum("skipped" in record.message for record in caplog.records) == 3


def test_staged_media_are_those_stored_as_named():
    url = atif_runs.stored("computer_use", media=False)
    source = T.TurnSource(url, "task-c1")
    assert T.AtifDocuments().staged_media(source) == []
    atif_runs.stored("computer_use")
    staged = T.AtifDocuments().staged_media(source)
    assert [path for path, _ in staged] == [
        SHOT, "media/69a8aa01fe5537076d023dd5e9cc4275b39fc0434aa6483a9f58b66ef2f3c896.png",
        "media/482405b002a62efa8516c36709a7a100637507670576fa73a20ad5420d55c66f.png"]
    assert all(media == T.media_url(url, path) for path, media in staged)


def test_the_default_compaction_gives_the_otel_compactions_events():
    view = task_document(atif_runs.fixture("two_turns"), ["task-91c2"])
    events, files = T.compact_atif_trajectory(view, TrajectoryFilter(), result_file_prefix="turn_01_")
    assert [(e["type"], e["tool"]) for e in events if e["type"] != "final_response"] == [
        ("tool_call", "Bash"), ("tool_result", "Bash"), ("tool_call", "Read"), ("tool_result", "Read"),
        ("tool_call", "Edit"), ("tool_result", "Edit"), ("tool_call", "Bash"), ("tool_result", "Bash")]
    assert events[0] == {"type": "tool_call", "tool": "Bash", "input": {"command": "pytest tests/test_parse.py -q"},
                         "timestamp": ""}
    assert events[3] == {"type": "tool_result", "tool": "Read",
                         "output_file": "/tmp/tool_call_results/turn_01_tool_call_result_1.json", "timestamp": ""}
    assert [path for path, _ in files] == ["/tmp/tool_call_results/turn_01_tool_call_result_1.json"]
    assert json.loads(files[0][1]).startswith("2026-01-01T03:04:05Z")
    assert events[-1] == {"type": "final_response", "text": "parse_date now accepts a trailing Z; all 3 tests in "
                          "tests/test_parse.py pass.", "timestamp": ""}
    thinking, _ = T.compact_atif_trajectory(view, TrajectoryFilter(include_thinking=True, include_tool_calls=False,
                                                                     include_tool_call_results=False,
                                                                     include_response=False))
    assert thinking == [{"type": "thinking", "text": "Run the failing test first to see the error.", "timestamp": ""}]


def test_the_default_compaction_writes_out_image_results_and_walks_sub_agents():
    events, files = T.compact_atif_trajectory(atif_runs.fixture("computer_use"), TrajectoryFilter())
    assert [e.get("output_file") for e in events if e["type"] == "tool_result"] == [
        f"/tmp/tool_call_results/tool_call_result_{n}.json" for n in (1, 2, 3)]
    assert json.loads(files[0][1])[0]["source"]["path"] == SHOT
    events, _ = T.compact_atif_trajectory(atif_runs.fixture("subagent"), TrajectoryFilter())
    assert [(e["type"], e.get("tool")) for e in events] == [
        ("tool_call", "Task"), ("tool_result", "Task"), ("tool_call", "Bash"), ("tool_result", "Bash"),
        ("final_response", None)]
    assert events[-1]["text"] == "There are three: a.toml, b.toml and c.toml."


def test_a_transcript_takes_actions_from_tool_calls_and_frames_from_result_images():
    url = atif_runs.stored("computer_use")
    documents = T.AtifDocuments()
    transcript, frames = T.transcript_from_atif(atif_runs.fixture("computer_use"),
                                                lambda path: documents.media(url, path))
    assert [(a.index, a.tool, a.args.get("action")) for a in transcript.actions] == [
        (1, "computer", "screenshot"), (2, "computer", "left_click"), (3, "computer", "type")]
    assert [base64.b64decode(a.screenshot) for a in transcript.actions] == [
        b"screenshot-1", b"screenshot-2", b"screenshot-3"]
    assert [(f.media_type, base64.b64decode(f.data)) for f in frames] == [
        ("image/png", b"screenshot-1"), ("image/png", b"screenshot-2"), ("image/png", b"screenshot-3")]
    assert [(m.after_action, m.text) for m in transcript.messages] == [(3, "B2 is now 42 and the file is saved.")]
    transcript, frames = T.transcript_from_atif(atif_runs.fixture("computer_use"), lambda path: None)
    assert frames == [] and all(a.screenshot is None for a in transcript.actions)
