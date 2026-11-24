"""A run that prescribed atif/1 is judged from each turn's copy: the turn's own task cut out of the conversation's
document, with the media it names, and a turn whose copy couldn't be had from its native trajectory."""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace

import pytest
from agentenv_protocol.atif import load, strip_profile

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.config import get_config
from agent_env.env.env import DeployedEnv
from agent_env.task_step.task_steps.verifiers.env_outcome_verifier import EnvOutcomeVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import TrajectoryFilter
from tst.unit.task_step.verifiers import test_judge_inputs as J
from tst.unit.task_step.verifiers import test_rubrics_verifier_golden as G
from tst.util import atif_runs

pytestmark = pytest.mark.usefixtures("local_stores")

FIRST, SECOND = "parse_date without changing the test", "Also cover a +05:30 offset."


def _turn_texts(prompt: str) -> list[str]:
    return re.split(r"===== Turn \d+ =====\n", prompt.split("<agent_trajectory>", 1)[1].split("</agent_trajectory>")[0])[1:]


async def _model_prompt(step_kw: dict, run) -> str:
    given = await J._model_judged(step_kw, run)(lambda context: context)
    return given["calls"][0]["prompt"]


@pytest.mark.asyncio
async def test_a_direct_judge_reads_each_turn_alone_under_its_marker():
    first, second = _turn_texts(await _model_prompt({"use_trajectory": True}, J._atif_run))
    first, second = json.loads(first.strip()), json.loads(second.strip())
    assert FIRST in first["steps"][0]["message"] and second["steps"][0]["message"] == SECOND
    assert [s["step_id"] for s in second["steps"]] == [1, 2, 3]
    assert "final_metrics" not in first and "final_metrics" not in second
    load(second)


@pytest.mark.asyncio
async def test_a_turn_of_a_document_another_prompt_shares_is_cut_to_its_own_task():
    prompt = await _model_prompt({"use_trajectory": True, "trajectory_filter": TrajectoryFilter()},
                                 lambda: J._atif_run(tasks=("task-91c3",)))
    events = json.loads(prompt.split("<agent_trajectory>", 1)[1].split("</agent_trajectory>")[0])
    assert [(e["type"], e["tool"]) for e in events[:2]] == [("tool_call", "Edit"), ("tool_result", "Edit")]
    assert events[-1] == {"type": "final_response", "text": "Added test_offset.", "timestamp": ""}
    assert "pytest tests/test_parse.py" not in prompt


@pytest.mark.asyncio
async def test_graded_up_to_a_turn_no_later_turn_leaks():
    def run():
        context = J._atif_run()
        context.metadata["user_overrides"] = {"step_params": {"verify": {"grade_up_to_turn": 1}}}
        return context

    prompt = await _model_prompt({"use_trajectory": True}, run)
    assert FIRST in prompt and SECOND not in prompt and "Added test_offset" not in prompt


@pytest.mark.asyncio
async def test_a_mixed_run_reads_each_turn_by_its_label():
    native, atif = _turn_texts(await _model_prompt({"use_trajectory": True}, lambda: J._atif_run(native_first=True)))
    assert native.strip() == J.TURNS[0]
    assert json.loads(atif.strip())["steps"][0]["message"] == SECOND


@pytest.mark.asyncio
async def test_final_frames_come_from_the_media_its_results_name():
    given = await J._model_judged(J._FINAL_FRAMES, lambda: J._atif_run("computer_use", tasks=("task-c1",)))(
        lambda context: context)
    assert [block["image_url"]["url"] for block in given["calls"][0]["images"]] == [
        "data:image/png;base64,c2NyZWVuc2hvdC0y", "data:image/png;base64,c2NyZWVuc2hvdC0z"]


@pytest.mark.asyncio
async def test_a_frame_whose_media_is_missing_or_mismatched_is_skipped(caplog):
    caplog.set_level(logging.WARNING)

    def run():
        context = J._atif_run("computer_use", tasks=("task-c1",))
        store = get_config().get_object_store()
        store.put(f"{atif_runs.folder()}media/69a8aa01fe5537076d023dd5e9cc4275b39fc0434aa6483a9f58b66ef2f3c896.png",
                  b"not the frame", allow_overwrite=True)
        return context

    given = await J._model_judged(J._FINAL_FRAMES, run)(lambda context: context)
    assert [block["image_url"]["url"] for block in given["calls"][0]["images"]] == [
        "data:image/png;base64,c2NyZWVuc2hvdC0x", "data:image/png;base64,c2NyZWVuc2hvdC0z"]
    assert any("skipped" in record.message and "sha256" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_a_per_criterion_judge_with_no_stored_media_is_skipped():
    step = G._step(**J._SCREENSHOTS)
    step._prompt_llm_judge = J._Judge(lambda ids: G._evidence(ids)).model
    done = await step.execute(J._atif_run("computer_use", tasks=("task-c1",), media=False))
    assert [row["id"] for row in done.metadata["verifications"]["vid"]["results"]] == ["no_screenshots"]


@pytest.mark.asyncio
async def test_an_agent_judge_gets_each_turn_and_its_media_in_a_folder_of_its_own(monkeypatch):
    runs = [await J._agent_judged(monkeypatch, {"use_trajectory": True},
                                  lambda: J._atif_run("computer_use", tasks=("task-c1",)))(lambda context: context)
            for _ in range(2)]
    folders = [{path.rsplit("/", 2)[0] if "/media/" in path else path.rsplit("/", 1)[0] for path in given["files"]}
               for given in runs]
    assert all(len(folder) == 1 for folder in folders) and folders[0] != folders[1]
    files, prompt = runs[0]["files"], runs[0]["calls"][0]["prompt"]
    (folder,) = folders[0]
    assert folder in prompt
    load(json.loads(files[f"{folder}/turn_01.json"]["text"]))
    assert sorted(path for path in files if "/media/" in path) == sorted(
        f"{folder}/{path}" for path in ["media/82e60066a2de261b6653bcdda90d1cb9a7af2f474cecd82e8d371e8384d72315.png",
                                        "media/69a8aa01fe5537076d023dd5e9cc4275b39fc0434aa6483a9f58b66ef2f3c896.png",
                                        "media/482405b002a62efa8516c36709a7a100637507670576fa73a20ad5420d55c66f.png"])
    assert "turn_01.json is an ATIF (Agent Trajectory Interchange Format, v1.8) document" in prompt
    assert "is relative to this directory" in prompt


@pytest.mark.asyncio
async def test_an_agent_judge_on_a_mixed_run_is_told_which_turns_are_atif(monkeypatch):
    given = await J._agent_judged(monkeypatch, {"use_trajectory": True},
                                  lambda: J._atif_run(native_first=True))(lambda context: context)
    prompt = given["calls"][0]["prompt"]
    assert "2 files named turn_01.json … turn_02.json" in prompt and "turn_02.json is an ATIF" in prompt
    assert "media/<sha256>" not in prompt
    texts = [file["text"] for path, file in sorted(given["files"].items())]
    assert texts[0] == J.TURNS[0] and json.loads(texts[1])["steps"][0]["message"] == SECOND


@pytest.mark.asyncio
async def test_an_agent_judge_on_compacted_turns_gets_each_turns_tool_results(monkeypatch):
    given = await J._agent_judged(monkeypatch, {"use_trajectory": True, "trajectory_filter": TrajectoryFilter()},
                                  J._atif_run)(lambda context: context)
    assert [path for path in given["files"] if path.startswith("/tmp/tool_call_results/")] == [
        "/tmp/tool_call_results/turn_01_tool_call_result_1.json"]
    assert "Each file is a compacted event list" in given["calls"][0]["prompt"]


@pytest.mark.asyncio
async def test_a_scripted_verify_gets_each_turn_by_its_label(monkeypatch):
    script = b"async def verify(url, trajectory):\n    return [{'id': 'c', 'score': 1.0, 'given': trajectory}]\n"
    monkeypatch.setattr(FileArtifact, "get", staticmethod(lambda id, version=None: SimpleNamespace(load=lambda: script)))
    context = J._atif_run()
    context.prompt_responses[0].target_agent_per_turn_trajectory_object_urls.insert(0, None)
    context.prompt_responses[0].target_agent_per_turn_trajectories.insert(0, atif_runs.native("task-0"))
    context.deployed_envs.append(DeployedEnv(env_id="e", env_version=1, mcp_url="http://mcp"))
    step = EnvOutcomeVerifierTaskStep(id="outcome", version=None, env_id="e", file_artifact_id="s", verifier_id="v")
    given = (await step.execute(context)).metadata["verifications"]["v"]["results"][0]["given"]
    assert given[0] is None
    assert [doc["steps"][0]["message"] for doc in given[1:]][1] == SECOND
    assert FIRST in given[1]["steps"][0]["message"]
    assert all("final_metrics" not in doc for doc in given[1:])


@pytest.mark.asyncio
async def test_a_copy_without_the_profile_is_graded_turn_by_turn():
    """Every profile field is optional: a copy whose tasks carry no ids is cut at the step each turn reached."""
    def run():
        context = J._atif_run()
        atif_runs.stored("two_turns", document=strip_profile(atif_runs.fixture("two_turns")))
        return context

    first, second = _turn_texts(await _model_prompt({"use_trajectory": True}, run))
    first, second = json.loads(first.strip()), json.loads(second.strip())
    assert FIRST in first["steps"][0]["message"] and second["steps"][0]["message"] == SECOND
    assert [s["step_id"] for s in second["steps"]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_a_scripted_verify_can_ask_for_each_turns_label(monkeypatch):
    script = b"async def verify(url, trajectory_formats):\n    return [{'id': 'c', 'score': 1.0, 'given': trajectory_formats}]\n"
    monkeypatch.setattr(FileArtifact, "get", staticmethod(lambda id, version=None: SimpleNamespace(load=lambda: script)))
    context = J._atif_run(native_first=True)
    context.deployed_envs.append(DeployedEnv(env_id="e", env_version=1, mcp_url="http://mcp"))
    step = EnvOutcomeVerifierTaskStep(id="outcome", version=None, env_id="e", file_artifact_id="s", verifier_id="v")
    given = (await step.execute(context)).metadata["verifications"]["v"]["results"][0]["given"]
    assert given == ["claude-code-stream-json", "atif/1"]
