"""cua_verifier: a computer-use task's rubric as rubrics_verifier's criteria, its score (binary criteria, weights,
negative penalties, must-pass gates), and a run against a computer at its driver's boundary (``tst.unit.computer_home``)
with the judge faked: the files the agent left read through the driver and rendered into the judge's prompt, a missing
one named, the golden files and the state commands' output beside them, the caller's guidance or the default, and the
files graded alone when no prompt step names them; state output that was cut marked incomplete, a Windows computer
refused."""

from __future__ import annotations

import io
import json
import re
from types import SimpleNamespace

import openpyxl
import pytest
from agentenv_protocol.computer import COMPUTER

from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.verifiers import cua_verifier
from agent_env.task_step.task_steps.verifiers.cua_verifier import (
    DEFAULT_GOLDEN_NOTE,
    DEFAULT_GUIDANCE,
    MAX_FILE_CHARS,
    RECORDS_KEY,
    CuaVerifierTaskStep,
    criteria_from_rubric,
    cua_rubric_score,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from tst.unit.computer_home import CARD, Computer

RUBRIC = {"criteria": [
    {"id": "sheet-exists", "title": "Provides report.xlsx.", "weight": 8,
     "annotations": {"criteria_type": "MUST-PASS", "criteria_category": "format_gate"}},
    {"id": "total", "title": "Cell B3 of report.xlsx totals the amounts.", "weight": 50,
     "annotations": {"criteria_type": "REGULAR", "target_result_files": ["report.xlsx"]}},
    {"id": "note", "title": "notes.txt says the report is done.", "weight": 25,
     "annotations": {"criteria_category": "must have criteria"}},
    {"id": "leak", "title": "The email reveals the customer's card number.", "weight": 30,
     "annotations": {"criteria_type": "NEGATIVE"}},
    {"id": "look", "title": "The table is legible.", "annotations": {"criteria_category": "visual"}},
]}


def test_the_rubric_becomes_criteria_by_kind_with_weights_and_annotations():
    criteria = {c["id"]: c for c in criteria_from_rubric(RUBRIC)}

    assert criteria["sheet-exists"] == {"id": "sheet-exists", "title": "Provides report.xlsx.", "kind": "must-pass",
                                        "weight": 8.0}
    assert criteria["total"]["kind"] == "regular" and criteria["total"]["applies_to"] == {"result_files": ["report.xlsx"]}
    assert criteria["note"]["kind"] == "regular"
    assert criteria["leak"] == {"id": "leak", "title": "The email reveals the customer's card number.",
                                "kind": "negative", "weight": -30.0, "negative": True}
    assert criteria["look"]["weight"] == 0.0 and criteria["look"]["visual"] is True
    assert criteria_from_rubric([{"id": "x", "title": "t", "weight": -2}])[0]["kind"] == "negative"


@pytest.mark.parametrize("rubric, problem", [
    ({"criteria": []}, "non-empty list"), ([{"title": "t"}], "needs an id"), ([{"id": "x"}], "needs a title"),
    ([{"id": "x", "title": "t", "weight": "5"}], "weight must be a number"),
])
def test_a_rubric_it_cant_grade_is_refused(rubric, problem):
    with pytest.raises(ValueError, match=problem):
        criteria_from_rubric(rubric)


def _rows(**scores) -> list[dict]:
    return [{**c, "score": scores.get(c["id"], 1.0)} for c in criteria_from_rubric(RUBRIC)]


@pytest.mark.parametrize("scores, expected", [
    ({}, 1.0),                                   # all met, the failure mode absent
    ({"note": 0.0}, 58 / 83),                    # a regular criterion missed
    ({"leak": 0.0}, (83 - 30) / 83),             # the failure mode present: its penalty
    ({"note": 0.0, "total": 0.0, "leak": 0.0}, 0.0),  # clamped at 0
    ({"sheet-exists": 0.0}, 0.0),                # a must-pass gate failed
    ({"total": 0.9}, 33 / 83),                   # binary: anything short of 1 is 0
])
def test_the_score_is_the_gated_weighted_average_less_the_penalties(scores, expected):
    assert cua_rubric_score(_rows(**scores)) == pytest.approx(expected)


def test_gates_alone_score_one_less_the_penalties():
    rows = [{"id": "g", "kind": "must-pass", "weight": 0.0, "score": 1.0},
            {"id": "n", "kind": "negative", "weight": -0.25, "score": 0.0}]
    assert cua_rubric_score(rows) == pytest.approx(0.75)


@pytest.fixture
def computer(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    return Computer(tmp_path / "home").install(monkeypatch)


def _workbook() -> bytes:
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Report"
    sheet["A1"], sheet["B1"], sheet["B2"], sheet["B3"] = "amount", 5, 7, "=SUM(B1:B2)"
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def _step(**fields) -> CuaVerifierTaskStep:
    params = dict(id="grade", version=None, env_id="desk", rubric=RUBRIC, use_agent_judge=False,
                  verifier_id="cua", result_files=["/HOME/Desktop/report.xlsx", "~/notes.txt", "~/missing.pdf"],
                  instruction="Total the amounts in report.xlsx and leave a note.")
    params.update(fields)
    return CuaVerifierTaskStep(**params)


def _prompted_ids(prompt: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r'"id": "(c\d+)"', prompt)))


def _judge(step: CuaVerifierTaskStep, prompts: list[str], failing: tuple[str, ...] = ()) -> None:
    """A judge that fails the criteria ``failing`` names and meets the rest, each with a justification of its own."""
    short = {f"c{i}": c["id"] for i, c in enumerate(step.criteria, start=1)}

    async def judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        prompts.append(prompt)
        return {"response": json.dumps({"results": [
            {"id": cid, "score": 0.0 if short[cid] in failing else 1.0,
             "justification": f"Read the file for {short[cid]}"} for cid in _prompted_ids(prompt)]}),
            "trajectory_object_url": f"mem://judge/{len(prompts)}.json"}

    step._prompt_llm_judge = judge


@pytest.mark.asyncio
async def test_the_files_left_in_the_computer_are_rendered_into_the_judges_prompt_and_graded(computer, local_stores,
                                                                                            tmp_path):
    (computer.home / "Desktop").mkdir()
    (computer.home / "Desktop" / "report.xlsx").write_bytes(_workbook())
    (computer.home / "notes.txt").write_text("The report is done.\n")
    (tmp_path / "golden").mkdir()
    (tmp_path / "golden" / "report.csv").write_text("amount,12\n")
    golden = FileArtifactUniverse.put_bundled("golden-files", files={"report.csv": tmp_path / "golden" / "report.csv"})
    step = _step(result_files=[f"{computer.home}/Desktop/report.xlsx", "~/notes.txt", "~/missing.pdf"],
                 expected_artifact_id=golden.id, expected_artifact_version=golden.version,
                 state_commands=[{"name": "mailbox", "command": "echo '{\"sent\": 1}'"}],
                 guidance="Grade it as the house style says.", golden_note="Golden files are examples.")
    prompts: list[str] = []
    _judge(step, prompts, failing=("note",))
    ctx = TaskStepContext(deployed_envs=[computer.deployed()])

    await step.execute(ctx)

    [prompt] = prompts
    assert "B3 = None [formula: =SUM(B1:B2)]" in prompt and "B1 = 5" in prompt
    assert "The report is done." in prompt
    assert "## Files the agent did NOT produce" in prompt and "- ~/missing.pdf" in prompt
    assert "Golden files are examples." in prompt and "amount,12" in prompt
    assert '<state name="mailbox" exit_code="0">\n{"sent": 1}' in prompt
    assert "Total the amounts in report.xlsx" in prompt and "Grade it as the house style says." in prompt
    assert DEFAULT_GUIDANCE not in prompt and DEFAULT_GOLDEN_NOTE not in prompt
    assert len(_prompted_ids(prompt)) == 5
    assert ctx.prompt_responses == []  # the stand-in the files were graded with is gone
    entry = ctx.metadata["verifications"]["cua"]
    assert entry["score"] == pytest.approx(58 / 83) and entry["gate_passed"] and "gates_failed" not in entry
    assert {row["id"]: row["kind"] for row in entry["results"]}["sheet-exists"] == "must-pass"
    record = ctx.metadata[RECORDS_KEY]["grade"]
    assert [(f["path"], f.get("rendered_as"), f.get("missing", False)) for f in record["files"]] == [
        (f"{computer.home}/Desktop/report.xlsx", "xlsx", False), ("~/notes.txt", "text", False),
        ("~/missing.pdf", None, True)]
    assert record["golden_files"] == [{"name": "report.csv", "bytes": 10, "rendered_as": "text"}]
    assert record["state"] == [{"name": "mailbox", "exit_code": 0, "timed_out": False, "cut": False, "truncated": False}]


@pytest.mark.asyncio
async def test_a_failed_must_pass_gate_zeroes_the_score_and_is_named(computer):
    step = _step(result_files=["~/notes.txt"])
    _judge(step, [], failing=("sheet-exists",))
    ctx = TaskStepContext(deployed_envs=[computer.deployed()])

    await step.execute(ctx)

    entry = ctx.metadata["verifications"]["cua"]
    assert entry["score"] == 0.0 and not entry["gate_passed"] and entry["gates_failed"] == ["sheet-exists"]


@pytest.mark.asyncio
async def test_a_file_is_read_in_chunks_and_one_too_large_is_named_not_read(computer, monkeypatch):
    from agent_env.task_step.task_steps.verifiers import cua_verifier

    monkeypatch.setattr(cua_verifier, "READ_CHUNK_BYTES", 10)
    monkeypatch.setattr(cua_verifier, "MAX_READ_BYTES", 100)
    (computer.home / "notes.txt").write_text("0123456789abcdefghijklmnopqrstuvwxyz\n")
    (computer.home / "huge.txt").write_text("x" * 101)
    step = _step(result_files=["~/notes.txt", "~/huge.txt"])
    prompts: list[str] = []
    _judge(step, prompts)
    ctx = TaskStepContext(deployed_envs=[computer.deployed()])

    await step.execute(ctx)

    assert "0123456789abcdefghijklmnopqrstuvwxyz\n" in prompts[0] and "a file of 101 bytes, too large" in prompts[0]
    assert sum(1 for call in computer.calls if call[0] == "run" and "tail" in call[1]["argv"][2]) == 4
    assert ctx.metadata[RECORDS_KEY]["grade"]["files"][1] == {"path": "~/huge.txt", "bytes": 101,
                                                              "rendered_as": "too_large"}


@pytest.mark.asyncio
async def test_a_dash_prefixed_file_is_read_whole_and_a_short_read_is_refused(computer, monkeypatch):
    from agent_env.task_step.task_steps.verifiers import cua_verifier

    (computer.home / "--report.csv").write_text("total,1500\n")
    step = _step(result_files=["~/--report.csv"])
    prompts: list[str] = []
    _judge(step, prompts)

    await step.execute(TaskStepContext(deployed_envs=[computer.deployed()]))

    assert "total,1500" in prompts[0]

    monkeypatch.setattr(cua_verifier, "_SIZE", 'test -f "$1" || exit 3; echo 999')  # a file that shrank since
    with pytest.raises(RuntimeError, match=r"gave 11 bytes from byte 0 of its 999"):
        await _step(result_files=["~/--report.csv"]).execute(TaskStepContext(deployed_envs=[computer.deployed()]))


@pytest.mark.asyncio
async def test_the_files_only_stand_in_never_joins_the_context(computer):
    step = _step(result_files=[])
    seen: list[list] = []

    async def judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        seen.append(list(context.prompt_responses))
        return {"response": json.dumps({"results": [{"id": cid, "score": 1.0, "justification": f"Checked {cid}"}
                                                    for cid in _prompted_ids(prompt)]})}

    step._prompt_llm_judge = judge
    ctx = TaskStepContext(deployed_envs=[computer.deployed()])

    await step.execute(ctx)

    assert seen == [[]] and ctx.prompt_responses == []


@pytest.mark.asyncio
async def test_with_a_prompt_id_the_judge_also_sees_that_prompts_response(computer):
    step = _step(prompt_id="p1", result_files=[])
    prompts: list[str] = []
    _judge(step, prompts)
    ctx = TaskStepContext(deployed_envs=[computer.deployed()],
                          prompt_responses=[PromptResponse(prompt_id="p1", prompt_text="do it", response="Done: B3")])

    await step.execute(ctx)

    assert "Done: B3" in prompts[0] and len(ctx.prompt_responses) == 1
    assert DEFAULT_GUIDANCE in prompts[0]


def test_it_round_trips_through_the_registry_keeping_a_files_only_grade():
    step = _step(computer="desk", guidance="Be strict.", state_commands=[{"name": "s", "command": ["cat", "/tmp/s.json"]}])
    data = step.to_dict()
    again = get_task_step_registry()["cua_verifier"].from_dict(data)

    assert data["prompt_id"] is None and again.grades_files_alone and again.to_dict() == data
    assert again.criteria == step.criteria and again.score_rows is not None


@pytest.mark.parametrize("fields, problem", [
    ({"guidance": "  "}, "guidance must be text"),
    ({"golden_note": 3}, "golden_note must be text"),
    ({"use_trajectory": True}, "use_trajectory needs a prompt_id"),
    ({"result_files": "a.txt"}, "result_files must be a list"),
    ({"state_commands": [{"name": "s"}]}, "each of state_commands"),
])
def test_bad_configs_are_refused(fields, problem):
    with pytest.raises(ValueError, match=problem):
        _step(**fields)


@pytest.mark.asyncio
@pytest.mark.parametrize("stdout, truncated, why", [
    ("x" * (MAX_FILE_CHARS + 5), False, f"cut after {MAX_FILE_CHARS} characters"),
    ("short", True, "cut short by the computer's driver"),
])
async def test_state_output_that_was_cut_is_marked_incomplete(computer, stdout, truncated, why):
    step = _step(result_files=["~/notes.txt"], state_commands=[{"name": "mailbox", "command": "cat /tmp/m.json"}])
    prompts: list[str] = []
    _judge(step, prompts)

    async def state(deployed, command):
        return {"exit_code": 0, "timed_out": False, "stdout": stdout, "truncated": truncated}

    step._state = state
    ctx = TaskStepContext(deployed_envs=[computer.deployed()])

    await step.execute(ctx)

    assert f"INCOMPLETE: {why}. What isn't shown can't support a criterion." in prompts[0]
    [record] = ctx.metadata[RECORDS_KEY]["grade"]["state"]
    assert record["cut"] == (len(stdout) > MAX_FILE_CHARS) and record["truncated"] == truncated


@pytest.mark.asyncio
async def test_a_windows_computer_is_refused_before_anything_is_read(computer):
    card = {**CARD, "capabilities": {**CARD["capabilities"], "extensions": [
        *CARD["capabilities"]["extensions"], {"uri": COMPUTER, "params": {"platform": {"os": "windows"}}}]}}
    step = _step(result_files=["~/notes.txt"])
    _judge(step, [])

    with pytest.raises(RuntimeError, match="is a windows computer, which it can't read"):
        await step.execute(TaskStepContext(deployed_envs=[computer.deployed(card=card)]))
    assert computer.calls == []


def test_preflight_refuses_a_stored_windows_computer(monkeypatch):
    desk = ComputerEnv.__new__(ComputerEnv)
    desk.id, desk.environment_name = "desk", "desk"
    desk.computer = SimpleNamespace(os=SimpleNamespace(type="windows"))
    monkeypatch.setattr(cua_verifier, "stored_env", lambda env_id: desk)
    monkeypatch.setattr(RubricsVerifierTaskStep, "preflight", lambda self: [])

    assert _step(result_files=["~/a.txt"]).preflight() == [
        "step 'grade': env 'desk''s computer is windows, which cua_verifier can't read: it reads files and runs state "
        "commands through sh and POSIX tools"]
    desk.computer = SimpleNamespace(os=SimpleNamespace(type="linux"))
    assert _step(result_files=["~/a.txt"]).preflight() == []
