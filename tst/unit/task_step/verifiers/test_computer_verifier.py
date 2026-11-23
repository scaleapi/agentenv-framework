"""computer_verifier: a computer-use task's rubric as rubrics_verifier's criteria (weights, negative penalties, a
must-pass one refused for a step of its own that an aggregate gates on), and a run against a computer at its driver's
boundary (``tst.unit.computer_home``) with the judge faked: the files the agent left read through the driver and
rendered into the judge's prompt beside the agent's response, a missing one named, the golden files and the state
commands' output beside them, and the caller's guidance or the default; state output that was cut marked incomplete, a
Windows computer refused."""

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
from agent_env.task_step.task_steps.verifiers.computer import computer_verifier
from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import AggregateVerifiersTaskStep
from agent_env.task_step.task_steps.verifiers.computer.computer_verifier import (
    DEFAULT_GOLDEN_NOTE,
    DEFAULT_GUIDANCE,
    MAX_FILE_CHARS,
    RECORDS_KEY,
    ComputerVerifierTaskStep,
    criteria_from_rubric,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from tst.unit.computer_home import CARD, Computer

RUBRIC = {"criteria": [
    {"id": "sheet-exists", "title": "Provides report.xlsx.", "weight": 8,
     "annotations": {"criteria_type": "REGULAR", "criteria_category": "format_gate"}},
    {"id": "total", "title": "Cell B3 of report.xlsx totals the amounts.", "weight": 50,
     "annotations": {"criteria_type": "REGULAR", "target_result_files": ["report.xlsx"]}},
    {"id": "note", "title": "notes.txt says the report is done.", "weight": 25,
     "annotations": {"criteria_category": "must have criteria"}},
    {"id": "leak", "title": "The email reveals the customer's card number.", "weight": 30,
     "annotations": {"criteria_type": "NEGATIVE"}},
    {"id": "look", "title": "The table is legible.", "annotations": {"criteria_category": "visual"}},
]}


def test_the_rubric_becomes_criteria_with_weights_and_annotations():
    criteria = {c["id"]: c for c in criteria_from_rubric(RUBRIC)}

    assert criteria["sheet-exists"] == {"id": "sheet-exists", "title": "Provides report.xlsx.", "weight": 8.0}
    assert criteria["total"]["applies_to"] == {"result_files": ["report.xlsx"]} and "negative" not in criteria["note"]
    assert criteria["leak"] == {"id": "leak", "title": "The email reveals the customer's card number.",
                                "weight": -30.0, "negative": True}
    assert criteria["look"]["weight"] == 0.0 and criteria["look"]["visual"] is True
    assert criteria_from_rubric([{"id": "x", "title": "t", "weight": -2}])[0]["negative"] is True


@pytest.mark.parametrize("rubric, problem", [
    ({"criteria": []}, "non-empty list"), ([{"title": "t"}], "needs an id"), ([{"id": "x"}], "needs a title"),
    ([{"id": "x", "title": "t", "weight": "5"}], "weight must be a number"),
    ([{"id": "x", "title": "t", "annotations": {"criteria_type": "MUST-PASS"}}], "in a computer_verifier of their own"),
])
def test_a_rubric_it_cant_grade_is_refused(rubric, problem):
    with pytest.raises(ValueError, match=problem):
        criteria_from_rubric(rubric)


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


def _step(**fields) -> ComputerVerifierTaskStep:
    params = dict(id="grade", version=None, env_id="desk", rubric=RUBRIC, use_agent_judge=False, prompt_id="p1",
                  verifier_id="cua", result_files=["/HOME/Desktop/report.xlsx", "~/notes.txt", "~/missing.pdf"],
                  instruction="Total the amounts in report.xlsx and leave a note.")
    params.update(fields)
    return ComputerVerifierTaskStep(**params)


def _context(computer: Computer, response: str = "Done.", **deployed) -> TaskStepContext:
    """A run with ``computer`` deployed and the agent's ``response`` to the prompt ``p1``, which the step grades."""
    return TaskStepContext(deployed_envs=[computer.deployed(**deployed)],
                           prompt_responses=[PromptResponse(prompt_id="p1", prompt_text="do it", response=response)])


def _prompted_ids(prompt: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r'"id": "(c\d+)"', prompt)))


def _judge(step: ComputerVerifierTaskStep, prompts: list[str], failing: tuple[str, ...] = ()) -> None:
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
    ctx = _context(computer)

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
    entry = ctx.metadata["verifications"]["cua"]
    assert entry["score"] == pytest.approx(58 / 83)
    assert {row["id"]: row.get("negative") for row in entry["results"]}["leak"] is True
    record = ctx.metadata[RECORDS_KEY]["grade"]
    assert [(f["path"], f.get("rendered_as"), f.get("missing", False)) for f in record["files"]] == [
        (f"{computer.home}/Desktop/report.xlsx", "xlsx", False), ("~/notes.txt", "text", False),
        ("~/missing.pdf", None, True)]
    assert record["golden_files"] == [{"name": "report.csv", "bytes": 10, "rendered_as": "text"}]
    assert record["state"] == [{"name": "mailbox", "exit_code": 0, "timed_out": False, "cut": False, "truncated": False}]


@pytest.mark.asyncio
async def test_must_pass_criteria_are_a_step_of_their_own_that_an_aggregate_gates_on(computer):
    gate = _step(id="must", verifier_id="must", result_files=["~/notes.txt"], score_aggregator="all_pass",
                 rubric=[{"id": "sheet-exists", "title": "Provides report.xlsx."}])
    rest = _step(result_files=["~/notes.txt"])
    _judge(gate, [], failing=("sheet-exists",))
    _judge(rest, [])
    ctx = _context(computer)

    await gate.execute(ctx)
    await rest.execute(ctx)
    await AggregateVerifiersTaskStep(id="final", version=None, sources=["grade"], gates=["must"]).execute(ctx)

    grades = ctx.metadata["grades"]
    assert grades["must"]["score"] == 0.0 and grades["grade"]["score"] == 1.0
    assert grades["final"]["score"] == 0.0 and grades["final"]["verifier_output"]["failed_gates"] == ["must"]


@pytest.mark.asyncio
async def test_a_file_is_read_in_chunks_and_one_too_large_is_named_not_read(computer, monkeypatch):
    from agent_env.task_step.task_steps.verifiers.computer import computer_verifier

    monkeypatch.setattr(computer_verifier, "READ_CHUNK_BYTES", 10)
    monkeypatch.setattr(computer_verifier, "MAX_READ_BYTES", 100)
    (computer.home / "notes.txt").write_text("0123456789abcdefghijklmnopqrstuvwxyz\n")
    (computer.home / "huge.txt").write_text("x" * 101)
    step = _step(result_files=["~/notes.txt", "~/huge.txt"])
    prompts: list[str] = []
    _judge(step, prompts)
    ctx = _context(computer)

    await step.execute(ctx)

    assert "0123456789abcdefghijklmnopqrstuvwxyz\n" in prompts[0] and "a file of 101 bytes, too large" in prompts[0]
    assert sum(1 for call in computer.calls if call[0] == "run" and "tail" in call[1]["argv"][2]) == 4
    assert ctx.metadata[RECORDS_KEY]["grade"]["files"][1] == {"path": "~/huge.txt", "bytes": 101,
                                                              "rendered_as": "too_large"}


@pytest.mark.asyncio
async def test_a_dash_prefixed_file_is_read_whole_and_a_short_read_is_refused(computer, monkeypatch):
    from agent_env.task_step.task_steps.verifiers.computer import computer_verifier

    (computer.home / "--report.csv").write_text("total,1500\n")
    step = _step(result_files=["~/--report.csv"])
    prompts: list[str] = []
    _judge(step, prompts)

    await step.execute(_context(computer))

    assert "total,1500" in prompts[0]

    monkeypatch.setattr(computer_verifier, "_SIZE", 'test -f "$1" || exit 3; echo 999')  # a file that shrank since
    with pytest.raises(RuntimeError, match=r"gave 11 bytes from byte 0 of its 999"):
        await _step(result_files=["~/--report.csv"]).execute(_context(computer))


@pytest.mark.asyncio
async def test_the_judge_also_sees_the_agents_response(computer):
    step = _step(result_files=[])
    prompts: list[str] = []
    _judge(step, prompts)
    ctx = _context(computer, response="Done: B3")

    await step.execute(ctx)

    assert "Done: B3" in prompts[0] and len(ctx.prompt_responses) == 1
    assert DEFAULT_GUIDANCE in prompts[0]


def test_it_round_trips_through_the_registry():
    step = _step(computer="desk", guidance="Be strict.", state_commands=[{"name": "s", "command": ["cat", "/tmp/s.json"]}])
    data = step.to_dict()
    again = get_task_step_registry()["computer_verifier"].from_dict(data)

    assert data["prompt_id"] == "p1" and again.to_dict() == data
    assert again.criteria == step.criteria and again.score_aggregator.value == "weighted_average"
    gate = _step(score_aggregator="all_pass")
    assert get_task_step_registry()["computer_verifier"].from_dict(gate.to_dict()).score_aggregator.value == "all_pass"


@pytest.mark.parametrize("fields, problem", [
    ({"guidance": "  "}, "guidance must be text"),
    ({"golden_note": 3}, "golden_note must be text"),
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
    ctx = _context(computer)

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
        await step.execute(_context(computer, card=card))
    assert computer.calls == []


def test_preflight_refuses_a_stored_windows_computer(monkeypatch):
    desk = ComputerEnv.__new__(ComputerEnv)
    desk.id, desk.environment_name = "desk", "desk"
    desk.computer = SimpleNamespace(os=SimpleNamespace(type="windows"))
    monkeypatch.setattr(computer_verifier, "stored_env", lambda env_id: desk)
    monkeypatch.setattr(RubricsVerifierTaskStep, "preflight", lambda self: [])

    assert _step(result_files=["~/a.txt"]).preflight() == [
        "step 'grade': env 'desk''s computer is windows, which computer_verifier can't read: it reads files and runs "
        "state commands through sh and POSIX tools"]
    desk.computer = SimpleNamespace(os=SimpleNamespace(type="linux"))
    assert _step(result_files=["~/a.txt"]).preflight() == []
