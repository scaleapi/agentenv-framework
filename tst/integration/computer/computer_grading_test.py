"""Grading a multi env's computer with a judge agent, end to end, composed of the steps each part takes: a multi env of
one ubuntu-24.04 computer with the reference driver deployed, files left in it as an agent would leave them (by
computer_exec), one in its home outside the folders a snapshot takes by default; snapshot_computer_env keeps those
files by their paths; load_artifact stages that snapshot and the task's golden files in the judge's sandbox; and
rubrics_verifier grades with that judge, which it tells where each was staged. The judge is faked at its call
boundary: it reads each file and its golden file in its own container, as a judge with Bash would, so the CSV matches
and the note, which leaves out the total, doesn't."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_agent_sandbox_provider
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from tst.integration.computer.conftest import run_context
from tst.util.a2a_test_agent import put_test_agent

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BUNDLE = Path(__file__).parent / "computer_grading"
RESULT, GOLDEN = "/tmp/grading/result", "/tmp/grading/golden"
# Each criterion, by the key the judge sees it under, and the file it compares with its golden file.
COMPARED = {"c1": ("desk/Desktop/budget.csv", "budget.csv"), "c2": ("desk/notes.txt", "notes.txt")}


def test_a_judge_grades_a_computers_files_against_golden_files_staged_in_its_sandbox(state, monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(RubricsVerifierTaskStep, "_invoke_judge_a2a", _comparing_judge(prompts))
    put_test_agent("computer-grading-echo")

    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "grade", "--no-emulate"])

    assert ran.exit_code == 0, ran.output
    [prompt] = prompts
    staged = {line.split("`")[3]: set(line.split("contains: ")[1].split(", "))
              for line in prompt.splitlines() if line.startswith("- `") and "contains: " in line}
    assert staged == {f"{RESULT}/": {"desk/Desktop/budget.csv", "desk/notes.txt"},
                      f"{GOLDEN}/": {"budget.csv", "notes.txt"}}, prompt
    grade = run_context(ran.output)["metadata"]["grades"]["grade"]
    assert {row["id"]: row["score"] for row in grade["verifier_criteria"]} == {"budget": 1.0, "note": 0.0}, (
        grade["verifier_criteria"])
    assert grade["score"] == 0.5, grade


def _comparing_judge(prompts: list[str]):
    """Meets each criterion whose file reads the same as its golden file, both read in the judge's own container."""

    async def judge(self, *, eval_prompt: str, judge_agent, **_) -> dict:
        prompts.append(eval_prompt)
        provider = (build_sandbox_provider(judge_agent.sandbox_type) if judge_agent.sandbox_type
                    else get_agent_sandbox_provider())
        sandbox = await provider.get_sandbox(judge_agent.sandbox_id)
        rows = []
        for key, (left, golden) in COMPARED.items():
            read = [await _read_in_judge(sandbox, judge_agent, path)
                    for path in (f"{RESULT}/{left}", f"{GOLDEN}/{golden}")]
            rows.append({"id": key, "score": 1.0 if read[0] == read[1] and read[0][0] == 0 else 0.0,
                         "justification": f"read {left} as {read[0]!r} and golden {golden} as {read[1]!r}"})
        return {"response": json.dumps({"results": rows})}

    return judge


async def _read_in_judge(sandbox, judge_agent, path: str) -> tuple[int, str, str]:
    """``cat path`` where load_artifact staged it: in the judge's container on a VM-backed sandbox."""
    if isinstance(sandbox, VmSandbox):
        container = judge_agent.installed_container or sandbox.container_name
        return await sandbox.exec_with_output("docker", "exec", container, "cat", path)
    return await sandbox.exec_with_output("cat", path)
