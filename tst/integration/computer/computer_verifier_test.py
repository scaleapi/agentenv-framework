"""computer_verifier end to end on a real computer, its judge faked at its call boundary: an ubuntu-24.04 computer with the
reference driver deployed, files and a mock mailbox's state left in it as an agent would leave them (by computer_exec),
and a CUA rubric graded on them beside the echo agent's response to the task. The files and state are read through the driver into the judge's prompt; the fake
judge meets each criterion unless it names a file the prompt lists as not produced. The rubric scores 1, and so does
its aggregate gated on a step whose one criterion is the CSV; gated instead on a step for a PDF the computer hasn't,
the aggregate scores 0, naming that gate."""

import json
import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.task_step.task_steps.verifiers.computer.computer_verifier import DEFAULT_GUIDANCE, ComputerVerifierTaskStep
from tst.integration.computer.conftest import run_context
from tst.util.a2a_test_agent import put_test_agent

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BUNDLE = Path(__file__).parent / "computer_verifier"
_CRITERION = re.compile(r'"id": "(c\d+)",\s*"title": "((?:[^"\\]|\\.)*)"')


def _fake_judge(prompts: list[str]):
    """Meets each criterion of the prompt unless its title names a file the prompt lists as not produced."""

    async def judge(self, prompt: str, *, model: str, context, **_) -> dict:
        prompts.append(prompt)
        missing = re.findall(r"^- (.+)$", prompt.split("## Files the agent did NOT produce", 1)[1].split("##")[0],
                             re.M) if "## Files the agent did NOT produce" in prompt else []
        rows = [{"id": cid, "score": 0.0 if any(Path(m).name in title for m in missing) else 1.0,
                 "justification": f"Checked {cid} against the files shown"}
                for cid, title in dict(_CRITERION.findall(prompt)).items()]
        return {"response": json.dumps({"results": rows})}

    return judge


def test_a_rubric_is_graded_on_what_the_agent_left_in_the_computer(state, monkeypatch):
    prompts: list[str] = []
    monkeypatch.setattr(ComputerVerifierTaskStep, "_prompt_llm_judge", _fake_judge(prompts))
    put_test_agent("computer-verifier-echo")

    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "grade", "--no-emulate"])

    assert ran.exit_code == 0, ran.output
    graded = next(p for p in prompts if "budget.csv" in p and "mailbox" in p)
    assert "total,1500" in graded and '"to": "ana@example.com"' in graded and DEFAULT_GUIDANCE in graded
    assert "Echo: Write ~/Desktop/budget.csv" in graded
    metadata = run_context(ran.output)["metadata"]
    files = metadata["computer_verifier"]["grade"]["files"]
    assert [(f["path"], f["rendered_as"]) for f in files] == [("~/Desktop/budget.csv", "text"),
                                                              ("~/notes.txt", "text")], files
    assert metadata["computer_verifier"]["grade"]["state"] == [
        {"name": "mailbox", "exit_code": 0, "timed_out": False, "cut": False, "truncated": False}]
    grades = metadata["grades"]
    assert grades["grade"]["score"] == 1.0 and grades["csv"]["score"] == 1.0, grades["grade"]
    assert grades["final"]["score"] == 1.0, grades["final"]
    assert metadata["computer_verifier"]["pdf"]["files"][0] == {"path": "~/Desktop/report.pdf", "missing": True}
    gated = grades["gated"]
    assert gated["score"] == 0.0 and gated["verifier_output"]["failed_gates"] == ["pdf"], gated
