"""The osworld_* steps on a computer env, graded in the worker: osworld_initialize runs a classic task's init_config on
the guest server its card advertises, with the env's control token, and refuses an OSWorld-V2 task; osworld_evaluate
grades each func with upstream's metric there, combined by its conj, an LLM-judge func by the judge the worker
registers, a failing func as a score of 0, a turn that errored as a skip unless the agent ran out of its budget, result
files staged first on a re-grade. upstream's runtime is stood in for here (runtime_test.py runs it)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_env.task_step.context import PromptResponse, StepContextError, TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.computer.osworld import evaluate as evaluate_module
from agent_env.task_step.task_steps.computer.osworld import guest, runtime
from agent_env.task_step.task_steps.computer.osworld.evaluate import OSWorldEvaluateTaskStep, budget_exhaustion
from agent_env.task_step.task_steps.computer.osworld.initialize import OSWorldInitializeTaskStep
from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import AggregateVerifiersTaskStep
from tst.unit.task_step.osworld.helpers import ENV_URL, TOKEN, card, computer

GUEST = f"{ENV_URL}/svc/osworld-server/osworld"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@dataclass
class Runtime:
    """upstream's setup and checks as the steps call them: setups recorded, each check answered from ``scores`` in
    turn, an exception raised."""

    setups: list[list[dict]] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    scores: list[Any] = field(default_factory=list)
    setup_error: Exception | None = None
    access: list[tuple[str, dict]] = field(default_factory=list)

    async def run_setup(self, guest_url, headers, config):
        self.access.append((guest_url, headers))
        self.setups.append(config)
        if self.setup_error is not None:
            raise self.setup_error

    async def run_evaluate(self, guest_url, headers, func, result, expected, options):
        self.access.append((guest_url, headers))
        self.checks.append({"func": func, "result": result, "expected": expected, "options": options})
        score = self.scores.pop(0) if self.scores else 1.0
        if isinstance(score, Exception):
            raise score
        return score


@pytest.fixture
def rt(monkeypatch):
    stand_in = Runtime()
    monkeypatch.setattr(runtime, "run_setup", stand_in.run_setup)
    monkeypatch.setattr(runtime, "run_evaluate", stand_in.run_evaluate)
    return stand_in


@dataclass
class Judge:
    """A judge the worker registers: the payloads it's given, answered from ``answers`` in turn."""

    answers: list[dict] = field(default_factory=list)
    payloads: list[dict] = field(default_factory=list)

    async def __call__(self, payload, context):
        self.payloads.append(payload)
        return self.answers.pop(0) if self.answers else {"score": 1.0}


@pytest.fixture
def judge(monkeypatch):
    registered = Judge()
    monkeypatch.setattr(evaluate_module, "_JUDGES", {})
    evaluate_module.register_judge("agent_judge", registered)
    return registered


def _context(env=None, *prompts: PromptResponse) -> TaskStepContext:
    context = TaskStepContext()
    context.deployed_envs.append(computer() if env is None else env)
    context.prompt_responses.extend(prompts)
    return context


def _prompt(**fields) -> PromptResponse:
    return PromptResponse(**{"prompt_id": "p", "response": "done", "prompt_text": "Save the sheet", **fields})


@pytest.mark.parametrize("data", [
    {"type": "osworld_initialize", "env_id": "desk", "init_config": [{"type": "sleep", "parameters": {"seconds": 1}}],
     "osworld_v2_task_path": "https://host/t.py"},
    {"type": "osworld_evaluate", "env_id": "desk", "evaluator": {"funcs": "compare_table"}, "verifier_id": "v",
     "osworld_v2_task_path": "https://host/t.py", "judge_model": "m"},
], ids=lambda data: data["type"])
def test_the_steps_are_registered_and_round_trip(data):
    step = get_task_step_registry()[data["type"]].from_dict({"id": "s", "version": 1, **data})
    assert {k: v for k, v in step.to_dict().items() if k in data} == data


def test_the_legacy_task_url_field_is_read_as_the_path():
    step = OSWorldInitializeTaskStep.from_dict({"id": "s", "env_id": "desk", "osworld_v2_task_url": "https://h/t.py"})
    assert step.osworld_v2_task_path == "https://h/t.py"


@pytest.mark.asyncio
async def test_initialize_runs_the_whole_config_on_the_card_endpoint_with_the_control_token(rt):
    config = [{"type": "download", "parameters": {"files": [{"url": "https://h/a", "path": "/home/user/a"}]}},
              {"type": "launch", "parameters": {"command": ["gedit"]}}]
    await OSWorldInitializeTaskStep(id="init", version=1, env_id="desk", init_config=config).execute(
        _context(computer(card("/apps/7/osworld"))))
    assert rt.setups == [config] and rt.access == [(f"{ENV_URL}/apps/7/osworld", HEADERS)]


@pytest.mark.asyncio
async def test_the_control_token_goes_only_to_a_guest_server_under_the_envs_address(rt):
    await OSWorldInitializeTaskStep(id="init", version=1, env_id="desk", init_config=[{"type": "sleep"}]).execute(
        _context(computer(card("https://elsewhere.example/g"))))
    assert rt.access == [("https://elsewhere.example/g", {})]


@pytest.mark.asyncio
async def test_initialize_fails_on_a_failed_setup_bounded(rt):
    rt.setup_error = RuntimeError("no such type " + "x" * 5000)
    with pytest.raises(RuntimeError, match="setup failed: RuntimeError: no such type") as raised:
        await OSWorldInitializeTaskStep(id="init", version=1, env_id="desk",
                                        init_config=[{"type": "sleep"}]).execute(_context())
    assert len(str(raised.value)) < 400


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [
    OSWorldInitializeTaskStep(id="s", version=1, env_id="desk", osworld_v2_task_path="https://host/t.py"),
    OSWorldEvaluateTaskStep(id="s", version=1, env_id="desk", osworld_v2_task_url="https://host/t.py")],
    ids=["initialize", "evaluate"])
async def test_an_osworld_v2_task_fails_the_step_saying_so(rt, step):
    with pytest.raises(ValueError, match="OSWorld-V2 tasks"):
        await step.execute(_context(None, _prompt()))
    assert rt.setups == [] and rt.checks == []


@pytest.mark.asyncio
async def test_a_step_on_an_env_not_deployed_is_refused(rt):
    with pytest.raises(StepContextError, match="not found"):
        await OSWorldInitializeTaskStep(id="init", version=1, env_id="other",
                                        init_config=[{"type": "sleep"}]).execute(_context())


@pytest.mark.asyncio
async def test_a_computer_without_the_app_is_refused(rt, monkeypatch):
    monkeypatch.setattr(guest, "CARD_WAIT_S", 0)
    monkeypatch.setattr(guest.protocol_v1, "get_card", AsyncMock(return_value={}))
    with pytest.raises(RuntimeError, match="osworld-server computer app"):
        await OSWorldInitializeTaskStep(id="init", version=1, env_id="desk",
                                        init_config=[{"type": "sleep"}]).execute(_context(computer({})))


@pytest.mark.asyncio
async def test_evaluate_skips_a_turn_that_errored(rt):
    context = _context(None, _prompt(error_type="INFRA_ERROR", error_message="sandbox gone"))
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "compare_table"}).execute(context)
    assert context.metadata["verifications"]["v"] == {
        "results": [{"id": "prompt_error", "score": 0.0, "result": False,
                     "message": "Skipped: prompt had error_type=INFRA_ERROR"}], "score": 0.0}
    assert rt.checks == []


@pytest.mark.asyncio
@pytest.mark.parametrize("conj, scores, expected", [
    # OSWorld's DesktopEnv.evaluate: ``and`` 0 when any func scores 0, else their mean; ``or`` their max.
    ("and", (1, 1), 1.0), ("and", (1, 0), 0.0), ("and", (1, 0.5), 0.75), ("or", (0, 1), 1.0), ("or", (0, 0), 0.0),
    ("or", (0.5, 0.3), 0.5)])
async def test_evaluate_grades_each_func_with_upstream_and_the_judge_and_combines_by_conj(rt, judge, conj, scores,
                                                                                        expected):
    rt.scores = [scores[0]]
    judge.answers = [{"score": scores[1], "message": f"got {scores[1]}"}]
    evaluator = {"funcs": ["compare_table", "agent_judge"], "conj": conj,
                 "result": [{"type": "vm_file", "path": "/home/user/a.xlsx"}, {"type": "vm_file", "path": "/b"}],
                 "expected": [{"type": "cloud_file", "path": "https://h/gold.xlsx"}, None],
                 "options": [{"rules": [1], "model": "copied-by-mistake"}, {}], "rubric": [{"criterion": "x"}]}
    context = _context(None, _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v", evaluator=evaluator,
                                  judge_model="judge-m").execute(context)
    assert rt.checks == [{"func": "compare_table", "result": evaluator["result"][0],
                          "expected": evaluator["expected"][0], "options": {"rules": [1]}}]
    assert rt.access == [(GUEST, HEADERS)]
    assert judge.payloads == [{"func": "agent_judge", "options": {"model": "judge-m"}, "result": evaluator["result"][1],
                               "rubric": [{"criterion": "x"}], "instruction": "Save the sheet"}]
    verification = context.metadata["verifications"]["v"]
    assert verification["score"] == expected
    assert [r["func"] for r in verification["results"]] == ["compare_table", "agent_judge"]
    assert verification["results"][0]["results"] == [
        {"id": "compare_table", "score": float(scores[0]), "result": True, "message": ""}]
    assert [(r["score"], r["result"]) for r in verification["results"]] == [(float(s), s == 1) for s in scores]


@pytest.mark.asyncio
async def test_a_judge_func_with_no_judge_registered_scores_0_saying_so(rt, monkeypatch):
    monkeypatch.setattr(evaluate_module, "_JUDGES", {})
    context = _context(None, _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": ["agent_judge_zip"]}).execute(context)
    [row] = context.metadata["verifications"]["v"]["results"]
    assert row["score"] == 0.0 and "no judge is registered for agent_judge_zip" in row["results"][0]["message"]


def test_a_judge_is_registered_for_a_judge_func_alone_and_a_variants_own_wins(monkeypatch):
    monkeypatch.setattr(evaluate_module, "_JUDGES", {})
    generic, zipped = Judge(), Judge()
    evaluate_module.register_judge("agent_judge", generic)
    evaluate_module.register_judge("agent_judge_zip", zipped)
    assert evaluate_module.judge_for("agent_judge_zip") is zipped
    assert evaluate_module.judge_for("agent_judge_blend") is generic
    with pytest.raises(ValueError, match="isn't an LLM-judge func"):
        evaluate_module.register_judge("compare_table", generic)


@pytest.mark.asyncio
async def test_aggregate_verifiers_reads_each_funcs_score_from_the_rows(rt):
    rt.scores = [1, 0.5]
    context = _context(None, _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": ["compare_table", "file_contains"], "conj": "and"}).execute(context)
    for aggregator, expected in [("weighted_average", 0.75), ("all_pass", 0.0), ("any_pass", 1.0)]:
        await AggregateVerifiersTaskStep(id=f"a-{aggregator}", version=1, verifier_ids=["v"], verifier_id=aggregator,
                                         score_aggregator=aggregator).execute(context)
        assert context.metadata["verifications"][aggregator]["score"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("key, value", [("options", [{}]), ("result", [{"type": "vm_file", "path": "/a"}] * 3),
                                        ("expected", {"type": "cloud_file"})])
async def test_evaluate_refuses_per_func_lists_that_dont_match_its_funcs(rt, key, value):
    """Zipped, a shorter list would grade on fewer checks than the task names."""
    evaluator = {"funcs": ["compare_table", "agent_judge"], key: value}
    with pytest.raises(ValueError, match=f"'{key}' must be a list of 2, one per func"):
        await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                      evaluator=evaluator).execute(_context(None, _prompt()))
    assert rt.checks == []


@pytest.mark.asyncio
async def test_evaluate_one_func_keeps_its_partial_score(rt):
    rt.scores = [0.4]
    context = _context(None, _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "compare_table"}).execute(context)
    assert context.metadata["verifications"]["v"]["score"] == 0.4


@pytest.mark.asyncio
async def test_evaluate_scores_a_failing_func_0_and_grades_the_rest(rt):
    rt.scores = [RuntimeError("getter failed"), 1]
    context = _context(None, _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": ["compare_table", "file_contains"], "conj": "or"}).execute(context)
    verification = context.metadata["verifications"]["v"]
    assert verification["score"] == 1.0 and verification["debug"]["error"] is True
    assert "getter failed" in verification["results"][0]["results"][0]["message"]


@pytest.mark.asyncio
async def test_evaluate_infeasible_is_graded_on_the_agents_answer(rt):
    context = _context(None, _prompt(response="[INFEASIBLE] no such app"))
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": ["infeasible"]}).execute(context)
    assert context.metadata["verifications"]["v"]["score"] == 1.0
    await OSWorldEvaluateTaskStep(id="e2", version=1, env_id="desk", verifier_id="w",
                                  evaluator={"funcs": "compare_table"}).execute(context)
    [[result]] = [r["results"] for r in context.metadata["verifications"]["w"]["results"]]
    assert result["id"] == "infeasible_on_feasible" and context.metadata["verifications"]["w"]["score"] == 0.0
    assert rt.checks == []


@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "Max turns (50) reached without completion.", "agent timed out after 1800s",
    "Reached maximum number of turns (50)", "codex CLI timed out after 600s", "claude CLI timed out after 900s",
    "gemini CLI timed out after 600s", "Letta Code timed out after 600s", "hermes timed out after 600s",
    "opencode timed out after 600s"])
async def test_evaluate_grades_a_run_that_ran_out_of_turns_and_marks_it(rt, message):
    """Each shipped agent's own words for running out of turns or time."""
    context = _context(None, _prompt(error_type="AGENT_ERROR", error_message=message))
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "compare_table"}).execute(context)
    verification = context.metadata["verifications"]["v"]
    assert verification["score"] == 1.0
    assert verification["partial_credit"]["reason"] == "budget_exhausted"
    assert context.metadata["partial_credit"] == {"v": verification["partial_credit"]}


@pytest.mark.asyncio
async def test_evaluate_runs_postconfig_once_and_grades_when_it_fails(rt, caplog):
    rt.setup_error = RuntimeError("window gone " + "TASK-CONTENT " * 100)
    postconfig = [{"type": "activate_window", "parameters": {"window_name": "x"}}]
    context = _context(None, _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "compare_table", "postconfig": postconfig}).execute(context)
    assert rt.setups == [postconfig] and len(rt.checks) == 1
    assert context.metadata["verifications"]["v"]["score"] == 1.0
    assert "postconfig failed, grading anyway" in caplog.text and caplog.text.count("TASK-CONTENT") < 40


def _regrade(context, files):
    context.metadata["user_overrides"] = {"step_params": {"e": {"stage_files": files}}}
    return context


@pytest.mark.asyncio
async def test_evaluate_stages_result_files_before_grading(rt):
    files = [{"url": "https://signed.example/out.xlsx", "path": "/home/user/out.xlsx"}]
    context = _regrade(_context(None, _prompt()), files)
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "compare_table"}).execute(context)
    assert rt.setups == [[{"type": "download", "parameters": {"files": files}}]] and len(rt.checks) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("files, fails", [([{"url": "s3://b/out.xlsx", "path": "/x"}], False),
                                          ([{"url": "https://h/out", "path": "/x"}], True)],
                         ids=["not-http", "download-failed"])
async def test_evaluate_refuses_to_grade_an_unstaged_computer(rt, files, fails):
    if fails:
        rt.setup_error = RuntimeError("404")
    context = _regrade(_context(None, _prompt()), files)
    with pytest.raises((ValueError, RuntimeError)):
        await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                      evaluator={"funcs": "compare_table"}).execute(context)
    assert rt.checks == []


@pytest.mark.parametrize("message", ["codex CLI produced no output for 300s (stalled)", "command timed out after 30s",
                                     "Tool 'click' timed out after 60s", "model returned 500"])
def test_a_stop_that_isnt_the_agents_budget_isnt_read_as_one(message):
    assert budget_exhaustion(_prompt(error_type="AGENT_ERROR", error_message=message)) is None


@pytest.mark.asyncio
async def test_evaluate_gives_the_judge_funcs_the_runs_judge_model_ahead_of_theirs_and_the_steps(rt, judge):
    evaluator = {"funcs": ["agent_judge", "agent_judge_zip", "compare_table"],
                 "options": [{"model": "task-m"}, {}, {}], "rubric": [{"criterion": "x"}]}
    context = _context(None, _prompt())
    context.metadata["user_overrides"] = {"judge_model": "run-m"}
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v", evaluator=evaluator,
                                  judge_model="step-m").execute(context)
    assert [p["options"] for p in judge.payloads] == [{"model": "run-m"}, {"model": "run-m"}]
    assert rt.checks[0]["options"] == {}
    assert context.metadata["models"]["judge"] == {"e": "run-m"}


@pytest.mark.asyncio
async def test_evaluate_records_the_judge_models_the_funcs_are_given(rt, judge):
    """Without a run override, a judge func's own model is the one that grades it, so the summary names it."""
    evaluator = {"funcs": ["agent_judge", "agent_judge_zip", "compare_table"],
                 "options": [{"model": "task-m"}, {}, {}], "rubric": [{"criterion": "x"}]}
    context = _context(None, _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v", evaluator=evaluator,
                                  judge_model="step-m").execute(context)
    assert [p["options"] for p in judge.payloads] == [{"model": "task-m"}, {"model": "step-m"}]
    assert context.metadata["models"]["judge"] == {"e": "step-m, task-m"}


@pytest.mark.asyncio
async def test_evaluate_records_no_judge_model_when_the_judge_picks_it(rt, judge):
    context = _context(None, _prompt())
    await OSWorldEvaluateTaskStep(id="e", version=1, env_id="desk", verifier_id="v",
                                  evaluator={"funcs": "agent_judge", "rubric": [{"criterion": "x"}]}).execute(context)
    assert "model" not in judge.payloads[0]["options"]
    assert context.metadata["models"]["judge"] == {"e": None}


# Each agent's own sentence for running out of turns or time, as its source writes it, and that sentence as it reads at
# a stop: a reworded message fails here instead of quietly turning partial credit into a skip.
_AGENTS = Path(__file__).parents[4] / "src" / "agent_env" / "agents" / "agents"
_BUDGET_STOPS = [
    ("codex/codex_bridge.py", 'f"Reached maximum number of turns ({max_turns})"', "Reached maximum number of turns (50)"),
    ("codex/codex_bridge.py", 'f"codex CLI timed out after {timeout_s:.0f}s"', "codex CLI timed out after 600s"),
    ("claude-code/claude_bridge.py", 'f"claude CLI timed out after {timeout_s}s"', "claude CLI timed out after 900s"),
    ("gemini/gemini_bridge.py", 'f"gemini CLI timed out after {timeout_s}s"', "gemini CLI timed out after 600s"),
    ("hermes/hermes_bridge.py", 'f"hermes timed out after {timeout_s}s"', "hermes timed out after 600s"),
    ("letta-code/letta_code_bridge.py", 'f"Letta Code timed out after {timeout_s}s"', "Letta Code timed out after 600s"),
    ("opencode/opencode_bridge.py", 'f"opencode timed out after {timeout_s}s"', "opencode timed out after 600s"),
]


@pytest.mark.parametrize("bridge, written, at_a_stop", _BUDGET_STOPS)
def test_each_agents_budget_stop_is_read_as_its_source_writes_it(bridge, written, at_a_stop):
    from agent_env.task_step.task_steps.computer.osworld.evaluate import budget_exhaustion
    assert written in (_AGENTS / bridge).read_text(), f"{bridge} no longer stops with {written}"
    assert budget_exhaustion(_prompt(error_type="AGENT_ERROR", error_message=at_a_stop)) == at_a_stop
