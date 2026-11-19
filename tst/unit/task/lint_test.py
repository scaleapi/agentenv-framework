"""agent-env task lint: each rule with a task that breaks it and one that doesn't, where its findings refuse a run, and
the bundles agent-env ships linting clean."""

import json
from typing import ClassVar

import pytest
from click.testing import CliRunner

from agent_env.bundle import BundleKind, parse_bundle
from agent_env.bundle.installed import CORE, installed_bundles
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.cli import cli
from agent_env.eval import Eval
from agent_env.eval.eval import EvalTask
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from agent_env.task.lint import Severity, lint
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import (
    AgentPromptResponseVerifierTaskStep,
)
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)

BOX = {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm"}
TESTS = {"id": "tests", "type": "verify_sandbox", "sandbox_name": "box", "verifier_id": "tests",
         "criteria": [{"type": "probe_file_exists", "criterion": "has it", "paths": ["a.txt"]}]}
MORE = {**TESTS, "id": "more", "verifier_id": "more"}
FINAL = {"id": "final", "type": "aggregate_verifiers", "verifier_id": "final", "verifier_ids": ["tests", "more"]}
NOTES = {"id": "notes", "type": "load_artifact", "sandbox_name": "box", "artifact_id": "shared-notes"}


def _task(*steps, **fields) -> Task:
    return Task.from_dict({"id": "t", "version": 1, "steps": [dict(step) for step in steps], **fields})


def _found(task, *args, **kwargs) -> list[tuple[str, str, str]]:
    return [(f.severity.value, f.rule, f.text) for f in lint(task, *args, **kwargs)]


def test_a_field_its_step_doesnt_read_is_an_error_naming_the_nearest_one_it_does():
    steps = [BOX, {**TESTS, "min_passing_scor": 0.8}]

    assert _found(_task(BOX, TESTS), steps) == [(
        "error", "unknown-field", "step 'tests': verify_sandbox doesn't read 'min_passing_scor' (did you mean "
                                  "'min_passing_score'?), so it would be ignored")]
    assert _found(_task(BOX, TESTS), [BOX, TESTS]) == []


def test_a_verifier_config_that_fails_when_it_grades_is_an_error_and_one_a_run_skips_is_a_warning():
    no_paths = {**TESTS, "criteria": [{"type": "probe_file_exists", "criterion": "has it"}]}
    unknown = {**TESTS, "criteria": [{"type": "probe_file_smells", "criterion": "has it"}]}

    assert _found(_task(BOX, no_paths)) == [("error", "step-config", "step 'tests': criterion #0: probe_file_exists "
                                                                       "needs paths")]
    assert _found(_task(BOX, unknown)) == [("warning", "step-config", "step 'tests': criterion #0: type "
                                            "'probe_file_smells' isn't one verify_sandbox checks, so a run skips it; it "
                                            "checks ['bash_cmd_succeeds', 'probe_dir_exists', 'probe_file_contains', "
                                            "'probe_file_exists']")]
    assert _found(_task(BOX, no_paths), config=False) == []


@pytest.mark.parametrize(("criterion", "problem"), [
    ({"type": "response_shouts"}, "unknown type 'response_shouts'"),
    ({"type": "response_contains"}, "response_contains needs non-empty needles"),
    ({"type": "response_regex_present"}, "response_regex_present needs a non-empty pattern"),
    ({"type": "response_regex_present", "pattern": "(unclosed"}, "pattern '(unclosed' doesn't compile"),
    ({"type": "response_regex_present", "pattern": "x", "flags": ["SHOUT"]}, "flags ['SHOUT'] aren't re flags"),
])
def test_an_agent_prompt_response_criterion_a_run_would_refuse_is_a_config_problem(criterion, problem):
    step = AgentPromptResponseVerifierTaskStep(id="check", version=1, prompt_id="ask", criteria=[criterion])
    fine = AgentPromptResponseVerifierTaskStep(id="check", version=1, prompt_id="ask", criteria=[
        {"type": "response_contains", "needles": ["42"]},
        {"type": "response_regex_present", "pattern": "4[0-9]", "flags": ["IGNORECASE", "IGNORECASE"]}])

    (found,) = step.config_problems()
    assert found.startswith(f"criterion #0: {problem}")
    assert fine.config_problems() == []


def test_a_unit_test_run_that_tests_nothing_is_a_warning_and_a_negative_timeout_an_error():
    def tests(**kw):
        return RunContainerUnitTestsVerifierTaskStep(id="tests", version=1, sandbox_name="b", container_name="c",
                                                     **{"command": "pytest", **kw})

    assert tests(command="  ").config_warnings() == [
        "command is empty, so it exits 0 and the tests pass without running anything"]
    assert tests(command="", reward_path="/r").config_warnings() == []
    assert tests(timeout_sec=0).config_warnings() == [
        "timeout_sec 0 means no time limit, so a hung test runs until the sandbox's own TTL"]
    assert tests(timeout_sec=-1).config_problems() == ["timeout_sec -1 is negative; give the tests at least 1 s"]
    assert (tests().config_problems(), tests().config_warnings()) == ([], [])


class _Legacy(TaskStep):
    """A verifier that still writes its legacy entry itself, as agentenvhub-sdk's do until they move onto the base."""

    type: ClassVar[str] = "legacy_lint_test"
    entity_refs = ()

    def __init__(self, id):
        super().__init__(id, 1)
        self.verifier_id = id

    async def execute(self, context):
        return context


def test_a_headline_that_grades_nothing_is_an_error_and_one_that_writes_no_named_grade_a_warning():
    built = _task(BOX, TESTS).steps
    legacy = Task(id="t", version=1, steps=[*built, _Legacy("cua")], headline_verifiers=["cua"])

    assert ("error", "headline-not-a-verifier",
            "headline_verifiers names 'box', a deploy_sandbox step, which grades nothing") in _found(
        _task(BOX, TESTS, headline_verifiers=["box"]))
    assert ("warning", "headline-without-grade",
            "headline_verifiers names 'cua', a legacy_lint_test step that writes no named grade yet, so a run's result "
            "will read it as ungraded") in _found(legacy)
    assert _found(_task(BOX, TESTS, headline_verifiers=["tests"])) == []


def test_an_aggregate_source_no_verifier_writes_is_an_error():
    assert _found(_task(BOX, TESTS, FINAL, headline_verifiers=["final"])) == [(
        "error", "aggregate-source", "step 'final': verifier_ids names ['more'], which no verifier in this task writes "
                                     "(it has ['final', 'tests'])")]
    assert _found(_task(BOX, TESTS, MORE, FINAL, headline_verifiers=["final"])) == []


def test_a_store_entity_read_with_no_version_warns_and_a_pinned_or_local_one_doesnt():
    pinned = {**NOTES, "artifact_id": None, "artifacts": [{"id": "shared-notes", "version": 2}]}
    local = {**NOTES, "artifact_id": "@local/~/triage/notes"}

    assert _found(_task(BOX, NOTES, TESTS)) == [(
        "warning", "unpinned-reference", "step 'notes': artifacts[0].id names artifact 'shared-notes' with no version, "
                                         "so each run takes its latest; set version to pin it")]
    assert _found(_task(BOX, NOTES, TESTS), unpinned=False) == []
    assert _found(_task(BOX, pinned, TESTS)) == _found(_task(BOX, local, TESTS)) == []


def test_a_task_that_grades_nothing_warns_and_several_grades_without_a_headline_name_the_covering_aggregate():
    assert _found(_task(BOX)) == [("warning", "no-verifier", "no step grades this task, so a run has no score")]
    assert _found(_task(BOX, TESTS, MORE, FINAL)) == [(
        "warning", "no-headlines", "3 verifiers and no headline_verifiers, so every grade is top-line and anything that "
                                   "needs one number must name one; 'final' aggregates the others, so name it in "
                                   "headline_verifiers")]
    assert _found(_task(BOX, TESTS, MORE))[0][2].endswith("name the one that is its reward in headline_verifiers")
    assert _found(_task(BOX, TESTS)) == []


def test_errors_come_before_warnings():
    found = lint(_task(BOX, NOTES, TESTS, FINAL), [BOX, NOTES, TESTS, {**FINAL, "sources": ["tests"]}])

    assert [f.severity for f in found] == sorted((f.severity for f in found), key=lambda s: s is not Severity.ERROR)
    assert {f.rule for f in found} == {"unknown-field", "aggregate-source", "unpinned-reference", "no-headlines"}


def _shipped():
    """agent-env's own bundles that hold tasks; the rest (drivers, computers, browsers) have none to lint."""
    return {bundle.name: bundle for bundle in installed_bundles()
            if bundle.dist == CORE and any((bundle.root / "tasks").glob("*.json"))}


# The shipped examples that show env triggers and no grading, which lint rightly says grade nothing.
_UNGRADED = {"advance-to-next-event", "next-env-event"}


@pytest.mark.usefixtures("local_stores")
@pytest.mark.parametrize("name", sorted(_shipped()))
def test_a_shipped_bundles_tasks_lint_clean(name):
    bundle = _shipped()[name]
    with namespace_routing():
        parsed = parse_bundle(bundle.root, id_root=bundle.id_root)
        tasks = [entry.name for entry in parsed.entries if entry.kind is BundleKind.TASK]
        plan = plan_bundle(resolve_bundle(parsed), tasks=tasks, installed=installed_bundles)

    expected = [f"tasks/{task}.json: warning: no step grades this task, so a run has no score" for task in tasks
                ] if name in _UNGRADED else []
    assert sorted(plan.lint_warnings) == sorted(expected)


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "triage"
    (root / "tasks").mkdir(parents=True)
    return root


def _write(root, name, body):
    (root / "tasks" / f"{name}.json").write_text(json.dumps(body))


def test_the_cli_lints_a_bundle_folder_refusing_its_errors_and_printing_its_warnings(bundle_dir):
    _write(bundle_dir, "ok", [BOX, TESTS, MORE])
    warned = CliRunner().invoke(cli, ["task", "lint", str(bundle_dir)])
    _write(bundle_dir, "bad", [BOX, {**TESTS, "min_passing_scor": 0.8}])
    refused = CliRunner().invoke(cli, ["task", "lint", str(bundle_dir)])

    assert warned.exit_code == 0, warned.output
    assert "tasks/ok.json: warning: 2 verifiers and no headline_verifiers" in warned.output
    assert warned.output.endswith("task lint: 1 warning\n")
    assert refused.exit_code == 1
    assert ("tasks/bad.json: step 'tests': verify_sandbox doesn't read 'min_passing_scor' (did you mean "
            "'min_passing_score'?)") in refused.output


def test_the_cli_lints_one_task_file_of_a_bundle_or_one_on_its_own(bundle_dir, tmp_path):
    _write(bundle_dir, "ok", [BOX, TESTS])
    alone = tmp_path / "alone.json"
    alone.write_text(json.dumps({"headline_verifiers": ["tests"], "steps": [BOX, NOTES, TESTS]}))

    in_bundle = CliRunner().invoke(cli, ["task", "lint", str(bundle_dir / "tasks/ok.json")])
    on_its_own = CliRunner().invoke(cli, ["task", "lint", str(alone)])

    assert (in_bundle.exit_code, in_bundle.output) == (0, "task lint: no problems\n")
    assert on_its_own.exit_code == 0, on_its_own.output
    assert f"{alone}: warning: step 'notes': artifacts[0].id names artifact 'shared-notes'" in on_its_own.output


def test_the_cli_lints_a_stored_task(local_stores):
    task = _task(BOX, TESTS, headline_verifiers=["box"])
    stored = Task.put(id="t", steps=task.steps, headline_verifiers=["box"])

    result = CliRunner().invoke(cli, ["task", "lint", "--id", "t"])

    assert result.exit_code == 1
    assert f"t v{stored.version}: headline_verifiers names 'box', a deploy_sandbox step, which grades nothing" in (
        result.output)


def test_task_validate_fails_on_a_lint_error_and_prints_a_warning(local_stores):
    Task.put(id="graded", steps=_task(BOX, NOTES, TESTS).steps)
    Task.put(id="headless", steps=_task(BOX, TESTS).steps, headline_verifiers=["box"])

    warned = CliRunner().invoke(cli, ["task", "validate", "--id", "graded"])
    refused = CliRunner().invoke(cli, ["task", "validate", "--id", "headless"])

    assert warned.exit_code == 0, warned.output
    assert "graded v1: warning: step 'notes': artifacts[0].id names artifact 'shared-notes'" in warned.output
    assert refused.exit_code == 1
    assert "  - headline_verifiers names 'box', a deploy_sandbox step, which grades nothing" in refused.output


def test_eval_run_refuses_an_eval_whose_task_has_a_lint_error_before_any_run(local_stores, monkeypatch):
    task = _task(BOX, TESTS, headline_verifiers=["box"])
    listed = Eval(id="e", version=1, tasks=[EvalTask(task_id="t", task_version=1)])
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: listed))
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: task))
    monkeypatch.setattr(Task, "run", lambda *a, **k: pytest.fail("a run started"))

    result = CliRunner().invoke(cli, ["eval", "run", "--id", "e"])

    assert result.exit_code == 1, result.output
    assert ("Task lint found 1 error(s), so no run started:\n  - t v1: headline_verifiers names 'box', a deploy_sandbox "
            "step, which grades nothing") in result.output


def test_the_cli_takes_a_path_or_an_id_but_not_both(bundle_dir):
    assert CliRunner().invoke(cli, ["task", "lint"]).exit_code == 2
    assert CliRunner().invoke(cli, ["task", "lint", str(bundle_dir), "--id", "t"]).exit_code == 2


def test_a_run_of_a_bundle_with_a_lint_error_is_refused(bundle_dir):
    _write(bundle_dir, "t", [BOX, {**TESTS, "criteria": [{"type": "probe_file_exists", "criterion": "x"}]}])

    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--dry-run"])

    assert result.exit_code != 0
    assert "tasks/t.json: step 'tests': criterion #0: probe_file_exists needs paths" in result.output
