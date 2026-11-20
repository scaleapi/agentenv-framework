"""agent-env task lint: each rule with a task that breaks it and one that doesn't, where its findings refuse a run, and
the bundles agent-env ships linting clean."""

import importlib
import json
from typing import ClassVar

import pytest
from click.testing import CliRunner

from agent_env.bundle import BundleError, BundleKind, parse_bundle
from agent_env.bundle.installed import CORE, installed_bundles
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import build_step, resolve_bundle
from agent_env.bundle.run import planned_tasks
from agent_env.cli import cli
from agent_env.eval import Eval
from agent_env.eval.eval import EvalTask
from agent_env.openenv import app
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from agent_env.task.lint import UNPINNED, Finding, Severity, lint, said, unpinned_summary
from agent_env.task_step.task_step import TaskStep
from agent_env.training import trl
from agent_env.training.nemo_gym import __main__ as nemo_cli, prepare
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
# A validator: it grades, under a key of its own, with no verifier_id.
ASSESS = {"id": "assess", "type": "verify_mcp_env_assessment", "env_id": "mcp", "prompt_id": "ask"}


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
                                             "needs paths, a non-empty list of strings, not None")]
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


@pytest.mark.parametrize(("criterion", "problem"), [
    (None, "must be an object, not NoneType"),
    ({"type": ["response_contains"]}, "unknown type ['response_contains']"),
    ({"type": "response_contains", "needles": "abc"}, "response_contains needs non-empty needles, a list of strings"),
    ({"type": "response_contains", "needles": [7]}, "response_contains needs non-empty needles, a list of strings"),
    ({"type": "response_regex_present", "pattern": 7}, "response_regex_present needs a non-empty pattern, a string"),
    ({"type": "response_regex_present", "pattern": "x", "flags": "IGNORECASE"}, "flags must be a list of re flag"),
])
def test_an_agent_prompt_response_criterion_of_the_wrong_type_is_a_config_problem(criterion, problem):
    step = AgentPromptResponseVerifierTaskStep(id="check", version=1, prompt_id="ask", criteria=[criterion])

    (found,) = step.config_problems()
    assert found.startswith(f"criterion #0: {problem}")


@pytest.mark.parametrize(("criterion", "problem"), [
    ({"type": ["probe_file_exists"], "criterion": "x"}, "type must be a string, not ['probe_file_exists']"),
    ({"type": "probe_file_exists", "criterion": "x", "paths": "a.txt"},
     "probe_file_exists needs paths, a non-empty list of strings, not 'a.txt'"),
    ({"type": "probe_file_contains", "criterion": "x", "paths": ["a.txt"], "expected": 7},
     "expected must be a string, not 7"),
    ({"type": "bash_cmd_succeeds", "criterion": "x", "bash_cmd": ["make"]},
     "bash_cmd_succeeds needs a bash_cmd, a non-empty string, not ['make']"),
])
def test_a_verify_sandbox_criterion_of_the_wrong_type_is_a_config_problem_and_no_warning(criterion, problem):
    step = build_step({**TESTS, "criteria": [criterion]})

    assert (step.config_problems(), step.config_warnings()) == ([f"criterion #0: {problem}"], [])


@pytest.mark.parametrize(("timeout", "problems"), [
    ("300", ["timeout_sec must be a number of seconds, not '300'"]),
    (None, ["timeout_sec must be a number of seconds, not None"]),
    (True, ["timeout_sec must be a number of seconds, not True"]),
    (1.5, []),
])
def test_a_unit_test_timeout_is_a_number_of_seconds(timeout, problems):
    tests = RunContainerUnitTestsVerifierTaskStep(id="tests", version=1, sandbox_name="b", container_name="c",
                                                  command="pytest", timeout_sec=timeout)

    assert tests.config_problems() == problems


def test_the_cli_names_each_value_of_the_wrong_type_in_a_task_file_and_exits_1(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([BOX, {**TESTS, "criteria": [None, {"type": "probe_file_exists", "paths": "a.txt"}]},
                                {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask",
                                 "criteria": [{"type": "response_regex_present", "pattern": 7}]}]))

    result = CliRunner().invoke(cli, ["task", "lint", str(path)])

    assert result.exit_code == 1, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert [line.split(": ", 1)[1] for line in result.output.splitlines() if line.startswith("  - ")] == [
        "step 'tests': criterion #0: must be an object, not NoneType",
        "step 'tests': criterion #1: probe_file_exists needs paths, a non-empty list of strings, not 'a.txt'",
        "step 'check': criterion #0: response_regex_present needs a non-empty pattern, a string, not 7"]


def test_a_value_of_the_wrong_type_a_verifier_would_trip_on_is_a_config_problem():
    criteria = [{"type": "probe_file_exists", "criterion": "has it", "paths": ["a.txt"]}, None, "probe"]
    tests = RunContainerUnitTestsVerifierTaskStep(id="tests", version=1, sandbox_name="b", container_name="c",
                                                  command=["pytest"], verifier_id=["tests"])

    assert _found(_task(BOX, {**TESTS, "criteria": criteria})) == [
        ("error", "step-config", "step 'tests': criterion #1: must be an object, not NoneType"),
        ("error", "step-config", "step 'tests': criterion #2: must be an object, not str")]
    assert tests.config_problems() == ["verifier_id must be a string, not list", "command must be a string, not list"]
    assert tests.config_warnings() == []
    assert _found(Task(id="t", version=1, steps=[*_task(BOX).steps, tests])) == [
        ("error", "step-config", "step 'tests': verifier_id must be a string, not list"),
        ("error", "step-config", "step 'tests': command must be a string, not list")]


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


class _Outside(TaskStep):
    """An outside step off the base with no verifier_id, as agentenvhub-sdk's run_lighthouse_audit, which writes its
    entry under a verification_id: lint can't tell what it writes."""

    type: ClassVar[str] = "outside_lint_test"
    entity_refs = ()

    def __init__(self, id):
        super().__init__(id, 1)

    async def execute(self, context):
        return context


def test_a_headline_that_grades_nothing_is_an_error_and_one_that_writes_no_named_grade_a_warning():
    built = _task(BOX, TESTS).steps
    legacy = Task(id="t", version=1, steps=[*built, _Legacy("cua")], headline_verifiers=["cua"])
    outside = Task(id="t", version=1, steps=[*built, _Outside("audit")], headline_verifiers=["audit"])

    assert ("error", "headline-not-a-verifier",
            "headline_verifiers names 'box', a deploy_sandbox step, which grades nothing") in _found(
        _task(BOX, TESTS, headline_verifiers=["box"]))
    for task, (name, kind) in [(legacy, ("cua", "legacy_lint_test")), (outside, ("audit", "outside_lint_test")),
                               (_task(BOX, ASSESS, headline_verifiers=["assess"]),
                                ("assess", "verify_mcp_env_assessment"))]:
        assert ("warning", "headline-without-grade",
                f"headline_verifiers names {name!r}, a {kind} step that writes no named grade, so a run's result will "
                "read it as ungraded") in _found(task)
    assert _found(_task(BOX, TESTS, headline_verifiers=["tests"])) == []


def test_an_aggregate_source_no_verifier_writes_is_an_error_unless_a_step_lint_cant_see_into_may_write_it():
    on_assessment = {**FINAL, "verifier_ids": ["tests", "mcp_tool_correctness"]}

    assert _found(_task(BOX, TESTS, FINAL, headline_verifiers=["final"])) == [(
        "error", "aggregate-source", "step 'final': verifier_ids names ['more'], which no verifier in this task writes "
                                     "(it has ['final', 'tests'])")]
    assert ("warning", "aggregate-source",
            "step 'final': verifier_ids names ['mcp_tool_correctness'], which no verifier in this task names as its "
            "verifier_id; unless ['assess'] write it, under keys lint can't see, the aggregate fails when it grades"
            ) in _found(_task(BOX, ASSESS, TESTS, on_assessment, headline_verifiers=["final"]))
    outside = Task(id="t", version=1, steps=[*_task(BOX, TESTS, FINAL).steps, _Outside("audit")],
                   headline_verifiers=["final"])
    assert [f.severity for f in lint(outside) if f.rule == "aggregate-source"] == [Severity.WARNING]
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
    assert _found(_task(BOX, ASSESS)) == []
    assert _found(Task(id="t", version=1, steps=[*_task(BOX).steps, _Outside("audit")])) == []
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
    assert sorted(f"{task}: {finding}" for task, finding in plan.lint_findings) == sorted(expected)


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

    _write(bundle_dir, "bad", [BOX, {**TESTS, "id": ["tests"]}])
    unparsed = CliRunner().invoke(cli, ["task", "lint", str(bundle_dir)])
    assert (unparsed.exit_code, unparsed.output.splitlines()[-1]) == (
        1, "  - tasks/bad.json: step #1's id is a JSON array; it must be a string")


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
    assert ("Task lint found 1 error(s), so no run started (--skip-lint runs anyway):\n  - t v1: headline_verifiers "
            "names 'box', a deploy_sandbox step, which grades nothing") in result.output


def test_eval_run_with_skip_lint_says_the_error_and_goes_on_to_run(local_stores, monkeypatch):
    task = _task(BOX, TESTS, headline_verifiers=["box"])
    listed = Eval(id="e", version=1, tasks=[EvalTask(task_id="t", task_version=1)])
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: listed))
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: task))
    monkeypatch.setattr(importlib.import_module("agent_env.cli.eval.run"), "print_banner",
                        lambda: (_ for _ in ()).throw(SystemExit(7)))

    result = CliRunner().invoke(cli, ["eval", "run", "--id", "e", "--skip-lint"])

    assert result.exit_code == 7, result.output
    assert "  t v1: error: headline_verifiers names 'box', a deploy_sandbox step, which grades nothing" in result.output


def test_eval_run_with_skip_lint_runs_past_a_steps_config_problem_as_it_did_before_it_linted(local_stores,
                                                                                             monkeypatch):
    task = _task(BOX, {**TESTS, "criteria": [{"type": "probe_file_exists", "criterion": "x"}]})
    listed = Eval(id="e", version=1, tasks=[EvalTask(task_id="t", task_version=1)])
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: listed))
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: task))
    monkeypatch.setattr(importlib.import_module("agent_env.cli.eval.run"), "print_banner",
                        lambda: (_ for _ in ()).throw(SystemExit(7)))

    refused = CliRunner().invoke(cli, ["eval", "run", "--id", "e"])
    skipped = CliRunner().invoke(cli, ["eval", "run", "--id", "e", "--skip-lint"])

    assert refused.exit_code == 1, refused.output
    assert "so no run started (--skip-lint runs anyway):\n  - t v1: step 'tests': criterion #0: probe_file_exists" in (
        refused.output)
    assert skipped.exit_code == 7, skipped.output
    assert "  t v1: error: step 'tests': criterion #0: probe_file_exists needs paths" in skipped.output


class _Planned(Exception):
    """What a stand-in for a planning call raises once it has recorded the arguments it was given."""


def _recording(calls):
    def record(*args, **kwargs):
        calls.append(kwargs)
        raise _Planned()
    return record


def test_selecting_a_bundles_tasks_lints_none_of_them_for_its_caller_checks_those_it_keeps(bundle_dir):
    _write(bundle_dir, "t", {"headline_verifiers": ["box"], "steps": [BOX, TESTS]})

    with namespace_routing():
        assert [entry.entry.name for entry in planned_tasks(bundle_dir)] == ["t"]
        with pytest.raises(BundleError, match="which grades nothing"):
            plan_bundle(resolve_bundle(parse_bundle(bundle_dir)), tasks=["t"])


@pytest.mark.parametrize("serves", ["build_app", "trl", "nemo_gym"])
def test_each_way_a_trainer_serves_a_bundle_passes_skip_lint_to_its_plan(serves, bundle_dir, tmp_path, monkeypatch):
    calls = []
    if serves == "build_app":
        monkeypatch.setattr(app, "prepare_bundle", _recording(calls))
        serve = lambda skip: app.build_app(bundle_dir, skip_lint=skip)  # noqa: E731
    elif serves == "trl":
        monkeypatch.setattr(trl, "build_app", _recording(calls))
        serve = lambda skip: trl.AgentEnv.local(bundle_dir, skip_lint=skip)  # noqa: E731
    else:
        monkeypatch.setattr(prepare, "build_app", _recording(calls))
        serve = lambda skip: prepare.write_server(bundle_dir, tmp_path / "gym", skip_lint=skip)  # noqa: E731

    for skip in (False, True):
        with pytest.raises(_Planned):
            serve(skip)

    assert [call["skip_lint"] for call in calls] == [False, True]


def test_nemo_gyms_cli_takes_skip_lint_and_its_server_config_keeps_it(bundle_dir, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(nemo_cli, "write_server", lambda *args, **kwargs: calls.append(kwargs) or tmp_path)

    result = CliRunner().invoke(nemo_cli.main, [str(bundle_dir), str(tmp_path / "gym"), "--skip-lint"])

    assert result.exit_code == 0, result.output
    assert calls[0]["skip_lint"] is True


def test_the_cli_takes_a_path_or_an_id_but_not_both(bundle_dir):
    assert CliRunner().invoke(cli, ["task", "lint"]).exit_code == 2
    assert CliRunner().invoke(cli, ["task", "lint", str(bundle_dir), "--id", "t"]).exit_code == 2


_HEADLESS = "headline_verifiers names 'box', a deploy_sandbox step, which grades nothing"


def test_a_run_of_a_bundle_with_a_lint_error_is_refused_unless_it_skips_lint_but_not_for_a_steps_config(bundle_dir):
    _write(bundle_dir, "t", {"headline_verifiers": ["box"], "steps": [BOX, TESTS]})
    _write(bundle_dir, "u", [BOX, {**TESTS, "criteria": [{"type": "probe_file_exists", "criterion": "x"}]}])

    refused = CliRunner().invoke(cli, ["run", str(bundle_dir), "--task", "t", "--dry-run"])
    skipped = CliRunner().invoke(cli, ["run", str(bundle_dir), "--task", "t", "--dry-run", "--skip-lint"])
    config = CliRunner().invoke(cli, ["run", str(bundle_dir), "--task", "u", "--dry-run", "--skip-lint"])

    assert refused.exit_code != 0
    assert f"tasks/t.json: {_HEADLESS}" in refused.output
    assert "task lint found 1 of these; --skip-lint runs the tasks anyway" in refused.output
    assert skipped.exit_code == 0, skipped.output
    assert f"tasks/t.json: error: {_HEADLESS}" in skipped.output
    assert "Would run:\n  tasks/t.json v1\n" in skipped.output
    assert config.exit_code != 0
    assert "tasks/u.json: step 'tests': criterion #0: probe_file_exists needs paths" in config.output
    assert "--skip-lint" not in config.output


def test_bundle_register_with_skip_lint_registers_a_task_lint_refuses_saying_why_once(bundle_dir):
    _write(bundle_dir, "t", {"headline_verifiers": ["box"], "steps": [BOX, TESTS]})

    refused = CliRunner().invoke(cli, ["bundle", "register", str(bundle_dir), "--yes"])
    skipped = CliRunner().invoke(cli, ["bundle", "register", str(bundle_dir), "--yes", "--skip-lint"])

    assert refused.exit_code != 0
    assert skipped.exit_code == 0, skipped.output
    assert skipped.output.count(f"tasks/t.json: error: {_HEADLESS}") == 1
    assert Task.get("triage-t").headline_verifiers == ["box"]


def test_unpinned_references_are_said_as_one_line_counting_tasks_and_kinds():
    def unpinned(kind):
        return Finding(Severity.WARNING, UNPINNED, f"names {kind} 'x' with no version", "s", kind)

    many = [(f"t{n} v1", unpinned("agent")) for n in range(3)] + [("t0 v1", unpinned("artifact")),
                                                                  ("t1 v1", unpinned("artifact"))]

    assert unpinned_summary(many) == (
        "warning: 3 tasks name 5 store entities with no version, so each run takes the latest (agent 3, artifact 2); "
        "agent-env task lint --id TASK lists a task's")
    assert unpinned_summary(many[:1]).startswith("warning: 1 task names 1 store entity with no version")
    assert unpinned_summary([("t v1", Finding(Severity.WARNING, "no-verifier", "grades nothing"))]) is None


def test_eval_run_says_its_tasks_unpinned_references_in_one_line_and_its_other_warnings_by_kind(local_stores,
                                                                                               monkeypatch):
    tasks = {f"t{n}": Task.from_dict({"id": f"t{n}", "version": 1, "steps": [BOX, {**NOTES, "id": f"notes{n}"}]})
             for n in range(4)}
    listed = Eval(id="e", version=1, tasks=[EvalTask(task_id=name, task_version=1) for name in tasks])
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: listed))
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: tasks[id]))
    monkeypatch.setattr(importlib.import_module("agent_env.cli.eval.run"), "print_banner",
                        lambda: (_ for _ in ()).throw(SystemExit(7)))

    result = CliRunner().invoke(cli, ["eval", "run", "--id", "e"])

    assert result.exit_code == 7, result.output
    assert ("  warning: 4 tasks name 4 store entities with no version, so each run takes the latest (artifact 4); "
            "agent-env task lint --id TASK lists a task's") in result.output
    assert "names artifact 'shared-notes'" not in result.output
    assert ("  warning: no step grades this task, so a run has no score (4 tasks: t0 v1, t1 v1, t2 v1 and 1 more; "
            "agent-env task lint lists each)") in result.output


def test_a_run_says_each_kind_of_finding_once_with_how_many_tasks_have_it_errors_first():
    doubt, other = (Finding(Severity.WARNING, "no-verifier", text) for text in ("grades nothing", "unpinned"))
    wrong = Finding(Severity.ERROR, "step-config", "needs paths", "tests")
    findings = [*((f"tasks/{n}.json", doubt) for n in "abcde"), ("tasks/a.json", other), ("tasks/c.json", wrong),
                ("tasks/d.json", wrong)]

    assert said(findings) == [
        "error: step 'tests': needs paths (2 tasks: tasks/c.json, tasks/d.json; agent-env task lint lists each)",
        "warning: grades nothing (5 tasks: tasks/a.json, tasks/b.json, tasks/c.json and 2 more; agent-env task lint "
        "lists each)",
        "tasks/a.json: warning: unpinned"]
