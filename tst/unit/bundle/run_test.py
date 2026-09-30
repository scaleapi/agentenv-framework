"""Running a bundle: which tasks run and at which versions, what each run keeps, what is refused before
any write, and what ``agent-env run`` prints and exits with."""

import asyncio
import json
import logging
import re

import pytest
from click.testing import CliRunner

import agent_env.bundle.run as run_module
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError, Outcome, TaskRun, dry_run_bundle, parse_bundle, run_bundle
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.cli import cli
from agent_env.config.runtime import Config
from agent_env.store import Filter
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep
from tst.unit.bundle._support import layout, local_store, plan_of

ROOT = "@local/~/triage"
DRY_RUN = "Dry run: nothing is written or run.\n"


class _Scored(TaskStep):
    """Records ``score`` for its verifier after a moment, counting how many runs overlap."""

    type = "scored_run_test"
    entity_refs = ()
    active = 0
    peak = 0
    seen: list[TaskStepContext] = []

    def __init__(self, score=1.0, verifier_id=None, **base):
        super().__init__(**base)
        self.score, self.verifier_id = score, verifier_id

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), score=data.get("score"), verifier_id=data.get("verifier_id"))

    def to_dict(self):
        return {**super().to_dict(), "score": self.score, "verifier_id": self.verifier_id}

    async def execute(self, context):
        _Scored.seen.append(context)
        _Scored.active += 1
        _Scored.peak = max(_Scored.peak, _Scored.active)
        try:
            await asyncio.sleep(0.05)
        finally:
            _Scored.active -= 1
        if self.verifier_id is not None:
            context.metadata.setdefault("verifications", {})[self.verifier_id] = {"score": self.score}
        return context


class _Failing(_Scored):
    type = "failing_run_test"

    async def execute(self, context):
        raise RuntimeError("boom")


class _Cancelled(_Scored):
    type = "cancelled_run_test"

    async def execute(self, context):
        raise asyncio.CancelledError


def _scored(score):
    return json.dumps([{"id": "check", "type": "scored_run_test", "verifier_id": "v", "score": score}])


LAYOUT = {
    "tasks/a.json": _scored(1.0),
    "tasks/b.json": _scored(0.5),
    "tasks/c.json": json.dumps([{"id": "boom", "type": "failing_run_test"}]),
    "tasks/unnamed.json": _scored(1.0),
    "evals/full.toml": 'tasks = ["a", "c"]\n',
    "evals/smoke.toml": 'tasks = ["a", "b"]\n',
}


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), **{cls.type: cls for cls in (_Scored, _Failing, _Cancelled)}}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    monkeypatch.setattr(_Scored, "seen", [])
    monkeypatch.setattr(_Scored, "peak", 0)


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores):
    monkeypatch.setenv("HOME", str(tmp_path))
    return layout(tmp_path / "triage", LAYOUT)


def _names(runs):
    return [run.entry.name for run in runs]


def test_every_eval_runs_its_tasks_once_and_the_tasks_no_eval_names_are_skipped(bundle_dir):
    result = run_bundle(bundle_dir)

    assert _names(result.runs) == ["a", "b", "c"]
    assert {eval_run.entry.name: _names(eval_run.runs) for eval_run in result.evals} == {
        "full": ["a", "c"], "smoke": ["a", "b"]}
    assert result.evals[0].runs[0] is result.evals[1].runs[0]
    assert len(_Scored.seen) == 2
    assert [result.path(entry) for entry in result.skipped] == ["tasks/unnamed.json"]


def test_each_run_keeps_its_outcome_scores_and_context_and_a_failure_doesnt_stop_the_others(bundle_dir):
    result = run_bundle(bundle_dir)
    a, b, c = result.runs

    assert [run.outcome for run in result.runs] == [Outcome.PASSED, Outcome.BELOW_ONE, Outcome.FAILED]
    assert result.failed
    assert (a.scores, b.scores, c.scores) == ({"check": 1.0}, {"check": 0.5}, {})
    assert isinstance(c.error, RuntimeError)
    assert [failure["step_id"] for failure in c.failed_steps] == ["boom"]
    assert c.instance_id.startswith(f"{ROOT}/c-")
    with namespace_routing():
        assert Task.get_instance(c.instance_id).status == "failed"
        assert Task.get_instance(a.instance_id).status == "completed"


def test_selected_tasks_and_evals_run_their_tasks_once_and_skip_nothing(bundle_dir):
    result = run_bundle(bundle_dir, tasks=["unnamed", "a"], evals=["smoke"])

    assert _names(result.runs) == ["a", "b", "unnamed"]
    assert _names(result.evals[0].runs) == ["a", "b"]
    assert result.skipped == ()
    assert not result.failed


def test_a_bundle_without_evals_runs_every_task(bundle_dir):
    for name in ("full", "smoke"):
        (bundle_dir / f"evals/{name}.toml").unlink()

    result = run_bundle(bundle_dir)

    assert _names(result.runs) == ["a", "b", "c", "unnamed"]
    assert (result.evals, result.skipped) == ((), ())


def test_each_task_runs_at_the_version_materializing_left(bundle_dir, monkeypatch):
    run_bundle(bundle_dir, evals=["smoke"])
    (bundle_dir / "tasks/a.json").write_text(_scored(0.0))
    write = run_module.materialize

    def and_then_another_version(plan, **hooks):
        done = write(plan, **hooks)
        Task.put(id=f"{ROOT}/a", steps=[_Failing(id="late", version=None)])
        return done

    monkeypatch.setattr(run_module, "materialize", and_then_another_version)
    result = run_bundle(bundle_dir, evals=["smoke"])

    assert [(run.entry.name, run.task.version, run.outcome) for run in result.runs] == [
        ("a", 2, Outcome.BELOW_ONE), ("b", 1, Outcome.BELOW_ONE)]
    assert result.evals[0].version == 1  # it names its tasks without a version, so it is reused


def test_at_most_four_tasks_run_at_once(bundle_dir):
    for index in range(7):
        (bundle_dir / f"tasks/t{index}.json").write_text(_scored(1.0))

    run_bundle(bundle_dir, tasks=[f"t{index}" for index in range(7)])

    assert _Scored.peak == 4


def test_the_model_and_sandbox_reach_each_run_and_one_invocation_shares_a_run_group(bundle_dir):
    run_bundle(bundle_dir, tasks=["a", "b"], model="m", sandbox="local")
    first = list(_Scored.seen)
    run_bundle(bundle_dir, tasks=["a"])

    assert [context.agent_model for context in first] == ["m", "m"]
    assert [context.metadata["user_overrides"] for context in first] == [
        {"env_sandbox": "local", "agent_sandbox": "local", "sandbox": "local"}] * 2
    assert [context.metadata["task_id"] for context in first] == [f"{ROOT}/a", f"{ROOT}/b"]
    assert first[0].metadata["run_group_id"] == first[1].metadata["run_group_id"]
    assert _Scored.seen[2].metadata["run_group_id"] != first[0].metadata["run_group_id"]
    assert "user_overrides" not in _Scored.seen[2].metadata


def test_an_eval_naming_a_store_task_is_refused_before_anything_is_written(bundle_dir):
    with namespace_routing():
        Task.put(id="shared-task", steps=[_Scored(id="check", version=None)])
    layout(bundle_dir, {"evals/full.toml": 'tasks = ["a", { task = "shared-task", version = 1 }]\n',
                        "evals/other.toml": 'tasks = ["shared-task"]\n'})

    with pytest.raises(BundleError) as caught:
        run_bundle(bundle_dir)

    refused = "is a store task, and a bundle's evals run only the bundle's own tasks for now; run it on its own with"
    assert caught.value.problems == (
        f"evals/full.toml: tasks[1]: 'shared-task' {refused} agent-env task run --id shared-task --version 1",
        f"evals/other.toml: tasks[0]: 'shared-task' {refused} agent-env task run --id shared-task",
    )
    assert not local_store().path.exists()
    assert _Scored.seen == []


def test_an_eval_that_isnt_selected_isnt_refused(bundle_dir):
    with namespace_routing():
        Task.put(id="shared-task", steps=[_Scored(id="check", version=None)])
    layout(bundle_dir, {"evals/other.toml": 'tasks = ["shared-task"]\n'})

    assert _names(run_bundle(bundle_dir, evals=["smoke"]).runs) == ["a", "b"]


def test_progress_is_reported_and_a_progress_callback_that_raises_stops_without_failing_a_run(bundle_dir):
    lines = []
    run_bundle(bundle_dir, tasks=["a"], on_progress=lines.append)

    assert [re.sub(r"\d+\.\ds", "<t>", line) for line in lines] == [
        "tasks/a.json: v1 (new)",
        "[tasks/a.json] step 1/1 check (scored_run_test)",
        "[tasks/a.json] step 1/1 check done in <t>",
        "[tasks/a.json] passed in <t>",
    ]

    calls = []

    def broken(line):
        calls.append(line)
        raise BrokenPipeError

    assert run_bundle(bundle_dir, tasks=["a"], on_progress=broken).runs[0].outcome is Outcome.PASSED
    assert calls == ["tasks/a.json: v1, unchanged"]


def test_a_cancelled_run_raises_rather_than_being_kept_as_a_failure(bundle_dir):
    (bundle_dir / "tasks/c.json").write_text(json.dumps([{"id": "stop", "type": "cancelled_run_test"}]))

    with pytest.raises(asyncio.CancelledError):
        run_bundle(bundle_dir, tasks=["a", "c"])


@pytest.mark.parametrize("metadata, outcome", [
    ({}, Outcome.UNSCORED),
    ({"verifications": {"v": {"score": 1}}}, Outcome.PASSED),
    ({"verifications": {"v": {"score": 0.9}}}, Outcome.BELOW_ONE),
    ({"verifications": {"v": {"score": "n/a"}}}, Outcome.UNSCORED),
    ({"verifications": {"v": {"score": 1}, "v-trajectory-mistakes": {"score": 0.4}}}, Outcome.BELOW_ONE),
    ({"verifications": {"v": {"score": 1}}, "failed_steps": [{"step_id": "check"}]}, Outcome.FAILED),
    ({"verifications": {"v": {"score": 1}}, "failed_steps": [{"step_id": "check", "retried": True}]}, Outcome.PASSED),
], ids=["unscored", "passed", "below-one", "not-a-number", "another-score-below-one", "tolerant-failure",
        "retried-failure"])
def test_an_outcome_is_the_worst_of_its_failures_and_scores(bundle_dir, metadata, outcome):
    task = Task(id="t", version=1, steps=[_Scored(id="check", version=None, verifier_id="v")])
    entry = plan_of(bundle_dir).bundle.entries[0].entry

    assert TaskRun(entry, task, TaskStepContext(metadata=metadata), None, 0.0).outcome is outcome


@pytest.fixture
def quiet_logs():
    """pytest's live logging swaps its own stdout back in to print a record, so CliRunner loses whatever is
    echoed after the first one."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def test_a_score_is_labelled_by_its_step_or_else_by_its_key(bundle_dir):
    task = Task(id="t", version=1, steps=[_Scored(id="check", version=None, verifier_id="v")])
    entry = plan_of(bundle_dir).bundle.entries[0].entry
    context = TaskStepContext(metadata={"verifications": {"v": {"score": 1}, "v-trajectory-mistakes": {"score": 0.4}}})

    assert TaskRun(entry, task, context, None, 0.0).scores == {"check": 1.0, "v-trajectory-mistakes": 0.4}


def test_a_call_from_a_running_event_loop_is_refused_before_anything_is_written(bundle_dir):
    async def from_async_code():
        return run_bundle(bundle_dir)

    with pytest.raises(RuntimeError, match=r"await asyncio.to_thread\(run_bundle, root\)"):
        asyncio.run(from_async_code())

    assert not local_store().path.exists()


def test_an_unknown_sandbox_is_refused_before_anything_is_written(bundle_dir):
    with pytest.raises(ValueError, match="Unknown sandbox backend: 'nowhere'"):
        run_bundle(bundle_dir, sandbox="nowhere")

    assert not local_store().path.exists()


def _normalized(output):
    output = re.sub(r"\d+\.\ds", "<t>", output)
    return re.sub(rf"instance {re.escape(ROOT)}/(\w+)-\w+", r"instance \1-<id>", output)


def test_the_cli_prints_every_run_and_eval_and_exits_1_when_a_run_failed(bundle_dir, quiet_logs):
    result = CliRunner().invoke(cli, ["run", str(bundle_dir)])

    assert result.exit_code == 1, result.output
    summary = _normalized(result.output).split("\n\n", 1)[1]
    assert summary == (
        "Tasks:\n"
        "  tasks/a.json v1: passed (check: 1), <t>, instance a-<id>\n"
        "  tasks/b.json v1: scored below 1 (check: 0.5), <t>, instance b-<id>\n"
        "  tasks/c.json v1: failed at step 'boom': RuntimeError: boom, <t>, instance c-<id>\n"
        "Evals:\n"
        "  evals/full.toml v1: 1 of 2 passed, 1 failed\n"
        "  evals/smoke.toml v1: 1 of 2 passed\n"
        "tasks/unnamed.json isn't named by any eval, so it didn't run; run it with --task unnamed\n"
    )


def test_the_cli_prints_a_raised_runs_traceback_only_with_verbose(bundle_dir, quiet_logs):
    quiet = CliRunner().invoke(cli, ["run", str(bundle_dir)])
    verbose = CliRunner().invoke(cli, ["--verbose", "run", str(bundle_dir)])

    assert (quiet.exit_code, verbose.exit_code) == (1, 1)
    assert quiet.stderr == ""
    assert verbose.stderr.startswith("\ntasks/c.json raised:\nTraceback (most recent call last):")
    assert verbose.stderr.rstrip().endswith("RuntimeError: boom")
    assert _normalized(verbose.stdout).split("\n\n", 1)[1] == _normalized(quiet.stdout).split("\n\n", 1)[1]


def test_the_cli_exits_0_when_no_run_failed_even_if_one_scored_below_1(bundle_dir, quiet_logs):
    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--eval", "smoke"])

    assert result.exit_code == 0, result.output
    assert "tasks/b.json v1: scored below 1" in result.output


def test_the_cli_prints_an_unknown_sandbox_as_one_line(bundle_dir, quiet_logs):
    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--sandbox", "nowhere"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: Unknown sandbox backend: 'nowhere'")


def test_the_cli_prints_a_bundle_problem_as_the_answer_it_is(bundle_dir, quiet_logs):
    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--task", "nope"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: --task 'nope': this bundle has no task with that name or id")


def _put_image(id):
    get_artifact_store().put_document(DockerImageArtifact(
        id=id, description=id, image_name="solver:v1", tar_gz_s3_url="file:///solver.tar.gz"))


def test_a_dry_run_selects_what_a_run_would_and_runs_nothing(bundle_dir):
    every = dry_run_bundle(bundle_dir)
    selected = dry_run_bundle(bundle_dir, tasks=["unnamed", "a"], evals=["smoke"])

    assert [entry.name for entry in every.runs] == ["a", "b", "c"]
    assert [every.path(entry) for entry in every.skipped] == ["tasks/unnamed.json"]
    assert ([entry.name for entry in selected.runs], selected.skipped) == (["a", "b", "unnamed"], ())
    assert _Scored.seen == []
    assert not local_store().path.exists()


def test_a_dry_run_refuses_what_a_run_refuses_before_anything_is_written(bundle_dir):
    with namespace_routing():
        Task.put(id="shared-task", steps=[_Scored(id="check", version=None)])
    layout(bundle_dir, {"evals/other.toml": 'tasks = ["shared-task"]\n'})

    with pytest.raises(BundleError, match="evals/other.toml: tasks\\[0\\]: 'shared-task' is a store task"):
        dry_run_bundle(bundle_dir)
    with pytest.raises(ValueError, match="Unknown sandbox backend: 'nowhere'"):
        dry_run_bundle(bundle_dir, evals=["smoke"], sandbox="nowhere")
    with pytest.raises(BundleError, match="--task 'nope': this bundle has no task with that name or id"):
        dry_run_bundle(bundle_dir, tasks=["nope"])
    assert not local_store().path.exists()


def test_the_cli_dry_run_prints_what_the_run_would_write_and_run_and_exits_0(bundle_dir, quiet_logs):
    with namespace_routing():
        FileArtifact.put_bytes("shared-notes", description="notes", filename="notes.txt", content=b"notes")
    layout(bundle_dir, {
        "artifacts/script/run.py": "print(1)\n",
        "artifacts/script/lib.py": "x = 1\n",
        "tasks/code.json": json.dumps([
            {"id": "code", "type": "run_code", "script_artifact_id": "script", "script_file": "run.py"},
            {"id": "notes", "type": "load_artifact", "sandbox_name": "box", "artifact_id": "shared-notes"},
        ]),
        "evals/smoke.toml": 'tasks = ["a", "b", "code"]\n',
    })

    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert result.output == (
        DRY_RUN
        + "artifacts/script: v1 (new)\n"
        "tasks/a.json: v1 (new)\n"
        "tasks/b.json: v1 (new)\n"
        "tasks/c.json: v1 (new)\n"
        "tasks/code.json: v1 (new)\n"
        "evals/full.toml: v1 (new)\n"
        "evals/smoke.toml: v1 (new)\n"
        "\n"
        "Store refs:\n"
        "  artifact shared-notes v1, the latest\n"
        "Would run:\n"
        "  tasks/a.json v1\n"
        "  tasks/b.json v1\n"
        "  tasks/c.json v1\n"
        "  tasks/code.json v1\n"
        "Evals:\n"
        "  evals/full.toml v1: tasks/a.json, tasks/c.json\n"
        "  evals/smoke.toml v1: tasks/a.json, tasks/b.json, tasks/code.json\n"
        "tasks/unnamed.json isn't named by any eval, so it wouldn't run; run it with --task unnamed\n"
        "Not preflighted, since each reads what the run would write first:\n"
        "  tasks/code.json: step 'code' (run_code)\n"
        + DRY_RUN
    )
    assert not local_store().path.exists()
    assert _Scored.seen == []


def test_the_cli_dry_run_prints_a_problem_as_one_line_and_exits_1(bundle_dir, quiet_logs):
    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--task", "nope", "--dry-run"])

    assert result.exit_code == 1
    assert result.output.startswith(f"{DRY_RUN}Error: --task 'nope': this bundle has no task with that name or id")
    assert "Traceback" not in result.output


def test_the_cli_dry_run_accepts_keep_and_model_and_ignores_them(bundle_dir, quiet_logs):
    plain = CliRunner().invoke(cli, ["run", str(bundle_dir), "--eval", "smoke", "--dry-run"])
    ignoring = CliRunner().invoke(cli, ["run", str(bundle_dir), "--eval", "smoke", "--dry-run", "--keep", "--model", "m"])

    assert (plain.exit_code, ignoring.exit_code) == (0, 0)
    assert ignoring.output == plain.output
    assert _Scored.seen == []
