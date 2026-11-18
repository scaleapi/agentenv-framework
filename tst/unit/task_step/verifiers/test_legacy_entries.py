"""run_container_unit_tests_verifier, verify_sandbox, env_outcome_verifier and aggregate_verifiers on the
VerifierTaskStep base write the legacy ``verifications`` entry they wrote before it, byte for byte with its key order,
and raise what they raised (``tst/data/verifier_golden/legacy_entries.json``, recorded from main at 25b862bc6, object
urls under ``<objects>/``), but for ``CHANGED``; and each writes its named grade beside it."""
import asyncio
import itertools
import json
import shlex
import uuid
from types import SimpleNamespace

import pytest

import agent_env.providers.sandbox_providers.sandbox_provider as sandbox_provider
from agent_env.artifact import FileArtifact
from agent_env.cli.task.run import _format_verifier_output
from agent_env.config import get_config
from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.verifiers import verify_sandbox as verify_sandbox_module
from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import AggregateVerifiersTaskStep
from agent_env.task_step.task_steps.verifiers.env_outcome_verifier import EnvOutcomeVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep as RunContainer,
)
from agent_env.task_step.task_steps.verifiers.verify_sandbox import VerifySandboxTaskStep
from tst.unit.computer_home import Computer
from tst.util.verifier_golden import GOLDEN

RECORDED = json.loads((GOLDEN / "legacy_entries.json").read_text())
REWARD = "/logs/verifier/reward.txt"
SCENARIOS = {}
# What the base changes on purpose: an aggregate's entry has its (empty) results, and a script that returns None is
# told so rather than by a log line's len().
CHANGED_RAISES = {
    "outcome_returns_none": "TypeError: verify() in artifact 's' returned None, not a list of criterion rows",
}


def scenario(fn):
    SCENARIOS[fn.__name__] = fn
    return fn


def outcome(step, context) -> tuple[str | None, str | None]:
    """What ``step`` raised running on ``context``, and its entry as JSON with the object store's urls normalised."""
    raised = None
    try:
        asyncio.run(step.execute(context))
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
    entry = (context.metadata.get("verifications") or {}).get(step.verifier_id)
    objects = get_config().get_object_store().object_url("x")[:-1]
    return raised, json.dumps(entry).replace(objects, "<objects>/")


class _ContainerSandbox:
    """The task container's files as a dict: ``rm -f`` and ``docker cp`` act on it, and the command runs ``test_sh``,
    which returns its exit code, stdout and stderr."""

    type = None

    def __init__(self, files, test_sh):
        self.files, self.vm_files, self.test_sh = dict(files), {}, test_sh

    def scoped_name(self, name):
        return name

    async def exec_script(self, script):
        words = shlex.split(script)
        if words[0] == "docker" and "rm" in words:
            for path in words[words.index("--") + 1:]:
                self.files.pop(path, None)
        elif words[:2] == ["rm", "-f"]:
            self.vm_files.pop(words[2], None)
        return ""

    async def exec_with_output(self, *args):
        if args[:2] == ("sudo", "cat"):
            return 0, self.vm_files[args[2]], ""
        return self.test_sh(self.files)

    async def docker_cp(self, source, destination):
        path = source.split(":", 1)[1]
        if path not in self.files:
            raise RuntimeError(f"no such file: {path}")
        self.vm_files[destination] = self.files[path]


def _pinned(monkeypatch):
    """The outputs' keys are named after the time and a uuid: pinned, so the entry is the same every run."""
    counter = itertools.count()
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(counter)))
    monkeypatch.setattr("agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier.time.time",
                        lambda: 1_790_000_000.0)
    monkeypatch.setattr(RunContainer, "_upload_text_artifact", staticmethod(
        lambda text, artifact_id, description, object_url: SimpleNamespace(id=artifact_id, version=1,
                                                                           object_url=object_url)))


def _in_container(monkeypatch, sandbox, *, instance_id="inst-1", step_params=None, seed=None):
    _pinned(monkeypatch)
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider",
                        lambda: SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=sandbox)))
    context = TaskStepContext(instance_id=instance_id)
    context.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    context.deployed_sandboxes.append(DeployedSandbox(sandbox_name="h", sandbox_id="sb-1", sandbox_mode="vm"))
    if step_params is not None:
        context.metadata["user_overrides"] = {"step_params": {"tests": step_params}}
    if seed is not None:
        context.metadata["seed"] = seed
    return context


def _tests(**fields):
    return RunContainer(id="tests", version=None, sandbox_name="h", container_name="c",
                        **{"command": "bash /tests/test.sh", **fields})


def _writes_reward(value, exit_code=0):
    def test_sh(files):
        if value is not None:
            files[REWARD] = value
        return exit_code, "3 passed", ""
    return test_sh


@scenario
def passing_tests(monkeypatch, tmp_path):
    return _tests(), _in_container(monkeypatch, _ContainerSandbox({}, lambda files: (0, "ok\n" * 400, "warn")))


@scenario
def failing_tests_raise_after_recording(monkeypatch, tmp_path):
    return _tests(verifier_id="unit"), _in_container(monkeypatch, _ContainerSandbox(
        {}, lambda files: (1, "1 failed", "E" * 3000)))


@scenario
def timeout_at_124(monkeypatch, tmp_path):
    return _tests(timeout_sec=5), _in_container(monkeypatch, _ContainerSandbox({}, lambda files: (124, "", "killed")))


@scenario
def timeout_at_137(monkeypatch, tmp_path):
    return _tests(), _in_container(monkeypatch, _ContainerSandbox({}, lambda files: (137, "", "")))


@scenario
def reward_one(monkeypatch, tmp_path):
    return _tests(reward_path=REWARD), _in_container(monkeypatch, _ContainerSandbox({}, _writes_reward("1\n")))


@scenario
def reward_half(monkeypatch, tmp_path):
    return _tests(reward_path=REWARD), _in_container(monkeypatch, _ContainerSandbox({}, _writes_reward("0.5")))


@scenario
def reward_clamped(monkeypatch, tmp_path):
    return _tests(reward_path=REWARD), _in_container(monkeypatch, _ContainerSandbox({}, _writes_reward("2")))


@scenario
def reward_missing_after_a_failing_exit(monkeypatch, tmp_path):
    return _tests(reward_path=REWARD), _in_container(monkeypatch, _ContainerSandbox(
        {REWARD: "1\n"}, _writes_reward(None, exit_code=1)))


@scenario
def reward_not_a_number(monkeypatch, tmp_path):
    return _tests(reward_path=REWARD), _in_container(monkeypatch, _ContainerSandbox({}, _writes_reward("FAIL")))


@scenario
def reward_then_time_out(monkeypatch, tmp_path):
    return _tests(reward_path=REWARD), _in_container(monkeypatch, _ContainerSandbox(
        {}, _writes_reward("1", exit_code=124)))


@scenario
def result_files(monkeypatch, tmp_path):
    files = {"/out/report.json": '{"tests": 3, "passed": 3}', "/out/log.txt": "plain"}
    return (_tests(result_paths=["/out/report.json", "/out/log.txt", "/out/missing.json"]),
            _in_container(monkeypatch, _ContainerSandbox(files, lambda files: (0, "", ""))))


@scenario
def result_file_over_the_inline_limit(monkeypatch, tmp_path):
    report = json.dumps({"tests": [{"name": f"test_{i}", "status": "passed", "duration": 0.01} for i in range(300)]})
    return (_tests(result_paths=["/out/ctrf.json"]),
            _in_container(monkeypatch, _ContainerSandbox({"/out/ctrf.json": report}, lambda files: (0, "x" * 5000, ""))))


@scenario
def step_params_and_seed(monkeypatch, tmp_path):
    return _tests(command="run <case>", env_vars={"STORED": "1"}), _in_container(
        monkeypatch, _ContainerSandbox({}, lambda files: (0, "", "")), seed={"case": "a b"},
        step_params={"command": "pytest -k <case>", "env_vars": {"TOKEN": "s3cret"}})


@scenario
def no_instance_id(monkeypatch, tmp_path):
    return _tests(), _in_container(monkeypatch, _ContainerSandbox({}, lambda files: (0, "", "")), instance_id=None)


@scenario
def container_not_deployed(monkeypatch, tmp_path):
    return (RunContainer(id="tests", version=None, sandbox_name="h", container_name="other", command="true"),
            _in_container(monkeypatch, _ContainerSandbox({}, lambda files: (0, "", ""))))


@scenario
def sandbox_not_deployed(monkeypatch, tmp_path):
    context = _in_container(monkeypatch, _ContainerSandbox({}, lambda files: (0, "", "")))
    context.deployed_sandboxes.clear()
    return _tests(), context


@scenario
def env_var_override_not_a_posix_name(monkeypatch, tmp_path):
    return _tests(), _in_container(monkeypatch, _ContainerSandbox({}, lambda files: (0, "", "")),
                                   step_params={"env_vars": {"A B": "x"}})


def _in_computer(monkeypatch, tmp_path):
    _pinned(monkeypatch)
    computer = Computer(tmp_path).install(monkeypatch)
    (tmp_path / "logs").mkdir()
    return TaskStepContext(deployed_envs=[computer.deployed()])


@scenario
def computer_reward(monkeypatch, tmp_path):
    return RunContainer(id="tests", version=None, env_id="desk", setup_commands=["touch setup-ran"],
                        env_vars={"ANSWER": "42"}, reward_path="logs/reward.txt", result_paths=["out.json"],
                        command='echo \'{"n": 1}\' > out.json; [ "$ANSWER" = 42 ] && echo 1 > logs/reward.txt; true'
                        ), _in_computer(monkeypatch, tmp_path)


@scenario
def computer_tests_fail(monkeypatch, tmp_path):
    return (RunContainer(id="tests", version=None, env_id="desk", command="echo boom >&2; exit 2"),
            _in_computer(monkeypatch, tmp_path))


@scenario
def computer_setup_fails(monkeypatch, tmp_path):
    return (RunContainer(id="tests", version=None, env_id="desk", setup_commands=["exit 3"], command="true"),
            _in_computer(monkeypatch, tmp_path))


@scenario
def computer_env_not_deployed(monkeypatch, tmp_path):
    return (RunContainer(id="tests", version=None, env_id="nope", command="true"), _in_computer(monkeypatch, tmp_path))


class _ProbeSandbox:
    """A VM whose probes see every file but ``missing``, whose bash checks fail when they run ``false``, and whose
    commands raise on ``raises_on`` or hang."""

    mode, sandbox_id, container_name = "vm", "sb-1", "agent-api"

    def __init__(self, missing=(), raises_on=None, hangs=False):
        self.missing, self.raises_on, self.hangs = set(missing), raises_on, hangs

    async def exec_with_output(self, *args):
        if self.hangs:
            await asyncio.sleep(1)
        if self.raises_on in args:
            raise ConnectionError("channel closed")
        if args[-3:] == ("ps", "--format", "{{.Names}}"):
            return 0, "agent-api\n", ""
        if "test" in args and args[-1] in self.missing:
            return 1, "", ""
        if "cat" in args:
            return 0, "hello world\n", ""
        if "bash" in args:
            return (1, "", "nope") if "false" in args[-1] else (0, "", "")
        return 0, "", ""


def _on_sandbox(monkeypatch, sandbox, **context):
    monkeypatch.setattr(verify_sandbox_module, "build_sandbox_provider", lambda spec: SimpleNamespace(
        get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=sandbox)))
    return TaskStepContext(**context) if context else TaskStepContext(deployed_sandboxes=[
        DeployedSandbox(sandbox_name="box", sandbox_id="sb-1", sandbox_mode="vm", sandbox_type="fake")])


PROBES = [
    {"type": "probe_file_exists", "criterion": "files", "paths": ["hello.txt", "/etc/hosts"]},
    {"type": "probe_file_contains", "criterion": "greets", "paths": ["hello.txt"], "expected": "hel+o", "weight": 2},
    {"type": "bash_cmd_succeeds", "criterion": "check", "bash_cmd": "python3 check.py", "id": "chk"},
    {"type": "response_contains", "criterion": "not ours", "weight": 5},
    {"type": "probe_dir_exists", "criterion": "dir", "paths": ["out"]},
]


def _probes(criteria=PROBES, **fields):
    return VerifySandboxTaskStep(id="fs", version=None, base_dir="/app", verifier_id="fs-vid",
                                 **{"sandbox_name": "box", **fields}, criteria=criteria)


@scenario
def probes_pass_beside_a_row_they_skip(monkeypatch, tmp_path):
    return _probes(), _on_sandbox(monkeypatch, _ProbeSandbox())


@scenario
def a_probe_fails(monkeypatch, tmp_path):
    return _probes(), _on_sandbox(monkeypatch, _ProbeSandbox(missing={"/app/out"}))


@scenario
def probes_weighted(monkeypatch, tmp_path):
    return _probes(score_aggregator="weighted_average"), _on_sandbox(monkeypatch, _ProbeSandbox(missing={"/app/out"}))


@scenario
def probes_any_pass(monkeypatch, tmp_path):
    return _probes(score_aggregator="any_pass"), _on_sandbox(monkeypatch, _ProbeSandbox(missing={"/app/out"}))


@scenario
def probes_in_the_agents_container(monkeypatch, tmp_path):
    agent = DeployedAgent(agent_name="solver", api_url="http://agent", sandbox_id="sb-1", sandbox_type="fake")
    return (_probes(sandbox_name=None, agent_name="solver"),
            _on_sandbox(monkeypatch, _ProbeSandbox(), deployed_agents=[agent]))


@scenario
def every_row_skipped(monkeypatch, tmp_path):
    return _probes([{"type": "response_contains", "criterion": "x"}]), _on_sandbox(monkeypatch, _ProbeSandbox())


@scenario
def no_probes(monkeypatch, tmp_path):
    return _probes([]), _on_sandbox(monkeypatch, _ProbeSandbox())


@scenario
def a_probe_that_raises(monkeypatch, tmp_path):
    return (_probes([{"type": "bash_cmd_succeeds", "criterion": "c", "bash_cmd": "true"}]),
            _on_sandbox(monkeypatch, _ProbeSandbox(raises_on="bash")))


@scenario
def a_probe_that_times_out(monkeypatch, tmp_path):
    return (_probes([{"type": "bash_cmd_succeeds", "criterion": "c", "bash_cmd": "true"}], shell_timeout_seconds=0.01),
            _on_sandbox(monkeypatch, _ProbeSandbox(hangs=True)))


@scenario
def probed_sandbox_not_deployed(monkeypatch, tmp_path):
    return _probes(sandbox_name="nope"), _on_sandbox(monkeypatch, _ProbeSandbox())


@scenario
def probed_agent_not_deployed(monkeypatch, tmp_path):
    return _probes(sandbox_name=None, agent_name="solver"), _on_sandbox(monkeypatch, _ProbeSandbox(),
                                                                        deployed_agents=[])


@scenario
def probed_agent_without_a_sandbox(monkeypatch, tmp_path):
    return _probes(sandbox_name=None, agent_name="solver"), _on_sandbox(
        monkeypatch, _ProbeSandbox(), deployed_agents=[DeployedAgent(agent_name="solver", api_url="http://a")])


def _outcome_script(monkeypatch, body, **fields):
    script = f"async def verify(url):\n    {body}\n".encode()
    monkeypatch.setattr(FileArtifact, "get", staticmethod(lambda id, version=None: SimpleNamespace(load=lambda: script)))
    step = EnvOutcomeVerifierTaskStep(id="outcome", version=None, env_id="e", file_artifact_id="s",
                                      verifier_id="eo-vid", **fields)
    context = TaskStepContext()
    context.deployed_envs.append(DeployedEnv(env_id="e", env_version=1, mcp_url="http://mcp"))
    return step, context


OUTCOME_ROWS = ("[{'id': 'a', 'score': 1.0, 'result': True, 'why': 'ok'}, {'id': 'b', 'score': 0.5, 'weight': 2}, "
                "{'id': 'c', 'result': False, 'weight': -1, 'skipped': True}, {'result': True, 'extra': {'n': [1, 2]}}]")


@scenario
def outcome_rows(monkeypatch, tmp_path):
    return _outcome_script(monkeypatch, f"return {OUTCOME_ROWS}")


@scenario
def outcome_rows_weighted(monkeypatch, tmp_path):
    return _outcome_script(monkeypatch, f"return {OUTCOME_ROWS}", score_aggregator="weighted_average")


@scenario
def outcome_no_rows(monkeypatch, tmp_path):
    return _outcome_script(monkeypatch, "return []")


@scenario
def outcome_returns_none(monkeypatch, tmp_path):
    return _outcome_script(monkeypatch, "return None")


@scenario
def outcome_returns_a_tuple(monkeypatch, tmp_path):
    return _outcome_script(monkeypatch, "return ({'id': 'a', 'result': True},)")


@scenario
def outcome_rows_that_arent_dicts(monkeypatch, tmp_path):
    return _outcome_script(monkeypatch, "return ['a', 'b']")


@scenario
def outcome_row_score_a_string(monkeypatch, tmp_path):
    return _outcome_script(monkeypatch, "return [{'score': '1'}]", score_aggregator="weighted_average")


@scenario
def outcome_script_raises(monkeypatch, tmp_path):
    return _outcome_script(monkeypatch, "raise ValueError('db unreachable')")


@scenario
def outcome_env_not_deployed(monkeypatch, tmp_path):
    step, _ = _outcome_script(monkeypatch, "return []")
    return step, TaskStepContext()


def _sources():
    context = TaskStepContext()
    context.metadata["verifications"] = {
        "fs": {"results": [{"id": "x", "score": 1.0, "result": True, "skipped": False},
                           {"id": "y", "skipped": True, "justification": "not handled"}], "score": 1.0},
        "judge": {"format": "rubric_binary", "results": [{"id": "r1", "score": 0.0, "result": False, "weight": 3}],
                  "score": 0.0},
        "inner": {"score": 0.25, "source_verifier_ids": ["fs"]},
        "no-rows": {"score": 1.0},
    }
    return context


def _aggregate(verifier_ids, **fields):
    return AggregateVerifiersTaskStep(id="final", version=None, verifier_ids=verifier_ids, verifier_id="final-vid",
                                      **fields)


@scenario
def aggregate_two_sources(monkeypatch, tmp_path):
    return _aggregate(["fs", "judge"]), _sources()


@scenario
def aggregate_all_pass(monkeypatch, tmp_path):
    return _aggregate(["fs", "judge"], score_aggregator="all_pass"), _sources()


@scenario
def aggregate_of_an_aggregate(monkeypatch, tmp_path):
    return _aggregate(["inner", "fs"]), _sources()


@scenario
def aggregate_of_a_source_without_rows(monkeypatch, tmp_path):
    return _aggregate(["no-rows"]), _sources()


@scenario
def aggregate_of_nothing(monkeypatch, tmp_path):
    return _aggregate([]), _sources()


@scenario
def aggregate_missing_a_source(monkeypatch, tmp_path):
    return _aggregate(["fs", "gone"]), _sources()


def _expected(name, step) -> tuple[str | None, dict | None]:
    entry = RECORDED[name]["entry"]
    if step.type == "aggregate_verifiers" and entry is not None:
        entry = {"results": [], **entry}
    return CHANGED_RAISES.get(name, RECORDED[name]["raised"]), entry


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_each_path_writes_the_entry_and_raises_what_it_did_before_the_base(name, local_stores, monkeypatch, tmp_path):
    step, context = SCENARIOS[name](monkeypatch, tmp_path)
    expected_raise, expected_entry = _expected(name, step)

    raised, entry = outcome(step, context)

    assert raised == expected_raise
    assert entry == json.dumps(expected_entry)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_each_path_writes_its_named_grade(name, local_stores, monkeypatch, tmp_path):
    step, context = SCENARIOS[name](monkeypatch, tmp_path)
    raised, entry = outcome(step, context)
    entry = json.loads(entry)

    grade = context.metadata["grades"][step.id]

    assert (grade["name"], grade["verifier_type"]) == (step.id, step.type)
    status = (grade["verifier_status"], grade["score"], grade["passed"])
    if entry is None:
        assert status == ("failed", None, False) and grade["verifier_error_reasoning"] == raised
    elif entry.get("timed_out"):
        assert status == ("failed", None, False)
        assert grade["verifier_error_reasoning"] == raised.removeprefix("RuntimeError: ")
    else:
        score = float(entry["score"])
        assert status == ("graded", score, score >= 1.0) and grade["verifier_error_reasoning"] is None
        assert len(grade["verifier_criteria"]) == max(len(entry["results"] or []), 1)
    assert "settings_error" not in grade["verifier_settings"]


def test_a_test_runs_output_is_the_grades_and_a_large_one_goes_to_the_object_store(local_stores, monkeypatch,
                                                                                  tmp_path):
    for name, inline in (("passing_tests", True), ("result_file_over_the_inline_limit", False)):
        step, context = SCENARIOS[name](monkeypatch, tmp_path)
        outcome(step, context)
        entry, grade = context.metadata["verifications"][step.verifier_id], context.metadata["grades"][step.id]

        output = {key: value for key, value in entry.items() if key not in ("results", "score")}
        if inline:
            assert grade["verifier_output"] == output
        else:
            ref = grade["verifier_output_ref"]
            assert grade["verifier_output"] is None and ref["bytes"] == len(json.dumps(output))
            assert json.loads(get_config().get_object_store().get(ref["object_url"])) == output


def test_a_test_runs_settings_name_its_env_vars_and_hold_the_runs_command_but_no_value(local_stores, monkeypatch,
                                                                                       tmp_path):
    step, context = step_params_and_seed(monkeypatch, tmp_path)
    outcome(step, context)

    grade = context.metadata["grades"]["tests"]

    assert grade["verifier_settings"] == {
        "scoring_rules": 1, "sandbox_name": "h", "container_name": "c", "env_id": None, "command": "pytest -k <case>",
        "setup_commands": [], "timeout_sec": 300, "env_vars": ["STORED", "TOKEN"], "result_paths": [],
        "reward_path": None, "score_aggregator": "all_pass", "overridden": ["command", "env_vars"],
    }
    assert "s3cret" not in json.dumps(grade)


def test_a_test_run_refused_for_an_env_var_name_still_records_its_settings(local_stores, monkeypatch, tmp_path):
    step, context = env_var_override_not_a_posix_name(monkeypatch, tmp_path)
    outcome(step, context)

    settings = context.metadata["grades"]["tests"]["verifier_settings"]

    assert settings["env_vars"] == ["A B"] and settings["overridden"] == ["env_vars"]
    assert settings["command"] == "bash /tests/test.sh"


def test_a_probe_row_the_step_skips_is_skipped_on_its_grade_and_left_out_of_its_score(local_stores, monkeypatch,
                                                                                     tmp_path):
    step, context = probes_pass_beside_a_row_they_skip(monkeypatch, tmp_path)
    outcome(step, context)

    grade = context.metadata["grades"]["fs"]

    assert [(row["id"], row["passed"], row["skipped"]) for row in grade["verifier_criteria"]] == [
        ("0", True, False), ("1", True, False), ("chk", True, False), ("3", False, True), ("4", True, False)]
    assert grade["score"] == 1.0


def test_an_aggregate_grade_is_one_row_for_itself_at_its_weighted_average(local_stores, monkeypatch, tmp_path):
    step, context = aggregate_two_sources(monkeypatch, tmp_path)
    outcome(step, context)

    grade = context.metadata["grades"]["final"]

    assert grade["verifier_criteria"] == [{"id": "final", "score": 0.25, "passed": False, "weight": 1.0,
                                           "skipped": False}]
    assert grade["verifier_output"] == {"source_verifier_ids": ["fs", "judge"]}
    assert grade["verifier_settings"] == {"scoring_rules": 1, "verifier_ids": ["fs", "judge"],
                                          "score_aggregator": "weighted_average"}


def test_task_run_prints_an_aggregates_score_without_an_empty_results_block(local_stores, monkeypatch, tmp_path):
    step, context = aggregate_two_sources(monkeypatch, tmp_path)
    outcome(step, context)

    assert _format_verifier_output("final-vid", context.metadata["verifications"]["final-vid"]) == [
        "score (verifier_id=final-vid): 0.25"]
    assert _format_verifier_output("fs", context.metadata["verifications"]["fs"])[0] == "verification (verifier_id=fs):"


STORED = [  # each step as main stored it
    (RunContainer(id="tests", version=None, sandbox_name="h", container_name="c", command="bash t.sh",
                  reward_path="/r.txt", env_vars={"A": "1"}),
     {"id": "tests", "type": "run_container_unit_tests_verifier", "version": None, "fail_task_on_error": True,
      "sandbox_name": "h", "container_name": "c", "env_id": None, "command": "bash t.sh", "verifier_id": "tests",
      "setup_commands": [], "timeout_sec": 300, "env_vars": {"A": "1"}, "result_paths": ["/r.txt"],
      "reward_path": "/r.txt"}),
    (VerifySandboxTaskStep(id="fs", version=None, verifier_id="v1", criteria=[]),
     {"id": "fs", "type": "verify_sandbox", "version": None, "fail_task_on_error": True, "criteria": [],
      "agent_name": "default-agent", "base_dir": "/app", "shell_timeout_seconds": 120, "score_aggregator": "all_pass",
      "verifier_id": "v1"}),
    (EnvOutcomeVerifierTaskStep(id="outcome", version=None, env_id="e", file_artifact_id="s", file_artifact_version=2,
                                verifier_id="outcome", prompt_id="p"),
     {"id": "outcome", "type": "env_outcome_verifier", "version": None, "fail_task_on_error": True,
      "verifier_id": "outcome", "env_id": "e", "file_artifact_id": "s", "file_artifact_version": 2,
      "score_aggregator": "all_pass", "prompt_id": "p"}),
    (AggregateVerifiersTaskStep(id="final", version=None, verifier_ids=["a", "b"], verifier_id="final-score"),
     {"id": "final", "type": "aggregate_verifiers", "version": None, "fail_task_on_error": True,
      "verifier_ids": ["a", "b"], "score_aggregator": "weighted_average", "verifier_id": "final-score"}),
]


@pytest.mark.parametrize("step, stored", STORED, ids=[stored["type"] for _, stored in STORED])
def test_each_verifiers_stored_form_is_unchanged(step, stored):
    assert step.to_dict() == stored
    assert type(step).from_dict(stored).to_dict() == stored
