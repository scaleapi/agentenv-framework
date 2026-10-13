"""What running a bundle tears down, and when: each run as it ends, nothing under ``keep`` until the result
is torn down, and every run that started on Ctrl-C or SIGTERM, which marks it cancelled. A second signal stops
the teardown. ``agent-env run`` prints what it tore down, what it keeps up and what is still up."""

import asyncio
import json
import logging
import re
import signal
import sys
import time

import pytest
from click.testing import CliRunner

from agent_env.bundle import Outcome, RunInterrupted, run_bundle
from agent_env.cli import cli
from agent_env.config import get_config
from agent_env.config.runtime import Config
from agent_env.store.document_store import Filter
from agent_env.task import TaskInstanceStore, teardown
from agent_env.env.env import DeployedGatewayEnv
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from tst.unit.bundle._support import layout, local_store

pytestmark = pytest.mark.usefixtures("sigint_handled")


class _Deploys(TaskStep):
    """Records a sandbox on the fake backend, then optionally signals this process once the other runs have
    started, naps, and fails or raises CancelledError."""

    type = "deploys_teardown_test"
    entity_refs = ()

    def __init__(self, nap=0.0, send=None, end=None, **base):
        super().__init__(**base)
        self.nap, self.send, self.end = nap, send, end

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), nap=data.get("nap", 0.0), send=data.get("send"), end=data.get("end"))

    def to_dict(self):
        return {**super().to_dict(), "nap": self.nap, "send": self.send, "end": self.end}

    async def execute(self, context):
        name = context.metadata["task_id"].rsplit("/", 1)[-1]
        context.deployed_sandboxes.append(DeployedSandbox(
            sandbox_name="box", sandbox_id=f"sb-{name}", sandbox_mode="vm", sandbox_type="fake",
            tunnel_urls={"8080": f"http://localhost:1{len(name)}"}))
        if self.send:
            await asyncio.sleep(0.2)
            signal.raise_signal(getattr(signal, self.send))
        await asyncio.sleep(self.nap)
        if self.end == "fail":
            raise RuntimeError("boom")
        if self.end == "cancel":
            raise asyncio.CancelledError
        return context


class _FakeSandbox:
    ON_THIS_MACHINE = False

    def __init__(self, log, sandbox_id):
        self.log, self.sandbox_id = log, sandbox_id

    async def terminate(self):
        if self.sandbox_id in self.log.signal_midway:
            signal.raise_signal(signal.SIGINT)
            await asyncio.sleep(0.2)
        if self.sandbox_id in self.log.second_signal:
            signal.raise_signal(signal.SIGINT)
            await asyncio.Event().wait()
        if self.sandbox_id in self.log.failing:
            raise RuntimeError("stuck")
        self.log.append(self.sandbox_id)


class _Terminated(list):
    def __init__(self):
        super().__init__()
        self.failing: set[str] = set()
        self.signal_midway: set[str] = set()  # a signal lands while it tears down, which then goes on
        self.second_signal: set[str] = set()  # a signal lands while it tears down, which then hangs


@pytest.fixture
def terminated(monkeypatch):
    log = _Terminated()

    class _Provider:
        async def get_sandbox(self, sandbox_id):
            return _FakeSandbox(log, sandbox_id)

    def build(name):
        if name != "fake":
            raise ValueError(f"unknown backend {name}")
        return _Provider()

    monkeypatch.setattr(teardown, "build_sandbox_provider", build)
    return log


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Deploys.type: _Deploys}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)


def _task(**step):
    return json.dumps([{"id": "deploy", "type": _Deploys.type, **step}])


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "lifecycle"


def _statuses(result):
    return {run.entry.name: (doc["status"], doc["error"]) for run in result.runs if run.instance_id
            for doc in [local_store().find_one("task_instances", Filter.of(instance_id=run.instance_id))]}


def test_each_run_is_torn_down_as_it_ends_whether_it_passed_or_failed(bundle_dir, terminated):
    layout(bundle_dir, {"tasks/ok.json": _task(), "tasks/bad.json": _task(end="fail")})

    result = run_bundle(bundle_dir)

    assert {run.entry.name: run.outcome for run in result.runs} == {"ok": Outcome.UNSCORED, "bad": Outcome.FAILED}
    assert sorted(terminated) == ["sb-bad", "sb-ok"]
    assert [[sandbox.sandbox_id for sandbox in run.torn_down.terminated] for run in result.runs] == [
        [f"sb-{run.entry.name}"] for run in result.runs]


def test_keep_leaves_every_run_up_until_the_result_is_torn_down(bundle_dir, terminated):
    layout(bundle_dir, {"tasks/ok.json": _task(), "tasks/bad.json": _task(end="fail"),
                        "evals/all.toml": 'tasks = ["ok", "bad"]\n'})

    kept = run_bundle(bundle_dir, keep=True)

    assert terminated == []
    assert [run.torn_down for run in kept.runs] == [None, None]

    torn = kept.teardown()

    assert sorted(terminated) == ["sb-bad", "sb-ok"]
    assert all(len(run.torn_down.terminated) == 1 for run in torn.runs)
    assert {id(run) for run in torn.evals[0].runs} == {id(run) for run in torn.runs}
    assert torn.teardown() == torn
    assert sorted(terminated) == ["sb-bad", "sb-ok"]


@pytest.mark.parametrize("send, reason", [("SIGINT", "cancelled by Ctrl-C"), ("SIGTERM", "cancelled by SIGTERM")])
def test_a_signal_cancels_the_runs_marks_and_tears_down_the_started_ones_and_raises_with_them(
        bundle_dir, terminated, send, reason):
    layout(bundle_dir, {"tasks/a.json": _task(nap=30, send=send),
                        **{f"tasks/{name}.json": _task(nap=30) for name in "bcde"}})
    lines = []

    with pytest.raises(RunInterrupted) as stopped:
        run_bundle(bundle_dir, on_progress=lines.append)

    result = stopped.value.result
    assert stopped.value.signum == getattr(signal, send)
    assert [(run.entry.name, run.outcome, run.started) for run in result.runs] == [
        *((name, Outcome.CANCELLED, True) for name in "abcd"), ("e", Outcome.CANCELLED, False)]
    assert sorted(terminated) == ["sb-a", "sb-b", "sb-c", "sb-d"]
    assert _statuses(result) == {name: ("cancelled", reason) for name in "abcd"}
    assert get_config().get_document_store().find_one("task_instances", Filter.of(task_id=result.runs[0].task.id)) is None
    assert "Cancelling: tearing down 4 runs (Ctrl-C again to stop now)" in lines


def test_a_second_signal_stops_the_teardown_and_leaves_what_it_didnt_reach(bundle_dir, terminated):
    layout(bundle_dir, {"tasks/a.json": _task(nap=30, send="SIGINT"), "tasks/b.json": _task(nap=30)})
    terminated.second_signal.add("sb-a")
    lines = []

    with pytest.raises(RunInterrupted) as stopped:
        run_bundle(bundle_dir, on_progress=lines.append)

    first, second = stopped.value.result.runs
    assert [sandbox.sandbox_id for sandbox in first.torn_down.left] == ["sb-a"]
    assert terminated == ["sb-b"] or [sandbox.sandbox_id for sandbox in second.torn_down.left] == ["sb-b"]
    assert _statuses(stopped.value.result) == {name: ("cancelled", "cancelled by Ctrl-C") for name in "ab"}
    assert "Stopping the teardown now" in lines


def test_a_first_signal_lets_a_finished_runs_teardown_finish(bundle_dir, terminated):
    layout(bundle_dir, {"tasks/done.json": _task(), "tasks/slow.json": _task(nap=30)})
    terminated.signal_midway.add("sb-done")

    with pytest.raises(RunInterrupted) as stopped:
        run_bundle(bundle_dir)

    done, slow = stopped.value.result.runs
    assert (done.outcome, slow.outcome) == (Outcome.UNSCORED, Outcome.CANCELLED)
    assert [sandbox.sandbox_id for sandbox in done.torn_down.terminated] == ["sb-done"]
    assert sorted(terminated) == ["sb-done", "sb-slow"]


def test_with_keep_a_second_signal_leaves_the_finished_runs_up_too(bundle_dir, terminated):
    layout(bundle_dir, {"tasks/done.json": _task(), "tasks/slow.json": _task(nap=30, send="SIGINT")})
    terminated.second_signal.add("sb-slow")

    with pytest.raises(RunInterrupted) as stopped:
        run_bundle(bundle_dir, keep=True)

    done, slow = stopped.value.result.runs
    assert terminated == []
    assert done.torn_down is None
    assert [sandbox.sandbox_id for sandbox in slow.torn_down.left] == ["sb-slow"]


def test_with_keep_a_signal_tears_down_the_runs_that_already_finished_too(bundle_dir, terminated):
    layout(bundle_dir, {"tasks/done.json": _task(), "tasks/slow.json": _task(nap=30, send="SIGINT")})

    with pytest.raises(RunInterrupted) as stopped:
        run_bundle(bundle_dir, keep=True)

    assert sorted(terminated) == ["sb-done", "sb-slow"]
    assert [run.outcome for run in stopped.value.result.runs] == [Outcome.UNSCORED, Outcome.CANCELLED]


@pytest.mark.parametrize("keep", [False, True])
def test_a_steps_own_cancellation_still_raises_once_every_run_is_torn_down(bundle_dir, terminated, keep):
    layout(bundle_dir, {"tasks/a.json": _task(end="cancel"), "tasks/b.json": _task(nap=0.3)})

    with pytest.raises(asyncio.CancelledError):
        run_bundle(bundle_dir, keep=keep)

    assert sorted(terminated) == ["sb-a", "sb-b"]


def test_a_signal_during_the_teardown_of_kept_runs_raises_with_what_it_reached(bundle_dir, terminated):
    layout(bundle_dir, {"tasks/ok.json": _task()})
    terminated.second_signal.add("sb-ok")
    kept = run_bundle(bundle_dir, keep=True)

    with pytest.raises(RunInterrupted) as stopped:
        kept.teardown()

    assert [sandbox.sandbox_id for sandbox in stopped.value.result.runs[0].torn_down.left] == ["sb-ok"]


@pytest.fixture
def quiet_logs():
    """pytest's live logging swaps its own stdout back in to print a record, so CliRunner loses whatever is
    echoed after the first one."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _summary(output):
    output = re.sub(r"\d+\.\ds", "<t>", output)
    return re.sub(r"instance @local/\S+/(\w+)-\w+", r"instance \1-<id>", output).split("\n\nTasks:\n", 1)[1]


def test_the_cli_prints_what_ran_and_what_it_tore_down_on_ctrl_c_and_exits_130(bundle_dir, terminated, quiet_logs):
    layout(bundle_dir, {"tasks/a.json": _task(nap=30, send="SIGINT"), "tasks/b.json": _task(nap=30),
                        "tasks/c.json": _task(nap=30), "tasks/d.json": _task(nap=30), "tasks/e.json": _task()})

    result = CliRunner().invoke(cli, ["run", str(bundle_dir)])

    assert result.exit_code == 130, result.output
    assert _summary(result.output) == (
        "  tasks/a.json v1: cancelled, <t>, instance a-<id>\n"
        "  tasks/b.json v1: cancelled, <t>, instance b-<id>\n"
        "  tasks/c.json v1: cancelled, <t>, instance c-<id>\n"
        "  tasks/d.json v1: cancelled, <t>, instance d-<id>\n"
        "  tasks/e.json v1: didn't start\n"
        "\n"
        "Tore down 4 sandboxes.\n"
    )


def test_the_cli_prints_a_sandbox_it_couldnt_tear_down_as_still_up(bundle_dir, terminated, quiet_logs):
    layout(bundle_dir, {"tasks/ok.json": _task()})
    terminated.failing.add("sb-ok")

    result = CliRunner().invoke(cli, ["run", str(bundle_dir)])

    assert result.exit_code == 0, result.output
    assert "[tasks/ok.json] couldn't tear down sb-ok: RuntimeError: stuck\n" in result.output
    assert result.output.endswith("\nStill up, 1 sandbox:\n  sb-ok  fake\n")


def test_the_cli_keep_prints_the_endpoints_holds_and_tears_down_on_ctrl_c(bundle_dir, terminated, quiet_logs,
                                                                          monkeypatch):
    layout(bundle_dir, {"tasks/ok.json": _task(), "tasks/bad.json": _task(end="fail")})
    holds = []

    def interrupted_sleep(seconds):
        holds.append(list(terminated))
        signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(time, "sleep", interrupted_sleep)

    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--keep"])

    assert result.exit_code == 1, result.output
    assert holds == [[]]
    assert result.output.split("\n\nKept up:\n", 1)[1] == (
        "  tasks/bad.json\n"
        "    sb-bad  fake  sandbox box\n"
        "      port 8080  http://localhost:13\n"
        "  tasks/ok.json\n"
        "    sb-ok  fake  sandbox box\n"
        "      port 8080  http://localhost:12\n"
        "\n"
        "Holding 2 sandboxes up; Ctrl-C tears them down.\n"
        "\n"
        "Tearing down 2 sandboxes (Ctrl-C again to stop now)\n"
        "\n"
        "Tore down 2 sandboxes.\n"
    )
    assert sorted(terminated) == ["sb-bad", "sb-ok"]


def test_the_cli_keep_with_nothing_deployed_does_not_hold(bundle_dir, quiet_logs, monkeypatch):
    layout(bundle_dir, {"tasks/nothing.json": json.dumps([])})
    monkeypatch.setattr(time, "sleep", lambda seconds: pytest.fail("held with nothing up"))

    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--keep"])

    assert "Nothing to keep up: no run left a sandbox." in result.output


def test_the_keep_listing_names_every_sandbox_it_counts_with_the_records_behind_it():
    context = TaskStepContext(
        deployed_sandboxes=[DeployedSandbox(sandbox_name="box", sandbox_id="sb-box", sandbox_mode="vm",
                                            sandbox_type="fake", tunnel_urls={"8080": "http://t"})],
        deployed_envs=[DeployedGatewayEnv(env_id="shop", env_version=1, gateway_url="http://g", mcp_url="http://m",
                                          db_web_url="http://pgweb", sandbox_id="sb-gw", sandbox_type="fake",
                                          sandbox_ids={"service_db": {"orders": "sb-db"}})],
        deployed_agents=[DeployedAgent(agent_name="solver", api_url="http://a", a2a_url="http://a2a",
                                       sandbox_id="sb-box", sandbox_type="fake")],
    )

    assert sys.modules["agent_env.cli.run"]._kept_up(context) == [
        "sb-box  fake  sandbox box, agent solver",
        "  port 8080  http://t",
        "  a2a  http://a2a",
        "sb-gw  fake  env shop",
        "  mcp  http://m",
        "  gateway  http://g",
        "  pgweb  http://pgweb",
        "sb-db  fake  env shop",
    ]


def test_with_verbose_an_interrupted_run_still_prints_the_tracebacks(bundle_dir, terminated, quiet_logs):
    layout(bundle_dir, {"tasks/a.json": _task(end="fail"), "tasks/b.json": _task(nap=30, send="SIGINT")})

    result = CliRunner().invoke(cli, ["--verbose", "run", str(bundle_dir)])

    assert result.exit_code == 130, result.output
    assert "tasks/a.json raised:" in result.stderr


def test_a_cancelled_mark_lands_after_a_failure_write_a_second_signal_cut_short(bundle_dir, terminated, monkeypatch):
    layout(bundle_dir, {"tasks/a.json": _task(nap=30, send="SIGINT")})
    record_failure = TaskInstanceStore.record_task_failure_sync

    def slow_failure_write(self, *args):
        signal.raise_signal(signal.SIGINT)  # the second signal lands while the failure write is on its way
        time.sleep(0.3)
        record_failure(self, *args)

    monkeypatch.setattr(TaskInstanceStore, "record_task_failure_sync", slow_failure_write)

    with pytest.raises(RunInterrupted) as stopped:
        run_bundle(bundle_dir)

    assert _statuses(stopped.value.result) == {"a": ("cancelled", "cancelled by Ctrl-C")}
