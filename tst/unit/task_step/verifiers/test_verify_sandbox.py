import asyncio
import logging
from types import SimpleNamespace

import pytest

from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.verifiers import verify_sandbox
from agent_env.task_step.task_steps.verifiers.verify_sandbox import VerifySandboxTaskStep

_CRITERIA = [
    {"type": "probe_file_exists", "criterion": "files", "paths": ["hello.txt", "/etc/hosts"]},
    {"type": "probe_file_contains", "criterion": "greets", "paths": ["hello.txt"], "expected": "hel+o"},
    {"type": "bash_cmd_succeeds", "criterion": "check", "bash_cmd": "python3 check.py"},
    {"type": "response_contains", "criterion": "not ours"},
]


class _FakeSandbox:
    def __init__(self, mode, container_name="agent-api"):
        self.mode = mode
        self.sandbox_id = "sb-1"
        self.container_name = container_name
        self.calls: list[tuple[str, ...]] = []

    async def exec_with_output(self, *args):
        self.calls.append(args)
        if args[-3:] == ("ps", "--format", "{{.Names}}"):
            return 0, f"{self.container_name}\n", ""
        if "cat" in args:
            return 0, "hello world\n", ""
        return 0, "", ""



class _HangingSandbox(_FakeSandbox):
    async def exec_with_output(self, *args):
        self.calls.append(args)
        await asyncio.sleep(1)
        return 0, "", ""


class _BashSandbox:
    """Runs each probe in a real bash, so a test sees how bash parses the wrapped command."""

    mode = "vm"
    sandbox_id = "sb-1"
    container_name = "unused"

    async def exec_with_output(self, *args):
        proc = await asyncio.create_subprocess_exec(
            *[a for a in args if a != "sudo"], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return proc.returncode, out.decode(), err.decode()

@pytest.fixture
def provide(monkeypatch):
    requested = []

    def install(sandbox):
        def build(spec):
            requested.append(spec)
            return SimpleNamespace(get_sandbox=_returning(sandbox))

        monkeypatch.setattr(verify_sandbox, "build_sandbox_provider", build)
        return requested

    return install


def _returning(sandbox):
    async def get_sandbox(sandbox_id):
        assert sandbox_id == sandbox.sandbox_id
        return sandbox

    return get_sandbox


def _sandbox_context(mode="vm"):
    return TaskStepContext(deployed_sandboxes=[
        DeployedSandbox(sandbox_name="box", sandbox_id="sb-1", sandbox_mode=mode, sandbox_type="fake"),
    ])


def _step(**kwargs):
    return VerifySandboxTaskStep(
        id="verify", version=None, base_dir="/app/greeting", verifier_id="hello",
        criteria=_CRITERIA, **kwargs,
    )


def test_agent_name_and_sandbox_name_are_exclusive():
    with pytest.raises(ValueError, match="either `agent_name` or `sandbox_name`"):
        _step(agent_name="solver", sandbox_name="box")


def test_a_defaulted_agent_name_yields_to_sandbox_name():
    step = VerifySandboxTaskStep.from_dict({
        "id": "verify", "version": None, "agent_name": TaskStep.DEFAULT_AGENT_NAME, "sandbox_name": "box",
    })
    assert (step.agent_name, step.sandbox_name) == (None, "box")


def test_neither_source_defaults_to_the_default_agent():
    step = _step()
    assert (step.agent_name, step.sandbox_name) == (TaskStep.DEFAULT_AGENT_NAME, None)


def test_sandbox_source_round_trips():
    data = _step(sandbox_name="box").to_dict()
    assert (data["agent_name"], data["sandbox_name"]) == (None, "box")
    assert VerifySandboxTaskStep.from_dict(data).to_dict() == data


@pytest.mark.asyncio
async def test_sandbox_source_probes_the_vm_host(provide):
    sandbox = _FakeSandbox("vm")
    requested = provide(sandbox)

    ctx = await _step(sandbox_name="box").execute(_sandbox_context())

    assert requested == ["fake"]
    assert sandbox.calls == [
        ("sudo", "test", "-e", "/app/greeting/hello.txt"),
        ("sudo", "test", "-e", "/etc/hosts"),
        ("sudo", "cat", "/app/greeting/hello.txt"),
        ("sudo", "bash", "-c", "cd /app/greeting && (python3 check.py\n)"),
    ]
    verification = ctx.metadata["verifications"]["hello"]
    assert verification["score"] == 1.0
    assert [r.get("result") for r in verification["results"]] == [True, True, True, None]
    assert verification["results"][3]["skipped"] is True


@pytest.mark.asyncio
async def test_sandbox_source_in_container_mode_runs_probes_as_is(provide):
    sandbox = _FakeSandbox("container")
    provide(sandbox)

    await _step(sandbox_name="box").execute(_sandbox_context("container"))

    assert sandbox.calls[-1] == ("bash", "-c", "cd /app/greeting && (python3 check.py\n)")


@pytest.mark.asyncio
async def test_agent_source_probes_inside_the_agent_container(provide):
    sandbox = _FakeSandbox("vm")
    provide(sandbox)
    context = TaskStepContext(deployed_agents=[
        DeployedAgent(agent_name="solver", api_url="http://agent", sandbox_id="sb-1", sandbox_type="fake"),
    ])

    ctx = await _step(agent_name="solver").execute(context)

    assert sandbox.calls == [
        ("sudo", "docker", "ps", "--format", "{{.Names}}"),
        ("sudo", "docker", "exec", "-u", "0", "agent-api", "test", "-e", "/app/greeting/hello.txt"),
        ("sudo", "docker", "exec", "-u", "0", "agent-api", "test", "-e", "/etc/hosts"),
        ("sudo", "docker", "exec", "-u", "0", "agent-api", "cat", "/app/greeting/hello.txt"),
        ("sudo", "docker", "exec", "-w", "/app/greeting", "agent-api", "bash", "-c", "python3 check.py"),
    ]
    assert ctx.metadata["verifications"]["hello"]["score"] == 1.0


@pytest.mark.asyncio
async def test_a_reattached_local_agent_is_probed_inside_its_own_container(provide, tmp_path, monkeypatch):
    sandbox, calls = await _reattached_local_agent(tmp_path, monkeypatch, running="a2a-agent-other\nagent-local-agent1\n")
    provide(sandbox)

    await _step(agent_name="solver").execute(_local_agent_context())

    assert calls == [
        ("sudo", "docker", "ps", "--format", "{{.Names}}"),
        ("sudo", "docker", "exec", "-u", "0", "agent-local-agent1", "test", "-e", "/app/greeting/hello.txt"),
        ("sudo", "docker", "exec", "-u", "0", "agent-local-agent1", "test", "-e", "/etc/hosts"),
        ("sudo", "docker", "exec", "-u", "0", "agent-local-agent1", "cat", "/app/greeting/hello.txt"),
        ("sudo", "docker", "exec", "-w", "/app/greeting", "agent-local-agent1", "bash", "-c", "python3 check.py"),
    ]


@pytest.mark.asyncio
async def test_a_reattached_local_agent_whose_container_is_gone_fails_rather_than_borrow_another(provide, tmp_path, monkeypatch):
    sandbox, calls = await _reattached_local_agent(tmp_path, monkeypatch, running="a2a-agent-other\n")
    provide(sandbox)

    with pytest.raises(RuntimeError, match="'agent-local-agent1' is not running"):
        await _step(agent_name="solver").execute(_local_agent_context())
    assert calls == [("sudo", "docker", "ps", "--format", "{{.Names}}")]


@pytest.mark.asyncio
async def test_unknown_sandbox_name_names_the_deployed_ones():
    with pytest.raises(RuntimeError, match=r"Sandbox 'nope' not found in context.deployed_sandboxes \(deployed: \['box'\]\)"):
        await _step(sandbox_name="nope").execute(_sandbox_context())


def test_an_agent_path_document_has_no_sandbox_name_key():
    assert "sandbox_name" not in VerifySandboxTaskStep(id="v", version=None, criteria=[]).to_dict()


def test_a_blank_agent_name_counts_as_unset_next_to_sandbox_name():
    step = VerifySandboxTaskStep(id="v", version=None, criteria=[], agent_name="", sandbox_name="box")
    assert (step.sandbox_name, step.agent_name) == ("box", None)


@pytest.mark.asyncio
async def test_a_probe_that_times_out_fails_its_row_and_says_what_timed_out(provide):
    provide(_HangingSandbox("vm"))
    step = VerifySandboxTaskStep(
        id="verify", version=None, base_dir="/app/greeting", verifier_id="hello", sandbox_name="box",
        shell_timeout_seconds=0.01,
        criteria=[
            {"type": "probe_file_contains", "criterion": "greets", "paths": ["hello.txt"], "expected": "x"},
            {"type": "bash_cmd_succeeds", "criterion": "check", "bash_cmd": "true"},
        ],
    )

    ctx = await step.execute(_sandbox_context())

    rows = ctx.metadata["verifications"]["hello"]["results"]
    assert [(r["result"], r["score"], r["justification"]) for r in rows] == [
        (False, 0.0, "timed out reading hello.txt after 0.01s"),
        (False, 0.0, "timed out after 0.01s"),
    ]



@pytest.mark.asyncio
@pytest.mark.parametrize("bash_cmd", ["test -d .  # a trailing comment", "cat <<EOF\nheredoc\nEOF"])
async def test_a_bash_check_may_end_in_a_comment_or_a_heredoc(provide, tmp_path, bash_cmd):
    provide(_BashSandbox())
    step = VerifySandboxTaskStep(
        id="verify", version=None, base_dir=str(tmp_path), verifier_id="hello", sandbox_name="box",
        criteria=[{"type": "bash_cmd_succeeds", "criterion": "check", "bash_cmd": bash_cmd}],
    )

    ctx = await step.execute(_sandbox_context())

    assert ctx.metadata["verifications"]["hello"]["score"] == 1.0


@pytest.mark.parametrize("blank", ["", "  ", "\t\n"])
def test_a_blank_sandbox_name_counts_as_unset(blank):
    step = _step(agent_name="solver", sandbox_name=blank)
    assert (step.agent_name, step.sandbox_name) == ("solver", None)
    assert "sandbox_name" not in step.to_dict()
    assert _step(sandbox_name=blank).agent_name == TaskStep.DEFAULT_AGENT_NAME


@pytest.mark.asyncio
async def test_the_sandbox_is_logged_before_container_discovery_can_fail(provide, caplog):
    class _NoDocker(_FakeSandbox):
        async def exec_with_output(self, *args):
            self.calls.append(args)
            return 1, "", "docker: command not found"

    provide(_NoDocker("vm"))
    context = TaskStepContext(deployed_agents=[
        DeployedAgent(agent_name="solver", api_url="http://agent", sandbox_id="sb-1", sandbox_type="fake"),
    ])

    with caplog.at_level(logging.INFO, logger=verify_sandbox.logger.name), pytest.raises(RuntimeError):
        await _step(agent_name="solver").execute(context)

    assert "Connected to sandbox sb-1 (mode=vm)" in caplog.text


def _local_agent_context():
    return TaskStepContext(deployed_agents=[
        DeployedAgent(agent_name="solver", api_url="http://agent", sandbox_id="local-agent1", sandbox_type="local"),
    ])


async def _reattached_local_agent(tmp_path, monkeypatch, *, running: str):
    """A local agent sandbox rebuilt from its work dir, answering `docker ps` with ``running`` and recording each command."""
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))
    (tmp_path / "agent-env-local-agent1-abc123").mkdir()
    (tmp_path / "agent-env-local-agent1-abc123" / ".agent-container-mode").write_text("agent-local-agent1")
    calls: list[tuple[str, ...]] = []

    async def exec_with_output(self, *args):
        calls.append(args)
        return 0, running if "ps" in args else "hello world\n", ""

    monkeypatch.setattr(LocalSandbox, "exec_with_output", exec_with_output)
    return await LocalSandboxProvider().get_sandbox("local-agent1"), calls
