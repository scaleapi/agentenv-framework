"""Behavior tests for RunCodeTaskStep and its in-sandbox runner: script resolution
(single-file and multi-file), creation-time preflight, output capture, and staging
layout. Only external I/O is mocked — the sandbox, and the artifact store's Mongo
and S3 reads; `_real_run` swaps in a sandbox that actually executes the script."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.artifact import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, EnvNeedsSandbox
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps import run_code as _run_code_module
from tst.unit.event_loop_probe import on_event_loop
from agent_env.task_step.task_steps.run_code import (
    RunCodeExecutionError,
    RunCodeTaskStep,
    RunCodeValidationError,
    _AgentTarget,
)

_RUNNER = Path(_run_code_module.__file__).parent / "run_code_runner.py"


def _make_context(*, with_agent=True, metadata=None):
    # Real internal objects (pure dataclasses) — only external I/O gets mocked.
    agent = DeployedAgent(
        agent_name="default-agent", api_url="http://agent", sandbox_id="sb-agent"
    )
    env = DeployedGatewayEnv(
        env_id="env-x", env_version=1, gateway_url="", mcp_url="",
        db_web_url=None, sandbox_id="sb-env",
    )
    return TaskStepContext(
        deployed_agents=[agent] if with_agent else [],
        deployed_envs=[env],
        metadata={} if metadata is None else metadata,
    )


# Subclassing FileArtifact would register a bogus type in the global artifact registry
# and break the golden-wire-format tests, so the S3 boundary is patched instead.
_DEFAULT_BODY = b"def run(inp):\n    return {}\n"


def _file_artifact(name="script.py", body=_DEFAULT_BODY):
    fa = FileArtifact(
        id=f"fa-{name}", description="demo", filename=name,
        content_type="text/x-python", s3_url=f"s3://bucket/{name}",
    )
    _BODIES[name] = body
    return fa


def _universe(members, *, refs=False):
    """A FileArtifactUniverse whose members are {filename: bytes}.

    `refs=True` populates only file_artifact_refs, the shape a real put() produces
    and the one get_file_artifacts prefers.
    """
    _BODIES.update(members)
    ids = {name: f"fa-{name}" for name in members}
    if refs:
        return FileArtifactUniverse(
            id="fau", version=1,
            file_artifact_refs={n: {"id": i, "version": 1} for n, i in ids.items()},
        )
    return FileArtifactUniverse(id="fau", version=1, file_artifact_ids=ids)


# Filename -> bytes for the patched S3 reads. A missing name must raise, not serve a
# default: a typo'd fixture would otherwise run a working script and pass for the
# wrong reason.
_BODIES: dict[str, bytes] = {}


@pytest.fixture(autouse=True)
def _isolate_bodies():
    _BODIES.clear()
    yield
    _BODIES.clear()


def _load_body(self):
    return _BODIES[self.filename]


def _get_member(artifact_id, version=None):
    """Stand in for FileArtifact.get (Mongo) so the real get_file_artifacts runs."""
    return _file_artifact(artifact_id.removeprefix("fa-"), _BODIES[artifact_id.removeprefix("fa-")])


@contextmanager
def _sandbox(exec_results, *, artifact="single", agent_mode=True):
    """Mock only the external boundary: the Sandbox (VMs/docker), the sandbox
    providers, and the artifact store (`Artifact.get`/`FileArtifact.get`→Mongo,
    `FileArtifact.load`/`put_bytes`→S3). `exec_results` is the ordered list of
    (exit_code, stdout, stderr) returned per exec, in order: the script run, the
    `wc -c` size check, then the `cat` read.

    `artifact` is "single", "wrong-type", or a prebuilt artifact.
    """
    sandbox = AsyncMock()
    sandbox.mode = "vm"
    sandbox.container_name = "agent-api"
    # Before the script runs: reset_dir's `rm -rf` + `mkdir -p` (agent mode only;
    # host mode clears via exec_script).
    prefix = [(0, "", ""), (0, "", "")] if agent_mode else []
    sandbox.exec_with_output = AsyncMock(side_effect=prefix + list(exec_results))
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)

    if artifact == "single":
        artifact = _file_artifact()
    elif artifact == "wrong-type":
        artifact = object()  # neither FileArtifact nor universe → the type guard rejects it

    put = MagicMock(return_value=FileArtifact(
        id="out", version=1, description="captured output", filename="stdout.txt",
        content_type="text/plain", s3_url="s3://bucket/out.txt",
    ))
    member_get = MagicMock(side_effect=_get_member)
    # Tests that care about platform-injected secrets patch this themselves; the rest
    # must not reach for real AWS/secret-store credentials.
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.get_agent_sandbox_provider",
        return_value=provider,
    ), patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.get_env_sandbox_provider",
        return_value=provider,
    ), patch("agent_env.artifact.Artifact.get", return_value=artifact), patch.object(
        FileArtifact, "put_bytes", put
    ), patch.object(FileArtifact, "load", _load_body), patch.object(
        FileArtifact, "get", member_get
    ):
        sandbox.member_get = member_get
        yield sandbox


def _step(**overrides):
    return RunCodeTaskStep(
        **{"id": "run-code-1", "version": None, "script_artifact_id": "fa", **overrides}
    )


class _RealSandbox:
    """Stages to a real directory and really runs `run_code_runner.py`, so tests whose
    subject is "the script runs" can't pass vacuously on canned exec results."""

    mode = "local"  # not "vm" → _AgentTarget uses an empty prefix, no docker exec

    def __init__(self, root):
        self.root = Path(root)

    async def write_file_from_text(self, content, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    async def exec_script(self, script, **_kw):
        subprocess.run(["bash", "-c", script], check=True, capture_output=True)
        return ""

    async def exec_with_output(self, *args):
        args = [a for a in args if a != "sudo"]
        if args[0] == "timeout":  # macOS has no coreutils `timeout`
            args = args[2:]
        proc = subprocess.run(args, capture_output=True)
        return proc.returncode, proc.stdout.decode(), proc.stderr.decode()


@contextmanager
def _real_run(tmp_path, artifact):
    """Like _sandbox, but the staged script actually executes."""
    sandbox = _RealSandbox(tmp_path)
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.get_agent_sandbox_provider",
        return_value=provider,
    ), patch("agent_env.artifact.Artifact.get", return_value=artifact), patch.object(
        FileArtifact, "load", _load_body
    ), patch.object(
        FileArtifact, "get", MagicMock(side_effect=_get_member)
    ), patch.object(_run_code_module, "_RUN_DIR", str(tmp_path)):
        yield sandbox


@pytest.mark.asyncio
async def test_result_is_stored_under_result_id():
    step = _step(args={"threshold": 3}, result_id="filter")
    with _sandbox([(0, "", ""), (0, "22", ""), (0, '{"keep": true, "n": 2}', "")]) as sandbox:
        ctx = await step.execute(_make_context())

    assert ctx.metadata["script_results"]["filter"] == {"keep": True, "n": 2}
    # The script must receive the step's args in its input — there's no outcome
    # proxy for this (the result is whatever the script returns), so check the
    # input we hand it.
    written = [str(c.args[0]) for c in sandbox.write_file_from_text.await_args_list]
    assert any('"threshold": 3' in w for w in written)
    # Agent mode must run the script *inside* the agent container, not on the VM
    # host (regression guard: an earlier version ran it on the host and failed).
    assert any("agent-api" in c.args for c in sandbox.exec_with_output.await_args_list)


@pytest.mark.asyncio
async def test_the_script_is_read_off_the_event_loop():
    on_loop: list[bool] = []

    def load(self):
        on_loop.append(on_event_loop())
        return _load_body(self)

    step = _step()
    with _sandbox([(0, "", ""), (0, "2", ""), (0, "{}", "")]), patch.object(FileArtifact, "load", load):
        await step.execute(_make_context())

    assert on_loop == [False]


@pytest.mark.asyncio
async def test_host_target_runs_with_no_agent_deployed():
    # `env_id` set + zero deployed agents: still succeeds. This is the whole point
    # of the host target — run code without a deploy_agent step (or LLM key).
    step = _step(env_id="env-x", result_id="filter")
    with _sandbox(
        [(0, "", ""), (0, "13", ""), (0, '{"keep": true}', "")], agent_mode=False
    ):
        ctx = await step.execute(_make_context(with_agent=False))

    assert ctx.metadata["script_results"]["filter"] == {"keep": True}


@pytest.mark.asyncio
async def test_host_target_on_an_env_outside_our_sandboxes_says_it_needs_one():
    ctx = _make_context(with_agent=False)
    ctx.deployed_envs = [DeployedEnv(env_id="env-x", env_version=1, env_provider_type="hosted")]
    with pytest.raises(EnvNeedsSandbox, match="^run_code on the env's host needs a sandbox; env 'env-x' runs outside"):
        await _step(env_id="env-x", result_id="filter")._host_target(ctx)


@pytest.mark.asyncio
async def test_host_target_honors_custom_env_sandbox_type():
    # custom sandbox_type must resolve via build_sandbox_provider, not the default provider
    step = _step(env_id="env-x", result_id="filter")
    ctx = _make_context(with_agent=False)
    ctx.deployed_envs[0].sandbox_type = "arp"

    sandbox = AsyncMock()
    sandbox.mode = "vm"
    sandbox.container_name = "agent-api"
    sandbox.exec_with_output = AsyncMock(
        # leading entry is the `env` read execute() does for redaction
        # host mode clears via exec_script, so exec_with_output sees only:
        # the script run, `wc -c`, then `cat`.
        side_effect=[(0, "", ""), (0, "13", ""), (0, '{"keep": true}', "")]
    )
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    artifact = _file_artifact()
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider",
        return_value=provider,
    ) as build, patch("agent_env.artifact.Artifact.get", return_value=artifact), patch.object(
        FileArtifact, "load", _load_body
    ):
        await step.execute(ctx)

    build.assert_called_once_with("arp")
    assert ctx.metadata["script_results"]["filter"] == {"keep": True}


def _sandbox_context(*, sandbox_type=None):
    ctx = _make_context(with_agent=False)
    ctx.deployed_sandboxes = [
        DeployedSandbox(sandbox_name="bundler", sandbox_id="sb-bundler", sandbox_mode="vm", sandbox_type=sandbox_type)
    ]
    return ctx


def _host_sandbox():
    sandbox = AsyncMock()
    sandbox.mode = "vm"
    sandbox.exec_with_output = AsyncMock(
        side_effect=[(0, "", ""), (0, "13", ""), (0, '{"keep": true}', "")]
    )
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    return sandbox, provider


@pytest.mark.asyncio
async def test_sandbox_target_runs_on_the_named_sandboxs_host():
    step = _step(sandbox_name="bundler", result_id="filter")
    sandbox, provider = _host_sandbox()
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider",
        return_value=provider,
    ), patch("agent_env.artifact.Artifact.get", return_value=_file_artifact()), patch.object(
        FileArtifact, "load", _load_body
    ):
        ctx = await step.execute(_sandbox_context())

    provider.get_sandbox.assert_awaited_once_with("sb-bundler")
    commands = [" ".join(map(str, c.args)) for c in sandbox.exec_with_output.await_args_list]
    assert not any("docker" in c for c in commands)
    sandbox.write_file_from_text.assert_not_awaited()
    written = [c.args[1].rsplit("/", 1)[-1] for c in sandbox.write_host_file.await_args_list]
    assert {"input.json", "runner.py"} <= set(written)
    assert ctx.metadata["script_results"]["filter"] == {"keep": True}


@pytest.mark.asyncio
async def test_sandbox_target_honors_the_sandboxs_type():
    step = _step(sandbox_name="bundler", result_id="filter")
    _, provider = _host_sandbox()
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider",
        return_value=provider,
    ) as build, patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider",
    ) as default, patch("agent_env.artifact.Artifact.get", return_value=_file_artifact()), patch.object(
        FileArtifact, "load", _load_body
    ):
        ctx = await step.execute(_sandbox_context(sandbox_type="modal_vm"))

    build.assert_called_once_with("modal_vm")
    default.assert_not_called()
    assert ctx.metadata["script_results"]["filter"] == {"keep": True}


@pytest.mark.asyncio
async def test_sandbox_target_names_a_sandbox_the_run_never_deployed():
    with pytest.raises(RuntimeError, match="^Sandbox 'ghost' not found in context.deployed_sandboxes$"):
        await _step(sandbox_name="ghost")._sandbox_target(_sandbox_context())


def _container_sandbox(*, vm_backed):
    sandbox = AsyncMock(spec=VmSandbox) if vm_backed else AsyncMock()
    sandbox.mode = "container"
    sandbox.container_name = "agent-local-1"
    # reset_dir's `rm -rf` + `mkdir -p`, then the script run, `wc -c`, then `cat`.
    sandbox.exec_with_output = AsyncMock(
        side_effect=[(0, "", ""), (0, "", ""), (0, "", ""), (0, "13", ""), (0, '{"keep": true}', "")]
    )
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    return sandbox, provider


@pytest.mark.asyncio
@pytest.mark.parametrize("vm_backed,prefix", [
    (False, ()),
    (True, ("sudo", "docker", "exec", "-u", "0", "agent-local-1")),
])
async def test_sandbox_target_runs_inside_a_container_mode_sandbox(vm_backed, prefix):
    step = _step(sandbox_name="bundler", result_id="filter")
    ctx = _sandbox_context()
    ctx.deployed_sandboxes[0].sandbox_mode = "container"
    sandbox, provider = _container_sandbox(vm_backed=vm_backed)
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider",
        return_value=provider,
    ), patch("agent_env.artifact.Artifact.get", return_value=_file_artifact()), patch.object(
        FileArtifact, "load", _load_body
    ):
        ctx = await step.execute(ctx)

    calls = [c.args for c in sandbox.exec_with_output.await_args_list]
    assert all(c[: len(prefix)] == prefix for c in calls)
    assert any("runner.py" in " ".join(c) for c in calls)
    sandbox.write_host_file.assert_not_awaited()
    written = [c.args[1].rsplit("/", 1)[-1] for c in sandbox.write_file_from_text.await_args_list]
    assert {"input.json", "runner.py"} <= set(written)
    assert ctx.metadata["script_results"]["filter"] == {"keep": True}


@pytest.mark.parametrize("targets", [
    {"env_id": "env-x", "sandbox_name": "bundler"},
    {"env_id": "env-x", "agent_name": "solver"},
    {"sandbox_name": "bundler", "agent_name": "solver"},
    {"sandbox_name": "bundler", "agent_name": RunCodeTaskStep.DEFAULT_AGENT_NAME},
    {"env_id": "env-x", "sandbox_name": "bundler", "agent_name": "solver"},
])
def test_the_three_targets_are_mutually_exclusive(targets):
    with pytest.raises(ValueError, match="set at most one of `env_id`, `sandbox_name` and `agent_name`"):
        _step(**targets)


@pytest.mark.parametrize("target", [{"env_id": "env-x"}, {"sandbox_name": "bundler"}])
@pytest.mark.parametrize("blank", ["", None])
def test_a_blank_agent_name_beside_a_host_target_is_unset(target, blank):
    step = _step(**target, agent_name=blank)
    assert step.agent_name is None
    assert RunCodeTaskStep.from_dict({**step.to_dict(), "agent_name": blank}).to_dict() == step.to_dict()


def test_blank_targets_fall_back_to_the_default_agent():
    step = _step(env_id="", sandbox_name="", agent_name="")
    assert (step.env_id, step.sandbox_name, step.agent_name) == (None, None, RunCodeTaskStep.DEFAULT_AGENT_NAME)


@pytest.mark.parametrize("target", [{"env_id": "env-x"}, {"sandbox_name": "bundler"}])
def test_a_host_target_stores_no_agent_name(target):
    assert _step(**target).to_dict()["agent_name"] is None


def test_no_target_runs_in_the_default_agent():
    assert _step().agent_name == RunCodeTaskStep.DEFAULT_AGENT_NAME


@pytest.mark.parametrize("target", [{"env_id": "env-x"}, {"sandbox_name": "bundler"}])
def test_a_stored_step_with_the_default_agent_name_beside_its_target_still_loads(target):
    stored = {**_step(**target).to_dict(), "agent_name": RunCodeTaskStep.DEFAULT_AGENT_NAME}
    step = RunCodeTaskStep.from_dict(stored)
    assert step.agent_name is None
    assert {k: getattr(step, k) for k in target} == target


def test_a_stored_step_with_a_named_agent_beside_its_target_is_rejected():
    stored = {**_step(env_id="env-x").to_dict(), "agent_name": "solver"}
    with pytest.raises(ValueError, match="set at most one of"):
        RunCodeTaskStep.from_dict(stored)


@pytest.mark.asyncio
async def test_agent_target_honors_the_agents_sandbox_type():
    # an agent deployed on another provider is reached there, not through the configured default
    step = _step(result_id="filter")
    ctx = _make_context()
    ctx.deployed_agents[0].sandbox_type = "modal"

    sandbox = AsyncMock()
    sandbox.mode = "vm"
    sandbox.container_name = "agent-api"
    sandbox.exec_with_output = AsyncMock(
        # the `env` read execute() does for redaction, the clear, the script run, `wc -c`, then `cat`
        side_effect=[(0, "", ""), (0, "", ""), (0, "", ""), (0, "13", ""), (0, '{"keep": true}', "")]
    )
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider",
        return_value=provider,
    ) as build, patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.get_agent_sandbox_provider",
    ) as default, patch("agent_env.artifact.Artifact.get", return_value=_file_artifact()), patch.object(
        FileArtifact, "load", _load_body
    ):
        await step.execute(ctx)

    build.assert_called_once_with("modal")
    default.assert_not_called()
    assert ctx.metadata["script_results"]["filter"] == {"keep": True}


@pytest.mark.asyncio
async def test_a_reattached_local_agent_runs_the_script_inside_its_container(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))
    (tmp_path / "agent-env-local-agent1-abc123").mkdir()
    (tmp_path / "agent-env-local-agent1-abc123" / ".agent-container-mode").write_text("agent-local-agent1")
    ctx = _make_context()
    ctx.deployed_agents[0].sandbox_id, ctx.deployed_agents[0].sandbox_type = "local-agent1", "local"

    target = await _step()._agent_target(ctx)

    assert target.prefix == ("sudo", "docker", "exec", "-u", "0", "agent-local-agent1")


@pytest.mark.asyncio
async def test_host_installed_agent_on_a_vm_runs_on_the_host():
    step = _step(result_id="filter")
    ctx = _make_context()
    ctx.deployed_agents[0].sandbox_type = "modal_vm"
    ctx.deployed_agents[0].on_host = True

    sandbox = AsyncMock()
    sandbox.mode = "vm"
    sandbox.container_name = "agent-api"
    sandbox.exec_with_output = AsyncMock(
        side_effect=[(0, "", ""), (0, "13", ""), (0, '{"keep": true}', "")]
    )
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider",
        return_value=provider,
    ), patch("agent_env.artifact.Artifact.get", return_value=_file_artifact()), patch.object(
        FileArtifact, "load", _load_body
    ):
        await step.execute(ctx)

    commands = [" ".join(map(str, c.args)) for c in sandbox.exec_with_output.await_args_list]
    commands += [str(c.args[0]) for c in sandbox.exec_script.await_args_list]
    assert not any("docker" in c for c in commands)
    sandbox.write_file_from_text.assert_not_awaited()
    written = [c.args[1].rsplit("/", 1)[-1] for c in sandbox.write_host_file.await_args_list]
    assert {"input.json", "runner.py"} <= set(written)
    assert ctx.metadata["script_results"]["filter"] == {"keep": True}


@pytest.mark.asyncio
async def test_timeout_is_reported():
    with _sandbox([(124, "", "")]):  # `timeout` coreutil exit code
        with pytest.raises(RunCodeExecutionError, match="timed out"):
            await _step(timeout_seconds=5).execute(_make_context())


@pytest.mark.asyncio
async def test_script_crash_surfaces_stderr():
    with _sandbox([(1, "", "Traceback: boom")]):
        with pytest.raises(RunCodeExecutionError, match="boom"):
            await _step().execute(_make_context())


class _LocalExec:
    """A sandbox stand-in that runs commands against the real filesystem, decoding
    strictly like the production sandbox (providers/sandbox.py). Lets the size/read
    path run real `wc`/`cat` against real files instead of a canned mock string."""

    mode = "local"  # not "vm" → _AgentTarget uses an empty prefix (no docker exec)

    async def exec_with_output(self, *args):
        proc = subprocess.run(list(args), capture_output=True)
        return proc.returncode, proc.stdout.decode(), proc.stderr.decode()


@pytest.mark.asyncio
async def test_oversized_result_is_rejected(tmp_path):
    # Real file + real `wc -c`: an over-cap result is rejected on byte size before
    # any content is read. The payload is raw (non-ASCII) UTF-8 — the size check
    # must not depend on the runner's json.dump escaping (ensure_ascii). The old
    # `head -c cap+1` read would have sliced this mid-codepoint and crashed strict
    # UTF-8 decode instead of reporting "exceeds".
    (tmp_path / "output.json").write_text(
        json.dumps({"blob": "é" * 500}, ensure_ascii=False), encoding="utf-8"
    )
    step = _step()
    step.MAX_OUTPUT_BYTES = 50  # below the file's byte size
    target = _AgentTarget(_LocalExec(), prefix=())
    with pytest.raises(RunCodeValidationError, match="exceeds"):
        await step._read_output(target, str(tmp_path))


@pytest.mark.asyncio
async def test_read_output_parses_real_multibyte_file(tmp_path):
    # The cat+parse path round-trips raw multibyte UTF-8 JSON without decode errors.
    payload = {"ok": True, "txt": "café ☕"}
    (tmp_path / "output.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    target = _AgentTarget(_LocalExec(), prefix=())
    assert await _step()._read_output(target, str(tmp_path)) == payload


@pytest.mark.asyncio
async def test_non_json_result_is_rejected():
    with _sandbox([(0, "", ""), (0, "8", ""), (0, "not json", "")]):
        with pytest.raises(RunCodeValidationError, match="not valid JSON"):
            await _step().execute(_make_context())


@pytest.mark.asyncio
async def test_missing_agent_is_a_clear_error():
    with _sandbox([]):
        with pytest.raises(RuntimeError, match="not found in context.deployed_agents"):
            await _step(agent_name="ghost").execute(_make_context())


@pytest.mark.asyncio
async def test_non_file_artifact_is_rejected():
    with _sandbox([], artifact="wrong-type"):
        with pytest.raises(RunCodeValidationError, match="must be a FileArtifact or FileArtifactUniverse"):
            await _step().execute(_make_context())


# --- run_code_runner (the in-sandbox bootstrap) ---


def test_runner_calls_entrypoint_with_input_and_writes_result(tmp_path):
    (tmp_path / "script.py").write_text(
        "def run(inp):\n    return {'doubled': inp['args']['n'] * 2}\n"
    )
    (tmp_path / "input.json").write_text(
        json.dumps({"args": {"n": 21}, "results": {}})
    )
    subprocess.run(
        [sys.executable, str(_RUNNER), str(tmp_path), "run", "script.py"], check=True
    )
    assert json.loads((tmp_path / "output.json").read_text()) == {"doubled": 42}


def test_runner_exits_nonzero_when_entrypoint_missing(tmp_path):
    # execute() relies on a nonzero exit to surface the failure as RunCodeExecutionError.
    (tmp_path / "script.py").write_text("def run(inp):\n    return {}\n")
    (tmp_path / "input.json").write_text("{}")
    result = subprocess.run(
        [sys.executable, str(_RUNNER), str(tmp_path), "nope", "script.py"], capture_output=True
    )
    assert result.returncode != 0


def test_runner_imports_sibling_modules_from_the_run_dir(tmp_path):
    (tmp_path / "helpers.py").write_text("VALUE = {'from': 'helper'}\n")
    (tmp_path / "run.py").write_text(
        "import helpers\ndef run(inp):\n    return helpers.VALUE\n"
    )
    (tmp_path / "input.json").write_text(json.dumps({"args": {}, "results": {}}))
    subprocess.run(
        [sys.executable, str(_RUNNER), str(tmp_path), "run", "run.py"], check=True
    )
    assert json.loads((tmp_path / "output.json").read_text()) == {"from": "helper"}


def test_config_round_trips_through_to_dict_from_dict():
    step = _step(
        entrypoint="filter_prs",
        script_artifact_version=3,
        script_file="run.py",
        args={"repo": "x"},
        result_id="r1",
        timeout_seconds=120,
        env_id="env-x",
        depends_on=[{"task_step_id": "deploy-env-1"}],
    )
    d = step.to_dict()
    assert d["type"] == "run_code"
    assert d["env_id"] == "env-x"  # the field whose drop broke the first host run
    assert RunCodeTaskStep.from_dict(d).to_dict() == d


def test_sandbox_name_round_trips_through_to_dict_from_dict():
    d = _step(sandbox_name="bundler").to_dict()
    assert d["sandbox_name"] == "bundler"
    assert RunCodeTaskStep.from_dict(d).to_dict() == d


# --- multi-file script artifacts (FileArtifactUniverse) ---


@pytest.mark.asyncio
async def test_universe_with_one_py_member_runs():
    universe = _universe({"run.py": b"def run(inp):\n    return {'ok': True}\n"})
    with _sandbox(
        [(0, "", ""), (0, "13", ""), (0, '{"ok": true}', "")], artifact=universe
    ) as sandbox:
        ctx = await _step(result_id="r").execute(_make_context())

    assert ctx.metadata["script_results"]["r"] == {"ok": True}
    staged = [str(c.args[1]) for c in sandbox.write_file_from_text.await_args_list]
    assert any(p.endswith("/run.py") for p in staged)
    assert any("run.py" in c.args for c in sandbox.exec_with_output.await_args_list)


@pytest.mark.asyncio
async def test_universe_members_are_staged_so_the_entry_imports_its_siblings(tmp_path):
    # Only reachable if helpers.py was staged AND importable from the run dir.
    universe = _universe({
        "run.py": b"import helpers\ndef run(inp):\n    return helpers.value()\n",
        "helpers.py": b"def value():\n    return {'from': 'helper'}\n",
        "screenshot.png": b"\x89PNG\r\n\x1a\n\xff\xfe binary",
    })
    with _real_run(tmp_path, universe):
        ctx = await _step(result_id="r").execute(_make_context())

    assert ctx.metadata["script_results"]["r"] == {"from": "helper"}
    # the binary member is skipped, so it never reaches the run dir or a UTF-8 decode
    assert not (tmp_path / "run-code-1" / "screenshot.png").exists()


@pytest.mark.asyncio
async def test_script_file_selects_among_several_py_members():
    universe = _universe({
        "run.py": b"def run(inp):\n    return {'which': 'run'}\n",
        "other.py": b"def run(inp):\n    return {'which': 'other'}\n",
    })
    with _sandbox(
        [(0, "", ""), (0, "18", ""), (0, '{"which": "other"}', "")], artifact=universe
    ) as sandbox:
        await _step(script_file="other.py").execute(_make_context())

    assert any("other.py" in c.args for c in sandbox.exec_with_output.await_args_list)
    assert not any("run.py" in c.args for c in sandbox.exec_with_output.await_args_list)


@pytest.mark.asyncio
async def test_ambiguous_universe_names_the_candidates():
    universe = _universe({"a.py": b"", "b.py": b""})
    with _sandbox([], artifact=universe):
        with pytest.raises(RunCodeValidationError, match=r"set script_file to one of.*a\.py"):
            await _step().execute(_make_context())


@pytest.mark.asyncio
async def test_several_run_py_says_so_rather_than_reporting_none():
    """Several run.py is a different problem from no run.py, and must not be described as none.

    Both land on "ambiguous, set script_file", but "no single 'run.py'" reads as "there is no
    run.py" and sends the author looking for a file that is not missing. A real universe with
    per-layout subdirectories hits this immediately.
    """
    universe = _universe(
        {"pkg_a/run.py": b"", "pkg_b/run.py": b"", "pkg_a/helpers.py": b""}
    )
    with _sandbox([], artifact=universe):
        with pytest.raises(RunCodeValidationError) as exc:
            await _step().execute(_make_context())

    message = str(exc.value)
    assert "2 files named 'run.py'" in message
    assert "pkg_a/run.py" in message and "pkg_b/run.py" in message
    assert "no 'run.py'" not in message, "several run.py must not be reported as none"


@pytest.mark.asyncio
async def test_unknown_script_file_lists_members():
    universe = _universe({"run.py": b"", "helpers.py": b""})
    with _sandbox([], artifact=universe):
        with pytest.raises(RunCodeValidationError, match="not a .py member"):
            await _step(script_file="nope.py").execute(_make_context())


@pytest.mark.asyncio
async def test_member_shadowing_staging_file_is_rejected():
    # A member called runner.py would overwrite the bootstrap we stage next to it.
    universe = _universe({"run.py": b"", "runner.py": b""})
    with _sandbox([], artifact=universe):
        with pytest.raises(RunCodeValidationError, match="shadow"):
            await _step().execute(_make_context())


# --- preflight: catch bad config before a run, not mid-run ---


def test_preflight_rejects_wrong_artifact_type():
    with _sandbox([], artifact="wrong-type"):
        problems = _step().preflight()
    assert len(problems) == 1
    assert "must be a FileArtifact or FileArtifactUniverse" in problems[0]


def test_preflight_reports_a_missing_artifact():
    from agent_env.store.base import NotFoundError

    with patch("agent_env.artifact.Artifact.get", side_effect=NotFoundError("nope")):
        problems = _step().preflight()
    assert len(problems) == 1
    assert "does not exist" in problems[0]


def test_preflight_lets_infrastructure_errors_propagate():
    # An expired credential is not a broken task config.
    with patch(
        "agent_env.artifact.Artifact.get", side_effect=RuntimeError("NoCredentialsError")
    ):
        with pytest.raises(RuntimeError, match="NoCredentialsError"):
            _step().preflight()


def test_preflight_passes_for_a_usable_artifact():
    with _sandbox([]):
        assert _step().preflight() == []


def test_preflight_does_not_download_the_script():
    # It must stay cheap enough to run over a whole task: documents only, no S3.
    def _explode(self):
        raise AssertionError("preflight must not load script content")

    with patch("agent_env.artifact.Artifact.get", return_value=_file_artifact()), patch.object(
        FileArtifact, "load", _explode
    ):
        assert _step().preflight() == []


def test_task_preflight_aggregates_across_steps():
    from agent_env.task import Task

    good, bad = _step(id="ok"), _step(id="broken")
    with _sandbox([], artifact="wrong-type"):
        problems = Task(id="t", version=None, steps=[good, bad]).preflight()
    # both steps point at the same bad artifact, so both report
    assert len(problems) == 2
    assert all("must be a FileArtifact or FileArtifactUniverse" in p for p in problems)


# --- staging layout ---


@pytest.mark.asyncio
async def test_single_file_artifact_runs_whatever_its_filename(tmp_path):
    # importlib returns spec=None for an unknown suffix, so staging under the
    # artifact's own filename cannot execute. Runs for real: the result is the proof.
    artifact = _file_artifact("notes.txt", b"def run(inp):\n    return {'ok': 1}\n")
    with _real_run(tmp_path, artifact):
        ctx = await _step(result_id="r").execute(_make_context())

    assert ctx.metadata["script_results"]["r"] == {"ok": 1}


@pytest.mark.asyncio
async def test_single_file_named_runner_py_does_not_shadow_the_bootstrap(tmp_path):
    # Staged under its own name it would be overwritten by the bootstrap and then
    # handed to itself as the entry, recursing until the interpreter gives up.
    artifact = _file_artifact("runner.py", b"def run(inp):\n    return {'ok': 2}\n")
    with _real_run(tmp_path, artifact):
        ctx = await _step(result_id="r").execute(_make_context())

    assert ctx.metadata["script_results"]["r"] == {"ok": 2}


@pytest.mark.asyncio
async def test_script_file_is_rejected_for_a_single_file_artifact():
    with _sandbox([]):
        with pytest.raises(RunCodeValidationError, match="single-file FileArtifact"):
            await _step(script_file="run.py").execute(_make_context())


@pytest.mark.asyncio
async def test_universe_over_the_staging_cap_is_rejected():
    universe = _universe({"run.py": b"x" * 64})
    step = _step()
    step.MAX_SCRIPT_BYTES = 10
    with _sandbox([], artifact=universe):
        with pytest.raises(RunCodeValidationError, match="staging cap"):
            await step.execute(_make_context())


@pytest.mark.asyncio
async def test_member_escaping_the_staging_dir_is_rejected():
    # Member names are unvalidated on write and staging writes as root.
    universe = _universe({"run.py": b"", "../../etc/evil.py": b""})
    with _sandbox([], artifact=universe):
        with pytest.raises(RunCodeValidationError, match="escape the staging directory"):
            await _step().execute(_make_context())


def test_preflight_uses_refs_when_ids_are_absent():
    # get_file_artifacts prefers refs, so preflight must read refs too, or it
    # validates names execute never stages. The real method runs here.
    universe = _universe({"run.py": _DEFAULT_BODY}, refs=True)
    with _sandbox([], artifact=universe) as sandbox:
        assert _step().preflight() == []
    # and it stayed cheap: no member document was resolved
    sandbox.member_get.assert_not_called()


def test_from_dict_without_script_file_still_loads():
    # Tasks stored before script_file existed.
    stored = {
        "id": "c4", "type": "run_code", "version": None, "fail_task_on_error": True,
        "script_artifact_id": "fa", "entrypoint": "run", "args": {},
        "result_id": "c4", "timeout_seconds": 600, "env_id": None,
        "agent_name": "default-agent",
    }
    step = RunCodeTaskStep.from_dict(stored)
    assert step.script_file is None
    assert step.to_dict()["script_file"] is None


@pytest.mark.asyncio
async def test_a_stale_run_dir_does_not_leak_into_this_run(tmp_path):
    # Sandboxes are reused, and run_dir is on sys.path: a leftover sibling module
    # would silently satisfy an import, and a leftover output.json would be read
    # back as this run's result.
    run_dir = tmp_path / "run-code-1"
    run_dir.mkdir()
    (run_dir / "helpers.py").write_text("def value():\n    return {'stale': True}\n")
    (run_dir / "output.json").write_text('{"stale": true}')

    artifact = _file_artifact(
        "script.py", b"import helpers\ndef run(inp):\n    return helpers.value()\n"
    )
    with _real_run(tmp_path, artifact):
        with pytest.raises(RunCodeExecutionError):
            await _step(result_id="r").execute(_make_context())


@pytest.mark.asyncio
async def test_a_nested_entry_imports_its_siblings(tmp_path):
    # run.py is matched by basename, so a universe can be laid out under a directory.
    # run_dir alone on sys.path does not resolve `import helpers` from src/run.py.
    universe = _universe({
        "src/run.py": b"import helpers\ndef run(inp):\n    return helpers.V\n",
        "src/helpers.py": b"V = {'nested': True}\n",
    })
    with _real_run(tmp_path, universe):
        ctx = await _step(result_id="r").execute(_make_context())

    assert ctx.metadata["script_results"]["r"] == {"nested": True}


@pytest.mark.asyncio
async def test_a_nested_entry_can_use_a_package_relative_import(tmp_path):
    # Importing the entry from its file path alone makes it top-level, and a relative
    # import then fails however the siblings are staged. A dotted import gives it the
    # package context. No __init__.py: run_dir on sys.path makes pkg a namespace package.
    universe = _universe({
        "pkg/run.py": b"from .helpers import value\ndef run(inp):\n    return value()\n",
        "pkg/helpers.py": b"def value():\n    return {'relative': True}\n",
    })
    with _real_run(tmp_path, universe):
        ctx = await _step(result_id="r", script_file="pkg/run.py").execute(_make_context())

    assert ctx.metadata["script_results"]["r"] == {"relative": True}


@pytest.mark.asyncio
async def test_a_directory_that_is_not_a_module_name_still_runs(tmp_path):
    # "my-scripts" is a legal member name and an illegal module name, which is why the
    # package is bound to a name of ours rather than to the directory's.
    universe = _universe({
        "my-scripts/run.py": b"from .helpers import V\ndef run(inp):\n    return V\n",
        "my-scripts/helpers.py": b"V = {'fallback': True}\n",
    })
    with _real_run(tmp_path, universe):
        ctx = await _step(result_id="r").execute(_make_context())

    assert ctx.metadata["script_results"]["r"] == {"fallback": True}


@pytest.mark.asyncio
async def test_a_directory_named_after_a_stdlib_module_still_runs(tmp_path):
    # The runner imports `os` itself, so resolving the entry through the directory's own
    # name would find that instead of the staged member and fail before the script runs.
    universe = _universe({
        "os/run.py": b"from .helpers import V\ndef run(inp):\n    return V\n",
        "os/helpers.py": b"V = {'shadowed': True}\n",
    })
    with _real_run(tmp_path, universe):
        ctx = await _step(result_id="r").execute(_make_context())

    assert ctx.metadata["script_results"]["r"] == {"shadowed": True}


