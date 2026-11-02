"""Behavior tests for RunCodeTaskStep and its in-sandbox runner: script resolution
(single-file and multi-file), creation-time preflight, output capture, and staging
layout. Only external I/O is mocked — the sandbox, and the artifact store's Mongo
and S3 reads; `_real_run` swaps in a sandbox that actually executes the script."""

from __future__ import annotations

import ast
import json
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_env.artifact import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, DeployedSandboxEnv, EnvNeedsSandbox
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandbox
from agent_env.providers.sandbox_providers.sandbox import Sandbox, VmSandbox
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps import run_code as _run_code_module
from tst.unit.computer_home import CARD, TOKEN, Computer
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


def _vm_sandbox_mock():
    """A VM sandbox at the boundary; tunnels and host ports are plain attributes of a real one, which reaches its own
    URLs as the base class does."""
    sandbox = AsyncMock(spec=VmSandbox)
    sandbox.mode = "vm"
    sandbox.container_name = "agent-api"
    sandbox.tunnel_urls = {}
    sandbox.host_ips = ()
    sandbox.host_port = MagicMock(side_effect=lambda port: port)
    sandbox.scoped_name = MagicMock(side_effect=lambda name: name)
    sandbox.url_on_host = MagicMock(side_effect=lambda url: Sandbox.url_on_host(sandbox, url))
    return sandbox


@contextmanager
def _sandbox(exec_results, *, artifact="single", agent_mode=True):
    """Mock only the external boundary: the Sandbox (VMs/docker), the sandbox
    providers, and the artifact store (`Artifact.get`/`FileArtifact.get`→Mongo,
    `FileArtifact.load`/`put_bytes`→S3). `exec_results` is the ordered list of
    (exit_code, stdout, stderr) returned per exec, in order: the script run, the
    `wc -c` size check, then the `cat` read.

    `artifact` is "single", "wrong-type", or a prebuilt artifact.
    """
    sandbox = _vm_sandbox_mock()
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


class _ContainerEnvSandbox(Sandbox):
    """A container sandbox, as Modal's is, whose commands really run here; it has no host to stage on or run on."""

    type = "modal"
    mode = "container"

    def __init__(self, tunnel_urls):
        self.sandbox_id = "sb-env"
        self.tunnel_urls = tunnel_urls
        self.vnc_url = None
        self.commands = []

    async def terminate(self):
        pass

    async def write_file_from_text(self, content, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)

    async def exec_with_output(self, *args):
        self.commands.append(args)
        proc = subprocess.run(list(args), capture_output=True)
        return proc.returncode, proc.stdout.decode(), proc.stderr.decode()


_CARD_URL = "https://abc.modal.host/.well-known/agent-env.json"


@pytest.mark.parametrize("deployed,expected", [
    (DeployedGatewayEnv(env_id="gw", env_version=1, sandbox_id="sb", gateway_url="https://abc.modal.host",
                        environment_card_url=_CARD_URL, environment_card={"name": "gw"}),
     {"id": "gw", "url": "http://127.0.0.1:18765", "mcp_url": "http://127.0.0.1:18765/mcp"}),
    (DeployedSandboxEnv(env_id="srv", env_version=1, sandbox_id="sb", env_provider_type="server",
                        environment_card_url=_CARD_URL, environment_card={"name": "srv"}),
     {"id": "srv", "url": "http://127.0.0.1:18765", "mcp_url": "http://127.0.0.1:18765/mcp"}),
    # a plugin's env, whose card puts its MCP endpoint elsewhere
    (DeployedSandboxEnv(env_id="plug", env_version=1, sandbox_id="sb", env_provider_type="a-plugins",
                        environment_card_url=_CARD_URL,
                        environment_card={"name": "plug",
                                          "additionalInterfaces": [{"transport": "mcp", "url": "/v2/mcp"}]}),
     {"id": "plug", "url": "http://127.0.0.1:18765", "mcp_url": "http://127.0.0.1:18765/v2/mcp"}),
    # a record from before the card, which kept only its MCP URL
    (DeployedSandboxEnv(env_id="old", env_version=1, sandbox_id="sb", mcp_url="https://abc.modal.host/mcp"),
     {"id": "old", "url": None, "mcp_url": "http://127.0.0.1:18765/mcp"}),
])
def test_an_envs_address_and_mcp_endpoint_come_from_its_card_whatever_deployed_it(deployed, expected):
    sandbox = _ContainerEnvSandbox({18765: "https://abc.modal.host"})

    assert _run_code_module._env_on_host(deployed, sandbox) == expected


def _written_input(sandbox):
    """The input.json the step staged, through whichever writer its target uses."""
    for call in sandbox.write_host_file.await_args_list:
        if str(call.args[1]).endswith("/input.json"):
            return json.loads(call.args[0].decode())
    for call in sandbox.write_file_from_text.await_args_list:
        if str(call.args[1]).endswith("/input.json"):
            return json.loads(call.args[0])
    raise AssertionError("no input.json was staged")


@pytest.mark.asyncio
async def test_a_script_on_an_envs_host_gets_the_envs_address_and_mcp_endpoint_as_that_host_reaches_them():
    ctx = _make_context(with_agent=False)
    ctx.deployed_envs[0] = DeployedGatewayEnv(
        env_id="env-x", env_version=1, sandbox_id="sb-env", gateway_url="https://abc.modal.host",
        environment_card_url=_CARD_URL, environment_card={"name": "env-x"},
    )
    with _sandbox([(0, "", ""), (0, "2", ""), (0, "{}", "")], agent_mode=False) as sandbox:
        sandbox.tunnel_urls = {18765: "https://abc.modal.host"}
        await _step(env_id="env-x").execute(ctx)

    assert _written_input(sandbox)["env"] == {
        "id": "env-x", "url": "http://127.0.0.1:18765", "mcp_url": "http://127.0.0.1:18765/mcp",
    }


@pytest.mark.asyncio
async def test_a_script_for_an_env_in_a_container_sandbox_runs_in_that_container_and_reaches_the_env_there(tmp_path):
    sandbox = _ContainerEnvSandbox({18765: "https://abc.modal.host"})
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    ctx = _make_context(with_agent=False)
    ctx.deployed_envs = [DeployedSandboxEnv(
        env_id="env-x", env_version=1, sandbox_id="sb-env", env_provider_type="server",
        environment_card_url=_CARD_URL, environment_card={"name": "srv"},
    )]
    artifact = _file_artifact(body=b"def run(inp):\n    return inp['env']\n")
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.get_env_sandbox_provider", return_value=provider,
    ), patch("agent_env.artifact.Artifact.get", return_value=artifact), patch.object(
        FileArtifact, "load", _load_body
    ), patch.object(_run_code_module, "_RUN_DIR", str(tmp_path)):
        ctx = await _step(env_id="env-x").execute(ctx)

    assert ctx.metadata["script_results"]["run-code-1"] == {
        "id": "env-x", "url": "http://127.0.0.1:18765", "mcp_url": "http://127.0.0.1:18765/mcp",
    }
    assert not any(command[0] in ("sudo", "docker") for command in sandbox.commands)


@pytest.mark.asyncio
async def test_a_script_in_an_agent_gets_no_env():
    with _sandbox([(0, "", ""), (0, "2", ""), (0, "{}", "")]) as sandbox:
        await _step().execute(_make_context())

    assert set(_written_input(sandbox)) == {"args", "results"}


@pytest.mark.asyncio
async def test_runs_of_one_step_on_this_machine_stage_into_dirs_of_their_own(tmp_path):
    """Local sandboxes share this machine's /tmp: a run of the same step on another must not clear this one's files."""
    sandboxes = {}
    for sid in ("local-aaaa1111", "local-bbbb2222"):
        (tmp_path / sid).mkdir()
        sandboxes[sid] = LocalSandbox(sandbox_id=sid, work_dir=tmp_path / sid)
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(side_effect=lambda sid: sandboxes[sid])
    artifact = _file_artifact(body=b"def run(inp):\n    return inp['args']\n")
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider", return_value=provider,
    ), patch("agent_env.artifact.Artifact.get", return_value=artifact), patch.object(
        FileArtifact, "load", _load_body
    ), patch.object(_run_code_module, "_RUN_DIR", str(tmp_path / "run")):
        for sid in sandboxes:
            ctx = _make_context(with_agent=False)
            ctx.deployed_sandboxes = [
                DeployedSandbox(sandbox_name="box", sandbox_id=sid, sandbox_mode="vm", sandbox_type="local")
            ]
            await _step(sandbox_name="box", args={"who": sid}).execute(ctx)
            assert ctx.metadata["script_results"]["run-code-1"] == {"who": sid}

    for sid in sandboxes:
        staged = json.loads((tmp_path / f"run-{sid}" / "run-code-1" / "input.json").read_text())
        assert staged["args"] == {"who": sid}


@pytest.mark.asyncio
async def test_host_target_honors_custom_env_sandbox_type():
    # custom sandbox_type must resolve via build_sandbox_provider, not the default provider
    step = _step(env_id="env-x", result_id="filter")
    ctx = _make_context(with_agent=False)
    ctx.deployed_envs[0].sandbox_type = "arp"

    sandbox = _vm_sandbox_mock()
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
    sandbox = _vm_sandbox_mock()
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


def test_a_stored_step_with_a_named_agent_beside_its_env_id_still_loads_on_the_host():
    stored = {**_step(env_id="env-x").to_dict(), "agent_name": "solver"}
    step = RunCodeTaskStep.from_dict(stored)
    assert (step.env_id, step.agent_name) == ("env-x", None)


def test_a_stored_step_with_a_named_agent_beside_its_sandbox_name_is_rejected():
    stored = {**_step(sandbox_name="bundler").to_dict(), "agent_name": "solver"}
    with pytest.raises(ValueError, match="set at most one of"):
        RunCodeTaskStep.from_dict(stored)


@pytest.mark.asyncio
async def test_agent_target_honors_the_agents_sandbox_type():
    # an agent deployed on another provider is reached there, not through the configured default
    step = _step(result_id="filter")
    ctx = _make_context()
    ctx.deployed_agents[0].sandbox_type = "modal"

    sandbox = _vm_sandbox_mock()
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

    sandbox = _vm_sandbox_mock()
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
@pytest.mark.parametrize("installed_in,expected", [(None, "agent-api"), ("agent-api", "agent-api"), ("task", "task")])
async def test_an_agent_on_a_vm_runs_the_script_in_the_container_it_was_installed_in(installed_in, expected):
    ctx = _make_context()
    ctx.deployed_agents[0].sandbox_type = "modal_vm"
    ctx.deployed_agents[0].installed_container = installed_in
    sandbox = _vm_sandbox_mock()
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    with patch(
        "agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider",
        return_value=provider,
    ):
        target = await _step()._agent_target(ctx)
    await target.write_file("/tmp/run/input.json", "{}")

    assert target.prefix == ("sudo", "docker", "exec", "-u", "0", expected)
    sandbox.write_file_from_text.assert_awaited_once_with(
        "{}", "/tmp/run/input.json", **({"container": installed_in} if installed_in else {}))


@pytest.mark.asyncio
async def test_timeout_is_reported():
    with _sandbox([(124, "", "")]):  # the runner's status at the limit
        with pytest.raises(RunCodeExecutionError, match="timed out"):
            await _step(timeout_seconds=5).execute(_make_context())


@pytest.mark.asyncio
async def test_script_crash_surfaces_stderr():
    with _sandbox([(1, "", "Traceback: boom")]):
        with pytest.raises(RunCodeExecutionError, match="boom"):
            await _step().execute(_make_context())


@pytest.mark.asyncio
async def test_a_long_stderr_keeps_its_end_and_says_how_much_was_cut():
    stderr = "x" * 500 + "\nValueError: the cause\n"
    step = _step()
    step._STDERR_TAIL_BYTES = 100
    with _sandbox([(1, "", stderr)]):
        with pytest.raises(RunCodeExecutionError) as exc:
            await step.execute(_make_context())
    assert str(exc.value).endswith(f"[first {len(stderr) - 100:,} bytes of stderr cut]\n{stderr[-100:]}")


def test_a_cut_inside_a_character_drops_the_rest_of_it_and_counts_it():
    step = _step()
    step._STDERR_TAIL_BYTES = 5  # the last 5 bytes of "éééé" start inside its second é
    assert step._stderr_tail("éééé") == "[first 4 bytes of stderr cut]\néé"
    assert step._stderr_tail("éé") == "éé"


@pytest.mark.asyncio
async def test_a_script_past_its_timeout_is_stopped_and_reported(tmp_path):
    artifact = _file_artifact(body=b"import time\ndef run(inp):\n    time.sleep(300)\n")
    started = time.monotonic()
    with _real_run(tmp_path, artifact):
        with pytest.raises(RunCodeExecutionError, match="timed out after 1s"):
            await _step(timeout_seconds=1).execute(_make_context())
    assert time.monotonic() - started < 30


@pytest.mark.asyncio
async def test_a_crash_reports_all_of_stderr_through_to_the_exception(tmp_path):
    body = b"import sys\ndef run(inp):\n    print('x' * 5000, file=sys.stderr)\n    raise ValueError('the cause')\n"
    with _real_run(tmp_path, _file_artifact(body=body)):
        with pytest.raises(RunCodeExecutionError) as exc:
            await _step().execute(_make_context())
    assert "x" * 5000 in str(exc.value)
    assert str(exc.value).rstrip().endswith("ValueError: the cause")


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


def test_runner_parses_as_python_3_9():
    # macOS's /usr/bin/python3 is 3.9, and the runner runs on whatever python3 a host has.
    ast.parse(_RUNNER.read_text(), feature_version=(3, 9))


def _runner_args(run_dir, body, timeout):
    """Stage the runner, script and input as run_code does, the runner inside the run dir."""
    (run_dir / "runner.py").write_text(_RUNNER.read_text())
    (run_dir / "script.py").write_text(body)
    (run_dir / "input.json").write_text(json.dumps({"args": {}, "results": {}}))
    return [sys.executable, str(run_dir / "runner.py"), str(run_dir), "run", "script.py", str(timeout)]


def _supervised(run_dir, body, timeout=60):
    return subprocess.run(_runner_args(run_dir, body, timeout), capture_output=True, text=True, timeout=60)


def _running(pid):
    """True while `pid` is a live process; a zombie awaiting its reaper counts as gone."""
    stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(stat) and not stat.startswith("Z")


def _gone_within(pid, seconds=5.0):
    deadline = time.monotonic() + seconds
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _running(pid)


# Writes the pid of a `sleep` it starts, so a test can check the whole group was stopped.
_STARTS_A_CHILD = """\
import os, subprocess, time
def run(inp):
    child = subprocess.Popen(["sleep", "300"])
    with open(os.path.join(os.path.dirname(__file__), "pid.tmp"), "w") as f:
        f.write(str(child.pid))
    os.rename(f.name, f.name[:-len(".tmp")])
    time.sleep(300)
"""


def _wait_for_pid(run_dir):
    deadline = time.monotonic() + 30
    while not (run_dir / "pid").exists():
        assert time.monotonic() < deadline, "the script never started its child"
        time.sleep(0.05)
    return int((run_dir / "pid").read_text())


def test_a_supervised_script_writes_its_result(tmp_path):
    result = _supervised(tmp_path, "def run(inp):\n    return {'ok': True}\n")
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "output.json").read_text()) == {"ok": True}


def test_a_supervised_scripts_exit_status_and_stderr_pass_through(tmp_path):
    result = _supervised(tmp_path, "import sys\ndef run(inp):\n    print('boom', file=sys.stderr)\n    sys.exit(3)\n")
    assert (result.returncode, result.stderr) == (3, "boom\n")


def test_a_script_killed_by_a_signal_exits_128_plus_its_number(tmp_path):
    result = _supervised(tmp_path, "import os, signal\ndef run(inp):\n    os.kill(os.getpid(), signal.SIGKILL)\n")
    assert result.returncode == 128 + signal.SIGKILL


def test_the_limit_stops_the_script_and_everything_it_started(tmp_path):
    started = time.monotonic()
    proc = subprocess.Popen(_runner_args(tmp_path, _STARTS_A_CHILD, 1))
    child = _wait_for_pid(tmp_path)
    assert proc.wait(timeout=30) == 124
    assert time.monotonic() - started < 30
    assert _gone_within(child)


def test_a_script_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path):
    body = (
        "import signal, time\ndef run(inp):\n    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n    time.sleep(300)\n"
    )
    started = time.monotonic()
    result = _supervised(tmp_path, body, timeout=1)
    elapsed = time.monotonic() - started
    assert result.returncode == 124
    assert 5 <= elapsed < 30


# Starts the runner with SIGINT and SIGHUP ignored, as a background job is (CI runs the tier as one). The script must
# still get their defaults, as it does under GNU timeout.
_WITH_SIGNALS_IGNORED = (
    "import os, signal, sys\n"
    "for s in (signal.SIGINT, signal.SIGHUP):\n    signal.signal(s, signal.SIG_IGN)\n"
    "os.execv(sys.executable, [sys.executable] + sys.argv[1:])\n"
)


def _start_runner(run_dir, body, timeout):
    return subprocess.Popen([sys.executable, "-c", _WITH_SIGNALS_IGNORED, *_runner_args(run_dir, body, timeout)[1:]])


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_signal_to_the_runner_stops_the_script_and_everything_it_started(tmp_path, signum):
    proc = _start_runner(tmp_path, _STARTS_A_CHILD, 60)
    child = _wait_for_pid(tmp_path)
    proc.send_signal(signum)
    assert proc.wait(timeout=30) == 128 + signum
    assert _gone_within(child)


def test_a_forwarded_signal_the_script_ignores_is_followed_by_sigkill(tmp_path):
    ignore_term = "def run(inp):\n    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    body = _STARTS_A_CHILD.replace("def run(inp):\n", ignore_term)
    proc = _start_runner(tmp_path, "import signal\n" + body, 60)
    child = _wait_for_pid(tmp_path)
    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 128 + signal.SIGKILL
    assert 5 <= time.monotonic() - started < 30
    assert _gone_within(child)


def test_artifact_files_named_after_the_supervisors_imports_never_run_in_it(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "link").symlink_to(run_dir)  # reached through a symlink, as macOS's /tmp is
    for name in ("signal", "subprocess", "time"):
        (run_dir / f"{name}.py").write_text("raise SystemExit('ran in the supervisor')\n")
    result = _supervised(tmp_path / "link", "def run(inp):\n    return {'ok': True}\n")
    assert result.returncode == 0, result.stderr
    assert json.loads((run_dir / "output.json").read_text()) == {"ok": True}


def test_the_script_still_imports_its_own_module_named_like_one_the_supervisor_uses(tmp_path):
    (tmp_path / "signal.py").write_text("VALUE = 'mine'\n")
    result = _supervised(tmp_path, "import signal\ndef run(inp):\n    return {'from': signal.VALUE}\n")
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "output.json").read_text()) == {"from": "mine"}


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




@pytest.fixture
def computer(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir()
    return Computer(tmp_path / "home").install(monkeypatch)


async def _run_in(computer, body, deployed=None, **step):
    """Execute a run_code step for the computer's env with a script of ``body``."""
    ctx = TaskStepContext(deployed_envs=[deployed or computer.deployed()])
    with patch("agent_env.artifact.Artifact.get", return_value=_file_artifact(body=body)), \
            patch.object(FileArtifact, "load", _load_body):
        return await _step(env_id="desk", **step).execute(ctx)


@pytest.mark.asyncio
async def test_a_script_for_a_computer_runs_inside_it_and_reaches_its_driver_on_loopback_with_its_token(computer):
    body = b"import os\ndef run(inp):\n    return {'env': inp['env'], 'cwd': os.getcwd()}\n"

    ctx = await _run_in(computer, body)

    result = ctx.metadata["script_results"]["run-code-1"]
    assert result["env"] == {"id": "desk", "url": "http://127.0.0.1:18765", "mcp_url": "http://127.0.0.1:18765/mcp",
                             "headers": {"Authorization": f"Bearer {TOKEN}"}}
    assert computer.headers and all(sent == {"Authorization": f"Bearer {TOKEN}"} for sent in computer.headers)
    assert Path(result["cwd"]).resolve() == computer.home.resolve()
    run_dir = ".agentenv/run_code/run-code-1"
    assert [call[1] for call in computer.calls if call[0] == "data/add"] == [
        [{"path": f"{run_dir}/script.py"}], [{"path": f"{run_dir}/input.json"}], [{"path": f"{run_dir}/runner.py"}]]
    (script,) = [params for kind, params, *_ in computer.calls if kind == "run" and params["argv"][0] == "python3"]
    assert script["argv"] == ["python3", f"{run_dir}/runner.py", run_dir, "run", "script.py", "600"]
    assert script["timeout_s"] == 610
    assert ("data/get", [f"{run_dir}/output.json"]) in computer.calls
    assert not (computer.home / run_dir).exists()


@pytest.mark.asyncio
async def test_a_failed_script_in_a_computer_is_reported_and_leaves_no_run_folder(computer):
    with pytest.raises(RunCodeExecutionError, match="(?s)exited 1: .*ValueError: no"):
        await _run_in(computer, b"def run(inp):\n    raise ValueError('no')\n")

    assert not (computer.home / ".agentenv/run_code/run-code-1").exists()


@pytest.mark.asyncio
async def test_a_run_the_computer_stopped_at_its_timeout_is_reported_as_one(computer, monkeypatch):
    monkeypatch.setattr(computer, "_run", lambda params: {"exit_code": None, "stdout": "", "stderr": "",
                                                          "truncated": False, "timed_out": True}
                        if params["argv"][0] == "python3" else Computer._run(computer, params))

    with pytest.raises(RunCodeExecutionError, match="timed out after 30s"):
        await _run_in(computer, _DEFAULT_BODY, timeout_seconds=30)


@pytest.mark.asyncio
async def test_a_run_whose_call_drops_fails_naming_why_and_the_next_one_runs_afresh(computer, monkeypatch):
    invoke, dropped = computer.invoke_extension, []

    async def drop_once(base_url, card, uri, params=None, timeout=30, verify=True, *, method=None, headers=None):
        if params["argv"][0] == "python3" and not dropped:
            dropped.append(params["request_id"])
            raise httpx.RemoteProtocolError("the driver went away")
        return await invoke(base_url, card, uri, params, timeout, verify, method=method)

    monkeypatch.setattr("agentenv_protocol.client.invoke_extension", drop_once)
    with pytest.raises(RuntimeError, match=re.escape(
            "env 'desk': the connection to its driver broke during computer-exec/v1 run (RemoteProtocolError: the "
            "driver went away)")):
        await _run_in(computer, _DEFAULT_BODY, id="read.v1")
    await _run_in(computer, _DEFAULT_BODY, id="read.v1")

    runs = [params for kind, params, *_ in computer.calls if kind == "run"]
    staged = [params["argv"] for params in runs if params["argv"][0] == "pkill"]
    assert staged == [["pkill", "-f", r"\.agentenv/run_code/read\.v1/runner\.py"]] * 2
    assert not re.search(staged[0][2], "python3 /home/u/.agentenv/run_code/read-v1/runner.py")
    scripts = [params["request_id"] for params in runs if params["argv"][0] == "python3"]
    assert len(scripts) == 1 and scripts[0] != dropped[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("card, step, problem", [
    ({**CARD, "capabilities": {**CARD["capabilities"], "operations": ["data/get"]}}, {},
     "env 'desk' serves computer-exec/v1 but not data/add, which running code in it needs"),
    (CARD, {"timeout_seconds": 3591}, "timeout_seconds 3591 is above the 3590 s a script in env 'desk' may run: it "
                                      "runs a command for at most 3600 s"),
], ids=["no-data-add", "timeout"])
async def test_a_computer_run_its_driver_cant_carry_out_is_refused_before_anything_is_staged(computer, card, step,
                                                                                          problem):
    with pytest.raises(RunCodeValidationError, match=problem):
        await _run_in(computer, _DEFAULT_BODY, deployed=computer.deployed(card=card), **step)

    assert computer.calls == []
