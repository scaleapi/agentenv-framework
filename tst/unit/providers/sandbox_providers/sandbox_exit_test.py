"""Whether a sandbox still runs the process it was made to run, and what that process last wrote, as a computer's boot
asks: a local sandbox's container from Docker, its output and errors as Docker logged them; a Modal sandbox's process
from Modal, its output once it has exited, read once; and nothing from a sandbox whose provider can't tell."""

import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent_env.providers.sandbox_providers.local_docker.provider as ls
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox


def _local(tmp_path, owns: bool = True) -> ls.LocalSandbox:
    sandbox = ls.LocalSandbox(sandbox_id="local-disk", work_dir=tmp_path)
    if owns:
        (tmp_path / ls._CONTAINER_MODE_MARKER).write_text(sandbox.container_name)
    return sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize(("state", "status"), [("exited 3", 3), ("dead 137", 137), ("running 0", None),
                                                ("created 0", None)])
async def test_a_local_sandbox_says_its_container_exited_with_dockers_exit_code(tmp_path, monkeypatch, state, status):
    asked = []
    monkeypatch.setattr(ls, "docker", lambda *args, timeout=None: asked.append(args) or state)

    assert await _local(tmp_path).exit_status() == status
    assert asked == [("container", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}",
                      "agent-local-disk")]


@pytest.mark.asyncio
async def test_a_local_sandbox_reads_its_containers_last_lines_its_output_and_errors_as_docker_logged_them(
        tmp_path, monkeypatch):
    ran = []

    def run(args, **kwargs):
        ran.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"agentenv-init: can't mount /dev/vda2\r\nreboot: Power down\r\n")

    monkeypatch.setattr(ls.subprocess, "run", run)

    assert await _local(tmp_path).output_tail(200) == ["agentenv-init: can't mount /dev/vda2", "reboot: Power down"]
    ((args, kwargs),) = ran
    assert args == ["docker", "logs", "--tail", "200", "agent-local-disk"]
    assert (kwargs["stdout"], kwargs["stderr"]) == (subprocess.PIPE, subprocess.STDOUT)


@pytest.mark.asyncio
async def test_a_local_sandbox_says_nothing_of_a_container_it_doesnt_run_or_docker_wont_say(tmp_path, monkeypatch):
    def failing(*args, timeout=None):
        raise ls.DockerFailed("docker container inspect failed: No such container", 1)

    monkeypatch.setattr(ls, "docker", failing)
    monkeypatch.setattr(ls.subprocess, "run", lambda args, **kwargs: SimpleNamespace(returncode=1, stdout=b"Error"))
    (tmp_path / "theirs").mkdir()

    assert await _local(tmp_path).exit_status() is None and await _local(tmp_path).output_tail(5) == []
    vm = _local(tmp_path / "theirs", owns=False)
    monkeypatch.setattr(ls, "docker", MagicMock(side_effect=AssertionError("asked Docker of no container")))
    assert await vm.exit_status() is None and await vm.output_tail(5) == []


def _modal_sb(status, stdout="", stderr=""):
    sb = MagicMock()
    sb.object_id = "sb-1"
    sb.poll.aio = AsyncMock(return_value=status)
    sb.stdout.read.aio = AsyncMock(return_value=stdout)
    sb.stderr.read.aio = AsyncMock(return_value=stderr)
    return sb


@pytest.mark.asyncio
@pytest.mark.parametrize("made", [lambda sb: ModalVmSandbox(sb, {}), lambda sb: ModalSandbox(sb, {})],
                         ids=["modal_vm", "modal"])
async def test_a_modal_sandbox_reads_its_process_output_once_it_has_exited_and_keeps_it(made):
    running = _modal_sb(None)

    assert await made(running).exit_status() is None and await made(running).output_tail(5) == []
    running.stdout.read.aio.assert_not_awaited()

    ended = _modal_sb(2, stdout="[    1.0] Linux version 6.8\nagentenv-init: no filesystem labelled agentenv-root\n",
                      stderr="launch: the guest powered off\n")
    sandbox = made(ended)
    assert await sandbox.exit_status() == 2
    assert await sandbox.output_tail(5) == ["[    1.0] Linux version 6.8",
                                            "agentenv-init: no filesystem labelled agentenv-root",
                                            "launch: the guest powered off"]
    assert await sandbox.output_tail(1) == ["launch: the guest powered off"]
    ended.stdout.read.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_modal_sandbox_whose_output_cant_be_read_gives_none(caplog):
    sb = _modal_sb(2)
    sb.stdout.read.aio = AsyncMock(side_effect=RuntimeError("stream gone"))

    assert await ModalVmSandbox(sb, {}).output_tail(5) == []
    assert "Couldn't read the output of Modal sandbox sb-1" in caplog.text
