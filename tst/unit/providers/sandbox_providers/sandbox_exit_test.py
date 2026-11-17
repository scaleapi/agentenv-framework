"""Whether a sandbox still runs the process it was made to run, and what that process last wrote, as a computer's boot
asks: a local sandbox's container from Docker, its output and errors as Docker logged them; a Modal sandbox's process
from Modal, its output once it has exited, read once, a chunk at a time, into a bounded tail of each stream, neither
crowding out the other's last lines; and nothing from a sandbox whose provider can't tell."""

import asyncio
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent_env.providers.sandbox_providers.local_docker.provider as ls
from agent_env.providers.sandbox_providers import modal_sandbox
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


class _Stream:
    """A Modal sandbox's stream, handed a chunk at a time, which ``reads`` counts."""

    def __init__(self, *chunks):
        self.chunks, self.reads = chunks, 0

    def __aiter__(self):
        return self._chunks()

    async def _chunks(self):
        self.reads += 1
        for chunk in self.chunks:
            yield chunk


def _modal_sb(status, stdout=(), stderr=()):
    sb = MagicMock()
    sb.object_id = "sb-1"
    sb.poll.aio = AsyncMock(return_value=status)
    sb.stdout, sb.stderr = _Stream(*stdout), _Stream(*stderr)
    return sb


@pytest.mark.asyncio
@pytest.mark.parametrize("made", [lambda sb: ModalVmSandbox(sb, {}), lambda sb: ModalSandbox(sb, {})],
                         ids=["modal_vm", "modal"])
async def test_a_modal_sandbox_reads_its_process_output_once_it_has_exited_and_keeps_it(made):
    running = _modal_sb(None, stdout=["never read\n"])

    assert await made(running).exit_status() is None and await made(running).output_tail(5) == []
    assert running.stdout.reads == 0

    ended = _modal_sb(2, stdout=["[    1.0] Linux version 6.8\r\nagentenv-init: no filesystem lab", "elled agentenv-root\n",
                                 "\n"], stderr=["launch: the guest powered off"])
    sandbox = made(ended)
    assert await sandbox.exit_status() == 2
    assert await sandbox.output_tail(5) == ["[    1.0] Linux version 6.8",
                                            "agentenv-init: no filesystem labelled agentenv-root",
                                            "launch: the guest powered off"]
    assert await sandbox.output_tail(1) == ["agentenv-init: no filesystem labelled agentenv-root"]
    assert (ended.stdout.reads, ended.stderr.reads) == (1, 1)


@pytest.mark.asyncio
async def test_a_modal_sandbox_keeps_a_bounded_tail_of_each_stream_and_neither_crowds_out_the_others_last_lines(
        monkeypatch):
    monkeypatch.setattr(modal_sandbox, "_OUTPUT_KEPT_LINES", 300)
    console = [f"console {n}\n" for n in range(5000)] + ["agentenv-init: can't mount /dev/vda2\n"]
    sandbox = ModalVmSandbox(_modal_sb(0, stdout=console, stderr=[f"socat {n}\n" for n in range(400)]), {})

    tail = await sandbox.output_tail(200)

    assert len(sandbox._ended[0]) == len(sandbox._ended[1]) == 300
    assert tail == [*(f"console {n}" for n in range(4901, 5000)), "agentenv-init: can't mount /dev/vda2",
                    *(f"socat {n}" for n in range(300, 400))]
    few = ModalVmSandbox(_modal_sb(0, stdout=["the guest's last words\n"], stderr=[f"socat {n}\n" for n in range(400)]),
                         {})
    assert (await few.output_tail(200))[0] == "the guest's last words" and len(await few.output_tail(200)) == 200


@pytest.mark.asyncio
async def test_a_modal_sandbox_whose_stream_breaks_keeps_what_it_read_and_all_the_other_stream_says(caplog):
    class Breaks(_Stream):
        async def _chunks(self):
            yield "launch: the computer's QEMU exited\n"
            raise RuntimeError("stream gone")

    class Slow(_Stream):
        async def _chunks(self):
            for line in ("[    1.0] Linux version 6.8\n", "agentenv-init: the image names no command\n"):
                await asyncio.sleep(0.05)
                yield line

    sb = _modal_sb(2)
    sb.stdout, sb.stderr = Slow(), Breaks()

    assert await ModalVmSandbox(sb, {}).output_tail(5) == [
        "[    1.0] Linux version 6.8", "agentenv-init: the image names no command", "launch: the computer's QEMU exited"]
    assert "Couldn't read all the output of Modal sandbox sb-1: builtins.RuntimeError: stream gone" in caplog.text


@pytest.mark.asyncio
async def test_a_modal_sandbox_whose_output_takes_too_long_to_read_keeps_what_it_read(monkeypatch, caplog):
    class Endless(_Stream):
        async def _chunks(self):
            yield "agentenv-init: no filesystem labelled agentenv-root\n"
            await asyncio.Event().wait()

    monkeypatch.setattr(modal_sandbox, "_OUTPUT_READ_SECONDS", 0.2)
    sb = _modal_sb(2)
    sb.stdout = Endless()

    assert await ModalVmSandbox(sb, {}).output_tail(5) == ["agentenv-init: no filesystem labelled agentenv-root"]
    assert "reading took past 0.2 s" in caplog.text
