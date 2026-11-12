"""A computer's boot when its sandbox stops running it: it fails within seconds of the sandbox's container, or the
launcher booting its disk, exiting, rather than at its readiness timeout, naming the exit status and what the sandbox
last wrote, its guest init's and launcher's reasons first; one never ready says what it last wrote too; and a sandbox
whose provider can't tell fails as before."""

import asyncio

import httpx
import pytest

from agent_env.env.envs.computer.utils import boot
from agent_env.providers.sandbox_providers.sandbox import Sandbox

URL = "http://computer"
CONSOLE = ["[    0.912] EXT4-fs (vda2): mounted filesystem", "agentenv-init: no DHCP lease on eth0",
           "\x1b[0;31magentenv-init: the image's user ghost isn't in /etc/passwd\x1b[0m\r",
           "[    3.120] reboot: Power down", ""]


class Running(Sandbox):
    """A sandbox that runs its computer for ``runs`` asks, then exits with ``status``, having written ``written``."""

    def __init__(self, status=None, written=(), runs=0):
        self.sandbox_id, self.tunnel_urls, self.status, self.written, self.runs = "sb-1", {}, status, list(written), runs
        self.asked = 0

    async def terminate(self):
        pass

    async def exit_status(self):
        self.asked += 1
        return self.status if self.asked > self.runs else None

    async def output_tail(self, lines):
        return self.written[-lines:]


class Unasked(Sandbox):
    sandbox_id, tunnel_urls = "sb-2", {}

    async def terminate(self):
        pass


@pytest.fixture(autouse=True)
def unanswered(monkeypatch):
    async def get_card(url, timeout=5):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(boot.protocol_v1, "get_card", get_card)
    monkeypatch.setattr(boot, "_POLL_S", 0.01)
    monkeypatch.setattr(boot, "_EXIT_POLL_S", 0.05)


@pytest.mark.asyncio
async def test_a_disk_whose_guest_init_fails_stops_the_boot_within_seconds_with_its_reason_and_last_lines():
    sandbox = Running(status=0, written=CONSOLE, runs=2)

    with pytest.raises(boot.ComputerStopped) as stopped:
        await asyncio.wait_for(boot.ready(URL, 180, "t0ken", sandbox=sandbox), 5)

    assert str(stopped.value) == (
        "the computer at http://computer stopped before it was ready, exiting with 0; its sandbox last wrote:\n"
        "  [    0.912] EXT4-fs (vda2): mounted filesystem\n"
        "  agentenv-init: no DHCP lease on eth0\n"
        "  agentenv-init: the image's user ghost isn't in /etc/passwd\n"
        "  [    3.120] reboot: Power down")
    assert isinstance(stopped.value, boot.ComputerFailed) and sandbox.asked == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "said"), [
    (3, "exiting with 3; its sandbox last wrote:\n  launch: KVM can't run this arm64 disk on this amd64 machine"),
    (137, "exiting with 137 (SIGKILL); its sandbox last wrote:\n  launch: KVM can't run this arm64 disk"),
    (-9, "exiting with -9 (SIGKILL)"),
    (2, "exiting with 2"),
], ids=["refused", "killed", "killed-as-python-says", "plain"])
async def test_a_launcher_that_exits_names_its_status_and_a_signal_that_ended_it(status, said):
    sandbox = Running(status=status, written=["launch: KVM can't run this arm64 disk on this amd64 machine"])

    with pytest.raises(boot.ComputerStopped) as stopped:
        await boot.ready(URL, 180, sandbox=sandbox)

    assert said in str(stopped.value)


@pytest.mark.asyncio
async def test_a_computer_never_ready_says_what_its_sandbox_last_wrote_at_its_timeout():
    console = [f"[  {n}.0] systemd[1]: Started unit {n}." for n in range(40)]

    with pytest.raises(TimeoutError) as timed_out:
        await boot.ready(URL, 0.3, sandbox=Running(written=console))

    assert str(timed_out.value) == (
        "the computer at http://computer wasn't ready within 0 s (ConnectError: Connection refused); its sandbox last "
        "wrote:\n" + "\n".join(f"  [  {n}.0] systemd[1]: Started unit {n}." for n in range(35, 40)))


@pytest.mark.asyncio
async def test_a_sandbox_whose_provider_cant_tell_waits_out_the_timeout_and_says_what_it_did_before():
    with pytest.raises(TimeoutError) as timed_out:
        await boot.ready(URL, 0.3, sandbox=Unasked())

    assert str(timed_out.value) == ("the computer at http://computer wasn't ready within 0 s (ConnectError: Connection "
                                    "refused)")


@pytest.mark.asyncio
async def test_a_sandbox_that_cant_be_asked_or_read_keeps_the_boot_waiting_then_times_out_as_before():
    class Unreachable(Running):
        async def exit_status(self):
            self.asked += 1
            raise RuntimeError("docker: Cannot connect to the Docker daemon")

        async def output_tail(self, lines):
            raise RuntimeError("docker: Cannot connect to the Docker daemon")

    sandbox = Unreachable()

    with pytest.raises(TimeoutError, match=r"\(ConnectError: Connection refused\)$"):
        await boot.ready(URL, 0.3, sandbox=sandbox)

    assert sandbox.asked > 1


@pytest.mark.asyncio
async def test_a_computer_that_says_it_failed_adds_what_its_sandbox_last_wrote(monkeypatch):
    async def get_card(url, timeout=5):
        return {"name": "desk"}

    async def status(url, card, uri, params, timeout, method, headers):
        return {"ready": False, "failed": "display: Xvfb exited"}

    monkeypatch.setattr(boot.protocol_v1, "get_card", get_card)
    monkeypatch.setattr(boot.protocol_v1, "invoke_extension", status)

    with pytest.raises(boot.ComputerFailed) as failed:
        await boot.ready(URL, 5, sandbox=Running(written=["Xvfb: no screens found"]))

    assert str(failed.value) == ("the computer at http://computer failed: display: Xvfb exited; its sandbox last wrote:\n"
                                 "  Xvfb: no screens found")
    assert not isinstance(failed.value, boot.ComputerStopped)


def test_what_a_boot_names_of_its_sandboxs_last_lines_is_its_reasons_and_its_last_lines_in_order():
    lines = ["launch: the proxy in front of a forwarded port exited (1), so the computer is stopped",
             *(f"agentenv-init: line {n}" for n in range(5)), *(f"kernel {n}" for n in range(10)), "  ", "\x1b[2K"]

    assert boot.said_why(lines) == ["agentenv-init: line 2", "agentenv-init: line 3", "agentenv-init: line 4",
                                    "kernel 5", "kernel 6", "kernel 7", "kernel 8", "kernel 9"]
    assert boot.said_why(["only", "a few"]) == ["only", "a few"] and boot.said_why([]) == []
