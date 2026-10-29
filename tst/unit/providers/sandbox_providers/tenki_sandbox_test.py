"""Offline unit tests for the Tenki sandbox: exec, stdin transfer, lifecycle and egress."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from tenki import InvalidStateError, SandboxError, SessionNotFoundError, SessionTerminatedError, StreamClosedError
from tenki_sandbox.models import SessionEgressPolicy

from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.tenki import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.tenki.sandbox import TenkiSandbox, allows_host, policy_from_info
from tst.unit.providers.sandbox_providers.tenki_fakes import FakeSession, result

ALLOWLIST = NetworkMode.ALLOWLIST


def _sandbox(session: FakeSession | None = None, *, policy=NetworkPolicy()) -> TenkiSandbox:
    return TenkiSandbox(
        session or FakeSession(state="RUNNING"), tunnel_urls={}, network_policy=policy,
        command_timeout=600,
    )


def _transient() -> SandboxError:
    error = SandboxError("unavailable")
    error.retryable = True
    return error


async def _no_sleep(_seconds):
    return None


@pytest.mark.asyncio
async def test_exec_runs_as_root_with_the_lifetime_budget_and_drops_sudo():
    session = FakeSession()
    session.results = [result(stdout=b"hi\n")]
    assert (await _sandbox(session).exec_with_output("sudo", "echo", "hi"))[:2] == (0, "hi\n")
    assert session.exec_calls == [{"argv": ["echo", "hi"], "timeout": 600, "privileged": True}]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [_transient(), SandboxError("sandbox run stream closed before exit")])
async def test_a_lost_stream_is_exit_minus_one_with_no_stdout(error):
    session = FakeSession()
    session.results = [error]
    exit_code, stdout, stderr = await _sandbox(session).exec_with_output("true")
    assert (exit_code, stdout) == (-1, "")
    assert type(error).__name__ in stderr


@pytest.mark.asyncio
async def test_a_terminated_session_raises_rather_than_reads_as_a_lost_stream():
    session = FakeSession()
    session.results = [SessionTerminatedError("session terminated")]
    with pytest.raises(SessionTerminatedError):
        await _sandbox(session).exec_with_output("true")


@pytest.mark.asyncio
@pytest.mark.parametrize(("fields", "exit_code", "note"), [
    ({"signal": "killed", "reason": "signaled"}, 137, "killed by signal: killed"),
    ({"reason": 'exec: "nope": executable file not found in $PATH'}, 127, "executable file not found"),
])
async def test_tenkis_minus_one_for_a_signal_or_a_failed_start_gets_the_shells_code(fields, exit_code, note):
    session = FakeSession()
    session.results = [result(exit_code=-1, **fields)]
    code, _, stderr = await _sandbox(session).exec_with_output("nope")
    assert code == exit_code
    assert note in stderr


@pytest.mark.asyncio
async def test_exec_script_retries_a_lost_stream(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    session = FakeSession()
    session.results = [_transient(), result(stdout=b"done")]
    assert await _sandbox(session).exec_script("echo done", max_retries=1) == "done"
    assert len(session.exec_calls) == 2


@pytest.mark.asyncio
async def test_other_errors_propagate():
    session = FakeSession()
    session.results = [SessionNotFoundError("gone")]
    with pytest.raises(SessionNotFoundError):
        await _sandbox(session).exec_with_output("true")


@pytest.mark.asyncio
async def test_a_timed_out_command_is_exit_124():
    session = FakeSession()
    session.results = [result(stdout=b"partial", timed_out=True)]
    exit_code, stdout, stderr = await _sandbox(session).exec_with_output("sleep", "999")
    assert (exit_code, stdout) == (124, "partial")
    assert "timed out after 600s" in stderr


@pytest.mark.asyncio
async def test_exec_with_stdin_streams_every_piece_as_root():
    session = FakeSession()
    session.results = [result(stdout=b"ok")]

    async def pieces():
        yield b"ab"
        yield b"cd"

    assert await _sandbox(session)._exec_with_stdin("cat > /x", pieces()) == (0, "ok", "")
    assert (session.processes[0].stdin, session.processes[0].stdin_closed) == ([b"ab", b"cd"], True)
    assert session.exec_calls[0] == {"argv": ["bash", "-c", "cat > /x"], "timeout": 600, "privileged": True}


@pytest.mark.asyncio
async def test_exec_with_stdin_kills_the_process_when_the_reader_fails():
    session = FakeSession()

    async def pieces():
        yield b"ab"
        raise OSError("disk read failed")

    with pytest.raises(OSError):
        await _sandbox(session)._exec_with_stdin("cat > /x", pieces())
    assert session.processes[0].calls == ["close_stdin", "kill", "wait"]


@pytest.mark.asyncio
async def test_a_wait_that_fails_reaps_the_process():
    session = FakeSession()
    process_start = session.start

    async def start(*argv, **kwargs):
        process = await process_start(*argv, **kwargs)
        process.wait_error = StreamClosedError("stream closed")
        return process

    session.start = start

    async def pieces():
        yield b"ab"

    with pytest.raises(StreamClosedError):
        await _sandbox(session)._exec_with_stdin("cat > /x", pieces())
    assert session.processes[0].calls == ["close_stdin", "wait", "close_stdin", "kill", "wait"]


@pytest.mark.asyncio
async def test_a_reap_that_is_cancelled_still_kills_and_waits():
    calls = []

    async def cancelled():
        calls.append("close_stdin")
        raise asyncio.CancelledError

    async def record(name):
        calls.append(name)

    process = SimpleNamespace(close_stdin=cancelled, kill=lambda: record("kill"),
                              wait=lambda timeout=None: record("wait"))
    with pytest.raises(asyncio.CancelledError):
        await TenkiSandbox._reap(process)
    assert calls == ["close_stdin", "kill", "wait"]


@pytest.mark.asyncio
async def test_a_failing_reap_does_not_mask_the_input_error():
    session = FakeSession()

    async def pieces():
        yield b"ab"
        raise OSError("disk read failed")

    async def stream_gone(*_args, **_kwargs):
        raise StreamClosedError("stream closed")

    process_start = session.start

    async def start(*argv, **kwargs):
        process = await process_start(*argv, **kwargs)
        process.close_stdin = process.kill = process.wait = stream_gone
        return process

    session.start = start
    with pytest.raises(OSError, match="disk read failed"):
        await _sandbox(session)._exec_with_stdin("cat > /x", pieces())


@pytest.mark.asyncio
async def test_stdin_stops_reading_once_the_command_ended(monkeypatch):
    session = FakeSession()
    read = []

    async def pieces():
        for i in range(5):
            read.append(i)
            yield b"x"

    async def ended(_process):
        return True

    monkeypatch.setattr(TenkiSandbox, "_wait_for_stdin_drain", staticmethod(ended))
    await _sandbox(session)._exec_with_stdin("cat > /x", pieces())
    assert read == [0]


@pytest.mark.asyncio
async def test_terminate_stops_the_data_plane_renewal_of_a_session_already_ending():
    calls = []
    session = FakeSession(state="TERMINATING")

    async def reset():
        calls.append("reset")

    session._reset_data_plane = reset
    await _sandbox(session).terminate()
    assert calls == ["reset"]


def _stalled_process(*, error: BaseException | None) -> SimpleNamespace:
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(sandbox_module._STDIN_QUEUE_LIMIT + 4):
        queue.put_nowait(object())
    done = asyncio.Event()
    done.set()
    return SimpleNamespace(_outbound=queue, _done=done, _error=error)


@pytest.mark.asyncio
async def test_the_stdin_drain_raises_the_error_of_a_stream_that_died():
    with pytest.raises(StreamClosedError):
        await asyncio.wait_for(
            TenkiSandbox._wait_for_stdin_drain(_stalled_process(error=StreamClosedError("stream closed"))), 1,
        )


@pytest.mark.asyncio
async def test_the_stdin_drain_returns_once_the_command_ended():
    assert await asyncio.wait_for(TenkiSandbox._wait_for_stdin_drain(_stalled_process(error=None)), 1)


@pytest.mark.asyncio
async def test_write_host_file_goes_over_stdin_to_any_path():
    session = FakeSession()
    await _sandbox(session).write_host_file(b"bytes", "/etc/agent/config.json")
    assert session.exec_calls[-1]["argv"] == ["bash", "-c", "cat > /etc/agent/config.json"]
    assert session.processes[-1].stdin == [b"bytes"]


@pytest.mark.asyncio
async def test_terminate_closes_the_session_once():
    session = FakeSession(state="RUNNING")
    sandbox = _sandbox(session)
    await sandbox.terminate()
    await sandbox.terminate()
    assert session.closed == 1


@pytest.mark.asyncio
async def test_a_failed_terminate_can_be_retried():
    session = FakeSession(state="RUNNING")
    session.close_error = SandboxError("unavailable")
    sandbox = _sandbox(session)
    with pytest.raises(SandboxError):
        await sandbox.terminate()
    session.close_error = None
    await sandbox.terminate()
    assert session.closed == 1


@pytest.mark.asyncio
async def test_terminate_tolerates_a_session_that_is_already_gone():
    session = FakeSession(state="RUNNING")
    session.close_error = SessionNotFoundError("gone")
    await _sandbox(session).terminate()


class _TerminatedElsewhere(FakeSession):
    """Another handle terminated it: this one's cached state is stale and close is refused."""

    def __init__(self, server_state: str):
        super().__init__(state="RUNNING")
        self.close_error = InvalidStateError("cannot terminate session in current state")
        self.server_state = server_state

    async def refresh(self):
        self.info.state = self.server_state
        return self.info


@pytest.mark.asyncio
@pytest.mark.parametrize("server_state", ["TERMINATING", "TERMINATED"])
async def test_terminate_tolerates_a_session_another_handle_terminated(server_state):
    await _sandbox(_TerminatedElsewhere(server_state)).terminate()


@pytest.mark.asyncio
async def test_terminate_still_raises_when_the_session_is_live():
    with pytest.raises(InvalidStateError):
        await _sandbox(_TerminatedElsewhere("PAUSING")).terminate()


@pytest.mark.asyncio
async def test_wait_for_vm_reports_the_bootstrap_failure():
    session = FakeSession()
    session.results = [result(stderr=b"E: Unable to locate package docker.io", exit_code=100)]
    with pytest.raises(RuntimeError, match="Unable to locate package"):
        await _sandbox(session).wait_for_vm()


@pytest.mark.asyncio
async def test_setup_requires_compose_v2():
    session = FakeSession()
    session.results = [result(), result(stderr=b"docker: 'compose' is not a docker command.", exit_code=1)]
    with pytest.raises(RuntimeError, match="no Docker Compose v2"):
        await _sandbox(session).setup_vm_for_gateway([8080])


@pytest.mark.parametrize(("host", "allowed"), [
    ("pypi.org", True),
    ("files.pythonhosted.org", True),
    ("a.b.pythonhosted.org", False),
    ("pythonhosted.org", False),
    ("10.2.3.4", True),
    ("11.0.0.1", False),
    ("evil.com", False),
])
def test_allows_host_follows_tenki_one_label_wildcards(host, allowed):
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org", "*.pythonhosted.org"), allow_cidrs=("10.0.0.0/8",))
    assert allows_host(policy, host) is allowed
    assert allows_host(NetworkPolicy(), host)


def test_policy_from_info():
    def info(**fields):
        return SimpleNamespace(**{"outbound_enabled": True, "egress": SessionEgressPolicy(), **fields})

    assert policy_from_info(info()) == NetworkPolicy()
    assert policy_from_info(info(outbound_enabled=False)) == NetworkPolicy(mode=ALLOWLIST)
    assert policy_from_info(info(egress=SessionEgressPolicy(allow_domains=("a.com",)))) == NetworkPolicy(
        mode=ALLOWLIST, allow_hosts=("a.com",),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("policy", "signed", "curled"), [
    (NetworkPolicy(), "https://bucket.s3.amazonaws.com/obj?sig", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",)), "https://bucket.s3.amazonaws.com/obj?sig", False),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("*.s3.amazonaws.com",)), "https://bucket.s3.amazonaws.com/o", True),
    (None, "https://bucket.s3.amazonaws.com/obj?sig", False),
    (NetworkPolicy(), None, False),
])
async def test_a_download_the_vm_cannot_fetch_goes_over_stdin(monkeypatch, policy, signed, curled):
    store = SimpleNamespace(signed_get_url=lambda _url: signed)
    monkeypatch.setattr(sandbox_module, "get_config", lambda: SimpleNamespace(get_object_store_at=lambda _url: store))
    pushed = []

    async def push(sandbox, object_store, object_url, vm_path):
        pushed.append(vm_path)

    monkeypatch.setattr(sandbox_module, "push_object_over_stdin", push)
    session = FakeSession()
    await _sandbox(session, policy=policy)._download_object_to_vm("s3://b/obj", "/tmp/obj")
    assert any("curl" in " ".join(call["argv"]) for call in session.exec_calls) is curled
    assert bool(pushed) is not curled
