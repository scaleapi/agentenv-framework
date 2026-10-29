"""Offline unit tests for the boxd sandbox: exec, readiness, lifecycle and egress."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from boxd import APIConnectionError, APIStatusError, ConflictError, NotFoundError

from agent_env.providers.sandbox_providers.boxd import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.boxd.sandbox import (
    _KEEPALIVE,
    KEEPALIVE_SEC,
    BoxdSandbox,
    allows_host,
    policy_from_entries,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy

from tst.unit.providers.sandbox_providers.boxd_fakes import FakeClient, fake_client_cls, result

ALLOWLIST = NetworkMode.ALLOWLIST


async def _sandbox(*, policy=NetworkPolicy()) -> tuple[BoxdSandbox, FakeClient]:
    client = fake_client_cls()()
    machine = await client.machines.create("agentenv-test")
    return BoxdSandbox(client, machine.id, tunnel_urls={}, network_policy=policy), client


async def _no_sleep(_seconds):
    return None


@pytest.mark.asyncio
async def test_exec_runs_argv_shell_quoted_with_no_input_and_keeps_sudo():
    sandbox, client = await _sandbox()
    client.machines.results = [result(stdout=b"hi there\n")]
    assert (await sandbox.exec_with_output("sudo", "echo", "hi there"))[:2] == (0, "hi there\n")
    line = _KEEPALIVE.format(interval=KEEPALIVE_SEC, line="sudo echo 'hi there' </dev/null")
    assert client.machines.exec_calls == [{"machine_id": "m-1", "command": line, "encoding": "buffer"}]


@pytest.mark.asyncio
async def test_exec_keeps_a_quiet_command_alive_and_drops_the_keepalive_bytes():
    """boxd stops relaying a command that prints nothing for 30 minutes; the NULs that prevent it never reach callers."""
    sandbox, client = await _sandbox()
    client.machines.results = [result(stdout=b"done\n", stderr=b"\0\0err\n\0", exit_code=3)]
    assert await sandbox.exec_with_output("sleep", "3600") == (3, "done\n", "err\n")
    command = client.machines.exec_calls[0]["command"]
    assert command.startswith(f"( while sleep {KEEPALIVE_SEC} </dev/null >/dev/null 2>&1; do printf '\\0' >&2; done ) &")
    assert command.endswith("; c=$?; kill $k 2>/dev/null; exit $c")


@pytest.mark.asyncio
async def test_exec_forwards_the_exit_code_and_keeps_stdout_and_stderr_apart():
    sandbox, client = await _sandbox()
    client.machines.results = [result(stdout=b"out\n", stderr=b"err\n", exit_code=3)]
    assert await sandbox.exec_with_output("bash", "-c", "echo out; echo err >&2; exit 3") == (3, "out\n", "err\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    APIConnectionError("connection reset"),
    APIStatusError("stream timeout", code="deadline_exceeded", grpc_code=4),
    APIStatusError("stream reset", code="internal", grpc_code=13),
])
async def test_a_lost_stream_is_exit_minus_one_with_no_stdout(error):
    sandbox, client = await _sandbox()
    client.machines.results = [error]
    exit_code, stdout, stderr = await sandbox.exec_with_output("true")
    assert (exit_code, stdout) == (-1, "")
    assert type(error).__name__ in stderr


@pytest.mark.asyncio
async def test_an_exit_minus_one_from_boxd_drops_its_partial_stdout():
    sandbox, client = await _sandbox()
    client.machines.results = [result(stdout=b"partial", exit_code=-1)]
    assert (await sandbox.exec_with_output("true"))[:2] == (-1, "")


@pytest.mark.asyncio
async def test_exec_script_retries_a_lost_stream(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    sandbox, client = await _sandbox()
    client.machines.results = [APIConnectionError("unavailable"), result(stdout=b"done")]
    assert await sandbox.exec_script("echo done", max_retries=1) == "done"
    assert len(client.machines.exec_calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    NotFoundError("VM is destroyed", code="not_found", grpc_code=5),
    ConflictError("VM is stopped, so it won't be resumed automatically", code="failed_precondition", grpc_code=9),
])
async def test_a_gone_or_stopped_machine_raises_rather_than_reads_as_a_lost_stream(error):
    sandbox, client = await _sandbox()
    client.machines.results = [error]
    with pytest.raises(type(error)):
        await sandbox.exec_with_output("true")


@pytest.mark.asyncio
async def test_wait_for_vm_waits_for_the_machine_then_docker():
    sandbox, client = await _sandbox()
    await sandbox.wait_for_vm()
    assert client.machines.ready_calls == ["m-1"]
    assert "docker compose version" in client.machines.exec_calls[0]["command"]


@pytest.mark.asyncio
async def test_wait_for_vm_retries_a_docker_probe_whose_stream_dropped(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    sandbox, client = await _sandbox()
    client.machines.results = [APIConnectionError("unavailable"), result(stdout=b"Docker Compose version v2")]
    await sandbox.wait_for_vm()
    assert len(client.machines.exec_calls) == 2


@pytest.mark.asyncio
async def test_wait_for_vm_reports_a_docker_that_never_answers():
    sandbox, client = await _sandbox()
    client.machines.results = [result(stderr=b"docker daemon did not become ready", exit_code=1)]
    with pytest.raises(RuntimeError, match=r"Docker not ready on boxd machine m-1: (?s:.*)docker daemon did not become ready"):
        await sandbox.setup_vm_for_gateway([8080])


@pytest.mark.asyncio
async def test_terminate_deletes_the_machine_and_tolerates_one_already_gone():
    sandbox, client = await _sandbox()
    await sandbox.terminate()
    await sandbox.terminate()
    assert client.machines.deleted == ["m-1"]


@pytest.mark.parametrize(("host", "allowed"), [
    ("pypi.org", True),
    ("files.pythonhosted.org", True),
    ("a.b.pythonhosted.org", True),
    ("pythonhosted.org", False),
    ("203.0.113.9", True),
    ("198.51.100.7", True),
    ("198.51.100.8", False),
    ("evil.com", False),
])
def test_allows_host_follows_boxd_wildcards_at_any_depth(host, allowed):
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org", "*.pythonhosted.org", "198.51.100.7"),
                           allow_cidrs=("203.0.113.0/24",))
    assert allows_host(policy, host) is allowed
    assert allows_host(NetworkPolicy(), host)


def test_policy_from_entries():
    assert policy_from_entries([]) == NetworkPolicy()
    assert policy_from_entries(["1.1.1.1", "203.0.113.0/24", "example.com"]) == NetworkPolicy(
        mode=ALLOWLIST, allow_hosts=("example.com",), allow_cidrs=("1.1.1.1", "203.0.113.0/24"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("policy", "signed", "curled"), [
    (NetworkPolicy(), "https://bucket.s3.amazonaws.com/obj?sig", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",)), "https://bucket.s3.amazonaws.com/obj?sig", False),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("bucket.s3.amazonaws.com",)), "https://bucket.s3.amazonaws.com/o", True),
    (None, "https://bucket.s3.amazonaws.com/obj?sig", False),
    (NetworkPolicy(), None, False),
])
async def test_a_download_the_machine_cannot_fetch_goes_over_exec(monkeypatch, policy, signed, curled):
    store = SimpleNamespace(signed_get_url=lambda _url: signed)
    monkeypatch.setattr(sandbox_module, "get_config", lambda: SimpleNamespace(get_object_store_at=lambda _url: store))
    pushed = []

    async def push(object_store, object_url, vm_path):
        pushed.append(vm_path)

    sandbox, client = await _sandbox(policy=policy)
    monkeypatch.setattr(sandbox, "_write_unsigned_object", push)
    await sandbox._download_object_to_vm("s3://b/obj", "/tmp/obj")
    assert any("curl" in call["command"] for call in client.machines.exec_calls) is curled
    assert bool(pushed) is not curled
