"""Offline unit tests for the kosa8 sandbox: exec, its limits, file writes, readiness and egress."""

from __future__ import annotations

import httpx
import pytest

from agent_env.providers.sandbox_providers.kosa8 import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.kosa8.sandbox import (
    Kosa8Api,
    Kosa8Error,
    Kosa8Sandbox,
    allows_host,
    denied_entries,
)
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from tst.unit.providers.sandbox_providers.kosa8_fakes import API_KEY, FakeKosa8, command_of, error, frames

ALLOWLIST = NetworkMode.ALLOWLIST


@pytest.fixture(autouse=True)
def _no_resend_delays(monkeypatch):
    monkeypatch.setattr(sandbox_module, "RESEND_DELAYS", (0, 0, 0, 0, 0))


@pytest.fixture
def api():
    fake = FakeKosa8()
    fake.sandboxes["sbx_1"] = {"id": "sbx_1", "state": "running", "name": "agentenv-x", "polls": 0}
    return fake


def _sandbox(api: FakeKosa8, policy: NetworkPolicy | None = NetworkPolicy()) -> Kosa8Sandbox:
    return Kosa8Sandbox(Kosa8Api(API_KEY, transport=api.transport()), "sbx_1", tunnel_urls={}, network_policy=policy)


@pytest.mark.asyncio
async def test_exec_runs_argv_as_root_with_no_input(api):
    api.exec_answers = [{"stdout": b"hi there\n"}]
    assert await _sandbox(api).exec_with_output("sudo", "echo", "hi there") == (0, "hi there\n", "")
    [argv] = api.execs
    assert argv == ["bash", "-c", 'exec "$@" </dev/null', "kosa8-exec", "echo", "hi there"]


@pytest.mark.asyncio
async def test_exec_keeps_stdout_and_stderr_apart_and_skips_heartbeats(api):
    api.exec_answers = [frames(b"out\n", b"err\n", 3, heartbeats=2)]
    assert await _sandbox(api).exec_with_output("bash", "-c", "exit 3") == (3, "out\n", "err\n")


@pytest.mark.asyncio
async def test_a_command_past_kosa8s_argv_limits_goes_through_a_file(api):
    script = "echo " + "x" * 20000
    await _sandbox(api).exec_with_output("sudo", "bash", "-c", script)
    [argv] = api.execs
    assert "mapfile" in argv[2] and argv[4].startswith("/workspace/.agentenv/argv-")
    assert command_of(argv, api, "sbx_1") == ["bash", "-c", script]
    assert api.dirs == [("sbx_1", "/workspace/.agentenv")]
    many = [str(index) for index in range(300)]
    await _sandbox(api).exec_with_output("printf", *many)
    assert command_of(api.execs[-1], api, "sbx_1") == ["printf", *many]


@pytest.mark.asyncio
async def test_a_stream_that_ends_without_an_exit_status_is_exit_minus_one_with_no_stdout(api):
    api.exec_answers = [frames(b"partial", exit_code=None), frames(b"partial", error_text="host lost")]
    sandbox = _sandbox(api)
    code, stdout, stderr = await sandbox.exec_with_output("cat", "big")
    assert (code, stdout) == (-1, "") and "before the command's exit status" in stderr
    code, stdout, stderr = await sandbox.exec_with_output("cat", "big")
    assert (code, stdout) == (-1, "") and "host lost" in stderr


@pytest.mark.asyncio
async def test_a_dropped_stream_is_exit_minus_one_and_exec_script_retries_it(api):
    api.exec_answers = [httpx.ReadError("reset"), {"stdout": b"ok\n"}]
    assert await _sandbox(api).exec_script("echo ok", max_retries=1) == "ok\n"
    assert len(api.execs) == 2


@pytest.mark.asyncio
async def test_an_exec_refused_before_it_started_is_resent(api):
    api.exec_answers = [error(409, "starting"), error(429, "rate_limited"), httpx.ConnectError("refused"),
                        {"stdout": b"ok\n"}]
    assert await _sandbox(api).exec_with_output("true") == (0, "ok\n", "")
    assert len(api.execs) == 4


@pytest.mark.asyncio
async def test_an_exec_that_may_have_run_is_not_resent(api):
    api.exec_answers = [error(502, "host_failed")]
    code, _, stderr = await _sandbox(api).exec_with_output("touch", "/x")
    assert code == -1 and "host_failed" in stderr
    assert len(api.execs) == 1


@pytest.mark.asyncio
async def test_an_exec_kosa8_refuses_raises(api):
    api.exec_answers = [error(402, "payment_required")]
    with pytest.raises(Kosa8Error, match="402 payment_required"):
        await _sandbox(api).exec_with_output("true")


@pytest.mark.asyncio
async def test_write_host_file_uploads_through_the_files_api_in_pieces(api, monkeypatch):
    monkeypatch.setattr(sandbox_module, "FILE_PIECE_BYTES", 4)
    sandbox = _sandbox(api)
    await sandbox.write_host_file(b"0123456789", "/etc/agentenv/secret")
    pieces = sorted(path for (_, path) in api.files if "/upload-" in path)
    assert [api.files[("sbx_1", path)] for path in pieces] == [b"0123", b"4567", b"89"]
    scripts = [command_of(argv, api, "sbx_1")[2] for argv in api.execs]
    assert scripts[0] == "mkdir -p /etc/agentenv"
    assert scripts[1].startswith("cat ") and scripts[1].count(".agentenv/upload-") == 6
    assert all("0123" not in script for script in scripts)


@pytest.mark.asyncio
async def test_write_host_file_writes_an_empty_file(api):
    await _sandbox(api).write_host_file(b"", "/tmp/empty")
    assert [data for (_, path), data in api.files.items() if "/upload-" in path] == [b""]


@pytest.mark.asyncio
async def test_wait_for_vm_names_the_sandbox_when_docker_never_readies(api):
    api.exec_result = lambda argv: {"exit": 1, "stderr": b"docker daemon did not become ready"}
    with pytest.raises(RuntimeError, match="Docker not ready on kosa8 sandbox sbx_1"):
        await _sandbox(api).wait_for_vm()


@pytest.mark.parametrize(("policy", "host", "allowed"), [
    (NetworkPolicy(), "anything.example", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("s3.amazonaws.com",)), "s3.amazonaws.com", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("*.amazonaws.com",)), "b.s3.amazonaws.com", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("*.amazonaws.com",)), "amazonaws.com", False),
    (NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("203.0.113.0/24",)), "203.0.113.9", True),
    (NetworkPolicy(mode=ALLOWLIST), "storage.googleapis.com", False),
])
def test_allows_host_follows_kosa8s_matching(policy, host, allowed):
    assert allows_host(policy, host) is allowed


@pytest.mark.parametrize(("policy", "url", "fetches"), [
    (NetworkPolicy(), "https://bucket.s3.amazonaws.com/x?sig=1", True),
    (NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",)), "https://bucket.s3.amazonaws.com/x", False),
    (None, "https://bucket.s3.amazonaws.com/x", False),
    (NetworkPolicy(), None, False),
])
def test_the_sandbox_downloads_only_what_its_policy_admits(api, policy, url, fetches):
    assert _sandbox(api, policy)._can_download(url) is fetches


@pytest.mark.parametrize(("enforced", "denied"), [
    ("any", ()),
    ("deny", ()),
    ("allow:pypi.org", ()),
    ("allow:*.amazonaws.com;deny:secret.s3.amazonaws.com,10.0.0.0/8", ("secret.s3.amazonaws.com", "10.0.0.0/8")),
    ("block:*.amazonaws.com", ("*.amazonaws.com",)),
])
def test_denied_entries_are_what_a_compiled_policy_denies(enforced, denied):
    assert denied_entries(enforced) == denied


@pytest.mark.parametrize(("enforced", "url", "fetches"), [
    ("block:*.amazonaws.com", "https://bucket.s3.amazonaws.com/x?sig=1", False),
    ("block:gist.github.com", "https://bucket.s3.amazonaws.com/x?sig=1", True),
    ("allow:*.amazonaws.com;deny:bucket.s3.amazonaws.com", "https://bucket.s3.amazonaws.com/x", False),
    ("allow:*.amazonaws.com;deny:other.s3.amazonaws.com", "https://bucket.s3.amazonaws.com/x", True),
])
def test_a_host_the_enforced_policy_denies_is_pushed_rather_than_downloaded(api, enforced, url, fetches):
    sandbox = _sandbox(api)
    sandbox.enforce(enforced)
    assert sandbox._can_download(url) is fetches
