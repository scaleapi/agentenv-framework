"""Offline unit tests for the Daytona sandbox: exec over sessions, readiness, terminate, the SDK handle, uploads and
downloads."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from daytona import (
    DaytonaBadGatewayError,
    DaytonaConflictError,
    DaytonaConnectionError,
    DaytonaConnectionTimeoutError,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaServiceUnavailableError,
    DaytonaSessionEndedError,
    DaytonaTimeoutError,
)

from agent_env.providers.sandbox_providers import sandbox as base_sandbox_module
from agent_env.providers.sandbox_providers.daytona import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.daytona.image import DOCKER_START_SCRIPT
from agent_env.providers.sandbox_providers.daytona.sandbox import DaytonaSandbox
from agent_env.providers.sandbox_providers.daytona.transfer import UPLOAD_CHUNK_BYTES
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy

from tst.unit.providers.sandbox_providers.daytona_fakes import FakeDaytona, FakeFs, FakeRaw, joining_handler

MIB = 1024 * 1024
SIGNED = "https://bucket.s3.amazonaws.com/obj?X-Amz-Signature=abc"


def allow(*hosts: str) -> NetworkPolicy:
    return NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=hosts)


def _sandbox(handler=None, *, policy: NetworkPolicy | None = NetworkPolicy(),
             **raw_fields) -> tuple[DaytonaSandbox, FakeRaw]:
    raw = FakeRaw(**raw_fields)
    if handler is not None:
        raw.process.handler = handler
    sandbox, _ = _on_clients(raw, policy=policy)
    return sandbox, raw


def _on_clients(raw: FakeRaw, *,
                policy: NetworkPolicy | None = NetworkPolicy()) -> tuple[DaytonaSandbox, list[FakeDaytona]]:
    """A sandbox over ``raw`` whose provider's current client is ``current[0]``, which a test may replace."""
    current = [FakeDaytona(sandboxes={raw.id: raw})]
    sandbox = DaytonaSandbox(lambda: current[0], raw, tunnel_urls={8080: "https://u"}, network_policy=policy)
    sandbox._DELETE_RETRY_DELAY = 0
    return sandbox, current


def _fast(sandbox: DaytonaSandbox) -> DaytonaSandbox:
    sandbox._VM_READY_TIMEOUT = 0.2
    sandbox._VM_READY_POLL_INTERVAL = 0.01
    return sandbox


def _parts(raw: FakeRaw) -> list[bytes]:
    paths = [path for path in raw.fs.files if path.startswith("/tmp/_ae_upload_")]
    return [raw.fs.files[path] for path in sorted(paths, key=lambda path: int(path.rsplit(".", 1)[1]))]


async def _no_sleep(_seconds):
    return None


class _Store:
    def __init__(self, objects: dict[str, bytes], signed: dict[str, str | None]):
        self._objects = objects
        self._signed = signed

    def signed_get_url(self, object_url: str) -> str | None:
        return self._signed[object_url]

    def open(self, object_url: str) -> io.BytesIO:
        return io.BytesIO(self._objects[object_url])


def _use_store(monkeypatch, store: _Store) -> None:
    config = SimpleNamespace(get_object_store_at=lambda object_url: store)
    monkeypatch.setattr(sandbox_module, "get_config", lambda: config)
    monkeypatch.setattr(base_sandbox_module, "get_config", lambda: config)


def test_sandbox_presents_as_a_daytona_vm():
    sandbox, raw = _sandbox()
    assert (sandbox.type, sandbox.mode, sandbox.sandbox_id, sandbox.vnc_url) == ("daytona", "vm", raw.id, None)
    assert sandbox.tunnel_urls == {8080: "https://u"} and sandbox.network_policy == NetworkPolicy()


@pytest.mark.asyncio
@pytest.mark.parametrize("argv", [
    ("sudo", "bash", "-c", "echo 'a b'; exit 3"),
    ("printf", "%s\n", "$HOME", "`id`", 'say "hi"', "", "two\nlines", "a;b && c | d", "*"),
])
async def test_exec_runs_the_exact_argv(argv):
    sandbox, raw = _sandbox()
    await sandbox.exec_with_output(*argv)
    assert raw.process.commands == [list(argv)]


@pytest.mark.asyncio
async def test_exec_forwards_the_exit_code_and_keeps_stdout_and_stderr_apart():
    sandbox, _ = _sandbox(lambda argv: (3, "out\n", "err\n"))
    assert await sandbox.exec_with_output("true") == (3, "out\n", "err\n")


@pytest.mark.asyncio
async def test_a_missing_exit_code_reads_as_minus_one_and_missing_output_as_empty():
    sandbox, _ = _sandbox(lambda argv: (None, None, None))
    assert await sandbox.exec_with_output("true") == (-1, "", "")


@pytest.mark.asyncio
async def test_each_command_runs_unbounded_in_a_session_of_its_own_that_is_deleted_after():
    sandbox, raw = _sandbox()
    for _ in range(3):
        await sandbox.exec_with_output("true")
    process = raw.process
    assert len(set(process.created_sessions)) == 3 and process.deleted_sessions == process.created_sessions
    assert process.open_sessions == set() and process.timeouts == [None, None, None]


@pytest.mark.asyncio
async def test_every_session_is_deleted_even_when_the_command_raises():
    sandbox, raw = _sandbox(lambda argv: DaytonaError("boom"))
    with pytest.raises(DaytonaError, match="boom"):
        await sandbox.exec_with_output("true")
    assert raw.process.open_sessions == set() and len(raw.process.deleted_sessions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    DaytonaConnectionError, DaytonaConnectionTimeoutError, DaytonaBadGatewayError, DaytonaServiceUnavailableError,
    DaytonaSessionEndedError, DaytonaTimeoutError,
])
async def test_a_transport_failure_is_exit_minus_one_with_no_stdout(error):
    sandbox, raw = _sandbox(lambda argv: error("dropped"))
    exit_code, stdout, stderr = await sandbox.exec_with_output("true")
    assert (exit_code, stdout) == (-1, "") and f"{error.__name__}: dropped" in stderr
    assert raw.process.open_sessions == set()


@pytest.mark.asyncio
async def test_a_transport_failure_reading_the_handle_is_exit_minus_one():
    raw = FakeRaw()
    sandbox, current = _on_clients(raw)

    async def unreachable(sandbox_id, request_timeout=None):
        raise DaytonaConnectionError("reset")

    current[0] = FakeDaytona()
    current[0].get = unreachable
    exit_code, stdout, stderr = await sandbox.exec_with_output("true")
    assert (exit_code, stdout) == (-1, "") and "DaytonaConnectionError" in stderr
    assert raw.process.commands == []


@pytest.mark.asyncio
async def test_exec_script_retries_a_dropped_connection(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    outcomes = [DaytonaConnectionError("reset"), (0, "ok\n", "")]
    sandbox, raw = _sandbox(lambda argv: outcomes.pop(0))
    assert await sandbox.exec_script("echo ok", max_retries=1) == "ok\n"
    assert len(raw.process.commands) == 2


@pytest.mark.asyncio
async def test_a_dropped_connection_without_retries_fails_the_script_with_exit_minus_one():
    sandbox, _ = _sandbox(lambda argv: DaytonaConnectionError("reset"))
    with pytest.raises(RuntimeError, match=r"exit -1"):
        await sandbox.exec_script("echo ok")


@pytest.mark.asyncio
async def test_handle_keeps_the_sdk_handle_while_the_client_is_unchanged():
    raw = FakeRaw()
    sandbox, current = _on_clients(raw)
    assert await sandbox.handle() is raw and await sandbox.handle() is raw
    assert current[0].lookups == []


@pytest.mark.asyncio
async def test_handle_reads_the_sandbox_again_once_on_a_new_client():
    raw = FakeRaw()
    sandbox, current = _on_clients(raw)
    fresh = FakeRaw(id=raw.id)
    current[0] = FakeDaytona(sandboxes={raw.id: fresh})
    assert await sandbox.handle() is fresh and await sandbox.handle() is fresh
    assert current[0].lookups == [raw.id]
    await sandbox.exec_with_output("true")
    assert (raw.process.commands, fresh.process.commands) == ([], [["true"]])


@pytest.mark.asyncio
async def test_terminate_deletes_the_sandbox():
    sandbox, raw = _sandbox()
    await sandbox.terminate()
    assert (raw.deletes, raw.state) == (1, "destroyed")


@pytest.mark.asyncio
async def test_terminate_of_a_gone_sandbox_is_not_an_error():
    sandbox, raw = _sandbox(delete_error=DaytonaNotFoundError("gone"))
    await sandbox.terminate()
    assert raw.deletes == 1


@pytest.mark.asyncio
async def test_terminate_of_a_sandbox_gone_before_its_handle_is_read_again_is_not_an_error():
    raw = FakeRaw()
    sandbox, current = _on_clients(raw)
    current[0] = FakeDaytona()
    await sandbox.terminate()
    assert current[0].lookups == [raw.id] and raw.deletes == 0


@pytest.mark.asyncio
async def test_terminate_of_a_sandbox_daytona_is_already_deleting_is_not_an_error():
    sandbox, raw = _sandbox(delete_error=DaytonaConflictError("state change in progress"), desired_state="destroyed")
    await sandbox.terminate()
    assert raw.deletes == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("second", [DaytonaConflictError("Sandbox state change in progress"),
                                    DaytonaNotFoundError("Sandbox not found")])
async def test_a_second_terminate_after_a_completed_delete_is_not_an_error(second):
    sandbox, raw = _sandbox()
    await sandbox.terminate()
    raw.delete_failures = [second]
    await sandbox.terminate()
    assert (raw.deletes, raw.delete_failures) == (2, [])


@pytest.mark.asyncio
async def test_terminate_retries_a_409_for_another_state_change_in_progress():
    sandbox, raw = _sandbox(delete_failures=[DaytonaConflictError("state change in progress")])
    await sandbox.terminate()
    assert (raw.deletes, raw.state) == (2, "destroyed")


@pytest.mark.asyncio
async def test_terminate_gives_up_on_a_409_that_never_clears():
    sandbox, raw = _sandbox(delete_error=DaytonaConflictError("state change in progress"))
    with pytest.raises(DaytonaConflictError):
        await sandbox.terminate()
    assert raw.deletes == sandbox._DELETE_ATTEMPTS


@pytest.mark.asyncio
async def test_terminate_surfaces_other_failures():
    sandbox, raw = _sandbox(delete_error=DaytonaError("server error"))
    with pytest.raises(DaytonaError, match="server error"):
        await sandbox.terminate()
    assert raw.deletes == 1


@pytest.mark.asyncio
async def test_wait_for_vm_starts_dockerd_then_polls_docker_info_with_bounded_commands_until_it_answers():
    answers = iter([(1, "", "no socket"), (1, "", "no socket"), (0, "29.9.1\n", "")])

    def handler(argv):
        return next(answers) if argv[:3] == ["sudo", "docker", "info"] else (0, "", "")

    sandbox, raw = _sandbox(handler)
    await _fast(sandbox).wait_for_vm()
    assert raw.process.commands[0] == ["sudo", "bash", "-c", DOCKER_START_SCRIPT]
    assert sum(cmd[:3] == ["sudo", "docker", "info"] for cmd in raw.process.commands) == 3
    assert all(timeout is not None and timeout >= 1 for timeout in raw.process.timeouts)


@pytest.mark.asyncio
async def test_wait_for_vm_gives_up_with_the_dockerd_log_tail():
    def handler(argv):
        if argv[:2] == ["sudo", "tail"]:
            return (0, "level=fatal msg=boom\n", "")
        return (1, "", "no socket") if argv[:3] == ["sudo", "docker", "info"] else (0, "", "")

    sandbox, _ = _sandbox(handler)
    with pytest.raises(RuntimeError, match="level=fatal msg=boom"):
        await _fast(sandbox).wait_for_vm()


@pytest.mark.asyncio
async def test_setup_for_gateway_requires_the_compose_v2_plugin():
    def handler(argv):
        return (1, "", "'compose' is not a docker command") if argv[2:4] == ["compose", "version"] else (0, "", "")

    sandbox, _ = _sandbox(handler)
    with pytest.raises(RuntimeError, match="Compose v2"):
        await _fast(sandbox).setup_vm_for_gateway([8080])


@pytest.mark.asyncio
async def test_setup_for_gateway_opens_no_firewall_ports():
    sandbox, raw = _sandbox(lambda argv: (0, "v5\n", ""))
    await _fast(sandbox).setup_vm_for_gateway([8080])
    assert not any("iptables" in " ".join(cmd) for cmd in raw.process.commands)


@pytest.mark.asyncio
async def test_a_large_write_goes_up_in_parts_and_is_joined_as_root_with_a_checksum():
    sandbox, raw = _sandbox()
    raw.process.handler = joining_handler(raw)
    payload = bytes(range(256)) * (9 * MIB // 256) + b"tail-17-bytes!!!!"
    await sandbox.write_host_file(payload, "/opt/smoke/payload.bin")
    parts = _parts(raw)
    assert len(parts) == 3 and all(len(part) <= UPLOAD_CHUNK_BYTES for part in parts) and b"".join(parts) == payload
    join = next(cmd for cmd in raw.process.commands if "sha256sum" in cmd[-1])
    assert join[:3] == ["sudo", "bash", "-c"] and "> /opt/smoke/payload.bin" in join[-1]
    assert any("rm -f" in cmd[-1] for cmd in raw.process.commands)


@pytest.mark.asyncio
async def test_an_empty_write_is_joined_and_checked_too():
    sandbox, raw = _sandbox()
    raw.process.handler = joining_handler(raw)
    await sandbox.write_host_file(b"", "/opt/empty")
    assert b"".join(_parts(raw)) == b""
    assert any("> /opt/empty" in cmd[-1] and "sha256sum" in cmd[-1] for cmd in raw.process.commands)


@pytest.mark.asyncio
async def test_a_checksum_mismatch_fails_the_write():
    sandbox, raw = _sandbox()
    raw.process.handler = joining_handler(raw, corrupt=True)
    with pytest.raises(RuntimeError, match="does not match the upload"):
        await sandbox.write_host_file(b"hello", "/opt/x")


@pytest.mark.asyncio
async def test_a_failed_part_upload_propagates_and_still_cleans_up_parts():
    sandbox, raw = _sandbox(fs=FakeFs(fail_on=lambda path: OSError("upload refused") if path.endswith(".1") else None))
    raw.process.handler = joining_handler(raw)
    with pytest.raises(OSError, match="upload refused"):
        await sandbox.write_host_file(b"a" * (5 * MIB), "/opt/x")
    assert any("rm -f" in cmd[-1] for cmd in raw.process.commands)
    assert not any("sha256sum" in cmd[-1] for cmd in raw.process.commands)


@pytest.mark.asyncio
async def test_uploads_go_through_the_handle_on_the_current_client():
    raw = FakeRaw()
    sandbox, current = _on_clients(raw)
    fresh = FakeRaw(id=raw.id)
    fresh.process.handler = joining_handler(fresh)
    current[0] = FakeDaytona(sandboxes={raw.id: fresh})
    await sandbox.write_host_file(b"hello", "/opt/x")
    assert (raw.fs.files, b"".join(_parts(fresh))) == ({}, b"hello")


@pytest.mark.parametrize(("policy", "url", "downloads"), [
    (NetworkPolicy(), SIGNED, True),
    (allow("bucket.s3.amazonaws.com"), SIGNED, True),
    (allow("*.amazonaws.com"), SIGNED, True),
    (allow("pypi.org"), SIGNED, False),
    (allow("*.bucket.s3.amazonaws.com"), SIGNED, False),
    (allow(), SIGNED, False),
    (None, SIGNED, False),
    (NetworkPolicy(), None, False),
    (NetworkPolicy(), "file:///tmp/obj", False),
])
def test_can_download_only_what_the_policy_admits(policy, url, downloads):
    sandbox, _ = _sandbox(policy=policy)
    assert sandbox._can_download(url) is downloads


@pytest.mark.asyncio
@pytest.mark.parametrize(("policy", "signed", "curled"), [
    (NetworkPolicy(), SIGNED, True),
    (allow("bucket.s3.amazonaws.com"), SIGNED, True),
    (allow("pypi.org"), SIGNED, False),
    (None, SIGNED, False),
    (NetworkPolicy(), None, False),
])
async def test_an_object_is_curled_when_the_policy_admits_its_url_and_uploaded_otherwise(monkeypatch, policy, signed,
                                                                                         curled):
    payload = b"object bytes" * 1000
    _use_store(monkeypatch, _Store({"s3://b/obj": payload}, {"s3://b/obj": signed}))
    sandbox, raw = _sandbox(policy=policy)
    raw.process.handler = joining_handler(raw)
    await sandbox._download_object_to_vm("s3://b/obj", "/tmp/obj")
    curls = [cmd[-1] for cmd in raw.process.commands if cmd[-1].startswith("curl ")]
    assert [SIGNED in curl and curl.endswith("-o /tmp/obj") for curl in curls] == ([True] if curled else [])
    assert b"".join(_parts(raw)) == (b"" if curled else payload)
    assert (raw.network_updates, sandbox.network_policy) == ([], policy)


@pytest.mark.asyncio
@pytest.mark.parametrize(("policy", "curled"), [
    (NetworkPolicy(), {"img-a", "img-b"}),
    (allow("a.example.com"), {"img-a"}),
    (allow("pypi.org"), set()),
    (None, set()),
])
async def test_tarballs_the_policy_cannot_download_are_uploaded_through_the_file_api(monkeypatch, policy, curled):
    names = ("img-a", "img-b")
    objects = {f"s3://b/{name}.tar.gz": name.encode() * 100 for name in names}
    signed = {f"s3://b/{name}.tar.gz": f"https://{name[-1]}.example.com/{name}?sig=1" for name in names}
    _use_store(monkeypatch, _Store(objects, signed))

    def images(argv):
        return (0, "REPOSITORY TAG\nimg-a 1\nimg-b 1\n", "") if argv == ["sudo", "docker", "images"] else (0, "", "")

    sandbox, raw = _sandbox(policy=policy)
    raw.process.handler = joining_handler(raw, otherwise=images)
    artifacts = [SimpleNamespace(image_name=f"{name}:1", tar_gz_object_url=f"s3://b/{name}.tar.gz") for name in names]
    await sandbox._load_tarballs(artifacts)
    (load,) = [cmd[-1] for cmd in raw.process.commands if "docker load" in cmd[-1]]
    joins = [cmd[-1] for cmd in raw.process.commands if "sha256sum" in cmd[-1]]
    assert {name for name in names if signed[f"s3://b/{name}.tar.gz"] in load} == curled
    assert {name for index, name in enumerate(names)
            if any(f"_docker_image_{raw.id}_{index}.tar.gz" in join for join in joins)} == set(names) - curled
    assert (raw.network_updates, sandbox.network_policy) == ([], policy)
