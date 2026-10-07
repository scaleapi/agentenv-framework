"""Unit tests for the Sailbox adapter over a fake SDK Sailbox."""

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.providers.sandbox_providers.sail_vm.sandbox import SailVmSandbox, egress_document, policy_from_document
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy


class _HostLost(Exception):
    pass


class _NotFound(Exception):
    pass


_SDK = SimpleNamespace(SailboxHostLostError=_HostLost, TransportError=_HostLost, NotFoundError=_NotFound)
_ALLOWLIST = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("10.0.0.0/8",))


async def _chunks(*parts: bytes):
    for part in parts:
        yield part


def _process(stdout=b"", stderr=b"", exit_code=0):
    process = MagicMock()
    process.stdout_bytes = _chunks(stdout[:3], stdout[3:])
    process.stderr_bytes = _chunks(stderr)
    process.wait = AsyncMock(return_value=SimpleNamespace(exit_code=exit_code))
    return process


def _sandbox(*outcomes, policy=NetworkPolicy()):
    sailbox = MagicMock(sailbox_id="sb_1")
    sailbox.exec.aio = AsyncMock(side_effect=list(outcomes))
    sailbox.terminate.aio = AsyncMock()
    sailbox.fs.write.aio = AsyncMock()
    applied = {"document": egress_document(policy) if policy is not None else {"rules": []}}

    async def set_policy(document):
        await asyncio.sleep(0)
        applied["document"] = document

    sailbox.set_egress_policy.aio = AsyncMock(side_effect=set_policy)
    sdk = SimpleNamespace(**vars(_SDK), Sailbox=SimpleNamespace(get=SimpleNamespace(aio=AsyncMock(
        side_effect=lambda _id: SimpleNamespace(egress_policy=SimpleNamespace(policy_id=None, document=applied["document"]))
    ))))
    return SailVmSandbox(sailbox, sdk=sdk, tunnel_urls={}, network_policy=policy), sailbox


@pytest.mark.asyncio
async def test_exec_returns_exact_bytes_and_exit_code_and_drops_a_leading_sudo():
    binary = bytes(range(256))
    sandbox, sailbox = _sandbox(_process(stdout=binary, stderr=b"warn\n", exit_code=7))

    process = await sandbox.exec("sudo", "bash", "-c", "echo 'a b'", "sudo")

    assert (await process.stdout.read(), await process.stderr.read(), await process.wait()) == (binary, b"warn\n", 7)
    args, kwargs = sailbox.exec.aio.await_args
    assert args == (["bash", "-c", "echo 'a b'", "sudo"],)
    assert kwargs["output_mode"] == "pipe"
    assert kwargs["timeout"] is None
    assert len(kwargs["idempotency_key"]) == 32


@pytest.mark.asyncio
async def test_stdout_streams_chunk_by_chunk_for_large_reads():
    sandbox, _ = _sandbox(_process(stdout=b"abcdefgh"))
    process = await sandbox.exec("cat", "big")
    assert [chunk async for chunk in process.stdout] == [b"abc", b"defgh"]
    assert await process.wait() == 0


@pytest.mark.asyncio
async def test_a_host_lost_while_the_command_runs_is_exit_minus_one():
    process = _process(stdout=b"partial")
    process.wait = AsyncMock(side_effect=_HostLost("migrated"))
    sandbox, _ = _sandbox(process)
    assert (await sandbox.exec_with_output("true"))[0] == -1


@pytest.mark.asyncio
async def test_output_nobody_reads_is_drained_by_wait_rather_than_blocking_the_command():
    process = MagicMock()

    async def many_chunks():
        for _ in range(1000):
            yield b"x" * 1024

    process.stdout_bytes = many_chunks()
    process.stderr_bytes = _chunks(b"")
    process.wait = AsyncMock(return_value=SimpleNamespace(exit_code=0))
    sandbox, _ = _sandbox(process)

    assert await asyncio.wait_for((await sandbox.exec("yes")).wait(), timeout=5) == 0


async def _failing_chunks(error):
    yield b"partial"
    raise error


@pytest.mark.asyncio
async def test_a_broken_stream_is_raised_not_reported_as_success():
    process = _process()
    process.stdout_bytes = _failing_chunks(ValueError("stream corrupted"))
    sandbox, _ = _sandbox(process)
    running = await sandbox.exec("cat", "f")
    with pytest.raises(ValueError, match="stream corrupted"):
        await running.stdout.read()
    with pytest.raises(ValueError, match="stream corrupted"):
        await running.wait()


@pytest.mark.asyncio
async def test_a_stream_cut_by_a_migration_ends_and_waits_to_exit_minus_one():
    process = _process()
    process.stdout_bytes = _failing_chunks(_HostLost("moved"))
    sandbox, _ = _sandbox(process)
    running = await sandbox.exec("cat", "f")
    assert await running.stdout.read() == b"partial"
    assert await running.wait() == -1


@pytest.mark.asyncio
async def test_exec_with_output_never_returns_the_partial_stdout_of_a_cut_stream():
    process = _process(stderr=b"moved\n")
    process.stdout_bytes = _failing_chunks(_HostLost("moved"))
    sandbox, _ = _sandbox(process)
    assert await sandbox.exec_with_output("cat", "trajectory.jsonl") == (-1, "", "moved\n")


@pytest.mark.asyncio
async def test_exec_script_retries_a_command_whose_output_stream_was_cut(monkeypatch):
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sandbox.asyncio.sleep", AsyncMock())
    cut = _process()
    cut.stdout_bytes = _failing_chunks(_HostLost("moved"))
    sandbox, sailbox = _sandbox(cut, _process(stdout=b"loaded\n"))

    assert await sandbox.exec_script("docker load < image.tar", max_retries=1) == "loaded\n"
    assert sailbox.exec.aio.await_count == 2


@pytest.mark.asyncio
async def test_each_exec_gets_its_own_idempotency_key():
    sandbox, sailbox = _sandbox(_process(), _process())
    await sandbox.exec("true")
    await sandbox.exec("true")
    keys = [call.kwargs["idempotency_key"] for call in sailbox.exec.aio.await_args_list]
    assert keys[0] != keys[1]


@pytest.mark.asyncio
async def test_a_lost_host_is_exit_minus_one_which_exec_script_retries(monkeypatch):
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sandbox.asyncio.sleep", AsyncMock())
    sandbox, sailbox = _sandbox(_HostLost("host gone"), _process(stdout=b"ok\n"))

    assert await sandbox.exec_script("echo ok", max_retries=1) == "ok\n"
    assert sailbox.exec.aio.await_count == 2


@pytest.mark.asyncio
async def test_other_sdk_errors_propagate():
    sandbox, _ = _sandbox(PermissionError("Invalid API key"))
    with pytest.raises(PermissionError):
        await sandbox.exec("true")


@pytest.mark.asyncio
async def test_wait_for_vm_starts_dockerd_once_when_it_is_not_running(monkeypatch):
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sail_vm.sandbox.asyncio.sleep", AsyncMock())
    sandbox, sailbox = _sandbox(
        _process(stderr=b"Cannot connect to the Docker daemon", exit_code=1),
        _process(),
        _process(stderr=b"still starting", exit_code=1),
        _process(stdout=b"20.10.24\n"),
    )

    await sandbox.wait_for_vm()

    commands = [call.args[0] for call in sailbox.exec.aio.await_args_list]
    assert commands[0][:2] == ["docker", "info"]
    assert commands[1][:2] == ["bash", "-c"] and "nohup dockerd" in commands[1][2]
    assert [c[:2] for c in commands[2:]] == [["docker", "info"], ["docker", "info"]]
    assert sailbox.exec.aio.await_args_list[0].kwargs["timeout"] == SailVmSandbox._DOCKER_PROBE_TIMEOUT


@pytest.mark.asyncio
async def test_wait_for_vm_reports_the_last_error_when_docker_never_answers(monkeypatch):
    monkeypatch.setattr(SailVmSandbox, "_VM_READY_TIMEOUT", 0)
    sandbox, _ = _sandbox(_process(stderr=b"daemon down", exit_code=1))
    with pytest.raises(RuntimeError, match="Docker not ready in Sailbox sb_1.*daemon down"):
        await sandbox.wait_for_vm()


@pytest.mark.asyncio
async def test_setup_requires_compose_v2(monkeypatch):
    monkeypatch.setattr(SailVmSandbox, "wait_for_vm", AsyncMock())
    sandbox, _ = _sandbox(_process(stderr=b"docker: 'compose' is not a docker command", exit_code=1))
    with pytest.raises(RuntimeError, match="no Docker Compose v2"):
        await sandbox.setup_vm_for_gateway([8080])


@pytest.mark.asyncio
async def test_host_files_are_written_with_the_native_filesystem_api():
    sandbox, sailbox = _sandbox(_process())
    await sandbox.write_host_file(b"\x00\x01" * 100_000, "/opt/data/blob.bin")
    sailbox.fs.write.aio.assert_awaited_once_with("/opt/data/blob.bin", b"\x00\x01" * 100_000)


@pytest.mark.asyncio
async def test_terminate_tolerates_an_already_deleted_sailbox():
    sandbox, sailbox = _sandbox()
    sailbox.terminate.aio.side_effect = _NotFound("sailbox not found")
    await sandbox.terminate()


@pytest.mark.asyncio
async def test_image_loading_fails_closed_when_the_policy_is_unknown():
    sandbox, _ = _sandbox(policy=None)
    with pytest.raises(RuntimeError, match="applied egress policy is unknown"):
        await sandbox.load_docker_images([object()])


@pytest.mark.asyncio
async def test_image_loading_widens_an_allowlist_with_the_signed_download_hosts(monkeypatch):
    sandbox, sailbox = _sandbox(policy=_ALLOWLIST)
    monkeypatch.setattr(
        SailVmSandbox, "_signed_image_urls",
        AsyncMock(return_value=["https://bucket.s3.amazonaws.com/a?sig=1", None]),
    )
    load = AsyncMock()
    monkeypatch.setattr(SailVmSandbox, "_load_docker_images", load)

    await sandbox.load_docker_images(["a", "b"])

    sailbox.set_egress_policy.aio.assert_awaited_once_with(
        {"allowlist": ["pypi.org", "bucket.s3.amazonaws.com", "10.0.0.0/8"]}
    )
    assert sandbox.network_policy.allow_hosts == ("pypi.org", "bucket.s3.amazonaws.com")
    load.assert_awaited_once_with(["a", "b"], ["https://bucket.s3.amazonaws.com/a?sig=1", None])


@pytest.mark.asyncio
async def test_image_loading_leaves_an_allow_all_policy_alone(monkeypatch):
    sandbox, sailbox = _sandbox()
    monkeypatch.setattr(SailVmSandbox, "_signed_image_urls", AsyncMock(return_value=["https://x.example/a"]))
    monkeypatch.setattr(SailVmSandbox, "_load_docker_images", AsyncMock())
    await sandbox.load_docker_images(["a"])
    sailbox.set_egress_policy.aio.assert_not_awaited()
    sandbox._sdk.Sailbox.get.aio.assert_not_awaited()


@pytest.mark.asyncio
async def test_widening_past_sails_allowlist_limit_is_refused(monkeypatch):
    full = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=tuple(f"h{i}.example" for i in range(128)))
    sandbox, sailbox = _sandbox(policy=full)
    monkeypatch.setattr(SailVmSandbox, "_signed_image_urls", AsyncMock(return_value=["https://bucket.example/a"]))
    with pytest.raises(RuntimeError, match="exceed Sail's 128-entry egress allowlist"):
        await sandbox.load_docker_images(["a"])
    sailbox.set_egress_policy.aio.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_host_under_an_allowed_wildcard_needs_no_new_entry(monkeypatch):
    full = NetworkPolicy(
        mode=NetworkMode.ALLOWLIST, allow_hosts=("*.s3.amazonaws.com", *(f"h{i}.example" for i in range(127)))
    )
    sandbox, sailbox = _sandbox(policy=full)
    monkeypatch.setattr(SailVmSandbox, "_signed_image_urls", AsyncMock(return_value=["https://bucket.s3.amazonaws.com/a"]))
    monkeypatch.setattr(SailVmSandbox, "_load_docker_images", AsyncMock())
    await sandbox.load_docker_images(["a"])
    sailbox.set_egress_policy.aio.assert_not_awaited()
    sandbox._sdk.Sailbox.get.aio.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_downloads_through_separate_handles_keep_each_others_hosts():
    sandbox, sailbox = _sandbox(policy=_ALLOWLIST)
    other_handle = SailVmSandbox(sailbox, sdk=sandbox._sdk, tunnel_urls={}, network_policy=_ALLOWLIST)

    await asyncio.gather(
        sandbox._allow_download_hosts(["https://a.example/x"], "download"),
        other_handle._allow_download_hosts(["https://b.example/y"], "download"),
    )

    assert set(sailbox.set_egress_policy.aio.await_args.args[0]["allowlist"]) >= {"a.example", "b.example"}


@pytest.mark.asyncio
async def test_a_host_already_allowed_is_not_reapplied(monkeypatch):
    sandbox, sailbox = _sandbox(policy=_ALLOWLIST)
    monkeypatch.setattr(SailVmSandbox, "_signed_image_urls", AsyncMock(return_value=["https://pypi.org/a"]))
    monkeypatch.setattr(SailVmSandbox, "_load_docker_images", AsyncMock())
    await sandbox.load_docker_images(["a"])
    sailbox.set_egress_policy.aio.assert_not_awaited()


def _object_store(monkeypatch, signed):
    store = MagicMock(
        signed_get_url=MagicMock(return_value=signed),
        open=MagicMock(side_effect=lambda _url: io.BytesIO(b"\x00payload")),
    )
    monkeypatch.setattr(
        "agent_env.providers.sandbox_providers.sail_vm.sandbox.get_config", lambda: MagicMock(get_object_store=lambda: store)
    )
    return store


@pytest.mark.asyncio
async def test_a_signed_object_download_allows_its_host_first(monkeypatch):
    _object_store(monkeypatch, "https://bucket.s3.amazonaws.com/f?sig=1")
    sandbox, sailbox = _sandbox(_process(), policy=_ALLOWLIST)

    await sandbox.load_s3_file("s3://bucket/f", "/tmp/f")

    sailbox.set_egress_policy.aio.assert_awaited_once_with(
        {"allowlist": ["pypi.org", "bucket.s3.amazonaws.com", "10.0.0.0/8"]}
    )
    script = sailbox.exec.aio.await_args.args[0][2]
    assert script.startswith("curl -fsSL") and "'https://bucket.s3.amazonaws.com/f?sig=1'" in script


@pytest.mark.asyncio
async def test_an_unsignable_object_is_streamed_through_the_filesystem_api(monkeypatch):
    _object_store(monkeypatch, None)
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sail_vm.sandbox._STREAM_CHUNK_BYTES", 4)
    sandbox, sailbox = _sandbox(policy=None)
    writer = MagicMock(write=AsyncMock())
    stream = MagicMock(__aenter__=AsyncMock(return_value=writer), __aexit__=AsyncMock(return_value=False))
    sailbox.fs.write_stream.aio = AsyncMock(return_value=stream)

    await sandbox.load_s3_file("file:///store/f", "/tmp/f")

    sailbox.fs.write_stream.aio.assert_awaited_once_with("/tmp/f")
    assert b"".join(call.args[0] for call in writer.write.await_args_list) == b"\x00payload"
    assert [len(call.args[0]) for call in writer.write.await_args_list] == [4, 4]
    stream.__aexit__.assert_awaited_once()
    sailbox.exec.aio.assert_not_awaited()


@pytest.mark.parametrize("policy", [NetworkPolicy(), _ALLOWLIST, NetworkPolicy(mode=NetworkMode.ALLOWLIST)])
def test_egress_documents_round_trip(policy):
    assert policy_from_document(egress_document(policy)) == policy


@pytest.mark.parametrize(
    "document",
    [None, {"no_network": True}, {"allowlist": ["a.example"], "blocked": ["b.example"]}, {"allowlist": "a.example"}],
)
def test_documents_agent_env_cannot_represent_are_unknown(document):
    assert policy_from_document(document) is None
