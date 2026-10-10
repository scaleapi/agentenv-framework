"""Offline unit tests for the Tensorlake VmSandbox adapter."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tensorlake.sandbox import NetworkConfig, SandboxConnectionError, SandboxNotFoundError

from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_VM
from agent_env.providers.sandbox_providers.tensorlake import sandbox as tl_sandbox
from agent_env.providers.sandbox_providers.tensorlake.sandbox import (
    TensorlakeSandbox,
    network_config_fields,
    network_policy_from_config,
)
from agent_env.store import set_object_store


def _artifact(url: str | None = "s3://bucket/image.tar.gz", *, context_only: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        tar_gz_object_url=url, context_only=context_only, image_name="img:latest", load_problem=lambda: None
    )


def _inner(policy: NetworkPolicy | None = None) -> MagicMock:
    """A fake SDK sandbox whose ``info`` reports the network policy ``update`` last applied."""
    applied = SimpleNamespace(network_policy=NetworkConfig(**network_config_fields(policy)) if policy else None)

    async def update(*, network=None, **kwargs):
        if network is not None:
            applied.network_policy = network

    sandbox = MagicMock()
    sandbox.sandbox_id = "tl-test"
    sandbox.terminate = AsyncMock()
    sandbox.update = AsyncMock(side_effect=update)
    sandbox.info = AsyncMock(return_value=applied)
    sandbox.write_file = AsyncMock()
    sandbox.run = AsyncMock(return_value=SimpleNamespace(exit_code=0, stdout="hello", stderr="warning"))
    return sandbox


def test_vm_identity_and_tunnel_urls():
    sandbox = TensorlakeSandbox(_inner(), tunnel_urls={8080: "https://8080-tl-test.sandbox.tensorlake.ai"})

    assert isinstance(sandbox, VmSandbox)
    assert sandbox.type == "tensorlake"
    assert sandbox.mode == SANDBOX_MODE_VM
    assert sandbox.sandbox_id == "tl-test"
    assert sandbox.vnc_url is None
    assert sandbox.tunnel_urls == {8080: "https://8080-tl-test.sandbox.tensorlake.ai"}


@pytest.mark.asyncio
async def test_exec_passes_argv_unchanged_and_keeps_sudo():
    inner = _inner()
    sandbox = TensorlakeSandbox(inner)

    result = await sandbox.exec_with_output("sudo", "bash", "-c", "docker images && echo 'hello world'")

    assert result == (0, "hello", "warning")
    inner.run.assert_awaited_once_with(
        "sudo", args=["bash", "-c", "docker images && echo 'hello world'"], timeout=None,
    )


@pytest.mark.asyncio
async def test_nonzero_exit_is_a_process_result():
    inner = _inner()
    inner.run.return_value = SimpleNamespace(exit_code=7, stdout="out", stderr="bad")

    assert await TensorlakeSandbox(inner).exec_with_output("sudo", "docker", "images") == (7, "out", "bad")


@pytest.mark.asyncio
async def test_setup_starts_docker_service_polls_until_ready_and_checks_compose(monkeypatch: pytest.MonkeyPatch):
    sandbox = TensorlakeSandbox(_inner())
    sandbox._exec_with_output_bounded = AsyncMock(
        side_effect=[(0, "", ""), (1, "", "not ready"), (0, "", ""), (0, "Docker Compose version v2.40.0", "")]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(tl_sandbox.asyncio, "sleep", sleep)

    await sandbox.setup_vm_for_gateway([18765])

    commands = [call.args for call in sandbox._exec_with_output_bounded.await_args_list]
    assert commands == [
        ("sudo", "systemctl", "start", "docker"),
        ("sudo", "docker", "info"),
        ("sudo", "docker", "info"),
        ("sudo", "docker", "compose", "version"),
    ]
    assert not any("iptables" in " ".join(command) for command in commands)
    sleep.assert_awaited_once_with(sandbox._VM_READY_POLL_INTERVAL)


@pytest.mark.asyncio
async def test_setup_rejects_an_image_without_compose_v2():
    sandbox = TensorlakeSandbox(_inner())
    sandbox._exec_with_output_bounded = AsyncMock(
        side_effect=[(0, "", ""), (0, "", ""), (1, "", "docker: 'compose' is not a docker command")]
    )

    with pytest.raises(RuntimeError, match="must include the Docker Compose v2 plugin"):
        await sandbox.setup_vm_for_gateway([])


@pytest.mark.asyncio
async def test_docker_readiness_timeout_reports_the_service_journal():
    inner = _inner()
    sandbox = TensorlakeSandbox(inner)
    sandbox._VM_READY_TIMEOUT = 0
    inner.run.return_value = SimpleNamespace(exit_code=0, stdout="dockerd failed to start", stderr="")

    with pytest.raises(RuntimeError, match=r"(?s)Docker not ready.*dockerd failed to start"):
        await sandbox.wait_for_vm()

    assert inner.run.await_args.args == ("sudo",)
    assert inner.run.await_args.kwargs["args"][:3] == ["journalctl", "-u", "docker"]


@pytest.mark.asyncio
async def test_a_failing_probe_counts_as_not_ready(monkeypatch: pytest.MonkeyPatch):
    inner = _inner()
    sandbox = TensorlakeSandbox(inner)
    sandbox._VM_READY_TIMEOUT = 1
    sandbox._DOCKER_READINESS_COMMAND_TIMEOUT = 0.25
    inner.run.side_effect = [
        SimpleNamespace(exit_code=0, stdout="", stderr=""),
        TimeoutError("docker info timed out"),
        SimpleNamespace(exit_code=0, stdout="", stderr=""),
    ]
    monkeypatch.setattr(tl_sandbox.asyncio, "sleep", AsyncMock())

    await sandbox.wait_for_vm()

    assert inner.run.await_args_list[1].kwargs["timeout"] <= 0.25


@pytest.mark.asyncio
async def test_terminate_terminates_the_sdk_sandbox():
    inner = _inner()

    await TensorlakeSandbox(inner).terminate()

    inner.terminate.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_terminate_of_a_sandbox_that_is_gone_is_not_an_error():
    inner = _inner()
    inner.terminate.side_effect = SandboxNotFoundError("tl-test")

    await TensorlakeSandbox(inner).terminate()


@pytest.mark.asyncio
async def test_a_lost_transport_is_exit_minus_one_so_idempotent_scripts_retry(monkeypatch: pytest.MonkeyPatch):
    inner = _inner()
    inner.run.side_effect = [
        SandboxConnectionError("stream ended without an exit event"),
        SimpleNamespace(exit_code=0, stdout="done", stderr=""),
    ]
    monkeypatch.setattr(tl_sandbox.asyncio, "sleep", AsyncMock())
    sandbox = TensorlakeSandbox(inner)

    assert await sandbox.exec_with_output("true") == (-1, "", "Connection error: stream ended without an exit event")
    inner.run.side_effect = [
        SandboxConnectionError("lost"),
        SimpleNamespace(exit_code=0, stdout="done", stderr=""),
    ]
    assert await sandbox.exec_script("echo done", max_retries=1) == "done"


def _sha256_reply(data: bytes) -> AsyncMock:
    return AsyncMock(return_value=f"{hashlib.sha256(data).hexdigest()}  /path\n")


@pytest.mark.asyncio
async def test_large_writes_upload_in_proxy_sized_parts_and_join_as_root():
    inner = _inner()
    sandbox = TensorlakeSandbox(inner)
    sandbox._UPLOAD_CHUNK_BYTES = 4
    sandbox.exec_script = _sha256_reply(b"0123456789")

    await sandbox._write_bytes_to_vm_path(b"0123456789", "/opt/agent/file.bin")

    uploads = sorted(call.args for call in inner.write_file.await_args_list)
    assert [content for _, content in uploads] == [b"0123", b"4567", b"89"]
    parts = [path for path, _ in uploads]
    stem = parts[0].removesuffix(".0")
    assert stem.startswith("/tmp/")
    assert parts == [f"{stem}.{index}" for index in range(3)]
    join, cleanup = (call.args[0] for call in sandbox.exec_script.await_args_list)
    assert join == (
        f'for i in $(seq 0 2); do cat "{stem}.$i" || exit 1; done > /opt/agent/file.bin'
        " && sha256sum /opt/agent/file.bin"
    )
    assert cleanup == f'for i in $(seq 0 2); do rm -f "{stem}.$i"; done'


@pytest.mark.asyncio
async def test_join_command_length_does_not_grow_with_the_part_count():
    inner = _inner()
    sandbox = TensorlakeSandbox(inner)
    sandbox._UPLOAD_CHUNK_BYTES = 1
    sandbox.exec_script = _sha256_reply(b"x" * 10_000)

    await sandbox._write_bytes_to_vm_path(b"x" * 10_000, "/opt/agent/file.bin")

    assert inner.write_file.await_count == 10_000
    assert all(len(call.args[0]) < 200 for call in sandbox.exec_script.await_args_list)


@pytest.mark.asyncio
async def test_empty_write_still_creates_the_file():
    inner = _inner()
    sandbox = TensorlakeSandbox(inner)
    sandbox.exec_script = _sha256_reply(b"")

    await sandbox._write_bytes_to_vm_path(b"", "/tmp/empty")

    inner.write_file.assert_not_awaited()
    (join,) = (call.args[0] for call in sandbox.exec_script.await_args_list)
    assert join.startswith("for i in $(seq 0 -1); do ")
    assert join.endswith("done > /tmp/empty && sha256sum /tmp/empty")


@pytest.mark.asyncio
async def test_unsigned_object_streams_through_the_file_api(tmp_path):
    payload = bytes(range(256)) * 40
    source = tmp_path / "image.tar.gz"
    source.write_bytes(payload)
    store = MagicMock()
    store.open = MagicMock(side_effect=lambda url: open(source, "rb"))
    inner = _inner()
    sandbox = TensorlakeSandbox(inner)
    sandbox._UPLOAD_CHUNK_BYTES = 1000
    sandbox.exec_script = _sha256_reply(payload)

    await sandbox._write_unsigned_object(store, "file:///store/image.tar.gz", "/tmp/image.tar.gz")

    store.open.assert_called_once_with("file:///store/image.tar.gz")
    uploads = sorted(inner.write_file.await_args_list, key=lambda call: int(call.args[0].rsplit(".", 1)[1]))
    assert b"".join(call.args[1] for call in uploads) == payload
    assert inner.write_file.await_count == 11
    assert all(len(call.args[0]) < 200 for call in sandbox.exec_script.await_args_list)


@pytest.mark.asyncio
async def test_parts_upload_concurrently_up_to_the_limit():
    inner = _inner()
    in_flight = peak = 0

    async def write_file(path, content):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1

    inner.write_file = AsyncMock(side_effect=write_file)
    sandbox = TensorlakeSandbox(inner)
    sandbox._UPLOAD_CHUNK_BYTES = 1
    sandbox._UPLOADS_IN_FLIGHT = 3
    sandbox.exec_script = _sha256_reply(b"x" * 9)

    await sandbox._write_bytes_to_vm_path(b"x" * 9, "/tmp/file.bin")

    assert peak == 3


@pytest.mark.asyncio
async def test_a_failed_part_raises_its_error_and_cleans_up():
    inner = _inner()
    inner.write_file = AsyncMock(side_effect=[None, SandboxConnectionError("proxy closed"), None, None])
    sandbox = TensorlakeSandbox(inner)
    sandbox._UPLOAD_CHUNK_BYTES = 1
    sandbox._UPLOADS_IN_FLIGHT = 1
    sandbox.exec_script = AsyncMock(return_value="")

    with pytest.raises(SandboxConnectionError, match="proxy closed"):
        await sandbox._write_bytes_to_vm_path(b"abcd", "/tmp/file.bin")

    (cleanup,) = (call.args[0] for call in sandbox.exec_script.await_args_list)
    assert "rm -f" in cleanup


@pytest.mark.asyncio
async def test_a_joined_file_with_the_wrong_digest_is_refused():
    inner = _inner()
    sandbox = TensorlakeSandbox(inner)
    sandbox.exec_script = _sha256_reply(b"other")

    with pytest.raises(RuntimeError, match="does not match the upload"):
        await sandbox._write_bytes_to_vm_path(b"data", "/tmp/file.bin")


def test_network_config_fields():
    assert network_config_fields(NetworkPolicy()) == {"allow_internet_access": True, "allow_out": [], "deny_out": []}
    assert network_config_fields(
        NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("api.example",), allow_cidrs=("10.0.0.0/8",))
    ) == {"allow_internet_access": True, "allow_out": ["api.example", "10.0.0.0/8"], "deny_out": []}
    # An empty allow_out means "allow everything" to Tensorlake, so an empty allowlist turns internet off.
    assert network_config_fields(NetworkPolicy(mode=NetworkMode.ALLOWLIST)) == {
        "allow_internet_access": False, "allow_out": [], "deny_out": [],
    }


@pytest.mark.parametrize(
    "policy",
    [
        NetworkPolicy(),
        NetworkPolicy(mode=NetworkMode.ALLOWLIST),
        NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("api.example", "*.cdn.example"), allow_cidrs=("8.8.8.8", "10.0.0.0/8")),
    ],
)
def test_network_policy_round_trips(policy: NetworkPolicy):
    applied = SimpleNamespace(**network_config_fields(policy))

    assert network_policy_from_config(applied, sandbox_id="tl-test") == policy


def test_no_applied_policy_reads_as_allow_all():
    assert network_policy_from_config(None, sandbox_id="tl-test") == NetworkPolicy()


@pytest.mark.parametrize(
    "applied",
    [
        SimpleNamespace(allow_internet_access=True, allow_out=[], deny_out=["10.0.0.0/8"]),
        SimpleNamespace(allow_internet_access=False, allow_out=["10.0.0.0/8"], deny_out=[]),
    ],
)
def test_a_policy_agent_env_never_applies_is_refused(applied):
    with pytest.raises(RuntimeError, match="Cannot recover the network policy"):
        network_policy_from_config(applied, sandbox_id="tl-test")


@pytest.mark.asyncio
async def test_apply_network_policy_updates_the_sandbox():
    inner = _inner()
    sandbox = TensorlakeSandbox(inner)
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("api.example",))

    await sandbox.apply_network_policy(policy)

    inner.update.assert_awaited_once_with(
        network=NetworkConfig(allow_internet_access=True, allow_out=["api.example"], deny_out=[])
    )
    assert sandbox.network_policy == policy


@pytest.mark.asyncio
async def test_restricted_image_loading_adds_the_signed_host_and_waits_for_it(monkeypatch: pytest.MonkeyPatch):
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("workload.example",), allow_cidrs=("10.0.0.0/8",))
    inner = _inner(policy)
    sandbox = TensorlakeSandbox(inner, network_policy=policy)
    sandbox.exec_script = AsyncMock(return_value="")
    artifact = _artifact()
    store = MagicMock()
    store.signed_get_url.return_value = "https://downloads.example/path?signature=x"
    set_object_store(store)
    base_load = AsyncMock()
    monkeypatch.setattr(VmSandbox, "_load_docker_images", base_load)

    await sandbox.load_docker_images([artifact])

    inner.update.assert_awaited_once_with(
        network=NetworkConfig(
            allow_internet_access=True,
            allow_out=["workload.example", "downloads.example", "10.0.0.0/8"],
            deny_out=[],
        )
    )
    assert sandbox.network_policy == policy.with_hosts(["downloads.example"])
    command, args = inner.run.await_args.args[0], inner.run.await_args.kwargs["args"]
    assert command == "sh" and "https://downloads.example/" in args[-1]
    base_load.assert_awaited_once_with([artifact], ["https://downloads.example/path?signature=x"])


@pytest.mark.asyncio
async def test_restricted_image_loading_skips_the_update_when_the_host_is_already_allowed(monkeypatch: pytest.MonkeyPatch):
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("downloads.example",))
    inner = _inner(policy)
    sandbox = TensorlakeSandbox(inner, network_policy=policy)
    store = MagicMock()
    store.signed_get_url.return_value = "https://downloads.example/image.tar.gz"
    set_object_store(store)
    monkeypatch.setattr(VmSandbox, "_load_docker_images", AsyncMock())

    await sandbox.load_docker_images([_artifact("s3://bucket/img")])

    inner.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_restricted_image_loading_pulls_ref_images_and_builds_context_images(monkeypatch: pytest.MonkeyPatch):
    sandbox = TensorlakeSandbox(_inner(), network_policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST))
    pull, build = AsyncMock(), AsyncMock()
    monkeypatch.setattr(VmSandbox, "pull_images", pull)
    monkeypatch.setattr(VmSandbox, "build_images", build)
    ref, context = _artifact(None), _artifact(None, context_only=True)

    await sandbox.load_docker_images([ref, context])

    pull.assert_awaited_once_with(["img:latest"])
    build.assert_awaited_once_with([context])


@pytest.mark.asyncio
async def test_restricted_object_download_adds_the_signed_host(monkeypatch: pytest.MonkeyPatch):
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST)
    sandbox = TensorlakeSandbox(_inner(policy), network_policy=policy)
    sandbox.exec_script = AsyncMock(return_value="")
    sandbox._exec_with_output_bounded = AsyncMock(return_value=(0, "", ""))
    store = MagicMock()
    store.signed_get_url.return_value = "https://downloads.example/context.tar.gz?signature=x"
    set_object_store(store)

    await sandbox.load_object_file("s3://bucket/context.tar.gz", "/tmp/context.tar.gz")

    assert sandbox.network_policy.allow_hosts == ("downloads.example",)
    assert "context.tar.gz?signature=x" in sandbox.exec_script.await_args.args[0]


@pytest.mark.asyncio
async def test_concurrent_downloads_through_two_wrappers_keep_each_others_hosts(monkeypatch: pytest.MonkeyPatch):
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("workload.example",))
    inner = _inner(policy)
    first, second = TensorlakeSandbox(inner, network_policy=policy), TensorlakeSandbox(inner, network_policy=policy)
    monkeypatch.setattr(TensorlakeSandbox, "_wait_for_egress", AsyncMock())

    await asyncio.gather(
        first._allow_download_hosts(["https://a.example/x"]),
        second._allow_download_hosts(["https://b.example/y"]),
    )

    applied = network_policy_from_config((await inner.info()).network_policy, sandbox_id="tl-test")
    assert set(applied.allow_hosts) == {"workload.example", "a.example", "b.example"}


@pytest.mark.asyncio
async def test_unknown_policy_refuses_image_load():
    sandbox = TensorlakeSandbox(_inner(), network_policy=None)

    with pytest.raises(RuntimeError, match="applied network policy is unknown"):
        await sandbox.load_docker_images([_artifact()])


@pytest.mark.asyncio
async def test_unreachable_download_host_is_a_warning_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(TensorlakeSandbox, "_EGRESS_PROPAGATION_TIMEOUT", 0.05)
    monkeypatch.setattr(TensorlakeSandbox, "_EGRESS_PROBE_INTERVAL", 0.01)
    sandbox = TensorlakeSandbox(_inner())
    sandbox._exec_with_output_bounded = AsyncMock(return_value=(7, "", ""))

    await sandbox._wait_for_egress(["downloads.example"])

    assert "still cannot reach" in caplog.text


@pytest.mark.asyncio
async def test_egress_wait_bounds_each_probe_by_the_time_left(monkeypatch: pytest.MonkeyPatch):
    clock = iter([100.0, 100.0, 158.0, 158.0, 161.0, 161.0])
    monkeypatch.setattr(tl_sandbox, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    sandbox = TensorlakeSandbox(_inner())
    sandbox._exec_with_output_bounded = AsyncMock(return_value=(7, "", ""))

    await sandbox._wait_for_egress(["downloads.example"])

    assert [call.kwargs["timeout"] for call in sandbox._exec_with_output_bounded.await_args_list] == [60.0, 2.0]
