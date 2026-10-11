"""Unit tests for the Vercel Sandbox adapter over a fake SDK handle."""

import asyncio
import io
import subprocess
import ssl
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from anyio import BrokenResourceError, EndOfStream
import httpx
import httpx2
from vercel.sandbox import (
    SandboxApiError,
    SandboxCredentialsError,
    SandboxResponseError,
    SandboxStreamError,
)

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers import vercel as vercel_module
from agent_env.providers.sandbox_providers.vercel import sandbox as vercel_sandbox_module
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.vercel.sandbox import (
    VercelSandbox,
    network_policy_from_vercel,
)


class _Reader:
    def __init__(self, value: str):
        self._value = value

    async def read(self, size: int = -1) -> str:
        value = self._value if size < 0 else self._value[:size]
        self._value = self._value[len(value):]
        return value

    async def readline(self) -> str:
        length = self._value.find("\n") + 1
        if length == 0:
            length = len(self._value)
        value, self._value = self._value[:length], self._value[length:]
        return value

    async def aclose(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._value:
            value, self._value = self._value, ""
            return value
        raise StopAsyncIteration


def _process(stdout="", stderr="", exit_code=0):
    process = MagicMock()
    process.stdout = _Reader(stdout)
    process.stderr = _Reader(stderr)
    process.wait = AsyncMock(return_value=exit_code)
    return process


def _raw(policy=None, routes=()):
    raw = MagicMock(name="sb-1")
    raw.name = "sb-1"
    raw.routes = [SimpleNamespace(port=port, url=url) for port, url in routes]
    raw.network_policy = policy
    raw.create_process = AsyncMock(return_value=_process())
    raw.get_process = AsyncMock(side_effect=lambda _id: _process(
        stdout=raw.create_process.return_value.stdout._value,
        stderr=raw.create_process.return_value.stderr._value,
    ))
    async def run_process(command, args, **kwargs):
        process = raw.create_process.return_value
        kwargs["stdout"].write(await process.stdout.read())
        kwargs["stderr"].write(await process.stderr.read())
        return SimpleNamespace(returncode=await process.wait())

    raw.run_process = AsyncMock(side_effect=run_process)
    raw.destroy = AsyncMock()
    raw.update_network_policy = AsyncMock()
    raw.fs.write_bytes = AsyncMock()
    return raw


def _sandbox(raw=None, policy=NetworkPolicy()):
    raw = raw or _raw()
    if raw.network_policy is None and policy is not None:
        raw.network_policy = vercel_sandbox_module.vercel_network_policy(policy)
    client = MagicMock(get_sandbox=AsyncMock(return_value=raw))
    return VercelSandbox(raw, client=client, tunnel_urls={}, network_policy=policy)


def _tarball():
    return DockerImageArtifact(id="image", description="", image_name="image:v1",
                               tar_gz_object_url="file:///store/image.tar.gz")


@pytest.mark.asyncio
async def test_exec_returns_exact_unicode_bytes_and_nonzero_exit():
    raw = _raw()
    raw.create_process = AsyncMock(
        return_value=_process(stdout="hello ☃", stderr="warn", exit_code=7)
    )
    sandbox = _sandbox(raw)

    process = await sandbox.exec("sudo", "bash", "-c", "echo test")

    assert (await process.stdout.read(), await process.stderr.read(), await process.wait()) == (
        b"hello \xe2\x98\x83", b"warn", 7
    )
    assert raw.run_process.await_args.args == ("bash", ["-c", "echo test"])
    assert raw.run_process.await_args.kwargs["sudo"] is True


@pytest.mark.asyncio
async def test_stdout_supports_async_iteration():
    raw = _raw()
    raw.create_process = AsyncMock(return_value=_process(stdout="one\ntwo\n"))
    process = await _sandbox(raw).exec("cat", "log")
    assert [chunk async for chunk in process.stdout] == [b"one\n", b"two\n"]


@pytest.mark.asyncio
async def test_terminate_treats_only_missing_resources_as_idempotent():
    raw = _raw()
    missing = type("Missing", (Exception,), {"status_code": 404})()
    raw.destroy = AsyncMock(side_effect=missing)
    await _sandbox(raw).terminate()

    genuine = type("Conflict", (Exception,), {"status_code": 409})()
    raw.destroy = AsyncMock(side_effect=genuine)
    with pytest.raises(BaseException):
        await _sandbox(raw).terminate()


@pytest.mark.asyncio
async def test_host_files_use_the_native_filesystem_api():
    raw = _raw()
    await _sandbox(raw)._write_bytes_to_vm_path(b"\x00\x01", "/opt/data/blob.bin")
    raw.fs.write_bytes.assert_awaited_once_with("/opt/data/blob.bin", b"\x00\x01")


@pytest.mark.asyncio
async def test_apply_network_policy_replaces_the_current_session_policy(monkeypatch):
    monkeypatch.setattr(vercel_sandbox_module, "vercel_network_policy", lambda policy: policy)
    raw = _raw()
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",))
    sandbox = _sandbox(raw)

    await sandbox.apply_network_policy(policy)

    raw.update_network_policy.assert_awaited_once_with(policy)
    assert sandbox.network_policy == policy


@pytest.mark.asyncio
async def test_restrictive_image_loading_adds_only_signed_download_hosts(monkeypatch):
    raw = _raw()
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",))
    sandbox = _sandbox(raw, policy=policy)
    monkeypatch.setattr(
        VercelSandbox,
        "_signed_image_urls",
        AsyncMock(return_value=["https://bucket.example/a?sig=1", None]),
    )
    monkeypatch.setattr(VercelSandbox, "_load_docker_images", AsyncMock())
    monkeypatch.setattr(
        vercel_sandbox_module,
        "vercel_network_policy",
        lambda policy: policy.with_hosts(["bucket.example"]),
    )

    await sandbox.load_docker_images([_tarball()])

    raw.update_network_policy.assert_awaited_once()
    applied = raw.update_network_policy.await_args.args[0]
    assert applied.allow_hosts == ("pypi.org", "bucket.example")
    assert sandbox.network_policy.allow_hosts == applied.allow_hosts


@pytest.mark.asyncio
async def test_unknown_reconnected_policy_stays_unknown_for_image_loading():
    sandbox = _sandbox(policy=None)
    sandbox._signed_image_urls = AsyncMock(return_value=["https://bucket.example/image"])
    with pytest.raises(RuntimeError, match="applied network policy is unknown"):
        await sandbox.load_docker_images([_tarball()])


@pytest.mark.asyncio
async def test_docker_setup_is_bounded_and_requires_compose(monkeypatch):
    exec_with_output = AsyncMock(return_value=(0, "", ""))
    monkeypatch.setattr(VercelSandbox, "_exec_with_output", exec_with_output)
    await _sandbox().wait_for_vm()

    command = exec_with_output.await_args.args
    assert command[:3] == ("sudo", "bash", "-c")
    assert "apt-get install -y -qq docker.io docker-compose-v2" in command[3]
    assert "docker compose version" in command[3]
    assert exec_with_output.await_args.kwargs["timeout"] == 300


def test_vercel_policies_round_trip_through_agent_env():
    custom = SimpleNamespace(
        mode="custom",
        allow={"pypi.org": (), "example.com": ()},
        subnets=SimpleNamespace(allow=("10.0.0.0/8",), deny=None),
    )
    assert network_policy_from_vercel(custom) == NetworkPolicy(
        mode=NetworkMode.ALLOWLIST,
        allow_hosts=("pypi.org", "example.com"),
        allow_cidrs=("10.0.0.0/8",),
    )
    assert network_policy_from_vercel(SimpleNamespace(mode="deny-all", allow={})) == NetworkPolicy(
        mode=NetworkMode.ALLOWLIST
    )


@pytest.mark.parametrize(
    "policy",
    [
        SimpleNamespace(mode="custom", allow={"pypi.org": (object(),)}, subnets=None),
        SimpleNamespace(mode="custom", allow={}, subnets=SimpleNamespace(allow=None, deny=["10.0.0.0/8"])),
        SimpleNamespace(mode="unknown", allow={}),
    ],
)
def test_vercel_rules_agent_env_cannot_represent_are_unknown(policy):
    assert network_policy_from_vercel(policy) is None


@pytest.mark.asyncio
async def test_byte_reads_and_lines_preserve_partial_utf8_and_eof():
    raw = _raw()
    raw.create_process.return_value = _process(stdout="☃a\nz")
    process = await _sandbox(raw).exec("printf", "test")
    reader = process.stdout
    assert await reader.read(0) == b""
    assert await reader.readexactly(1) == b"\xe2"
    assert await reader.readexactly(2) == b"\x98\x83"
    assert await reader.readline() == b"a\n"
    assert await reader.read(1) == b"z"
    assert await reader.read() == b""
    assert await reader.readline() == b""


@pytest.mark.asyncio
async def test_command_streams_route_both_outputs_through_one_native_operation():
    raw = _raw()
    raw.create_process.return_value = _process(stdout="one\ntwo ☃\n", stderr="warning")
    process = await _sandbox(raw).exec("printf", "test")
    stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
    assert (stdout, stderr) == ("one\ntwo ☃\n".encode(), b"warning")
    raw.run_process.assert_awaited_once()


@pytest.mark.asyncio
async def test_output_is_readable_before_the_command_finishes():
    raw = _raw()
    release = asyncio.Event()

    async def run(command, args, **kwargs):
        kwargs["stdout"].write("ready\n")
        await release.wait()
        kwargs["stderr"].write("finished\n")
        return SimpleNamespace(returncode=7)

    raw.run_process.side_effect = run
    process = await _sandbox(raw).exec("command")
    assert await asyncio.wait_for(process.stdout.readline(), timeout=1) == b"ready\n"
    waiting = asyncio.create_task(process.wait())
    await asyncio.sleep(0)
    assert not waiting.done()
    release.set()
    assert await waiting == 7
    assert await process.stderr.read() == b"finished\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ConnectionError("stream disconnected"),
        httpx2.ReadError("connection reset"),
        httpx2.WriteError("connection reset"),
        httpx2.RemoteProtocolError("incomplete response body"),
        httpx2.ReadTimeout("read timeout"),
        httpx2.WriteTimeout("write timeout"),
        BrokenResourceError(),
        EndOfStream(),
        ssl.SSLEOFError("TLS EOF"),
        SandboxResponseError("Sandbox process response is missing final metadata"),
        SandboxResponseError(
            "Sandbox process response final metadata is missing a return code"
        ),
    ],
)
async def test_lost_process_transport_maps_to_minus_one_with_partial_output_and_eof(error):
    raw = _raw()

    async def run(command, args, **kwargs):
        kwargs["stdout"].write("partial ☃\n")
        kwargs["stderr"].write("partial error\n")
        raise error

    raw.run_process.side_effect = run
    process = await _sandbox(raw).exec("command")
    stdout, stderr = await asyncio.gather(process.stdout.read(), process.stderr.read())
    assert (stdout, stderr, await process.wait()) == (
        "partial ☃\n".encode(),
        b"partial error\n",
        -1,
    )
    assert await process.stdout.read() == b""
    assert await process.stderr.read() == b""


@pytest.mark.asyncio
async def test_a_missing_returncode_result_maps_to_minus_one():
    raw = _raw()

    async def run(command, args, **kwargs):
        kwargs["stdout"].write("partial")
        return SimpleNamespace(returncode=None)

    raw.run_process.side_effect = run
    process = await _sandbox(raw).exec("command")
    assert await process.stdout.read() == b"partial"
    assert await process.wait() == -1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        SandboxResponseError("Sandbox process response included an unexpected NDJSON record"),
        SandboxStreamError("server rejected the command", code="command_refused"),
        SandboxApiError(
            httpx.Response(401, request=httpx.Request("POST", "https://vercel.example")),
            "unauthorized",
        ),
        SandboxCredentialsError("credentials are not configured"),
        OSError("unrelated filesystem failure"),
        ssl.SSLCertVerificationError("certificate verification failed"),
        httpx2.ConnectError("certificate verification failed"),
        httpx2.LocalProtocolError("invalid request headers"),
    ],
)
async def test_non_transport_sdk_and_os_errors_propagate(error):
    raw = _raw()

    async def run(command, args, **kwargs):
        kwargs["stdout"].write("partial")
        raise error

    raw.run_process.side_effect = run
    process = await _sandbox(raw).exec("command")
    with pytest.raises(type(error)) as raised:
        await asyncio.gather(process.stdout.read(), process.wait())
    assert raised.value is error


@pytest.mark.asyncio
async def test_malformed_returncodes_are_not_concealed():
    raw = _raw()
    raw.run_process = AsyncMock(return_value=SimpleNamespace(returncode="7"))
    process = await _sandbox(raw).exec("command")
    with pytest.raises(TypeError, match="malformed exit code"):
        await process.wait()


@pytest.mark.asyncio
async def test_cancellation_propagates_and_preserves_partial_output_with_eof():
    raw = _raw()
    released = asyncio.Event()

    async def run(command, args, **kwargs):
        kwargs["stdout"].write("partial\n")
        await released.wait()
        return SimpleNamespace(returncode=0)

    raw.run_process.side_effect = run
    process = await _sandbox(raw).exec("command")
    await asyncio.sleep(0)
    process._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await process.wait()
    assert await process.stdout.read() == b"partial\n"
    assert await process.stdout.read() == b""
    released.set()


@pytest.mark.asyncio
async def test_inherited_retry_is_opt_in_and_arbitrary_commands_do_not_retry():
    raw = _raw()
    sandbox = _sandbox(raw)

    async def run(command, args, **kwargs):
        if raw.run_process.await_count != 2:
            raise httpx2.ReadError("incomplete response")
        kwargs["stdout"].write("done")
        return SimpleNamespace(returncode=0)

    raw.run_process.side_effect = run
    assert await sandbox.exec_script("idempotent", max_retries=1) == "done"
    assert raw.run_process.await_count == 2
    with pytest.raises(RuntimeError, match="exit -1"):
        await sandbox.exec_script("arbitrary")
    assert raw.run_process.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", [False, True])
async def test_large_object_upload_checks_contents_before_replacing_destination(tmp_path, corrupt):
    raw = _raw()
    sandbox = _sandbox(raw)
    parts = []
    payload = bytes(range(256)) * 20000
    destination = tmp_path / "existing 'file.bin"
    destination.write_bytes(b"previous")

    async def write_bytes(path, data):
        parts.append((path, len(data)))
        Path(path).write_bytes(b"corrupted" if corrupt else data)

    async def execute(script):
        result = await asyncio.to_thread(subprocess.run, ["bash", "-c", script], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    raw.fs.write_bytes.side_effect = write_bytes
    sandbox.exec_script = execute
    if corrupt:
        with pytest.raises(RuntimeError, match="checksum"):
            await sandbox._write_stream_to_vm(io.BytesIO(payload), str(destination))
        assert destination.read_bytes() == b"previous"
    else:
        await sandbox._write_stream_to_vm(io.BytesIO(payload), str(destination))
        assert destination.read_bytes() == payload
    assert len(parts) == 3 and max(size for _, size in parts) <= 2 * 1024 * 1024
    assert all(not Path(path).exists() and not Path(path.removesuffix(".part")).exists() for path, _ in parts)


@pytest.mark.asyncio
async def test_signed_object_download_adds_host_and_preserves_current_policy(monkeypatch):
    raw = _raw()
    sandbox = _sandbox(raw, policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("stale.example",)))
    current = SimpleNamespace(mode="custom", allow={"current.example": ()}, subnets=None)
    sandbox._client = MagicMock(get_sandbox=AsyncMock(return_value=SimpleNamespace(network_policy=current)))
    store = MagicMock(signed_get_url=MagicMock(return_value="https://bucket.example/file?signature=abc"))
    config = MagicMock(get_object_store_at=MagicMock(return_value=store))
    monkeypatch.setattr(vercel_sandbox_module, "get_config", lambda: config, raising=False)
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sandbox.get_config", lambda: config)
    script = AsyncMock()
    sandbox.exec_script = script

    await sandbox.load_object_file("s3://bucket/file", "/tmp/a file")

    applied = raw.update_network_policy.await_args.args[0]
    assert set(applied.allow) == {"current.example", "bucket.example"}
    assert "stale.example" not in sandbox.network_policy.allow_hosts
    assert "https://bucket.example/file?signature=abc" in script.await_args.args[0]
    assert "'/tmp/a file'" in script.await_args.args[0]
    config.get_object_store_at.assert_called_once_with("s3://bucket/file")


@pytest.mark.asyncio
async def test_signed_download_refuses_an_unknown_current_policy(monkeypatch):
    sandbox = _sandbox(policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST))
    custom = SimpleNamespace(mode="custom", allow={"service.example": (object(),)}, subnets=None)
    sandbox._client = MagicMock(get_sandbox=AsyncMock(return_value=SimpleNamespace(network_policy=custom)))
    store = MagicMock(signed_get_url=MagicMock(return_value="https://bucket.example/file"))
    config = MagicMock(get_object_store_at=MagicMock(return_value=store))
    monkeypatch.setattr(vercel_sandbox_module, "get_config", lambda: config, raising=False)
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sandbox.get_config", lambda: config)
    sandbox.exec_script = AsyncMock()

    with pytest.raises(RuntimeError, match="applied network policy is unknown"):
        await sandbox.load_object_file("s3://bucket/file", "/tmp/file")

    sandbox._sandbox.update_network_policy.assert_not_awaited()
    sandbox.exec_script.assert_not_awaited()


@pytest.mark.asyncio
async def test_image_fallback_can_download_an_unsigned_object_without_deadlock(monkeypatch):
    sandbox = _sandbox(policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST))
    current = SimpleNamespace(mode="deny-all", allow={}, subnets=None)
    sandbox._client = MagicMock(get_sandbox=AsyncMock(return_value=SimpleNamespace(network_policy=current)))
    store = MagicMock(signed_get_url=MagicMock(return_value=None))
    config = MagicMock(get_object_store_at=MagicMock(return_value=store))
    monkeypatch.setattr(vercel_sandbox_module, "get_config", lambda: config, raising=False)
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sandbox.get_config", lambda: config)
    sandbox._signed_image_urls = AsyncMock(return_value=[None])
    sandbox._write_unsigned_object = AsyncMock()
    async def load(artifacts, urls):
        await sandbox._download_object_to_vm("file:///store/image", "/tmp/image")
    sandbox._load_docker_images = load

    await asyncio.wait_for(sandbox.load_docker_images([_tarball()]), 1)

    sandbox._write_unsigned_object.assert_awaited_once_with(store, "file:///store/image", "/tmp/image")
    sandbox._sandbox.update_network_policy.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [1, 3])
@pytest.mark.parametrize("error_type", [BrokenResourceError, ssl.SSLError])
async def test_interrupted_chunk_upload_retries_without_appending_partial_bytes(tmp_path, failures, error_type):
    raw = _raw()
    sandbox = _sandbox(raw)
    payload = b"binary\x00" * 400000
    destination = tmp_path / "destination"
    destination.write_bytes(b"previous")
    uploads = []

    async def write_bytes(path, data):
        uploads.append(path)
        Path(path).write_bytes(data[:100] if len(uploads) <= failures else data)
        if len(uploads) <= failures:
            raise error_type

    async def execute(script):
        result = await asyncio.to_thread(subprocess.run, ["bash", "-c", script], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout

    raw.fs.write_bytes.side_effect = write_bytes
    sandbox.exec_script = execute
    if failures == 3:
        with pytest.raises(error_type):
            await sandbox._write_stream_to_vm(io.BytesIO(payload), str(destination))
        assert len(uploads) == 3
        assert destination.read_bytes() == b"previous"
    else:
        await sandbox._write_stream_to_vm(io.BytesIO(payload), str(destination))
        assert len(uploads) == 3
        assert destination.read_bytes() == payload
    assert all(not Path(path).exists() for path in uploads)


@pytest.mark.asyncio
async def test_registry_and_context_images_use_the_vm_loading_contract(monkeypatch):
    sandbox = _sandbox(policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("registry.example",)))
    sandbox.exec_script = AsyncMock(return_value="")
    sandbox._signed_image_urls = AsyncMock(side_effect=AssertionError("non-tarball image was signed"))
    store = MagicMock(signed_get_url=MagicMock(return_value="https://bucket.example/context.tar.gz"))
    registry = MagicMock(auth=MagicMock(return_value=None))
    config = MagicMock(get_object_store_at=MagicMock(return_value=store),
                       get_image_store=MagicMock(return_value=registry))
    monkeypatch.setattr(vercel_sandbox_module, "get_config", lambda: config)
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sandbox.get_config", lambda: config)
    reference = DockerImageArtifact(id="ref", description="", image_name="registry.example/tool:v1")
    context = DockerImageArtifact(id="context", description="", image_name="context:v1",
                                  build_context_object_url="file:///store/context.tar.gz")

    await sandbox.load_docker_images([reference, context])

    scripts = [call.args[0] for call in sandbox.exec_script.await_args_list]
    assert "docker pull registry.example/tool:v1" in scripts
    assert any("docker build --platform linux/amd64 -f Dockerfile -t context:v1 ." in script for script in scripts)
    assert any("https://bucket.example/context.tar.gz" in script and "curl -fsSL" in script for script in scripts)
    store.signed_get_url.assert_called_once_with("file:///store/context.tar.gz")
    sandbox._signed_image_urls.assert_not_awaited()
    assert sandbox.network_policy.allow_hosts == ("registry.example", "bucket.example")


@pytest.mark.asyncio
async def test_unobtainable_image_is_refused_before_loading_any_image():
    sandbox = _sandbox()
    sandbox._signed_image_urls = AsyncMock(side_effect=AssertionError("validation was bypassed"))
    sandbox.exec_script = AsyncMock()
    tarball = DockerImageArtifact(id="tar", description="", image_name="tar:v1",
                                  tar_gz_object_url="file:///store/image.tar.gz")
    missing = DockerImageArtifact(id="missing", description="", image_name="missing:v1")

    with pytest.raises(RuntimeError, match="Can't load images: 'missing'"):
        await sandbox.load_docker_images([tarball, missing])

    sandbox._signed_image_urls.assert_not_awaited()
    sandbox.exec_script.assert_not_awaited()
