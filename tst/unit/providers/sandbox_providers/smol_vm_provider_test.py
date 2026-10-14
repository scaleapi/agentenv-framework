"""Local SmolVM lifecycle rules that must hold even when cloud credentials are present."""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace

import pytest

pytest.importorskip("smol")

from agent_env.providers.sandbox_providers.smol_vm import provider as module
from agent_env.providers.sandbox_providers.smol_vm import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.smol_vm.sandbox import SmolVmSandbox
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.store.routing import LocalRunImageStore


@pytest.mark.asyncio
async def test_agent_sandbox_returns_vm_to_load_saved_or_context_only_images(monkeypatch):
    provider = module.SmolVmSandboxProvider()
    create_vm = AsyncMock(return_value=SimpleNamespace(mode="vm"))
    monkeypatch.setattr(provider, "create_vm", create_vm)

    sandbox = await provider.create_sandbox(image_name="local-image", port=8080, env={"A2A_PORT": "8080"})

    assert sandbox.mode == "vm"
    assert create_vm.await_args.kwargs["exposed_ports"] == [8080]


@pytest.mark.asyncio
async def test_local_run_proxies_its_underlying_loopback_registries(monkeypatch):
    store = LocalRunImageStore(
        LocalRegistryImageStore("localhost:5001"), LocalRegistryImageStore("localhost:5000"),
    )
    monkeypatch.setattr(module, "get_config", lambda: SimpleNamespace(get_image_store=lambda: store))
    sandbox = SimpleNamespace(exec_script=AsyncMock())

    await module.SmolVmSandboxProvider()._ensure_local_registry_proxy(sandbox)

    commands = [call.args[0] for call in sandbox.exec_script.await_args_list]
    assert len(commands) == 2
    assert "TCP-LISTEN:5000" in commands[0]
    assert "TCP-LISTEN:5001" in commands[1]


@pytest.mark.asyncio
async def test_vm_creation_selects_local_and_can_publish_an_unstarted_application(monkeypatch):
    created = []

    class FakeMachine:
        id = "agentenv-test"

        def endpoint(self, port):
            return SimpleNamespace(http_url="http://127.0.0.1:44000")

        def write_file(self, path, data):
            created.append((path, data))

        def exec(self, command):
            assert command == ["uname", "-m"]
            return SimpleNamespace(exit_code=0, stdout="aarch64\n", stderr="")

        def delete(self):
            created.append(("deleted", None))

    def create(config, conn):
        assert conn.target == "local"
        assert config.wait_for_ports is False
        assert config.ttl_seconds == 900
        assert config.resources.network is True
        assert [(p.host, p.guest) for p in config.ports] == [(44000, 8080)]
        return FakeMachine()

    async def setup(self, sandbox):
        sandbox.extra_hosts = ("host.docker.internal:100.96.0.1",)

    monkeypatch.setattr(module, "Machine", SimpleNamespace(create=create))
    monkeypatch.setattr(module, "_host_port", lambda: 44000)
    monkeypatch.setattr(module.SmolVmSandboxProvider, "_setup_docker", setup)
    monkeypatch.setattr(module.SmolVmSandboxProvider, "_configure_host_access", setup)
    sandbox = await module.SmolVmSandboxProvider().create_vm(exposed_ports=[8080], memory=1024, timeout=900)
    assert sandbox.tunnel_urls[8080] == "http://127.0.0.1:44000"
    assert sandbox.platform == "linux/arm64"
    assert ("/storage/agentenv-ports.json", b'{"8080": 44000}') in created
    await sandbox.terminate()
    assert ("deleted", None) in created


@pytest.mark.asyncio
async def test_cancellation_during_native_creation_reclaims_vm(monkeypatch):
    started, release = threading.Event(), threading.Event()
    deleted = []

    class FakeMachine:
        def delete(self):
            deleted.append(True)

    def create(config, conn):
        started.set()
        assert release.wait(timeout=10)
        return FakeMachine()

    monkeypatch.setattr(module, "Machine", SimpleNamespace(create=create))
    monkeypatch.setattr(module, "_host_port", lambda: 44000)
    task = asyncio.create_task(module.SmolVmSandboxProvider().create_vm(memory=1024))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert deleted == [True]


@pytest.mark.asyncio
async def test_second_cancellation_during_create_still_deletes_machine(monkeypatch):
    started, release, deleted = threading.Event(), threading.Event(), threading.Event()

    class FakeMachine:
        def delete(self):
            deleted.set()

    def create(config, conn):
        started.set()
        assert release.wait(timeout=10)
        return FakeMachine()

    monkeypatch.setattr(module, "Machine", SimpleNamespace(create=create))
    monkeypatch.setattr(module, "_host_port", lambda: 44000)
    task = asyncio.create_task(module.SmolVmSandboxProvider().create_vm(memory=1024))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        for _ in range(100):
            if module._reapers:
                break
            await asyncio.sleep(0)
        assert module._reapers
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
    assert await asyncio.to_thread(deleted.wait, 5)


@pytest.mark.asyncio
async def test_second_cancellation_during_setup_still_deletes_machine(monkeypatch):
    started, deleted = asyncio.Event(), threading.Event()

    class FakeMachine:
        id = "agentenv-test"

        def delete(self):
            deleted.set()

    async def setup(self, sandbox):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(module, "Machine", SimpleNamespace(create=lambda config, conn: FakeMachine()))
    monkeypatch.setattr(module, "_host_port", lambda: 44000)
    monkeypatch.setattr(module.SmolVmSandboxProvider, "_setup_docker", setup)
    task = asyncio.create_task(module.SmolVmSandboxProvider().create_vm(memory=1024))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    for _ in range(100):
        if module._reapers:
            break
        await asyncio.sleep(0)
    assert module._reapers
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(deleted.wait, 5)


@pytest.mark.asyncio
async def test_reconnect_uses_explicit_agent_only_readiness(monkeypatch):
    options = []

    class FakeMachine:
        id = "agentenv-test"

        def exec(self, command):
            assert command == ["uname", "-m"]
            return SimpleNamespace(exit_code=0, stdout="aarch64\n", stderr="")

        def read_file(self, path):
            assert path == "/storage/agentenv-ports.json"
            return json.dumps({"8080": 44000}).encode()

        def endpoint(self, port):
            assert port == 8080
            return SimpleNamespace(http_url="http://127.0.0.1:44000")

    def connect(name, conn):
        assert name == "agentenv-test"
        options.append(conn)
        return FakeMachine()

    monkeypatch.setattr("agent_env.providers.sandbox_providers.smol_vm.sandbox.Machine", SimpleNamespace(connect=connect))
    sandbox = await SmolVmSandbox.reconnect("agentenv-test")
    assert sandbox.tunnel_urls == {8080: "http://127.0.0.1:44000"}
    assert sandbox.platform == "linux/arm64"
    assert options[0].target == "local" and options[0].wait_for_ports is False


def test_local_urls_are_rewritten_for_vm_host():
    assert module.SmolVmSandboxProvider.get_external_url("http://127.0.0.1:5000/v2/") == "http://host.docker.internal:5000/v2/"
    assert module.SmolVmSandboxProvider.get_external_url("https://example.org/path") == "https://example.org/path"


@pytest.mark.asyncio
async def test_vm_and_containers_resolve_grant_host_through_smol_gateway():
    sandbox = SimpleNamespace(
        exec_with_output=AsyncMock(return_value=(0, "100.96.0.1 STREAM host.smolvm.internal\n", "")),
        exec_script=AsyncMock(),
    )

    await module.SmolVmSandboxProvider()._configure_host_access(sandbox)

    assert sandbox.extra_hosts == ("host.docker.internal:100.96.0.1",)
    script = sandbox.exec_script.await_args.args[0]
    assert "host.docker.internal" in script and "100.96.0.1" in script
    assert "/etc/hosts" in script


@pytest.mark.asyncio
async def test_vm_host_entry_updates_without_dropping_other_guest_names(tmp_path):
    hosts = tmp_path / "hosts"
    hosts.write_text("127.0.0.1 localhost host.docker.internal\n::1 ip6-localhost\n")
    sandbox = SimpleNamespace(
        exec_with_output=AsyncMock(return_value=(0, "100.96.0.1 STREAM host.smolvm.internal\n", "")),
        exec_script=AsyncMock(),
    )
    provider = module.SmolVmSandboxProvider()
    for gateway in ("100.96.0.1", "100.96.0.2"):
        sandbox.exec_with_output.return_value = (0, f"{gateway} STREAM host.smolvm.internal\n", "")
        await provider._configure_host_access(sandbox)
        script = sandbox.exec_script.await_args.args[0].replace("/etc/hosts", str(hosts))
        subprocess.run(["bash", "-c", script], check=True)
        lines = hosts.read_text().splitlines()
        assert "127.0.0.1 localhost" in lines
        assert "::1 ip6-localhost" in lines
        assert [line for line in lines if "host.docker.internal" in line] == [f"{gateway} host.docker.internal"]


@pytest.mark.asyncio
async def test_local_object_download_uses_a_scoped_https_grant(monkeypatch, tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path))
    object_url = store.put("image.tar.gz", b"archive")
    issued = MagicMock(return_value=SimpleNamespace(url="https://host.docker.internal:1234/v1/grants/token"))
    monkeypatch.setattr(store, "issue_read_grant", issued)
    monkeypatch.setattr(sandbox_module, "get_config", lambda: SimpleNamespace(get_object_store_at=lambda url: store))
    machine = MagicMock(id="agentenv-test")
    sandbox = SmolVmSandbox(machine, {})
    sandbox.write_host_file = AsyncMock()
    sandbox.exec_script = AsyncMock()
    sandbox._write_unsigned_object = AsyncMock()

    await sandbox._download_object_to_vm(object_url, "/storage/image.tar.gz")

    issued.assert_called_once_with(object_url, expires_in=3600)
    sandbox._write_unsigned_object.assert_not_called()
    sandbox.write_host_file.assert_awaited_once()
    command = sandbox.exec_script.await_args.args[0]
    assert "SSL_CERT_FILE=/etc/agentenv-ca-bundle.pem" in command
    assert "https://host.docker.internal:1234/v1/grants/token" in command
    assert "-o /storage/image.tar.gz" in command


@pytest.mark.asyncio
async def test_unreachable_local_grant_falls_back_to_object_push(monkeypatch, tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path), grant_bind_host="172.17.0.1")
    object_url = store.put("image.tar.gz", b"archive")
    monkeypatch.setattr(sandbox_module, "get_config", lambda: SimpleNamespace(get_object_store_at=lambda url: store))
    sandbox = SmolVmSandbox(MagicMock(id="agentenv-test"), {})
    sandbox._write_unsigned_object = AsyncMock()

    await sandbox._download_object_to_vm(object_url, "/storage/image.tar.gz")
    sandbox._write_unsigned_object.assert_awaited_once_with(store, object_url, "/storage/image.tar.gz")


@pytest.mark.asyncio
async def test_loopback_advertised_grant_uses_object_push(monkeypatch, tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host="localhost")
    object_url = store.put("image.tar.gz", b"archive")
    monkeypatch.setattr(sandbox_module, "get_config", lambda: SimpleNamespace(get_object_store_at=lambda url: store))
    sandbox = SmolVmSandbox(MagicMock(id="agentenv-test"), {})
    sandbox._write_unsigned_object = AsyncMock()

    await sandbox._download_object_to_vm(object_url, "/storage/image.tar.gz")

    sandbox._write_unsigned_object.assert_awaited_once_with(store, object_url, "/storage/image.tar.gz")


@pytest.mark.parametrize("listed", [True, False], ids=["listed", "gone"])
@pytest.mark.asyncio
async def test_terminate_sandbox_deletes_a_listed_machine_without_setting_it_up(monkeypatch, listed):
    deleted = []
    connect = MagicMock(return_value=SimpleNamespace(delete=lambda: deleted.append(True)))
    machines = [SimpleNamespace(name="agentenv-vm-1")] if listed else []
    monkeypatch.setattr(module, "Machine", SimpleNamespace(list=lambda conn: machines, connect=connect))
    setup = AsyncMock()
    monkeypatch.setattr(module.SmolVmSandboxProvider, "_setup_docker", setup)

    await module.SmolVmSandboxProvider().terminate_sandbox("agentenv-vm-1")

    assert deleted == ([True] if listed else [])
    assert connect.called is listed
    setup.assert_not_called()
