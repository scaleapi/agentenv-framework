"""Local SmolVM lifecycle rules that must hold even when cloud credentials are present."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_env.providers.sandbox_providers.smol_vm import provider as module
from agent_env.providers.sandbox_providers.smol_vm.sandbox import SmolVmSandbox, _host_url


@pytest.mark.asyncio
async def test_vm_creation_selects_local_and_can_publish_an_unstarted_application(monkeypatch):
    created = []

    class FakeMachine:
        id = "agentenv-test"

        def endpoint(self, port):
            return SimpleNamespace(http_url=f"http://127.0.0.1:44000")

        def write_file(self, path, data):
            created.append((path, data))

        def delete(self):
            created.append(("deleted", None))

    def create(config, conn):
        assert conn.target == "local"
        assert config.wait_for_ports is False
        assert config.resources.network is True
        assert [(p.host, p.guest) for p in config.ports] == [(44000, 8080)]
        return FakeMachine()

    async def setup(self, sandbox):
        sandbox.extra_hosts = ("host.docker.internal:100.96.0.1",)

    monkeypatch.setattr(module, "Machine", SimpleNamespace(create=create))
    monkeypatch.setattr(module, "_host_port", lambda: 44000)
    monkeypatch.setattr(module.SmolVmSandboxProvider, "_setup_docker", setup)
    monkeypatch.setattr(module.SmolVmSandboxProvider, "_configure_host_access", setup)
    sandbox = await module.SmolVmSandboxProvider().create_vm(exposed_ports=[8080], memory=1024)
    assert sandbox.tunnel_urls[8080] == "http://127.0.0.1:44000"
    assert ("/storage/agentenv-ports.json", b'{"8080": 44000}') in created
    await sandbox.terminate()
    assert ("deleted", None) in created


@pytest.mark.asyncio
async def test_reconnect_uses_explicit_agent_only_readiness(monkeypatch):
    options = []

    class FakeMachine:
        id = "agentenv-test"

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
    assert options[0].target == "local" and options[0].wait_for_ports is False


def test_local_urls_are_rewritten_for_vm_host():
    assert _host_url("http://127.0.0.1:5000/v2/") == "http://host.smolvm.internal:5000/v2/"
    assert _host_url("https://example.org/path") == "https://example.org/path"
