"""Freestyle wire contract and lifecycle failures, with no network access."""

import asyncio
import json
import shlex
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from agent_env.config import ConfigError, reset_config
from agent_env.providers.sandbox_providers.freestyle import (
    FreestyleSandbox,
    FreestyleSandboxProvider,
)
from agent_env.providers.sandbox_providers.sandbox import (
    NetworkMode,
    NetworkPolicy,
    NetworkPolicyUnsupportedError,
)
from agent_env.providers.sandbox_providers.sandbox_provider import (
    build_sandbox_provider,
)


@pytest_asyncio.fixture
async def backend():
    requests = []
    vm = {"id": "vm-test", "resources": {"cpu": 1, "memory": 2048, "storage": 10240}}

    async def respond(request):
        requests.append(request)
        path = request.url.path
        if request.method == "POST" and path == "/v5/vms":
            vm.update(json.loads(request.content))
            return httpx.Response(200, json=vm)
        if request.method == "GET" and path == "/v5/tls":
            rules = [
                {
                    **rule,
                    "protocol": "http",
                    "destination": {**rule["destination"], "vmId": vm["id"]},
                }
                for rule in vm["tls"]["rules"]
            ]
            return httpx.Response(200, json={"rules": rules, "totalCount": len(rules)})
        if request.method == "GET":
            return httpx.Response(200, json=vm)
        if path.endswith("/exec-await"):
            return httpx.Response(
                200, json={"statusCode": 0, "stdout": "ok\n", "stderr": ""}
            )
        return httpx.Response(204)

    provider = FreestyleSandboxProvider(api_key="private-test-key")
    provider._client._client = httpx.AsyncClient(
        base_url="https://api.example.test",
        transport=httpx.MockTransport(respond),
        headers={"Authorization": "Bearer private-test-key"},
    )
    yield provider, requests, vm
    await provider.close()


@pytest.mark.asyncio
async def test_create_sizes_publishes_ports_and_keeps_attribution(backend):
    provider, requests, _ = backend
    sandbox = await provider.create_vm(
        cpu=1.5,
        memory=4096,
        disk_size_gb=12.25,
        timeout=600,
        exposed_ports=[8080, 8081, 8080],
        attribution={"run_id": "run-123"},
    )
    request = json.loads(requests[0].content)
    assert request["snapshotId"] == "freestyle/ubuntu"
    assert request["ttlSeconds"] == 600
    assert request["metadata"] == {"run_id": "run-123"}
    assert request["firewall"] == {
        "rules": [{"action": "allow", "source": {}, "destination": {"public": True}}]
    }
    assert [rule["destination"] for rule in request["tls"]["rules"]] == [
        {"port": 8080},
        {"port": 8081},
    ]
    assert json.loads(requests[1].content) == {
        "cpu": 2,
        "memory": 4096,
        "storage": 12544,
    }
    assert sandbox.type == "freestyle"
    assert sandbox.mode == "vm"
    assert sandbox.network_policy == NetworkPolicy()
    assert sandbox.tunnel_urls == {
        port: f"https://{request['slug']}-{port}.style.dev" for port in (8080, 8081)
    }
    assert "private-test-key" not in repr(provider)


@pytest.mark.asyncio
async def test_snapshot_minimum_is_preserved_without_downsizing(backend):
    provider, requests, _ = backend
    await provider.create_vm(
        image="sh-custom", memory=1024, disk_size_gb=1, setup_for_gateway=False
    )
    assert len(requests) == 1
    assert json.loads(requests[0].content)["snapshotId"] == "sh-custom"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpu": 0},
        {"cpu": float("nan")},
        {"disk_size_gb": float("inf")},
        {"memory": -1},
        {"memory": 1.5},
        {"timeout": 0},
        {"timeout": -1},
        {"exposed_ports": [0]},
        {"exposed_ports": [65536]},
        {"exposed_ports": [True]},
        {"boot_mode": "uefi"},
        {"attribution": {"freestyle.sh/reserved": "x"}},
        {"attribution": {"run_id": "x" * 64}},
        {"attribution": {"": "x"}},
        {"attribution": {str(i): "x" for i in range(65)}},
    ],
)
async def test_invalid_requests_fail_before_provisioning(backend, kwargs):
    provider, requests, _ = backend
    with pytest.raises(ValueError):
        await provider.create_vm(**kwargs)
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("create_vm", {}),
        ("create_sandbox", {"image_name": "img", "port": 8080, "env": {}}),
        ("create_container", {"image_name": "img", "port": 8080, "env": {}}),
    ],
)
async def test_restrictive_policy_is_rejected_on_every_entry_point(
    backend, method, kwargs
):
    provider, requests, _ = backend
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("example.test",))
    assert not provider.supports_network_policy(policy)
    with pytest.raises(NetworkPolicyUnsupportedError):
        await getattr(provider, method)(network_policy=policy, **kwargs)
    assert requests == []


@pytest.mark.asyncio
async def test_setup_failure_deletes_the_vm(backend, monkeypatch):
    provider, requests, _ = backend
    monkeypatch.setattr(
        FreestyleSandbox,
        "setup_vm_for_gateway",
        AsyncMock(side_effect=RuntimeError("Docker missing")),
    )
    with pytest.raises(RuntimeError, match="Docker missing"):
        await provider.create_vm()
    assert requests[-1].method == "DELETE"
    assert requests[-1].url.path == "/v5/vms/vm-test"
    assert not provider._cleanups


@pytest.mark.asyncio
async def test_cancellation_waits_for_creation_then_deletes(backend, monkeypatch):
    provider, _, _ = backend
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def request(method, path, **kwargs):
        calls.append((method, path))
        if method == "POST":
            started.set()
            await release.wait()
            return {"id": "vm-late"}
        return {}

    monkeypatch.setattr(provider._client, "request", request)
    task = asyncio.create_task(provider.create_vm())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == [("POST", "/v5/vms"), ("DELETE", "/v5/vms/vm-late")]


@pytest.mark.asyncio
async def test_lost_create_response_cleans_up_by_slug(backend, monkeypatch):
    provider, _, _ = backend
    calls = []

    async def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if method == "POST":
            raise httpx.ReadTimeout("lost response")
        return {}

    monkeypatch.setattr(provider._client, "request", request)
    with pytest.raises(httpx.ReadTimeout):
        await provider.create_vm()
    assert calls[1][:2] == ("DELETE", "/v5/vms/" + calls[0][2]["json"]["slug"])


@pytest.mark.asyncio
async def test_reconnect_recovers_ingress_and_reports_unknown_policy(backend):
    provider, requests, _ = backend
    created = await provider.create_vm(
        exposed_ports=[8080, 8081], setup_for_gateway=False
    )
    reconnected = await provider.get_sandbox(created.sandbox_id)
    assert reconnected.sandbox_id == created.sandbox_id
    assert reconnected.tunnel_urls == created.tunnel_urls
    assert reconnected.network_policy is None
    assert requests[-1].url.params["vmId"] == created.sandbox_id


@pytest.mark.asyncio
async def test_exec_preserves_argv_and_nonzero_exit(backend, monkeypatch):
    provider, _, _ = backend
    request = AsyncMock(
        return_value={"statusCode": 7, "stdout": "héllo", "stderr": "failed"}
    )
    monkeypatch.setattr(provider._client, "request", request)
    sandbox = FreestyleSandbox(provider._client, "vm-test", tunnel_urls={})
    argv = ("bash", "-c", "printf '%s' \"$1\"", "arg0", "a b; $(touch /tmp/bad)")
    assert await sandbox.exec_with_output("sudo", *argv) == (7, "héllo", "failed")
    payload = request.call_args.kwargs["json"]
    assert shlex.split(payload["command"]) == list(argv)
    assert payload["linuxUser"] == "root"
    assert payload["timeoutMs"] == 300000
    process = await sandbox.exec("true")
    assert b"".join([chunk async for chunk in process.stdout]) == "héllo".encode()


@pytest.mark.asyncio
async def test_guest_timeout_is_a_failure_without_transient_retry(backend, monkeypatch):
    provider, _, _ = backend
    request = AsyncMock(
        return_value={"statusCode": None, "stdout": "partial", "stderr": ""}
    )
    monkeypatch.setattr(provider._client, "request", request)
    sandbox = FreestyleSandbox(provider._client, "vm-test", tunnel_urls={})
    with pytest.raises(RuntimeError, match="exit 124"):
        await sandbox.exec_script("sleep 500", max_retries=2)
    assert request.await_count == 1


@pytest.mark.asyncio
async def test_host_file_write_uses_binary_filesystem_api(backend):
    provider, requests, _ = backend
    sandbox = FreestyleSandbox(provider._client, "vm-test", tunnel_urls={})
    data = bytes(range(256)) * 1024
    await sandbox.write_host_file(data, "/tmp/folder/file with spaces")
    request = requests[-1]
    assert request.method == "PUT"
    assert request.url.path == "/v5/vms/vm-test/fs/write"
    assert request.url.params["path"] == "/tmp/folder/file with spaces"
    assert request.content == data
    assert request.headers["content-type"] == "application/octet-stream"


def test_config_resolves_credentials_and_snapshot(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(
        '[sandbox.providers.freestyle.config]\napi_key = "env:TEST_FREESTYLE_KEY"\nsnapshot_id = "sh-custom"\n'
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    monkeypatch.setenv("TEST_FREESTYLE_KEY", "resolved-key")
    reset_config()
    provider = build_sandbox_provider("freestyle")
    assert isinstance(provider, FreestyleSandboxProvider)
    assert provider._client._api_key == "resolved-key"
    assert provider._snapshot_id == "sh-custom"
    assert provider._client._client is None


def test_missing_config_is_actionable():
    with pytest.raises(ConfigError, match="requires a non-empty 'api_key'"):
        FreestyleSandboxProvider.from_config()


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.test",
        "https://user:pass@api.example.test",
        "https://api.example.test?key=abc",
    ],
)
def test_credentials_require_a_safe_https_api_url(url):
    with pytest.raises(ValueError, match="HTTPS"):
        FreestyleSandboxProvider(api_key="private-key", api_url=url)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,attempts", [(503, 3), (429, 3), (401, 1), (404, 1)])
async def test_provisioning_cleanup_retries_only_transient_errors(
    backend, monkeypatch, status, attempts, caplog
):
    provider, _, _ = backend
    response = httpx.Response(
        status,
        request=httpx.Request("DELETE", "https://api.example.test/v5/vms/vm-test"),
    )
    request = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "delete failed", request=response.request, response=response
        )
    )
    monkeypatch.setattr(provider._client, "request", request)
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    creation = asyncio.get_running_loop().create_future()
    creation.set_result({"id": "vm-test"})
    await provider._reap_creation(creation, "agentenv-test")
    assert request.await_count == attempts
    assert bool(caplog.records) is (status != 404)


@pytest.mark.asyncio
async def test_reconnect_pages_and_excludes_egress_rules(backend, monkeypatch):
    provider, _, _ = backend
    vm = {"id": "vm-test", "slug": "agentenv-test"}
    egress = {
        "protocol": "http",
        "source": {"vmId": "vm-test"},
        "destination": {"public": True},
        "domain": "api.example.test",
    }
    ingress = {
        "protocol": "http",
        "source": {"public": True},
        "destination": {"vmId": "vm-test", "port": 8080},
        "domain": "agentenv-test-8080.style.dev",
    }
    request = AsyncMock(
        side_effect=[
            vm,
            {"rules": [egress], "totalCount": 2},
            {"rules": [ingress], "totalCount": 2},
        ]
    )
    monkeypatch.setattr(provider._client, "request", request)
    sandbox = await provider.get_sandbox("vm-test")
    assert sandbox.tunnel_urls == {8080: "https://agentenv-test-8080.style.dev"}
    assert request.call_args.kwargs["params"]["offset"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [204, 404, 403])
async def test_termination_is_idempotent_and_preserves_permission_errors(
    backend, monkeypatch, status
):
    provider, _, _ = backend
    response = httpx.Response(
        status,
        request=httpx.Request("DELETE", "https://api.example.test/v5/vms/vm-test"),
    )
    error = (
        None
        if status == 204
        else httpx.HTTPStatusError(
            "delete failed", request=response.request, response=response
        )
    )
    monkeypatch.setattr(
        provider._client, "request", AsyncMock(return_value={}, side_effect=error)
    )
    sandbox = FreestyleSandbox(provider._client, "vm-test", tunnel_urls={})
    if status == 403:
        with pytest.raises(httpx.HTTPStatusError):
            await sandbox.terminate()
    else:
        await sandbox.terminate()
