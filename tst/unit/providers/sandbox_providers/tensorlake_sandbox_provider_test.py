"""Offline unit tests for the Tensorlake sandbox provider."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from tensorlake.sandbox import RemoteAPIError, SandboxNotFoundError, SandboxPending

from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.tensorlake.provider import (
    HOST_IMAGE_DOCKERFILE,
    SANDBOX_STARTED_EVENT,
    TensorlakeSandboxProvider,
)
from agent_env.providers.sandbox_providers.tensorlake.sandbox import TensorlakeSandbox


def _info(*, ports=(), network=None):
    info = MagicMock()
    info.exposed_ports = list(ports)
    info.network_policy = network
    info.url_for_port.side_effect = lambda port: f"https://{port}-tl-test.sandbox.tensorlake.ai"
    return info


def _raw_sandbox() -> MagicMock:
    raw = MagicMock()
    raw.sandbox_id = "tl-test"
    raw.update = AsyncMock(side_effect=lambda **kwargs: _info(ports=kwargs.get("exposed_ports", ())))
    raw.terminate = AsyncMock()
    raw.info = AsyncMock(return_value=_info())
    return raw


def _provider(raw: MagicMock | None = None, **kwargs) -> tuple[TensorlakeSandboxProvider, MagicMock]:
    sdk = MagicMock()
    sdk.pending = MagicMock(sandbox_id="tl-test")
    sdk.pending.ready = AsyncMock(return_value=raw or _raw_sandbox())
    sdk.create = AsyncMock(return_value=sdk.pending)
    sdk.connect = AsyncMock(return_value=raw or _raw_sandbox())
    sdk.client = MagicMock()
    sdk.client.delete = AsyncMock()
    client_cls = MagicMock()
    client_cls.for_cloud.return_value.__aenter__ = AsyncMock(return_value=sdk.client)
    client_cls.for_cloud.return_value.__aexit__ = AsyncMock(return_value=None)
    sdk.client_cls = client_cls
    provider = TensorlakeSandboxProvider(
        api_key="tl-key", sandbox_cls=sdk, client_cls=client_cls, **kwargs,
    )
    return provider, sdk


@pytest.fixture(autouse=True)
def _no_other_platform_hosts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(TensorlakeSandboxProvider, "effective_network_policy", classmethod(
        lambda cls, policy: (policy or NetworkPolicy()).with_hosts(cls.EGRESS_HOSTS)
    ))


def test_from_config_requires_an_api_key():
    with pytest.raises(ConfigError, match="requires a non-empty 'api_key'"):
        TensorlakeSandboxProvider.from_config(image="my-image")


@pytest.mark.parametrize("key", ["sandbox_cls", "client_cls", "region"])
def test_from_config_refuses_unknown_keys(key: str):
    with pytest.raises(ConfigError, match=rf"unknown key\(s\): \['{key}'\]"):
        TensorlakeSandboxProvider.from_config(api_key="tl-key", **{key: "x"})


@pytest.mark.parametrize(("key", "value"), [("image", 1), ("image", " "), ("api_url", ["https://api.example"])])
def test_from_config_refuses_a_non_string_or_blank_value(key: str, value):
    with pytest.raises(ConfigError, match=f"'{key}' must be a non-empty string"):
        TensorlakeSandboxProvider.from_config(api_key="tl-key", **{key: value})


def test_from_config_takes_every_known_key():
    provider = TensorlakeSandboxProvider.from_config(api_key="tl-key", image="team-host", api_url="https://api.example")

    assert provider._client_kwargs() == {"api_key": "tl-key", "api_url": "https://api.example"}
    assert provider._image == "team-host"


@pytest.mark.parametrize("bad_key", ["", "   "])
def test_constructor_rejects_a_blank_api_key(bad_key: str):
    with pytest.raises(ValueError, match="api_key"):
        TensorlakeSandboxProvider(api_key=bad_key)


def test_ipv6_policies_are_unsupported():
    assert TensorlakeSandboxProvider.supports_network_policy(
        NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("api.example",), allow_cidrs=("10.0.0.0/8",))
    )
    assert not TensorlakeSandboxProvider.supports_network_policy(
        NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_cidrs=("2001:db8::/32",))
    )
    assert not TensorlakeSandboxProvider.supports_network_policy(
        NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("::1",))
    )


@pytest.mark.asyncio
async def test_create_vm_refuses_an_ipv6_policy_before_creating_anything():
    provider, sdk = _provider()

    with pytest.raises(NetworkPolicyUnsupportedError):
        await provider.create_vm(
            network_policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_cidrs=("2001:db8::/32",)),
            setup_for_gateway=False,
        )

    sdk.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_vm_logs_an_ignored_boot_mode_and_a_raised_disk(caplog: pytest.LogCaptureFixture):
    provider, _ = _provider()

    with caplog.at_level(logging.INFO):
        await provider.create_vm(boot_mode="uefi", cpu=2, memory=4096, disk_size_gb=10, setup_for_gateway=False)

    assert "Ignoring boot_mode=uefi" in caplog.text
    assert "disk=10GB to cpu=2.0, memory=4096MB, disk=30720MB" in caplog.text


@pytest.mark.asyncio
async def test_create_vm_raises_resources_to_the_floors_and_exposes_ports_publicly():
    raw = _raw_sandbox()
    provider, sdk = _provider(raw)

    sandbox = await provider.create_vm(
        cpu=1, memory=2048, disk_size_gb=10, timeout=900, exposed_ports=[8080, 8080, 9000], setup_for_gateway=False,
    )

    assert isinstance(sandbox, TensorlakeSandbox)
    sdk.create.assert_awaited_once_with(
        image="agentenv-dind-host-v1",
        cpus=2.0,
        memory_mb=4096,
        disk_mb=30 * 1024,
        timeout_secs=900,
        allow_internet_access=True,
        allow_out=[],
        deny_out=[],
        api_key="tl-key",
        api_url="https://api.tensorlake.ai",
        wait=False,
    )
    raw.update.assert_awaited_once_with(exposed_ports=[8080, 9000], allow_unauthenticated_access=True)
    assert sandbox.tunnel_urls == {
        8080: "https://8080-tl-test.sandbox.tensorlake.ai",
        9000: "https://9000-tl-test.sandbox.tensorlake.ai",
    }
    assert sandbox.network_policy == NetworkPolicy()


@pytest.mark.asyncio
async def test_create_vm_keeps_a_request_above_the_floors():
    provider, sdk = _provider(api_url="https://api.example")

    await provider.create_vm(cpu=4, memory=16384, disk_size_gb=50, setup_for_gateway=False)

    kwargs = sdk.create.await_args.kwargs
    assert (kwargs["cpus"], kwargs["memory_mb"], kwargs["disk_mb"]) == (4.0, 16384, 50 * 1024)
    assert kwargs["api_url"] == "https://api.example"


@pytest.mark.asyncio
async def test_creation_and_cleanup_use_one_endpoint_whatever_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TENSORLAKE_API_URL", "https://ambient.example")
    provider, sdk = _provider()
    sdk.pending.ready.side_effect = SandboxPending("tl-test", pending_reason="no_capacity")

    with pytest.raises(SandboxPending):
        await provider.create_vm(setup_for_gateway=False)

    assert sdk.create.await_args.kwargs["api_url"] == "https://api.tensorlake.ai"
    assert sdk.client_cls.for_cloud.call_args.kwargs["api_url"] == "https://api.tensorlake.ai"


@pytest.mark.asyncio
async def test_create_vm_raises_memory_to_one_gb_per_cpu():
    provider, sdk = _provider()

    await provider.create_vm(cpu=6, memory=4096, setup_for_gateway=False)

    assert sdk.create.await_args.kwargs["memory_mb"] == 6 * 1024


@pytest.mark.parametrize(
    ("resources", "message"),
    [({"cpu": 2, "memory": 32768}, "per CPU"), ({"disk_size_gb": 200}, "GB of disk")],
)
@pytest.mark.asyncio
async def test_create_vm_rejects_resources_tensorlake_cannot_give(resources, message):
    provider, sdk = _provider()

    with pytest.raises(ValueError, match=message):
        await provider.create_vm(setup_for_gateway=False, **resources)

    sdk.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_vm_sends_an_allowlist_with_the_ingress_domain():
    provider, sdk = _provider()
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("api.example",), allow_cidrs=("10.0.0.0/8",))

    sandbox = await provider.create_vm(network_policy=policy, setup_for_gateway=False)

    kwargs = sdk.create.await_args.kwargs
    assert kwargs["allow_internet_access"] is True
    assert kwargs["allow_out"] == ["api.example", "*.sandbox.tensorlake.ai", "10.0.0.0/8"]
    assert sandbox.network_policy == policy.with_hosts(["*.sandbox.tensorlake.ai"])


@pytest.mark.asyncio
async def test_create_vm_uses_a_configured_or_requested_image():
    provider, sdk = _provider(image="team-docker-host")

    await provider.create_vm(setup_for_gateway=False)
    assert sdk.create.await_args.kwargs["image"] == "team-docker-host"

    await provider.create_vm(image="request-image", setup_for_gateway=False)
    assert sdk.create.await_args.kwargs["image"] == "request-image"


@pytest.mark.asyncio
async def test_create_vm_explains_how_to_publish_an_unregistered_image():
    provider, sdk = _provider(api_url="https://api.example")
    sdk.create.side_effect = RemoteAPIError(
        400, "Image 'agentenv-dind-host-v1' is not registered in the server. Register it first."
    )

    with pytest.raises(ValueError) as raised:
        await provider.create_vm(setup_for_gateway=False)

    message = str(raised.value)
    assert "'agentenv-dind-host-v1' is not registered at https://api.example" in message
    assert "host_image.Dockerfile -n agentenv-dind-host-v1 --disk_mb 30720" in message
    assert isinstance(raised.value.__cause__, RemoteAPIError)


@pytest.mark.asyncio
async def test_create_vm_lets_other_api_errors_through():
    provider, sdk = _provider()
    error = RemoteAPIError(400, "cpus out of range")
    sdk.create.side_effect = error

    with pytest.raises(RemoteAPIError) as raised:
        await provider.create_vm(setup_for_gateway=False)

    assert raised.value is error


def test_the_host_image_dockerfile_ships_with_the_provider():
    assert HOST_IMAGE_DOCKERFILE.is_file()
    assert "docker-compose-plugin" in HOST_IMAGE_DOCKERFILE.read_text()


@pytest.mark.asyncio
async def test_create_vm_terminates_the_sandbox_when_setup_fails(monkeypatch: pytest.MonkeyPatch):
    raw = _raw_sandbox()
    provider, _ = _provider(raw)
    monkeypatch.setattr(TensorlakeSandbox, "setup_vm_for_gateway", AsyncMock(side_effect=RuntimeError("no docker")))

    with pytest.raises(RuntimeError, match="no docker"):
        await provider.create_vm(exposed_ports=[8080])

    raw.terminate.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_create_vm_deletes_a_sandbox_that_does_not_start():
    provider, sdk = _provider()
    sdk.pending.ready.side_effect = SandboxPending("tl-test", pending_reason="no_capacity")

    with pytest.raises(SandboxPending):
        await provider.create_vm(setup_for_gateway=False)

    sdk.client_cls.for_cloud.assert_called_once_with(api_key="tl-key", api_url="https://api.tensorlake.ai")
    sdk.client.delete.assert_awaited_once_with("tl-test")


@pytest.mark.asyncio
async def test_create_vm_deletes_the_sandbox_when_the_caller_cancels_the_start():
    provider, sdk = _provider()
    started = asyncio.Event()

    async def never_ready():
        started.set()
        await asyncio.Event().wait()

    sdk.pending.ready.side_effect = never_ready
    task = asyncio.create_task(provider.create_vm(setup_for_gateway=False))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    sdk.client.delete.assert_awaited_once_with("tl-test")


@pytest.mark.asyncio
async def test_a_cancelled_create_vm_deletes_the_sandbox_its_create_yields_before_it_ends():
    provider, sdk = _provider()
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_create(**kwargs):
        started.set()
        await release.wait()
        return sdk.pending

    sdk.create.side_effect = slow_create
    task = asyncio.create_task(provider.create_vm(setup_for_gateway=False))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    sdk.client.delete.assert_awaited_once_with("tl-test")
    sdk.pending.ready.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_create_vm_cancelled_twice_leaves_the_delete_to_a_callback():
    provider, sdk = _provider()
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_create(**kwargs):
        started.set()
        await release.wait()
        return sdk.pending

    sdk.create.side_effect = slow_create
    task = asyncio.create_task(provider.create_vm(setup_for_gateway=False))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    sdk.client.delete.assert_not_awaited()
    release.set()
    for _ in range(5):
        await asyncio.sleep(0)
    sdk.client.delete.assert_awaited_once_with("tl-test")


@pytest.mark.asyncio
async def test_create_vm_logs_the_attribution_in_the_started_event(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="agent_env.providers.sandbox_providers.tensorlake.provider")
    provider, _ = _provider()

    await provider.create_vm(setup_for_gateway=False, attribution={"run_id": "inst-1", "team": "t", "unset": None})

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == SANDBOX_STARTED_EVENT]
    assert record.tensorlake_sandbox_id == "tl-test"
    assert record.tensorlake_attribution == {"run_id": "inst-1", "team": "t"}
    assert record.run_id == "inst-1"
    assert "tl-key" not in caplog.text


@pytest.mark.asyncio
async def test_create_vm_ignores_a_sandbox_already_gone_during_cleanup():
    provider, sdk = _provider()
    sdk.pending.ready.side_effect = RuntimeError("failed")
    sdk.client.delete.side_effect = SandboxNotFoundError("tl-test")

    with pytest.raises(RuntimeError, match="failed"):
        await provider.create_vm(setup_for_gateway=False)


@pytest.mark.asyncio
async def test_create_sandbox_is_a_vm_with_one_port():
    raw = _raw_sandbox()
    provider, _ = _provider(raw)
    provider.create_vm = AsyncMock(return_value="vm")

    assert await provider.create_sandbox(image_name="agent:latest", port=8000, env={"A": "1"}, timeout=60) == "vm"
    assert provider.create_vm.await_args.kwargs["exposed_ports"] == [8000]


@pytest.mark.asyncio
async def test_get_sandbox_recovers_ports_and_policy():
    raw = _raw_sandbox()
    network = SimpleNamespace(allow_internet_access=True, allow_out=["api.example", "10.0.0.0/8"], deny_out=[])
    raw.info = AsyncMock(return_value=_info(ports=[8080], network=network))
    provider, sdk = _provider(raw)

    sandbox = await provider.get_sandbox("tl-test")

    sdk.connect.assert_awaited_once_with("tl-test", api_key="tl-key", api_url="https://api.tensorlake.ai")
    assert sandbox.tunnel_urls == {8080: "https://8080-tl-test.sandbox.tensorlake.ai"}
    assert sandbox.network_policy == NetworkPolicy(
        mode=NetworkMode.ALLOWLIST, allow_hosts=("api.example",), allow_cidrs=("10.0.0.0/8",)
    )


@pytest.mark.asyncio
async def test_get_sandbox_without_an_ingress_endpoint_still_reconnects_for_teardown():
    raw = _raw_sandbox()
    info = _info(ports=[8080])
    info.url_for_port.side_effect = lambda port: None
    raw.info = AsyncMock(return_value=info)
    provider, _ = _provider(raw)

    sandbox = await provider.get_sandbox("tl-test")
    await sandbox.terminate()

    assert sandbox.tunnel_urls == {}
    raw.terminate.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_sandbox_with_an_unknown_policy_fails_closed():
    raw = _raw_sandbox()
    raw.info = AsyncMock(
        return_value=_info(network=SimpleNamespace(allow_internet_access=True, allow_out=[], deny_out=["1.2.3.4"]))
    )
    provider, _ = _provider(raw)

    sandbox = await provider.get_sandbox("tl-test")

    assert sandbox.network_policy is None
