"""Unit tests for the Vercel provider; all SDK boundaries are fakes."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.config.errors import ConfigError
from agent_env.task import teardown
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps import teardown_sandboxes
from agent_env.providers.sandbox_providers import vercel as vercel_module
from agent_env.providers.sandbox_providers.vercel import sandbox as vercel_sandbox_module
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError, build_sandbox_provider
from agent_env.providers.sandbox_providers.vercel.provider import (
    VercelSandboxProvider,
    resource_shape,
)
from agent_env.providers.sandbox_providers.vercel import provider as provider_module


def _raw(routes=((8080, "https://sb-1.vercel.run"),), policy_mode="allow-all"):
    raw = MagicMock(name="sb-1")
    raw.name = "sb-1"
    raw.routes = [SimpleNamespace(port=port, url=url) for port, url in routes]
    raw.network_policy = SimpleNamespace(
        mode=policy_mode,
        allow={},
        subnets=SimpleNamespace(allow=(), deny=None),
    )
    raw.destroy = AsyncMock()
    return raw


def _client(raw):
    client = MagicMock()
    client.create_sandbox = AsyncMock(return_value=raw)
    client.get_sandbox = AsyncMock(return_value=raw)
    client.aclose = AsyncMock()
    return client


def _provider(raw=None, **config):
    raw = raw or _raw()
    client = _client(raw)
    provider = VercelSandboxProvider(client_factory=lambda: client, **config)
    provider._client = client
    return provider


@pytest.mark.parametrize(
    ("cpu", "memory", "shape"),
    [(1, 2048, (1, 2048)), (1, 8192, (4, 8192)), (3, 2048, (4, 8192)), (0.5, 5000, (4, 8192)), (1, 1, (1, 2048))],
)
def test_resource_shape_is_the_smallest_supported_cpu_count(cpu, memory, shape):
    assert resource_shape(cpu, memory) == shape


@pytest.mark.parametrize(
    ("cpu", "memory"),
    [(True, 2048), (0, 2048), (float("nan"), 2048), (1, True), (1, 0), (1, 64 * 1024 + 1), (33, 8192)],
)
def test_invalid_resources_are_refused(cpu, memory):
    with pytest.raises(ValueError):
        resource_shape(cpu, memory)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"image": ""}, "'image' must be a non-empty string"),
        ({"token": "t"}, "token, team_id and project_id together"),
        ({"failover_regions": "iad1"}, "'failover_regions' must be a list"),
        ({"failover_regions": ["iad1", "iad1"]}, "cannot contain duplicates"),
        ({"region": "iad1", "failover_regions": ["iad1"]}, "cannot include region"),
        ({"unknown": 1}, "unknown key"),
    ],
)
def test_from_config_rejects_invalid_config(config, message):
    with pytest.raises(ConfigError, match=message):
        VercelSandboxProvider.from_config(**config)


def test_provider_is_registered_and_does_not_claim_a_private_shared_network():
    provider = build_sandbox_provider("vercel")
    assert isinstance(provider, VercelSandboxProvider)
    assert provider.shares_network_with("vercel") is False


@pytest.mark.asyncio
async def test_create_vm_sends_the_validated_provider_configuration(monkeypatch):
    setup = AsyncMock()
    monkeypatch.setattr(vercel_module.VercelSandbox, "setup_vm_for_gateway", setup)
    monkeypatch.setattr(provider_module, "vercel_network_policy", lambda policy: policy)
    raw = _raw(routes=((8080, "https://sb-1.vercel.run"), (9000, "https://sb-2.vercel.run")))
    provider = _provider(
        raw,
        image="prepared-image:tag",
        region="sfo1",
        failover_regions=["cle1"],
        network_id="network-1",
    )
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",))

    sandbox = await provider.create_vm(
        cpu=1,
        memory=8192,
        disk_size_gb=20,
        timeout=900,
        exposed_ports=[8080, 9000, 8080],
        attribution={"run_id": "inst-1"},
        network_policy=policy,
    )

    client = provider._client
    kwargs = client.create_sandbox.await_args.kwargs
    assert kwargs["name"].startswith("agentenv-")
    assert kwargs["image"] == "prepared-image:tag"
    assert kwargs["ports"] == [8080, 9000]
    assert kwargs["execution_time_limit"] == 900
    assert (kwargs["resources"].vcpus, kwargs["resources"].memory) == (4, 8192)
    assert kwargs["persistent"] is False
    assert kwargs["network_id"] == "network-1"
    assert kwargs["region"] == "sfo1"
    assert kwargs["failover_regions"] == ["cle1"]
    assert kwargs["tags"] == {"run_id": "inst-1"}
    assert "*.vercel.run" in kwargs["network_policy"].allow_hosts
    assert (sandbox.sandbox_id, sandbox.mode) == ("sb-1", "vm")
    assert sandbox.tunnel_urls == {
        8080: "https://sb-1.vercel.run",
        9000: "https://sb-2.vercel.run",
    }
    setup.assert_awaited_once_with([8080, 9000])


@pytest.mark.asyncio
async def test_a_setup_failure_destroys_the_created_sandbox(monkeypatch):
    setup = AsyncMock(side_effect=RuntimeError("Docker never came up"))
    monkeypatch.setattr(vercel_module.VercelSandbox, "setup_vm_for_gateway", setup)
    monkeypatch.setattr(vercel_sandbox_module, "vercel_network_policy", lambda policy: policy)
    raw = _raw()
    provider = _provider(raw)
    with pytest.raises(RuntimeError, match="Docker never came up"):
        await provider.create_vm(exposed_ports=[])
    raw.destroy.assert_awaited_once_with(delete_orphan_snapshots=True)


@pytest.mark.asyncio
async def test_create_vm_logs_requested_and_allocated_resources_and_fixed_disk(monkeypatch, caplog):
    monkeypatch.setattr(vercel_module.VercelSandbox, "setup_vm_for_gateway", AsyncMock())
    monkeypatch.setattr(vercel_sandbox_module, "vercel_network_policy", lambda policy: policy)
    provider = _provider()

    with caplog.at_level(logging.INFO, logger=provider_module.logger.name):
        await provider.create_vm(
            cpu=0.5,
            memory=5000,
            disk_size_gb=10,
            exposed_ports=[],
            setup_for_gateway=False,
        )

    assert "requested_cpu=0.5 requested_memory_mb=5000 allocated_vcpus=4 allocated_memory_mb=8192" in caplog.text
    assert "requested_disk_gb=10 fixed_disk_gb=64" in caplog.text
    resources = provider._client.create_sandbox.await_args.kwargs["resources"]
    assert (resources.vcpus, resources.memory) == (4, 8192)


@pytest.mark.asyncio
async def test_create_vm_does_not_log_an_exact_resource_or_disk_match(monkeypatch, caplog):
    monkeypatch.setattr(vercel_module.VercelSandbox, "setup_vm_for_gateway", AsyncMock())
    monkeypatch.setattr(vercel_sandbox_module, "vercel_network_policy", lambda policy: policy)
    provider = _provider()

    with caplog.at_level(logging.INFO, logger=provider_module.logger.name):
        await provider.create_vm(
            cpu=1,
            memory=2048,
            disk_size_gb=64,
            exposed_ports=[],
            setup_for_gateway=False,
        )

    assert "resource shape adjusted" not in caplog.text
    assert "disk size is fixed" not in caplog.text


@pytest.mark.asyncio
async def test_a_missing_route_destroys_the_sandbox(monkeypatch):
    monkeypatch.setattr(vercel_module.VercelSandbox, "setup_vm_for_gateway", AsyncMock())
    monkeypatch.setattr(vercel_sandbox_module, "vercel_network_policy", lambda policy: policy)
    raw = _raw(routes=())
    provider = _provider(raw)
    with pytest.raises(RuntimeError, match="no public route for port\\(s\\) \\[8080\\]"):
        await provider.create_vm(exposed_ports=[8080])
    raw.destroy.assert_awaited_once_with(delete_orphan_snapshots=True)


@pytest.mark.asyncio
async def test_a_cancelled_create_reaps_the_sandbox_it_produces(monkeypatch):
    monkeypatch.setattr(vercel_sandbox_module, "vercel_network_policy", lambda policy: policy)
    raw = _raw()
    client = _client(raw)
    created = asyncio.Event()

    async def slow_create(**_kwargs):
        await created.wait()
        return raw

    client.create_sandbox = slow_create
    provider = VercelSandboxProvider(client_factory=lambda: client)
    caller = asyncio.ensure_future(provider.create_vm(exposed_ports=[]))
    await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    created.set()
    for _ in range(20):
        if raw.destroy.await_count:
            break
        await asyncio.sleep(0)
    raw.destroy.assert_awaited_once_with(delete_orphan_snapshots=True)


@pytest.mark.asyncio
async def test_get_sandbox_restores_routes_and_representable_policy():
    raw = _raw()
    raw.network_policy = SimpleNamespace(
        mode="custom",
        allow={"pypi.org": ()},
        subnets=SimpleNamespace(allow=("10.0.0.0/8",), deny=None),
    )
    provider = _provider(raw)
    sandbox = await provider.get_sandbox("sb-1")
    provider._client.get_sandbox.assert_awaited_once_with(name="sb-1")
    assert sandbox.tunnel_urls == {8080: "https://sb-1.vercel.run"}
    assert sandbox.network_policy == NetworkPolicy(
        mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("10.0.0.0/8",)
    )


@pytest.mark.asyncio
async def test_close_closes_the_long_lived_sdk_client():
    provider = _provider()
    client = provider._client
    await provider.create_vm(exposed_ports=[], setup_for_gateway=False)
    await provider.close()
    client.aclose.assert_awaited_once_with()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image": "ubuntu"}, "image overrides are unsupported"),
        ({"boot_mode": "custom"}, "boot modes are unsupported"),
        ({"disk_size_gb": 65}, "disk_size_gb"),
        ({"timeout": 0}, "timeout"),
        ({"exposed_ports": list(range(1, 17))}, "at most 15"),
        ({"exposed_ports": [0]}, "ports must be integers"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_create_arguments_fail_before_provisioning(kwargs, message):
    provider = _provider()
    with pytest.raises((ValueError,), match=message):
        await provider.create_vm(**kwargs)
    provider._client.create_sandbox.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_response", [False, True])
async def test_close_waits_for_cancelled_creation_and_reclaims_even_a_lost_response(lost_response):
    raw = _raw()
    client = _client(raw)
    started, release = asyncio.Event(), asyncio.Event()
    order = []

    async def create(**kwargs):
        started.set()
        await release.wait()
        if lost_response:
            raise RuntimeError("response lost after allocation")
        return raw

    async def destroy(**kwargs):
        order.append("destroy")

    async def close():
        order.append("close")

    client.create_sandbox = create
    raw.destroy.side_effect = destroy
    client.aclose.side_effect = close
    provider = VercelSandboxProvider(client_factory=lambda: client)
    caller = asyncio.create_task(provider.create_vm(setup_for_gateway=False))
    await started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    closing = asyncio.create_task(provider.close())
    await asyncio.sleep(0)
    assert not closing.done()
    assert order == []
    release.set()
    await asyncio.wait_for(closing, timeout=1)
    assert order == ["destroy", "close"]


def test_excess_tags_keep_run_and_step_then_sorted_keys_and_warn_without_values(caplog):
    original = {"z": "private-value", "run_id": "run", "d": "4", "pipeline_step": "step", "c": "3", "b": "2", "a": "1"}
    with caplog.at_level(logging.WARNING):
        tags = VercelSandboxProvider._tags(original)
        reordered = VercelSandboxProvider._tags(dict(reversed(list(original.items()))))
    assert tags == reordered == {"pipeline_step": "step", "run_id": "run", "a": "1", "b": "2", "c": "3"}
    assert len(original) == 7
    assert "d, z" in caplog.text
    assert "private-value" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["run", "step"])
async def test_cleanup_records_an_already_deleted_sandbox_as_torn_down(monkeypatch, entry):
    provider = _provider()
    client = provider._client
    provider._client.get_sandbox.side_effect = type("Missing", (Exception,), {"status_code": 404})("gone")
    monkeypatch.setattr(teardown, "build_sandbox_provider", lambda _: provider)
    monkeypatch.setattr(teardown_sandboxes, "build_sandbox_provider", lambda _: provider)
    context = TaskStepContext(deployed_sandboxes=[DeployedSandbox(
        sandbox_name="box", sandbox_id="gone", sandbox_mode="vm", sandbox_type="vercel",
    )])
    if entry == "run":
        report = await teardown.teardown_run(context)
        assert not report.failed and not report.still_up
        assert [box.sandbox_id for box in report.terminated] == ["gone"]
    else:
        await teardown_sandboxes.TeardownSandboxesTaskStep(
            id="cleanup", version=1, sandbox_names=["box"],
        ).execute(context)
    assert context.metadata[teardown_sandboxes.TORN_DOWN_KEY] == ["gone"]
    client.get_sandbox.assert_awaited_once_with(name="gone")
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_lookup_still_raises_and_preserves_the_sdk_error():
    provider = _provider()
    missing = type("Missing", (Exception,), {"status_code": 404})("gone")
    provider._client.get_sandbox.side_effect = missing
    with pytest.raises(SandboxNotFoundError) as caught:
        await provider.get_sandbox("gone")
    assert caught.value.__cause__ is missing


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [
    type("Unauthorized", (Exception,), {"status_code": 401})("unauthorized"),
    type("Forbidden", (Exception,), {"status_code": 403})("forbidden"),
    ConnectionError("offline"),
    asyncio.CancelledError(),
])
async def test_lookup_preserves_auth_transport_and_cancellation_errors(error):
    provider = _provider()
    provider._client.get_sandbox.side_effect = error
    with pytest.raises(type(error)) as caught:
        await provider.get_sandbox("box")
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["run", "step"])
async def test_cleanup_does_not_mark_permission_failure_as_torn_down(monkeypatch, entry):
    provider = _provider()
    provider._client.get_sandbox.side_effect = type("Forbidden", (Exception,), {"status_code": 403})("forbidden")
    monkeypatch.setattr(teardown, "build_sandbox_provider", lambda _: provider)
    monkeypatch.setattr(teardown_sandboxes, "build_sandbox_provider", lambda _: provider)
    context = TaskStepContext(deployed_sandboxes=[DeployedSandbox(
        sandbox_name="box", sandbox_id="box", sandbox_mode="vm", sandbox_type="vercel",
    )])
    if entry == "run":
        report = await teardown.teardown_run(context)
        assert len(report.failed) == 1 and report.still_up
    else:
        await teardown_sandboxes.TeardownSandboxesTaskStep(id="cleanup", version=1, sandbox_names=["box"]).execute(context)
    assert not context.metadata.get(teardown_sandboxes.TORN_DOWN_KEY)


@pytest.mark.parametrize("tags", [{}, {"a": "1"}, {str(i): str(i) for i in range(5)}])
def test_supported_tags_are_preserved_without_warning(tags, caplog):
    with caplog.at_level(logging.WARNING):
        assert VercelSandboxProvider._tags(tags) == tags
    assert not caplog.records


def test_invalid_tag_is_not_hidden_by_overflow():
    with pytest.raises(ValueError, match="non-empty"):
        VercelSandboxProvider._tags({**{str(i): "v" for i in range(6)}, "z": ""})


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 409])
async def test_terminate_by_id_cancels_streams_only_when_remote_is_gone(status):
    raw = _raw()
    started = asyncio.Event()

    async def stalled(*args, **kwargs):
        kwargs["stdout"].write("partial")
        started.set()
        await asyncio.Event().wait()

    raw.run_process = AsyncMock(side_effect=stalled)
    original = await _provider(raw).get_sandbox(raw.name)
    process = await original.exec("command")
    await started.wait()
    fresh = _provider(raw)
    error = type("LookupError", (Exception,), {"status_code": status})()
    fresh._client.get_sandbox.side_effect = error
    try:
        if status == 404:
            await fresh.terminate_sandbox(raw.name)
            assert not original._processes
            with pytest.raises(asyncio.CancelledError):
                await process.wait()
            assert await process.stdout.read() == b"partial"
        else:
            with pytest.raises(type(error)):
                await fresh.terminate_sandbox(raw.name)
            assert not process._task.done()
    finally:
        process._task.cancel()
        await asyncio.gather(process._task, return_exceptions=True)
