"""Focused unit tests for the E2B VM provider (all E2B calls are mocked)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.providers.e2b.provider import E2BSandboxProvider
from agent_env.providers.e2b.sandbox import E2BSandbox
from agent_env.providers.sandbox import NetworkMode, NetworkPolicy


class _Resolver:
    def __init__(self, template: str = "agent-env-v1-2c-4096m"):
        self.resolve = AsyncMock(return_value=template)


class _RawSandbox:
    sandbox_id = "e2b-sandbox-1"

    def __init__(self):
        self.kill = AsyncMock(return_value=True)

    def get_host(self, port: int) -> str:
        return f"{port}-{self.sandbox_id}.e2b.app"


class _AsyncSandboxSdk:
    create = AsyncMock(side_effect=lambda **_kwargs: _RawSandbox())
    connect = AsyncMock(side_effect=lambda *_args, **_kwargs: _RawSandbox())


@pytest.fixture(autouse=True)
def reset_sdk_mock_calls():
    _AsyncSandboxSdk.create.reset_mock()
    _AsyncSandboxSdk.connect.reset_mock()


@pytest.fixture
def sandbox_setup(monkeypatch):
    setup = AsyncMock()
    monkeypatch.setattr(E2BSandbox, "setup_vm_for_gateway", setup)
    return setup


@pytest.mark.asyncio
async def test_create_vm_uses_derived_template_and_preserves_attribution(
    sandbox_setup, caplog
):
    resolver = _Resolver()
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        template_resolver=resolver,
        sandbox_cls=_AsyncSandboxSdk,
    )

    sandbox = await provider.create_vm(
        cpu=2.0,
        memory=4096,
        disk_size_gb=99,
        timeout=123,
        exposed_ports=[8080, 9000],
        attribution={
            "product": "product-a",
            "customer": "customer-b",
            "team": "team-c",
            "project_id": "project-d",
        },
    )

    assert isinstance(sandbox, E2BSandbox)
    assert "Ignoring disk_size_gb=99" in caplog.text
    resolver.resolve.assert_awaited_once_with("agent-env-v1", cpu=2.0, memory_mb=4096)
    _AsyncSandboxSdk.create.assert_awaited_once_with(
        template="agent-env-v1-2c-4096m",
        timeout=123,
        api_key="e2b-secret",
        metadata={
            "product": "product-a",
            "customer": "customer-b",
            "team": "team-c",
            "project_id": "project-d",
            "agent_env_exposed_ports": "8080,9000",
        },
        network={"allow_public_traffic": True},
    )
    sandbox_setup.assert_awaited_once_with([8080, 9000])


@pytest.mark.asyncio
async def test_create_sandbox_provisions_a_vm_without_forwarding_image_or_env(
    sandbox_setup,
):
    resolver = _Resolver()
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        template_resolver=resolver,
        sandbox_cls=_AsyncSandboxSdk,
    )

    await provider.create_sandbox(
        image_name="registry.example/ignored:tag",
        port=8080,
        env={"IGNORED": "because-the-caller-loads-the-VM"},
        cpu=1.0,
        memory=2048,
        timeout=60,
    )

    resolver.resolve.assert_awaited_once_with("agent-env-v1", cpu=1.0, memory_mb=2048)
    assert (
        _AsyncSandboxSdk.create.await_args.kwargs["template"] == "agent-env-v1-2c-4096m"
    )
    assert "env" not in _AsyncSandboxSdk.create.await_args.kwargs
    assert "disk_size_gb" not in _AsyncSandboxSdk.create.await_args.kwargs
    sandbox_setup.assert_awaited_once_with([8080])


@pytest.mark.asyncio
async def test_reconnect_uses_configured_api_key(monkeypatch):
    raw = _RawSandbox()
    raw.get_info = AsyncMock(
        return_value={
            "metadata": {"agent_env_exposed_ports": "8080,9000"},
            "network": {
                "allowOut": ["only-this.example.com", "10.0.0.0/8"],
                "denyOut": ["0.0.0.0/0"],
            },
        }
    )
    reconnected = E2BSandbox(raw)
    reconnect = AsyncMock(return_value=reconnected)
    monkeypatch.setattr(E2BSandbox, "reconnect", reconnect)
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        sandbox_cls=_AsyncSandboxSdk,
    )

    result = await provider.get_sandbox("e2b-sandbox-1")

    assert result is reconnected
    reconnect.assert_awaited_once_with(
        "e2b-sandbox-1",
        api_key="e2b-secret",
        sandbox_cls=_AsyncSandboxSdk,
    )
    raw.get_info.assert_awaited_once_with()
    assert result.network_policy == NetworkPolicy(
        mode=NetworkMode.ALLOWLIST,
        allow_hosts=("only-this.example.com",),
        allow_cidrs=("10.0.0.0/8",),
    )
    assert result.tunnel_urls.get(8080) == "https://8080-e2b-sandbox-1.e2b.app"
    assert result.tunnel_urls.get(9000) == "https://9000-e2b-sandbox-1.e2b.app"


@pytest.mark.asyncio
async def test_reconnect_treats_missing_network_info_as_unknown(monkeypatch):
    raw = _RawSandbox()
    raw.get_info = AsyncMock(return_value={"metadata": {}})
    reconnect = AsyncMock(return_value=E2BSandbox(raw))
    monkeypatch.setattr(E2BSandbox, "reconnect", reconnect)
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        sandbox_cls=_AsyncSandboxSdk,
    )

    result = await provider.get_sandbox("e2b-sandbox-1")

    assert result.network_policy is None


@pytest.mark.asyncio
async def test_reconnect_with_unknown_network_policy_can_still_be_terminated(
    monkeypatch,
):
    raw = _RawSandbox()
    raw.get_info = AsyncMock(side_effect=RuntimeError("E2B API unavailable"))
    reconnect = AsyncMock(return_value=E2BSandbox(raw))
    monkeypatch.setattr(E2BSandbox, "reconnect", reconnect)
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        sandbox_cls=_AsyncSandboxSdk,
    )

    result = await provider.get_sandbox("e2b-sandbox-1")

    assert result.network_policy is None
    await result.terminate()
    raw.kill.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_reconnect_marks_unrepresentable_restrictive_policy_unknown(monkeypatch):
    raw = _RawSandbox()
    raw.get_info = AsyncMock(return_value={"allowInternetAccess": False, "network": {}})
    reconnect = AsyncMock(return_value=E2BSandbox(raw))
    monkeypatch.setattr(E2BSandbox, "reconnect", reconnect)
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        sandbox_cls=_AsyncSandboxSdk,
    )

    result = await provider.get_sandbox("e2b-sandbox-1")

    assert result.network_policy is None


@pytest.mark.asyncio
async def test_reconnect_marks_noncanonical_deny_rule_unknown(monkeypatch):
    raw = _RawSandbox()
    raw.get_info = AsyncMock(
        return_value={
            "network": {
                "allowOut": ["only-this.example.com"],
                "denyOut": ["10.0.0.0/8"],
            }
        }
    )
    reconnect = AsyncMock(return_value=E2BSandbox(raw))
    monkeypatch.setattr(E2BSandbox, "reconnect", reconnect)
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        sandbox_cls=_AsyncSandboxSdk,
    )

    result = await provider.get_sandbox("e2b-sandbox-1")

    assert result.network_policy is None


@pytest.mark.asyncio
async def test_rejects_image_override_before_resolving_or_creating():
    resolver = _Resolver()
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        template_resolver=resolver,
        sandbox_cls=_AsyncSandboxSdk,
    )

    with pytest.raises(ValueError, match="base_template is immutable"):
        await provider.create_vm(image="untrusted:latest")

    resolver.resolve.assert_not_awaited()
    _AsyncSandboxSdk.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_forwards_network_allowlist_to_e2b(sandbox_setup):
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        template_resolver=_Resolver(),
        sandbox_cls=_AsyncSandboxSdk,
    )

    await provider.create_vm(
        setup_for_gateway=False,
        network_policy=NetworkPolicy(
            mode=NetworkMode.ALLOWLIST,
            allow_hosts=("only-this.example.com",),
            allow_cidrs=("10.0.0.0/8",),
        ),
    )

    allow_out = _AsyncSandboxSdk.create.await_args.kwargs["network"]["allow_out"]
    assert "only-this.example.com" in allow_out
    assert "10.0.0.0/8" in allow_out
    assert _AsyncSandboxSdk.create.await_args.kwargs["network"]["deny_out"] == [
        "0.0.0.0/0"
    ]
    assert (
        _AsyncSandboxSdk.create.await_args.kwargs["network"]["allow_public_traffic"]
        is True
    )
    sandbox_setup.assert_not_awaited()


@pytest.mark.asyncio
async def test_allow_all_still_explicitly_enables_public_ingress(sandbox_setup):
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        template_resolver=_Resolver(),
        sandbox_cls=_AsyncSandboxSdk,
    )

    await provider.create_vm(setup_for_gateway=False)

    assert _AsyncSandboxSdk.create.await_args.kwargs["network"] == {
        "allow_public_traffic": True,
    }


@pytest.mark.asyncio
async def test_adapter_construction_failure_kills_raw_sandbox():
    raw = _RawSandbox()
    raw.get_host = MagicMock(side_effect=RuntimeError("host lookup failed"))
    sdk = MagicMock()
    sdk.create = AsyncMock(return_value=raw)
    provider = E2BSandboxProvider(
        api_key="e2b-secret",
        base_template="agent-env-v1",
        template_resolver=_Resolver(),
        sandbox_cls=sdk,
    )

    with pytest.raises(RuntimeError, match="host lookup failed"):
        await provider.create_vm(exposed_ports=[8080], setup_for_gateway=False)

    raw.kill.assert_awaited_once_with()
