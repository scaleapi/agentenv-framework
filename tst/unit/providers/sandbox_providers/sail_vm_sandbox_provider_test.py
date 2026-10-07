"""Unit tests for the Sail Sailbox provider; the SDK is a fake, never imported."""

import asyncio
import logging
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.sail_vm import _sdk
from agent_env.providers.sandbox_providers.sail_vm import provider as provider_module
from agent_env.providers.sandbox_providers.sail_vm.provider import (
    SANDBOX_STARTED_EVENT,
    SailVmSandboxProvider,
    sailbox_name,
    sailbox_shape,
)
from agent_env.providers.sandbox_providers.sail_vm.sandbox import SailVmSandbox
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError


class _SdkError(Exception):
    pass


def _listener(port, url=True):
    endpoint = SimpleNamespace(url=f"https://sb-1-{port}.sail.box") if url else None
    return SimpleNamespace(guest_port=port, endpoint=endpoint)


def _sailbox(ports=(), status="running"):
    sailbox = MagicMock(sailbox_id="sb_1", status=status, error_message="no capacity")
    sailbox.listeners.aio = AsyncMock(return_value=[_listener(port) for port in ports])
    sailbox.terminate.aio = AsyncMock()
    sailbox.egress_policy = SimpleNamespace(policy_id=None, document={})
    return sailbox


def _fake_sdk(sailbox):
    sdk = SimpleNamespace(
        App=SimpleNamespace(find=MagicMock(return_value=SimpleNamespace(id="app_1"))),
        Sailbox=SimpleNamespace(
            create=SimpleNamespace(aio=AsyncMock(return_value=sailbox)),
            get=SimpleNamespace(aio=AsyncMock(return_value=sailbox)),
        ),
        Image=SimpleNamespace(devbox=MagicMock(return_value="devbox-amd64")),
        AutoSleep=SimpleNamespace(
            never=lambda: "never", default=lambda: "default", not_before=lambda seconds: f"not_before:{seconds}"
        ),
        reset_transports=MagicMock(),
        NotFoundError=_SdkError,
        SailboxHostLostError=_SdkError,
        TransportError=_SdkError,
    )
    return sdk


@pytest.fixture(autouse=True)
def fresh_key_state(monkeypatch):
    monkeypatch.setattr(_sdk, "_installed_key", None)
    monkeypatch.setattr(_sdk, "_apps", {})
    monkeypatch.delenv(_sdk.API_KEY_ENV, raising=False)
    monkeypatch.delenv(_sdk.RUNTIME_THREADS_ENV, raising=False)


@pytest.fixture
def setup(monkeypatch):
    setup = AsyncMock()
    monkeypatch.setattr(SailVmSandbox, "setup_vm_for_gateway", setup)
    return setup


def _provider(sdk, **config):
    return SailVmSandboxProvider(api_key="sail-secret", sdk=sdk, **config)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "requires a non-empty 'api_key'"),
        ({"api_key": "  "}, "requires a non-empty 'api_key'"),
        ({"api_key": "k", "app": ""}, "'app' must be a non-empty string"),
        ({"api_key": "k", "min_size": "xl"}, "'min_size' must be one of"),
        ({"api_key": "k", "auto_sleep": "yes"}, "'auto_sleep' must be true or false"),
        ({"api_key": "k", "auto_sleep_min_idle_seconds": 0}, "from 1 to 3600"),
        ({"api_key": "k", "auto_sleep_min_idle_seconds": True}, "from 1 to 3600"),
        ({"api_key": "k", "runtime_threads": 257}, "from 1 to 256"),
        ({"api_key": "k", "region": "us"}, "unknown key"),
    ],
)
def test_from_config_rejects_invalid_config(config, message):
    with pytest.raises(ConfigError, match=message):
        SailVmSandboxProvider.from_config(**config)


def test_construction_neither_imports_the_sdk_nor_sets_the_key(monkeypatch):
    monkeypatch.delitem(sys.modules, "sail", raising=False)
    SailVmSandboxProvider.from_config(api_key="sail-secret")
    assert "sail" not in sys.modules
    assert _sdk.API_KEY_ENV not in os.environ


@pytest.mark.parametrize(
    ("cpu", "memory", "disk", "min_size", "shape"),
    [
        (1.0, 8192, 10, "s", ("s", 8, 10)),
        (0.5, 1024, 1, "s", ("s", 2, 8)),
        (2.0, 4096, 10, "s", ("m", 8, 32)),
        (1.0, 100 * 1024, 10, "s", ("m", 100, 32)),
        (1.0, 2048, 600, "s", ("l", 16, 600)),
        (1.0, 1500, 10.2, "m", ("m", 8, 32)),
    ],
)
def test_shape_is_the_smallest_covering_size_with_ceilings_rounded_up(cpu, memory, disk, min_size, shape):
    assert sailbox_shape(cpu, memory, disk, min_size=min_size) == shape


@pytest.mark.parametrize(("cpu", "memory", "disk"), [(9, 8192, 10), (1, 300 * 1024, 10), (1, 8192, 2000)])
def test_a_request_no_size_fits_is_refused(cpu, memory, disk):
    with pytest.raises(ValueError, match="no Sailbox size fits"):
        sailbox_shape(cpu, memory, disk)


def test_name_carries_slugged_attribution_in_key_order_within_128_chars():
    name = sailbox_name({"run_id": "inst-9f3c", "project_id": "p/123", "team": "env pod"})
    prefix, random, *rest = name.split("-", 2)
    assert (prefix, len(random)) == ("ae", 8)
    assert name.endswith("p-123-inst-9f3c-env-pod")
    assert len(sailbox_name({"k": "x" * 300})) == 128


@pytest.mark.asyncio
async def test_create_vm_sends_the_shape_lifetime_ports_and_policy(setup):
    sailbox = _sailbox(ports=[8080, 9000])
    sdk = _fake_sdk(sailbox)

    sandbox = await _provider(sdk).create_vm(
        cpu=2.0, memory=4096, disk_size_gb=20, timeout=900, exposed_ports=[8080, 9000, 8080],
        attribution={"run_id": "inst-1"},
    )

    kwargs = sdk.Sailbox.create.aio.await_args.kwargs
    assert kwargs["app"].id == "app_1"
    assert kwargs["image"] == "devbox-amd64"
    sdk.Image.devbox.assert_called_once_with("amd64")
    assert kwargs["name"].startswith("ae-") and kwargs["name"].endswith("-inst-1")
    assert {k: kwargs[k] for k in ("size", "memory_limit_gib", "disk_limit_gib", "max_lifetime_seconds")} == {
        "size": "m", "memory_limit_gib": 8, "disk_limit_gib": 32, "max_lifetime_seconds": 900,
    }
    assert kwargs["ingress_ports"] == [8080, 9000]
    assert kwargs["auto_sleep"] == "never"
    assert kwargs["egress_policy"] == {}
    assert "api_key" not in kwargs and "env" not in kwargs
    assert isinstance(sandbox, SailVmSandbox)
    assert (sandbox.type, sandbox.mode, sandbox.sandbox_id) == ("sail_vm", "vm", "sb_1")
    assert sandbox.tunnel_urls == {8080: "https://sb-1-8080.sail.box", 9000: "https://sb-1-9000.sail.box"}
    assert sandbox.network_policy == NetworkPolicy()
    setup.assert_awaited_once_with([8080, 9000])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "expected"),
    [({}, "never"), ({"auto_sleep": True}, "default"), ({"auto_sleep_min_idle_seconds": 600}, "not_before:600")],
)
async def test_auto_sleep_is_off_unless_configured(setup, config, expected):
    sdk = _fake_sdk(_sailbox())
    await _provider(sdk, **config).create_vm(exposed_ports=[])
    assert sdk.Sailbox.create.aio.await_args.kwargs["auto_sleep"] == expected


@pytest.mark.asyncio
async def test_allowlist_becomes_a_sail_allowlist_with_the_platform_floor(setup):
    sdk = _fake_sdk(_sailbox())
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("10.0.0.0/8",))

    sandbox = await _provider(sdk).create_vm(exposed_ports=[], network_policy=policy)

    entries = sdk.Sailbox.create.aio.await_args.kwargs["egress_policy"]["allowlist"]
    assert entries[0] == "pypi.org" and entries[-1] == "10.0.0.0/8"
    assert {"*.sail.box", "*.e2b.app"} <= set(entries)
    assert sandbox.network_policy.allow_cidrs == ("10.0.0.0/8",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy",
    [
        NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_cidrs=("2001:db8::/32",)),
        NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=tuple(f"h{i}.example" for i in range(130))),
    ],
)
async def test_an_unenforceable_policy_is_refused_before_provisioning(policy):
    sdk = _fake_sdk(_sailbox())
    assert SailVmSandboxProvider.supports_network_policy(policy) is False
    with pytest.raises(NetworkPolicyUnsupportedError):
        await _provider(sdk).create_vm(exposed_ports=[], network_policy=policy)
    sdk.Sailbox.create.aio.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_sailbox_is_terminated_and_reported():
    sailbox = _sailbox(status="failed")
    with pytest.raises(RuntimeError, match="failed to start: no capacity"):
        await _provider(_fake_sdk(sailbox)).create_vm(exposed_ports=[])
    sailbox.terminate.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_setup_failure_terminates_the_sailbox(setup):
    sailbox = _sailbox()
    setup.side_effect = RuntimeError("docker never came up")
    with pytest.raises(RuntimeError, match="docker never came up"):
        await _provider(_fake_sdk(sailbox)).create_vm(exposed_ports=[])
    sailbox.terminate.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_tunnel_urls_wait_until_every_port_is_routed(setup, monkeypatch):
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sail_vm.provider._LISTENER_POLL_INTERVAL", 0)
    sailbox = _sailbox()
    sailbox.listeners.aio = AsyncMock(side_effect=[[_listener(8080, url=False)], [_listener(8080)]])
    sandbox = await _provider(_fake_sdk(sailbox)).create_vm(exposed_ports=[8080])
    assert sandbox.tunnel_urls == {8080: "https://sb-1-8080.sail.box"}


@pytest.mark.asyncio
async def test_a_port_that_never_routes_fails_the_create(setup, monkeypatch):
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sail_vm.provider._LISTENER_POLL_INTERVAL", 0)
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sail_vm.provider._LISTENER_TIMEOUT", 0)
    sailbox = _sailbox()
    with pytest.raises(RuntimeError, match=r"no public URL for port\(s\) \[8080\]"):
        await _provider(_fake_sdk(sailbox)).create_vm(exposed_ports=[8080])
    sailbox.terminate.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_cancelled_create_terminates_the_sailbox_it_produces():
    sailbox = _sailbox()
    sdk = _fake_sdk(sailbox)
    created = asyncio.Event()

    async def slow_create(**_kwargs):
        await created.wait()
        return sailbox

    sdk.Sailbox.create.aio = slow_create
    caller = asyncio.ensure_future(_provider(sdk).create_vm(exposed_ports=[]))
    await asyncio.sleep(0.01)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    created.set()
    for _ in range(5):
        await asyncio.sleep(0)
    sailbox.terminate.aio.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failing_orphan_termination_is_retried_then_reported(monkeypatch, caplog):
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sail_vm.provider.asyncio.sleep", AsyncMock())
    sandbox = MagicMock(sandbox_id="sb_1")
    sandbox.terminate = AsyncMock(side_effect=RuntimeError("api down"))

    await provider_module._reap(sandbox)

    assert sandbox.terminate.await_count == provider_module._REAP_ATTEMPTS
    assert "Orphaned Sailbox sb_1 is still running" in caplog.text


@pytest.mark.asyncio
async def test_create_logs_the_attribution_join_event_without_the_key(setup, caplog):
    caplog.set_level(logging.INFO, logger="agent_env.providers.sandbox_providers.sail_vm.provider")
    await _provider(_fake_sdk(_sailbox())).create_vm(exposed_ports=[], attribution={"run_id": "inst-1", "team": "t"})

    (record,) = [r for r in caplog.records if getattr(r, "event", None) == SANDBOX_STARTED_EVENT]
    assert record.sail_sailbox_id == "sb_1"
    assert record.sail_app_name == "agent-env"
    assert record.sail_attribution == {"run_id": "inst-1", "team": "t"}
    assert record.run_id == "inst-1"
    assert "sail-secret" not in caplog.text


@pytest.mark.asyncio
async def test_create_sandbox_is_a_bare_vm_that_ignores_image_and_env(setup):
    sdk = _fake_sdk(_sailbox(ports=[8080]))
    await _provider(sdk).create_sandbox(image_name="registry.example/agent:1", port=8080, env={"TOKEN": "workload-token-value"})
    kwargs = sdk.Sailbox.create.aio.await_args.kwargs
    assert kwargs["ingress_ports"] == [8080]
    assert "workload-token-value" not in repr(kwargs)


@pytest.mark.asyncio
async def test_image_overrides_are_refused():
    with pytest.raises(ValueError, match="image overrides are unsupported"):
        await _provider(_fake_sdk(_sailbox())).create_vm(image="ubuntu:22.04")


@pytest.mark.asyncio
async def test_create_container_removes_the_registry_login_from_the_vm(monkeypatch):
    sandbox = MagicMock(spec=SailVmSandbox)
    sandbox.exec_script = AsyncMock()
    monkeypatch.setattr(
        "agent_env.providers.sandbox_providers.sandbox_provider.SandboxProvider.create_container",
        AsyncMock(return_value=sandbox),
    )
    result = await _provider(_fake_sdk(_sailbox())).create_container(image_name="r/i:1", port=80, env={})
    assert result is sandbox
    sandbox.exec_script.assert_awaited_once_with("rm -f /root/.docker/config.json")


@pytest.mark.asyncio
async def test_get_sandbox_restores_ports_and_the_applied_policy():
    sailbox = _sailbox(ports=[8080])
    sailbox.egress_policy = SimpleNamespace(policy_id=None, document={"allowlist": ["pypi.org", "10.0.0.0/8"]})
    sdk = _fake_sdk(sailbox)

    sandbox = await _provider(sdk).get_sandbox("sb_1")

    sdk.Sailbox.get.aio.assert_awaited_once_with("sb_1")
    assert sandbox.tunnel_urls == {8080: "https://sb-1-8080.sail.box"}
    assert sandbox.network_policy == NetworkPolicy(
        mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("10.0.0.0/8",)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "applied",
    [
        None,
        SimpleNamespace(policy_id="ep_1", document={}),
        SimpleNamespace(policy_id=None, document={"allowlist": ["a.example"], "rules": []}),
    ],
)
async def test_get_sandbox_leaves_an_unrepresentable_policy_unknown(applied):
    sailbox = _sailbox()
    sailbox.egress_policy = applied
    sandbox = await _provider(_fake_sdk(sailbox)).get_sandbox("sb_1")
    assert sandbox.network_policy is None


def test_the_key_is_set_only_while_the_sdk_builds_its_client():
    sdk = _fake_sdk(_sailbox())
    seen = {}

    def find(**kwargs):
        seen["key"] = os.environ.get(_sdk.API_KEY_ENV)
        seen["threads"] = os.environ.get(_sdk.RUNTIME_THREADS_ENV)
        return SimpleNamespace(id="app_1")

    sdk.App.find = MagicMock(side_effect=find)
    _sdk.connect("sail-secret", "agent-env", runtime_threads=16, sdk=sdk)

    assert seen == {"key": "sail-secret", "threads": "16"}
    assert _sdk.API_KEY_ENV not in os.environ
    assert _sdk.RUNTIME_THREADS_ENV not in os.environ
    sdk.reset_transports.assert_called_once()


def test_operator_settings_are_overridden_for_the_build_and_then_restored(monkeypatch):
    monkeypatch.setenv(_sdk.API_KEY_ENV, "operator-key")
    monkeypatch.setenv(_sdk.RUNTIME_THREADS_ENV, "4")
    sdk = _fake_sdk(_sailbox())
    sdk.App.find = MagicMock(side_effect=lambda **_: (os.environ[_sdk.API_KEY_ENV], os.environ[_sdk.RUNTIME_THREADS_ENV]))

    _, app = _sdk.connect("sail-secret", "agent-env", runtime_threads=16, sdk=sdk)

    assert app == ("sail-secret", "16")
    assert (os.environ[_sdk.API_KEY_ENV], os.environ[_sdk.RUNTIME_THREADS_ENV]) == ("operator-key", "4")


def test_the_key_is_installed_once_per_process_and_apps_are_cached():
    sdk = _fake_sdk(_sailbox())
    for _ in range(3):
        _sdk.connect("sail-secret", "agent-env", sdk=sdk)
    _sdk.connect("sail-secret", "other-app", sdk=sdk)

    sdk.reset_transports.assert_called_once()
    assert [c.kwargs["name"] for c in sdk.App.find.call_args_list] == ["agent-env", "other-app"]


def test_a_second_key_in_the_same_process_is_refused():
    sdk = _fake_sdk(_sailbox())
    _sdk.connect("sail-secret", "agent-env", sdk=sdk)
    with pytest.raises(ConfigError, match="one Sail API key"):
        _sdk.connect("another-key", "agent-env", sdk=sdk)


def test_a_rejected_key_can_be_retried():
    sdk = _fake_sdk(_sailbox())
    sdk.App.find = MagicMock(side_effect=[PermissionError("Invalid API key"), SimpleNamespace(id="app_1")])
    with pytest.raises(PermissionError):
        _sdk.connect("sail-secret", "agent-env", sdk=sdk)
    assert _sdk.API_KEY_ENV not in os.environ
    _sdk.connect("sail-secret", "agent-env", sdk=sdk)
    assert sdk.reset_transports.call_count == 2
