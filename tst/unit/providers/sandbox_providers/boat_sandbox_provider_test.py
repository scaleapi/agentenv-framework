"""Offline unit tests for the boat sandbox provider: config, sizing, labels, create, cleanup, reconnect and terminate."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest
import urllib3

from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.boat.config import (
    ATTRIBUTION_LABEL_PREFIX,
    CREATE_LABEL,
    DEFAULT_API_URL,
    LABEL_VALUE_MAX_BYTES,
    MANAGED_LABEL,
    MAX_LABELS,
    PORTS_LABEL,
    parse_settings,
    ports_from_labels,
    sandbox_labels,
    sandbox_name,
    sandbox_type,
)
from agent_env.providers.sandbox_providers.boat.provider import BoatSandboxProvider
from agent_env.providers.sandbox_providers.boat.sandbox import BoatSandbox
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import (
    _BUILTIN_SANDBOX_PROVIDERS,
    SandboxNotFoundError,
)

from tst.unit.providers.sandbox_providers.boat_fakes import SETTINGS, FakeBoatApi, FakeCommands, api_error


@pytest.fixture(autouse=True)
def no_waits(monkeypatch):
    for name in ("_RETRY_BASE_DELAY", "_DELETE_DELAY", "_READY_POLL_INTERVAL"):
        monkeypatch.setattr(BoatSandboxProvider, name, 0.0)


def provider(api: FakeBoatApi, commands: FakeCommands | None = None) -> BoatSandboxProvider:
    return BoatSandboxProvider(SETTINGS, api_factory=lambda settings: api, run_command=commands or FakeCommands())


def docker_ready(argv: list[str]) -> tuple[int, str, str]:
    return (0, "Docker Compose version v5.3.1\n", "") if argv == ["bash", "-c", BoatSandbox.docker_ready_script()] else (0, "", "")


# --- config ---


def test_settings_need_an_api_key_and_take_no_unknown_keys():
    with pytest.raises(ConfigError, match="requires a non-empty 'api_key'"):
        parse_settings({})
    with pytest.raises(ConfigError, match="unknown key"):
        parse_settings({"api_key": "boat_x", "region": "eu"})
    settings = parse_settings({"api_key": " boat_x ", "api_url": "https://boat.example/api/v1/"})
    assert (settings.api_key, settings.api_url) == ("boat_x", "https://boat.example/api/v1")
    assert parse_settings({"api_key": "boat_x"}).api_url == DEFAULT_API_URL
    assert "boat_x" not in repr(settings)


def test_from_config_builds_a_provider_without_showing_the_key():
    built = BoatSandboxProvider.from_config(api_key="boat_secret")
    assert "boat_secret" not in repr(built)
    assert _BUILTIN_SANDBOX_PROVIDERS["boat"].endswith("boat.provider:BoatSandboxProvider")


@pytest.mark.parametrize("cpu, memory_mb, disk_gb, expected", [
    (1, 2048, 10, "small"),
    (2, 4096, 51, "small"),
    (1, 8192, 10, "default"),
    (4, 4096, 10, "default"),
    (2, 4096, 60, "large"),
    (8, 16384, 100, "large"),
    (16, 32768, 251, "xlarge"),
])
def test_a_request_gets_the_smallest_type_that_covers_it(cpu, memory_mb, disk_gb, expected):
    assert sandbox_type(cpu=cpu, memory_mb=memory_mb, disk_gb=disk_gb) == expected


def test_a_request_bigger_than_the_largest_type_is_refused():
    with pytest.raises(ValueError, match="largest type, 'xlarge'"):
        sandbox_type(cpu=17, memory_mb=1024, disk_gb=10)


def test_labels_carry_ports_the_create_and_attribution_within_boats_limits():
    labels = sandbox_labels({"project_id": "p 1", "a/b": "x" * 300, "skip": None}, [8000, 8080], "tok")
    assert labels[MANAGED_LABEL] == "true" and labels[CREATE_LABEL] == "tok"
    assert ports_from_labels(labels) == [8000, 8080]
    assert labels[f"{ATTRIBUTION_LABEL_PREFIX}project_id"] == "p 1"
    assert len(labels[f"{ATTRIBUTION_LABEL_PREFIX}a/b"].encode()) == LABEL_VALUE_MAX_BYTES
    assert f"{ATTRIBUTION_LABEL_PREFIX}skip" not in labels


def test_attribution_past_boats_label_limit_is_logged_and_left_off(caplog):
    attribution = {f"key{n}": str(n) for n in range(MAX_LABELS)}
    with caplog.at_level(logging.WARNING):
        labels = sandbox_labels(attribution, [], "tok")
    assert len(labels) == MAX_LABELS
    assert "at most 20 labels" in caplog.text


def test_names_are_unique_and_name_the_attribution():
    first, second = sandbox_name({"run_id": "r1"}), sandbox_name({"run_id": "r1"})
    assert first != second and first.startswith("ae-") and first.endswith("-r1")


# --- create ---


@pytest.mark.asyncio
async def test_create_vm_sends_an_isolated_sandbox_and_returns_its_hosted_urls():
    api = FakeBoatApi()
    commands = FakeCommands(handler=docker_ready)
    sandbox = await provider(api, commands).create_vm(cpu=1, memory=8192, exposed_ports=[8000, 8000, 9000],
                                                      timeout=1800, attribution={"run_id": "r1"})
    request = api.creates[0]["request"]
    assert (request.type, request.ttl_seconds, request.no_env, request.snapshots) == ("default", 1800, True, False)
    assert [(port.port, port.public) for port in request.hosted_ports] == [(8000, True), (9000, True)]
    assert request.labels[PORTS_LABEL] == "8000,9000" and request.labels[f"{ATTRIBUTION_LABEL_PREFIX}run_id"] == "r1"
    assert isinstance(sandbox, BoatSandbox) and (sandbox.type, sandbox.mode) == ("boat", "vm")
    assert sandbox.tunnel_urls == {8000: api.url(sandbox.sandbox_id, 8000), 9000: api.url(sandbox.sandbox_id, 9000)}
    assert sandbox.network_policy == NetworkPolicy()
    assert commands.commands == [["bash", "-c", BoatSandbox.docker_ready_script()]]
    assert api.hosts == []


@pytest.mark.asyncio
async def test_a_port_the_create_could_not_host_is_hosted_again():
    api = FakeBoatApi(hosted_port_errors={8000})
    sandbox = await provider(api).create_vm(exposed_ports=[8000], setup_for_gateway=False)
    assert api.hosts == [(sandbox.sandbox_id, 8000, True)]
    assert sandbox.tunnel_urls == {8000: api.url(sandbox.sandbox_id, 8000)}


@pytest.mark.asyncio
async def test_create_sandbox_is_a_vm_with_the_agents_port():
    api = FakeBoatApi()
    sandbox = await provider(api, FakeCommands(handler=docker_ready)).create_sandbox(image_name="agent:1", port=8000,
                                                                                     env={"A": "1"})
    assert list(sandbox.tunnel_urls) == [8000]


@pytest.mark.asyncio
async def test_an_egress_policy_or_an_image_is_refused_before_anything_is_made(caplog):
    api = FakeBoatApi()
    with pytest.raises(NetworkPolicyUnsupportedError):
        await provider(api).create_vm(network_policy=NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("x.org",)))
    with pytest.raises(ValueError, match="image overrides"):
        await provider(api).create_vm(image="ubuntu:24.04")
    assert api.creates == []
    assert not BoatSandboxProvider.supports_network_policy(NetworkPolicy(mode=NetworkMode.ALLOWLIST))
    with caplog.at_level(logging.WARNING):
        await provider(api).create_vm(boot_mode="uefi", setup_for_gateway=False)
    assert "Ignoring boot_mode=uefi" in caplog.text


@pytest.mark.asyncio
async def test_a_refused_create_retries_with_a_new_key_and_an_unanswered_one_with_the_same():
    api = FakeBoatApi(create_errors=[api_error(429, "rate_limited"), api_error(500, "http_500")])
    sandbox = await provider(api).create_vm(setup_for_gateway=False)
    keys = [create["key"] for create in api.creates]
    assert keys[0] != keys[1] and keys[1] == keys[2]
    # The 500 stands for a lost answer: its sandbox was made, and the retry's answer is another in this fake.
    assert sandbox.sandbox_id in api.sandboxes_by_id


@pytest.mark.asyncio
async def test_a_request_boat_rejects_raises_at_once():
    api = FakeBoatApi(create_errors=[api_error(403, "trial_machine_class_not_allowed")])
    with pytest.raises(Exception, match="trial_machine_class_not_allowed"):
        await provider(api).create_vm(cpu=8, memory=16384, setup_for_gateway=False)
    assert len(api.creates) == 1 and api.label_queries == []


@pytest.mark.asyncio
async def test_a_create_that_never_answers_deletes_what_it_may_have_made(monkeypatch):
    monkeypatch.setattr(BoatSandboxProvider, "_CREATE_ATTEMPTS", 2)
    api = FakeBoatApi(create_errors=[urllib3.exceptions.ProtocolError("reset")] * 2)
    with pytest.raises(urllib3.exceptions.ProtocolError):
        await provider(api).create_vm(setup_for_gateway=False)
    token = api.creates[0]["request"].labels[CREATE_LABEL]
    assert api.label_queries == [[f"{CREATE_LABEL}={token}"]]
    assert api.sandboxes_by_id == {} and len(api.deletes) == 2


@pytest.mark.asyncio
async def test_a_create_cancelled_while_it_waits_to_retry_deletes_what_the_unanswered_attempt_made(monkeypatch):
    monkeypatch.setattr(BoatSandboxProvider, "_RETRY_BASE_DELAY", 30.0)
    api = FakeBoatApi(create_errors=[api_error(500, "http_500")])
    task = asyncio.create_task(provider(api).create_vm(setup_for_gateway=False))
    for _ in range(100):
        if api.sandboxes_by_id:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    token = api.creates[0]["request"].labels[CREATE_LABEL]
    assert api.label_queries == [[f"{CREATE_LABEL}={token}"]] and api.sandboxes_by_id == {}


@pytest.mark.asyncio
async def test_a_cancelled_create_deletes_the_sandbox_once_boat_answers():
    api = FakeBoatApi(create_gate=threading.Event())
    task = asyncio.create_task(provider(api).create_vm(setup_for_gateway=False))
    assert await asyncio.to_thread(api.create_entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    api.create_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert api.sandboxes_by_id == {} and len(api.deletes) == 1


@pytest.mark.asyncio
async def test_a_sandbox_that_fails_setup_is_deleted_before_the_error_is_raised():
    api = FakeBoatApi()
    commands = FakeCommands(handler=lambda argv: (1, "", "docker daemon did not become ready\n"))
    with pytest.raises(RuntimeError, match="docker daemon did not become ready"):
        await provider(api, commands).create_vm(exposed_ports=[8000])
    assert api.sandboxes_by_id == {} and len(api.deletes) == 1


@pytest.mark.asyncio
async def test_a_sandbox_that_fails_to_start_is_deleted():
    api = FakeBoatApi(states=["provisioning", "error"])
    with pytest.raises(RuntimeError, match="state 'error'"):
        await provider(api).create_vm(setup_for_gateway=False)
    assert api.sandboxes_by_id == {}


# --- reconnect and terminate ---


@pytest.mark.asyncio
async def test_a_fresh_provider_reconnects_with_the_ports_and_urls_the_labels_record():
    api = FakeBoatApi()
    made = await provider(api).create_vm(exposed_ports=[8000, 9000], setup_for_gateway=False)
    again = await provider(api).get_sandbox(made.sandbox_id)
    assert again.tunnel_urls == made.tunnel_urls and again.network_policy == NetworkPolicy()
    assert [host[1:] for host in api.hosts] == [(8000, True), (9000, True)]


@pytest.mark.asyncio
async def test_a_gone_or_stopped_sandbox_is_not_found():
    api = FakeBoatApi(states=["ready", "archived"])
    made = await provider(api).create_vm(setup_for_gateway=False)
    with pytest.raises(SandboxNotFoundError, match="stopped"):
        await provider(api).get_sandbox(made.sandbox_id)
    with pytest.raises(SandboxNotFoundError, match="not found"):
        await provider(api).get_sandbox("bx_missing")


@pytest.mark.asyncio
async def test_terminate_is_idempotent_and_terminate_sandbox_needs_no_handle():
    api = FakeBoatApi()
    made = await provider(api).create_vm(setup_for_gateway=False)
    await made.terminate()
    await made.terminate()
    await provider(api).terminate_sandbox(made.sandbox_id)
    assert api.deletes == [made.sandbox_id] * 3


@pytest.mark.asyncio
async def test_a_delete_that_keeps_failing_is_raised(monkeypatch):
    api = FakeBoatApi()
    made = await provider(api).create_vm(setup_for_gateway=False)

    def failing(**kwargs):
        raise api_error(500, "http_500")

    monkeypatch.setattr(api, "delete_sandbox", failing)
    with pytest.raises(RuntimeError, match="could not be deleted"):
        await provider(api).terminate_sandbox(made.sandbox_id)


@pytest.mark.asyncio
async def test_close_closes_the_sdk_clients_connections():
    api = FakeBoatApi()
    await provider(api).close()
    assert api.api_client.rest_client.pool_manager.cleared


@pytest.mark.asyncio
async def test_close_works_on_the_real_sdk_client():
    """The SDK's ApiClient has no close of its own; the provider clears its connection pool instead."""
    built = BoatSandboxProvider(SETTINGS)
    await built.close()
    await built.close()
