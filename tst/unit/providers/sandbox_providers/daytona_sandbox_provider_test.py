"""Offline unit tests for the Daytona sandbox provider: config, create, cleanup, reconnect, terminate and clients."""

from __future__ import annotations

import asyncio
import logging

import pytest
from daytona import (
    AsyncDaytona,
    CreateSandboxFromImageParams,
    CreateSandboxFromSnapshotParams,
    DaytonaAuthenticationError,
    DaytonaBadRequestError,
    DaytonaConflictError,
    DaytonaError,
    DaytonaForbiddenError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaTimeoutError,
)

from agent_env.config import reset_config
from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers import sandbox_provider as sandbox_provider_module
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.daytona.config import (
    DEFAULT_API_URL,
    IMAGE_CREATE_TIMEOUT,
    LABEL_VALUE_MAX,
    MANAGED_LABEL,
    PORTS_LABEL,
    SNAPSHOT_CREATE_TIMEOUT,
    parse_settings,
    sandbox_labels,
    sandbox_name,
    sandbox_resources,
    ttl_minutes,
)
from agent_env.providers.sandbox_providers.daytona.image import DEFAULT_BASE_IMAGE, HOST_IMAGE_VERSION
from agent_env.providers.sandbox_providers.daytona.provider import DaytonaSandboxProvider
from agent_env.providers.sandbox_providers.daytona.sandbox import DaytonaSandbox
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError, build_sandbox_provider

from tst.unit.providers.sandbox_providers.daytona_fakes import (
    PROXY_DOMAIN,
    FakeClients,
    FakeDaytona,
    FakeProcess,
    FakeRaw,
    FakeSnapshot,
)

CONFLICT = DaytonaConflictError("Sandbox state change in progress")
DAYTONA_HOSTS = ("*.daytonaproxy01.net", "*.daytonaproxy01.eu")


def allow(*hosts: str, cidrs: tuple[str, ...] = ()) -> NetworkPolicy:
    return NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=hosts, allow_cidrs=cidrs)


def _provider(fake: FakeDaytona | None = None, **config) -> tuple[DaytonaSandboxProvider, FakeDaytona]:
    fake = fake or FakeDaytona()
    provider = DaytonaSandboxProvider(parse_settings({"api_key": "k", **config}), client_factory=lambda settings: fake)
    return _fast(provider), fake


def _on(clients: FakeClients) -> DaytonaSandboxProvider:
    return _fast(DaytonaSandboxProvider(parse_settings({"api_key": "k"}), client_factory=clients))


def _fast(provider: DaytonaSandboxProvider) -> DaytonaSandboxProvider:
    provider._RATE_LIMIT_BASE_DELAY = 0
    provider._RECLAIM_DELAY = 0
    return provider


def _failing_setup() -> FakeProcess:
    return FakeProcess(handler=lambda argv: (1, "", "compose missing") if "compose" in argv else (0, "", ""))


async def _drain(turns: int = 50) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


@pytest.fixture
def no_delete_delay(monkeypatch):
    monkeypatch.setattr(DaytonaSandbox, "_DELETE_RETRY_DELAY", 0)


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    path = tmp_path / ".agentenv" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text('[sandbox.providers.daytona.config]\napi_key = "env:DAYTONA_TEST_FAKE_KEY"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(path))
    monkeypatch.setenv("DAYTONA_TEST_FAKE_KEY", "resolved-test-key")
    reset_config()
    sandbox_provider_module.reset_sandbox_provider()
    yield path
    reset_config()
    sandbox_provider_module.reset_sandbox_provider()


def test_only_an_api_key_is_required():
    settings = parse_settings({"api_key": "k"})
    assert (settings.api_url, settings.target, settings.snapshot, settings.base_image, settings.egress_hosts) == (
        DEFAULT_API_URL, None, None, DEFAULT_BASE_IMAGE, ())


def test_every_key_is_read():
    settings = parse_settings({
        "api_key": " k ", "api_url": "https://d.example/api", "target": "eu", "snapshot": "my-dind",
        "egress_hosts": [" *.internal.example "],
    })
    assert (settings.api_key, settings.api_url, settings.target, settings.snapshot, settings.egress_hosts) == (
        "k", "https://d.example/api", "eu", "my-dind", ("*.internal.example",))
    assert parse_settings({"api_key": "k", "base_image": "ubuntu:24.04"}).base_image == "ubuntu:24.04"


@pytest.mark.parametrize(("config", "message"), [
    ({}, "requires a non-empty 'api_key'"),
    ({"api_key": "  "}, "requires a non-empty 'api_key'"),
    ({"api_key": "k", "foo": 1}, "unknown key(s): ['foo']"),
    ({"api_key": "k", "target": 3}, "'target' must be a non-empty string"),
    ({"api_key": "k", "snapshot": "s", "base_image": "ubuntu:24.04"}, "either 'snapshot' or 'base_image'"),
    ({"api_key": "k", "egress_hosts": "a.example"}, "'egress_hosts' must be a list"),
    ({"api_key": "k", "egress_hosts": ["*"]}, "'egress_hosts' must be a list"),
])
def test_bad_settings_name_the_section_and_the_problem(config, message):
    with pytest.raises(ConfigError) as raised:
        DaytonaSandboxProvider.from_config(**config)
    assert "[sandbox.providers.daytona.config]" in str(raised.value) and message in str(raised.value)


def test_neither_the_settings_nor_the_provider_show_the_api_key():
    provider = DaytonaSandboxProvider.from_config(api_key="secret-key", target="eu")
    assert "secret-key" not in repr(provider._settings)
    assert repr(provider) == f"DaytonaSandboxProvider(api_url={DEFAULT_API_URL!r}, target='eu')"


def test_resources_meet_the_docker_floor_and_round_up():
    small = sandbox_resources(cpu=1.0, memory_mb=1024, disk_gb=3)
    large = sandbox_resources(cpu=2.5, memory_mb=8193, disk_gb=20.2)
    assert (small.cpu, small.memory, small.disk) == (2, 4, 10)
    assert (large.cpu, large.memory, large.disk) == (3, 9, 21)


@pytest.mark.parametrize(("timeout", "minutes"), [(1, 1), (60, 1), (61, 2), (7200, 120)])
def test_timeout_seconds_become_whole_ttl_minutes(timeout, minutes):
    assert ttl_minutes(timeout) == minutes


def test_labels_carry_attribution_ports_the_managed_marker_and_the_host_image_version():
    labels = sandbox_labels({"run_id": "run 42/x", "team": "env-pod", "skip": None}, [8080, 5432], host_image=True)
    assert labels[MANAGED_LABEL] == "true" and labels[PORTS_LABEL] == "8080,5432"
    assert labels["agentenv.host-image"] == HOST_IMAGE_VERSION
    assert (labels["agentenv.attr.run_id"], labels["agentenv.attr.team"]) == ("run-42-x", "env-pod")
    assert "agentenv.attr.skip" not in labels


def test_a_sandbox_booted_from_a_snapshot_carries_no_host_image_label():
    assert "agentenv.host-image" not in sandbox_labels({}, [], host_image=False)


def test_attribution_keys_with_nothing_label_safe_are_skipped():
    labels = sandbox_labels({"!!!": "x", "team": "env-pod"}, [], host_image=True)
    assert [key for key in labels if key.startswith("agentenv.attr.")] == ["agentenv.attr.team"]


def test_label_values_are_bounded():
    assert len(sandbox_labels({"long": "v" * 1000}, [], host_image=True)["agentenv.attr.long"]) == LABEL_VALUE_MAX


def test_names_are_unique_prefixed_and_short():
    first = sandbox_name({"run_id": "r1", "pipeline_step": "task_box" * 20})
    second = sandbox_name({"run_id": "r1"})
    assert first.startswith("ae-") and second.startswith("ae-") and first != second
    assert len(first) <= 63 and first[-1] != "-"


def test_an_unknown_config_key_fails_the_build(config_file):
    config_file.write_text(config_file.read_text() + 'surprise = "x"\n')
    reset_config()
    with pytest.raises(ConfigError, match="unknown key"):
        build_sandbox_provider("daytona")


def test_daytona_works_in_a_fallback_chain_with_the_local_provider(config_file):
    chain = build_sandbox_provider("daytona,local")
    assert isinstance(chain, ChainedSandboxProvider)
    assert isinstance(chain.links[0], DaytonaSandboxProvider)


@pytest.mark.asyncio
async def test_the_default_client_is_an_sdk_client_for_the_configured_api():
    provider = DaytonaSandboxProvider.from_config(api_key="k", api_url="https://d.example/api", target="eu")
    client = provider._client()
    assert isinstance(client, AsyncDaytona) and provider._client() is client
    assert (client._api_key, client._api_url, client._target) == ("k", "https://d.example/api", "eu")
    await provider.close()


def test_the_provider_makes_vms_and_containers_of_its_registry_name():
    assert DaytonaSandboxProvider.SUPPORTED_SANDBOX_MODES == {"vm", "container"}
    assert DaytonaSandbox.type == "daytona"


@pytest.mark.asyncio
async def test_create_vm_builds_the_host_image_at_the_requested_size_as_a_public_ephemeral_sandbox():
    provider, fake = _provider()
    sandbox = await provider.create_vm(cpu=4, memory=8192, disk_size_gb=20, timeout=900,
                                       exposed_ports=[8080, 8080, 5432], setup_for_gateway=False,
                                       attribution={"run_id": "r1"})
    params = fake.created_params[0]
    assert isinstance(params, CreateSandboxFromImageParams)
    assert params.image.dockerfile().splitlines()[0] == f"FROM {DEFAULT_BASE_IMAGE}"
    assert (params.resources.cpu, params.resources.memory, params.resources.disk) == (4, 8, 20)
    assert (params.public, params.ephemeral, params.auto_stop_interval, params.ttl_minutes) == (True, True, 0, 15)
    assert (params.domain_allow_list, params.network_allow_list, params.network_block_all) == (None, None, None)
    assert params.labels[MANAGED_LABEL] == "true" and params.labels[PORTS_LABEL] == "8080,5432"
    assert params.labels["agentenv.host-image"] == HOST_IMAGE_VERSION
    assert params.labels["agentenv.attr.run_id"] == "r1" and params.name.startswith("ae-") and "r1" in params.name
    assert fake.create_timeouts == [IMAGE_CREATE_TIMEOUT]
    assert isinstance(sandbox, DaytonaSandbox) and (sandbox.type, sandbox.mode) == ("daytona", "vm")
    assert sandbox.tunnel_urls == {port: f"https://{port}-{sandbox.sandbox_id}.{PROXY_DOMAIN}" for port in (8080, 5432)}
    assert sandbox.network_policy == NetworkPolicy()


@pytest.mark.asyncio
async def test_a_configured_base_image_is_what_the_host_image_builds_on():
    provider, fake = _provider(base_image="ubuntu:24.04")
    await provider.create_vm(setup_for_gateway=False)
    assert fake.created_params[0].image.dockerfile().splitlines()[0] == "FROM ubuntu:24.04"


@pytest.mark.asyncio
async def test_a_configured_snapshot_is_booted_instead_of_the_image():
    provider, fake = _provider(snapshot="my-dind")
    fake.snapshot.add(FakeSnapshot("my-dind", cpu=2, mem=8, disk=10))
    await provider.create_vm(cpu=2, memory=8192, disk_size_gb=10, setup_for_gateway=False)
    params = fake.created_params[0]
    assert isinstance(params, CreateSandboxFromSnapshotParams) and params.snapshot == "my-dind"
    assert fake.snapshot.gets == ["my-dind"]
    assert fake.create_timeouts == [SNAPSHOT_CREATE_TIMEOUT]
    assert "agentenv.host-image" not in params.labels


@pytest.mark.asyncio
async def test_a_snapshot_sandbox_is_logged_at_the_snapshot_size_not_the_request(caplog):
    provider, fake = _provider(snapshot="my-dind")
    fake.snapshot.add(FakeSnapshot("my-dind", cpu=8, mem=32, disk=50))
    with caplog.at_level(logging.INFO, logger="agent_env.providers.sandbox_providers.daytona.provider"):
        await provider.create_vm(cpu=1, memory=2048, disk_size_gb=10, setup_for_gateway=False)
    assert "cpu=8 memory=32GiB disk=50GiB" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(("cpu", "memory", "disk"), [(3, 8192, 10), (2, 8193, 10), (2, 8192, 10.5)])
async def test_a_request_bigger_than_the_snapshot_is_refused_before_creating(cpu, memory, disk):
    provider, fake = _provider(snapshot="my-dind")
    fake.snapshot.add(FakeSnapshot("my-dind", cpu=2, mem=8, disk=10))
    with pytest.raises(ValueError, match="snapshot 'my-dind' boots sandboxes of 2 vCPU, 8 GiB memory and 10 GiB disk"):
        await provider.create_vm(cpu=cpu, memory=memory, disk_size_gb=disk)
    assert fake.created_params == []


@pytest.mark.asyncio
async def test_create_sandbox_is_a_bare_vm_exposing_the_port():
    provider, fake = _provider()
    sandbox = await provider.create_sandbox(image_name="ignored:1", port=9000, env={"SECRET": "s"}, timeout=600)
    assert list(sandbox.tunnel_urls) == [9000] and sandbox.mode == "vm"
    assert "SECRET" not in str(fake.created_params[0])


@pytest.mark.asyncio
async def test_setup_for_gateway_runs_after_create():
    provider, fake = _provider()
    sandbox = await provider.create_vm(exposed_ports=[8080])
    assert ["sudo", "docker", "compose", "version"] in fake.sandboxes[sandbox.sandbox_id].process.commands


@pytest.mark.asyncio
async def test_an_image_override_is_refused():
    clients = FakeClients()
    with pytest.raises(ValueError, match="image overrides"):
        await _on(clients).create_vm(image="ubuntu:24.04")
    assert clients.made == []


@pytest.mark.asyncio
async def test_a_boot_mode_is_ignored_with_a_warning(caplog):
    provider, fake = _provider()
    with caplog.at_level(logging.WARNING, logger="agent_env.providers.sandbox_providers.daytona.provider"):
        await provider.create_vm(boot_mode="uefi", setup_for_gateway=False)
    assert len(fake.created_params) == 1
    assert "Ignoring boot_mode=uefi" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [
    allow("1.2.3.4"), allow(cidrs=("10.0.0.0/8",)), allow("pypi.org", cidrs=("10.0.0.0/8",)),
])
async def test_unenforceable_policies_are_refused_before_a_client_is_made(policy):
    clients = FakeClients()
    assert not DaytonaSandboxProvider.supports_network_policy(policy)
    with pytest.raises(NetworkPolicyUnsupportedError):
        await _on(clients).create_vm(network_policy=policy)
    assert clients.made == []


@pytest.mark.asyncio
async def test_a_restricted_create_allows_the_daytona_proxy_and_the_configured_hosts():
    provider, fake = _provider(egress_hosts=["*.internal.example"])
    sandbox = await provider.create_vm(network_policy=allow("pypi.org"), setup_for_gateway=False)
    sent = fake.created_params[0].domain_allow_list.split(",")
    assert sent[0] == "pypi.org" and {*DAYTONA_HOSTS, "*.internal.example"} <= set(sent)
    assert sandbox.network_policy.allow_hosts == tuple(sent)
    assert DaytonaSandboxProvider.EGRESS_HOSTS == DAYTONA_HOSTS


def test_every_restricted_policy_reaches_daytonas_preview_proxy_whatever_the_provider():
    effective = sandbox_provider_module.SandboxProvider.effective_network_policy(allow())
    assert set(DAYTONA_HOSTS) <= set(effective.allow_hosts)


@pytest.mark.asyncio
@pytest.mark.parametrize(("error", "message"), [
    (DaytonaForbiddenError, "Failed to create sandbox: Network access is restricted and cannot be overridden at the "
                            "sandbox level. Remove domainAllowList from the request."),
    (DaytonaBadRequestError, "Failed to create sandbox: Domain allow list cannot contain more than 100 domains"),
    (DaytonaBadRequestError, "Failed to create sandbox: networkAllowList and domainAllowList are mutually exclusive"),
    (DaytonaBadRequestError, 'Failed to create sandbox: Invalid domain: "x_y.example". Must be a valid ASCII domain'),
])
async def test_daytona_refusing_the_egress_lists_is_a_policy_error(error, message):
    def refuse(params, raw):
        raise error(message)

    provider, fake = _provider(FakeDaytona(create_hook=refuse))
    with pytest.raises(NetworkPolicyUnsupportedError, match="Daytona refused the egress policy"):
        await provider.create_vm(network_policy=allow("pypi.org"), setup_for_gateway=False)
    assert fake.lookups == []


@pytest.mark.asyncio
async def test_an_unrelated_refusal_of_a_restricted_create_keeps_its_own_error():
    def refuse(params, raw):
        raise DaytonaBadRequestError("Failed to create sandbox: 64 CPUs is more than the maximum allowed")

    provider, _ = _provider(FakeDaytona(create_hook=refuse))
    with pytest.raises(DaytonaBadRequestError):
        await provider.create_vm(network_policy=allow("pypi.org"), setup_for_gateway=False)


@pytest.mark.asyncio
async def test_a_policy_phrase_in_the_refusal_of_an_unrestricted_create_keeps_its_own_error():
    def refuse(params, raw):
        raise DaytonaForbiddenError("Failed to create sandbox: Network access is restricted for this organization")

    provider, _ = _provider(FakeDaytona(create_hook=refuse))
    with pytest.raises(DaytonaForbiddenError):
        await provider.create_vm(setup_for_gateway=False)


@pytest.mark.asyncio
async def test_rate_limited_creates_are_retried():
    failures = [DaytonaRateLimitError("slow down"), DaytonaRateLimitError("slow down")]

    def throttle(params, raw):
        if failures:
            raise failures.pop(0)

    provider, fake = _provider(FakeDaytona(create_hook=throttle))
    sandbox = await provider.create_vm(setup_for_gateway=False)
    assert len(fake.created_params) == 3 and sandbox.sandbox_id in fake.sandboxes


@pytest.mark.asyncio
async def test_rate_limiting_that_never_clears_gives_up_with_nothing_to_clean_up():
    def throttle(params, raw):
        raise DaytonaRateLimitError("slow down")

    provider, fake = _provider(FakeDaytona(create_hook=throttle))
    with pytest.raises(DaytonaRateLimitError):
        await provider.create_vm(setup_for_gateway=False)
    assert len(fake.created_params) == provider._RATE_LIMIT_ATTEMPTS
    assert fake.lookups == []


@pytest.mark.asyncio
async def test_a_create_the_api_rejected_outright_looks_nothing_up():
    def reject(params, raw):
        raise DaytonaAuthenticationError("Invalid credentials")

    provider, fake = _provider(FakeDaytona(create_hook=reject))
    with pytest.raises(DaytonaAuthenticationError):
        await provider.create_vm(setup_for_gateway=False)
    assert fake.lookups == []


@pytest.mark.asyncio
async def test_a_failed_setup_deletes_the_sandbox():
    provider, fake = _provider()
    fake.make_raw = lambda: FakeRaw(process=_failing_setup())
    with pytest.raises(RuntimeError, match="Compose v2"):
        await provider.create_vm(exposed_ports=[8080])
    assert [raw.deletes for raw in fake.sandboxes.values()] == [1]


@pytest.mark.asyncio
async def test_a_failed_setup_retries_a_delete_that_fails_once():
    provider, fake = _provider()
    fake.make_raw = lambda: FakeRaw(process=_failing_setup(), delete_failures=[DaytonaError("502")])
    with pytest.raises(RuntimeError):
        await provider.create_vm(exposed_ports=[8080])
    (raw,) = fake.sandboxes.values()
    assert (raw.deletes, raw.state) == (2, "destroyed")


@pytest.mark.asyncio
async def test_a_setup_cancelled_mid_way_deletes_the_sandbox_before_the_cancellation_is_raised():
    gate = asyncio.Event()
    provider, fake = _provider()
    fake.make_raw = lambda: FakeRaw(process=FakeProcess(gate=gate))
    task = asyncio.ensure_future(provider.create_vm(exposed_ports=[8080]))
    while not fake.sandboxes:
        await asyncio.sleep(0)
    (raw,) = fake.sandboxes.values()
    await raw.process.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert raw.deletes == 1 and raw.process.open_sessions == set()


@pytest.mark.asyncio
async def test_a_setup_cancelled_again_during_its_cleanup_is_not_held_up_and_the_cleanup_finishes():
    gate, delete_gate = asyncio.Event(), asyncio.Event()
    provider, fake = _provider()
    fake.make_raw = lambda: FakeRaw(process=FakeProcess(gate=gate), delete_gate=delete_gate)
    task = asyncio.ensure_future(provider.create_vm(exposed_ports=[8080]))
    while not fake.sandboxes:
        await asyncio.sleep(0)
    (raw,) = fake.sandboxes.values()
    await raw.process.entered.wait()
    task.cancel()
    await _drain(5)
    task.cancel()
    done, _ = await asyncio.wait([task], timeout=1)
    assert done and task.cancelled() and raw.deletes == 0
    delete_gate.set()
    await _drain()
    assert (raw.deletes, raw.state) == (1, "destroyed")


@pytest.mark.asyncio
async def test_a_create_cancelled_in_flight_is_raised_once_its_sandbox_is_deleted():
    started, release = asyncio.Event(), asyncio.Event()

    async def slow(params, raw):
        started.set()
        await release.wait()

    provider, fake = _provider(FakeDaytona(create_hook=slow))
    task = asyncio.ensure_future(provider.create_vm(setup_for_gateway=False))
    await started.wait()
    task.cancel()
    await _drain()
    assert not task.done()  # held until the create answers and its sandbox is deleted
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert [raw.deletes for raw in fake.sandboxes.values()] == [1]


@pytest.mark.asyncio
async def test_a_cancelled_create_retries_a_delete_that_fails_once():
    started, release = asyncio.Event(), asyncio.Event()

    async def slow(params, raw):
        raw.delete_failures.append(DaytonaError("502"))
        started.set()
        await release.wait()

    provider, fake = _provider(FakeDaytona(create_hook=slow))
    task = asyncio.ensure_future(provider.create_vm(setup_for_gateway=False))
    await started.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    (raw,) = fake.sandboxes.values()
    assert (raw.deletes, raw.state) == (2, "destroyed")


@pytest.mark.asyncio
async def test_a_create_that_fails_after_making_a_sandbox_deletes_it_by_name():
    async def made_then_failed(params, raw):
        fake.sandboxes[raw.id] = raw
        raise DaytonaTimeoutError("Sandbox failed to become ready within the timeout period")

    provider, fake = _provider(FakeDaytona(create_hook=made_then_failed))
    with pytest.raises(DaytonaTimeoutError):
        await provider.create_vm(setup_for_gateway=False)
    (raw,) = fake.sandboxes.values()
    assert fake.lookups == [raw.name] and raw.deletes == 1


@pytest.mark.asyncio
async def test_cleanup_waits_out_a_409_from_another_state_change_and_still_deletes():
    provider, fake = _provider()
    fake.make_raw = lambda: FakeRaw(process=_failing_setup(), delete_failures=[CONFLICT])
    with pytest.raises(RuntimeError):
        await provider.create_vm(exposed_ports=[8080])
    (raw,) = fake.sandboxes.values()
    assert (raw.deletes, raw.state) == (2, "destroyed")


@pytest.mark.asyncio
async def test_cleanup_takes_a_409_for_a_sandbox_already_being_deleted_as_done():
    provider, fake = _provider()
    fake.make_raw = lambda: FakeRaw(process=_failing_setup(), delete_failures=[CONFLICT], desired_state="destroyed")
    with pytest.raises(RuntimeError):
        await provider.create_vm(exposed_ports=[8080])
    (raw,) = fake.sandboxes.values()
    assert raw.deletes == 1


@pytest.mark.asyncio
async def test_cleanup_by_name_waits_out_a_409_too():
    async def made_then_failed(params, raw):
        raw.delete_failures.append(CONFLICT)
        fake.sandboxes[raw.id] = raw
        raise DaytonaTimeoutError("Sandbox failed to become ready within the timeout period")

    provider, fake = _provider(FakeDaytona(create_hook=made_then_failed))
    with pytest.raises(DaytonaTimeoutError):
        await provider.create_vm(setup_for_gateway=False)
    (raw,) = fake.sandboxes.values()
    assert (raw.deletes, raw.state) == (2, "destroyed")


@pytest.mark.asyncio
async def test_a_caller_cancelled_as_the_create_finishes_waits_for_a_stuck_cleanup_only_so_long():
    gate = asyncio.Event()
    caller: list[asyncio.Task] = []

    async def cancel_the_caller(params, raw):
        raw.delete_gate = gate
        caller[0].cancel()

    provider, fake = _provider(FakeDaytona(create_hook=cancel_the_caller))
    provider._CANCEL_WAIT = 0.1
    caller.append(asyncio.ensure_future(provider.create_vm(setup_for_gateway=False)))
    done, _ = await asyncio.wait(caller, timeout=1)
    assert done and caller[0].cancelled()
    gate.set()
    await _drain()
    assert [raw.deletes for raw in fake.sandboxes.values()] == [1]


@pytest.mark.asyncio
async def test_a_caller_cancelled_as_the_create_fails_still_gets_the_sandbox_cleaned_up():
    caller: list[asyncio.Task] = []

    async def cancel_then_fail(params, raw):
        fake.sandboxes[raw.id] = raw
        caller[0].cancel()
        raise DaytonaTimeoutError("Sandbox failed to become ready within the timeout period")

    provider, fake = _provider(FakeDaytona(create_hook=cancel_then_fail))
    caller.append(asyncio.ensure_future(provider.create_vm(setup_for_gateway=False)))
    with pytest.raises(asyncio.CancelledError):
        await caller[0]
    await _drain()
    assert [raw.deletes for raw in fake.sandboxes.values()] == [1]


@pytest.mark.asyncio
async def test_reconnect_restores_ports_urls_and_policy():
    provider, fake = _provider()
    created = await provider.create_vm(network_policy=allow("pypi.org"), exposed_ports=[8080], setup_for_gateway=False)
    other, _ = _provider(fake)
    again = await other.get_sandbox(created.sandbox_id)
    assert (again.sandbox_id, again.tunnel_urls, again.network_policy) == (
        created.sandbox_id, created.tunnel_urls, created.network_policy)


@pytest.mark.asyncio
async def test_reconnect_to_an_unexpressible_policy_leaves_it_unknown():
    provider, fake = _provider()
    created = await provider.create_vm(setup_for_gateway=False)
    raw = fake.sandboxes[created.sandbox_id]
    raw.domain_allow_list, raw.network_block_all = "pypi.org", True
    assert (await provider.get_sandbox(created.sandbox_id)).network_policy is None


@pytest.mark.asyncio
async def test_get_sandbox_only_reads_the_sandbox_and_its_preview_links():
    provider, fake = _provider()
    created = await provider.create_vm(network_policy=allow("pypi.org"), exposed_ports=[8080, 5432],
                                       setup_for_gateway=False)
    raw = fake.sandboxes[created.sandbox_id]
    before = (raw.domain_allow_list, raw.network_allow_list, raw.network_block_all, raw.labels.copy())
    raw.previewed.clear()
    await provider.get_sandbox(created.sandbox_id)
    assert fake.lookups == [created.sandbox_id] and raw.previewed == [8080, 5432]
    assert (raw.domain_allow_list, raw.network_allow_list, raw.network_block_all, raw.labels) == before
    assert (raw.deletes, raw.network_updates, raw.process.commands, len(fake.created_params)) == (0, [], [], 1)


@pytest.mark.asyncio
async def test_get_sandbox_of_a_sandbox_daytona_no_longer_knows_is_not_found():
    provider, fake = _provider()
    created = await provider.create_vm(setup_for_gateway=False)
    await provider.terminate_sandbox(created.sandbox_id)
    for sandbox_id in (created.sandbox_id, "sb-never"):
        with pytest.raises(SandboxNotFoundError, match=sandbox_id):
            await provider.get_sandbox(sandbox_id)


@pytest.mark.asyncio
async def test_terminate_sandbox_deletes_by_id_and_terminating_again_is_done():
    provider, fake = _provider()
    created = await provider.create_vm(exposed_ports=[8080], setup_for_gateway=False)
    raw = fake.sandboxes[created.sandbox_id]
    raw.previewed.clear()
    await provider.terminate_sandbox(created.sandbox_id)
    await provider.terminate_sandbox(created.sandbox_id)
    assert (raw.deletes, raw.state, raw.previewed) == (1, "destroyed", [])


@pytest.mark.asyncio
async def test_terminate_sandbox_of_a_sandbox_never_made_is_done():
    provider, fake = _provider()
    await provider.terminate_sandbox("sb-never")
    assert fake.lookups == ["sb-never"]


@pytest.mark.asyncio
async def test_terminate_sandbox_of_a_sandbox_gone_before_its_delete_is_done():
    provider, fake = _provider()
    created = await provider.create_vm(setup_for_gateway=False)
    raw = fake.sandboxes[created.sandbox_id]
    raw.delete_failures = [DaytonaNotFoundError("Sandbox not found")]
    await provider.terminate_sandbox(created.sandbox_id)
    assert raw.deletes == 1


@pytest.mark.asyncio
async def test_terminate_sandbox_waits_out_a_409_from_another_state_change(no_delete_delay):
    provider, fake = _provider()
    created = await provider.create_vm(setup_for_gateway=False)
    raw = fake.sandboxes[created.sandbox_id]
    raw.delete_failures = [CONFLICT]
    await provider.terminate_sandbox(created.sandbox_id)
    assert (raw.deletes, raw.state) == (2, "destroyed")


@pytest.mark.asyncio
async def test_terminate_sandbox_raises_what_isnt_a_gone_sandbox():
    provider, fake = _provider()
    created = await provider.create_vm(setup_for_gateway=False)
    fake.sandboxes[created.sandbox_id].delete_error = DaytonaError("server error")
    with pytest.raises(DaytonaError, match="server error"):
        await provider.terminate_sandbox(created.sandbox_id)


@pytest.mark.asyncio
async def test_every_sandbox_of_a_provider_shares_one_client_per_event_loop():
    clients = FakeClients()
    provider = _on(clients)
    created = await provider.create_vm(exposed_ports=[8080])
    await (await provider.get_sandbox(created.sandbox_id)).exec_with_output("true")
    await created.terminate()
    assert len(clients.made) == 1 and clients.settings == [provider._settings]


def test_each_event_loop_gets_its_own_client():
    clients = FakeClients()
    provider = _on(clients)

    async def use():
        created = await provider.create_vm(setup_for_gateway=False)
        await provider.get_sandbox(created.sandbox_id)

    asyncio.run(use())
    asyncio.run(use())
    assert len(clients.made) == 2
    assert all((len(client.created_params), len(client.lookups)) == (1, 1) for client in clients.made)


def test_close_closes_only_the_running_loops_client():
    clients = FakeClients()
    provider = _on(clients)

    async def create_and_close():
        await provider.create_vm(setup_for_gateway=False)
        await provider.close()

    asyncio.run(provider.create_vm(setup_for_gateway=False))
    asyncio.run(provider.close())
    asyncio.run(create_and_close())
    assert [client.closed for client in clients.made] == [0, 1]


@pytest.mark.asyncio
async def test_providers_do_not_share_clients_and_close_closes_only_their_own():
    clients = FakeClients()
    first, second = _on(clients), _on(clients)
    await first.create_vm(setup_for_gateway=False)
    await second.create_vm(setup_for_gateway=False)
    await first.close()
    await first.close()
    assert len(clients.made) == 2 and [client.closed for client in clients.made] == [1, 0]


@pytest.mark.asyncio
async def test_a_sandbox_made_before_close_reads_its_handle_again_on_the_next_client():
    clients = FakeClients()
    provider = _on(clients)
    created = await provider.create_vm(setup_for_gateway=False)
    await provider.close()
    assert (await created.exec_with_output("echo", "hi"))[0] == 0
    assert len(clients.made) == 2 and clients.made[1].lookups == [created.sandbox_id]
    assert clients.sandboxes[created.sandbox_id].process.commands == [["echo", "hi"]]
