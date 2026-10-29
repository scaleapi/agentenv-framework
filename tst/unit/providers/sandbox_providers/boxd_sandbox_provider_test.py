"""Offline unit tests for the boxd sandbox provider."""

from __future__ import annotations

import asyncio

import pytest
from boxd import APIConnectionError, APIStatusError, ConflictError, RateLimitError

from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.boxd import provider as provider_module
from agent_env.providers.sandbox_providers.boxd.provider import (
    DEFAULT_BASE_URL,
    BoxdSandboxProvider,
    egress_entries,
    machine_name,
    machine_size,
)
from agent_env.providers.sandbox_providers.boxd.sandbox import _KEEPALIVE, DOCKER_READY, KEEPALIVE_SEC, BoxdSandbox
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers import sandbox_provider as sandbox_provider_module
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError

from tst.unit.providers.sandbox_providers.boxd_fakes import FakeSnapshot, fake_client_cls, result

ALLOWLIST = NetworkMode.ALLOWLIST


@pytest.fixture
def client_cls():
    return fake_client_cls()


def _provider(client_cls, **kwargs) -> BoxdSandboxProvider:
    return BoxdSandboxProvider(api_key="bxd_test", client_cls=client_cls, **kwargs)


def _machines(client_cls):
    return client_cls.instances[0].machines


async def _no_sleep(_seconds):
    return None


@pytest.mark.parametrize(("cpu", "memory", "size"), [
    (1.0, 8192, (2, "8G")),
    (0.5, 1024, (1, "4G")),
    (1, 4096, (1, "4G")),
    (2.5, 2048, (4, "16G")),
    (1, 16385, (8, "32G")),
    (8, 32768, (8, "32G")),
])
def test_machine_size_is_the_smallest_that_covers_cpu_and_memory(cpu, memory, size):
    assert machine_size(cpu, memory, 10) == size


@pytest.mark.parametrize(("cpu", "memory", "disk"), [(9, 1024, 10), (1, 32769, 10), (1, 1024, 101)])
def test_machine_size_refuses_rather_than_shrinks(cpu, memory, disk):
    with pytest.raises(ValueError, match="boxd cannot size a machine"):
        machine_size(cpu, memory, disk)


@pytest.mark.parametrize("policy", [
    NetworkPolicy(),
    NetworkPolicy(mode=ALLOWLIST),
    NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org", "*.npmjs.org", "203.0.113.7"),
                  allow_cidrs=("198.51.100.0/24",)),
])
def test_supports_allow_all_and_allowlists_boxd_takes(policy):
    assert BoxdSandboxProvider.supports_network_policy(policy)


@pytest.mark.parametrize("entry", [
    "2001:db8::/32", "::1", "10.0.0.0/8", "172.20.1.1", "192.168.0.0/16", "127.0.0.1", "169.254.169.254",
    "224.0.0.0/4", "0.0.0.0/0", "localhost", "*.com", "example.com:443", "https://example.com", "example.com/path",
])
def test_refuses_an_allowlist_entry_boxd_refuses(entry):
    as_host = NetworkPolicy(mode=ALLOWLIST, allow_hosts=(entry,))
    assert not BoxdSandboxProvider.supports_network_policy(as_host)


def test_egress_entries_are_hosts_then_addresses_without_repeats():
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("a.com", "*.b.com", "a.com"), allow_cidrs=("1.1.1.0/24",))
    assert egress_entries(policy) == ["a.com", "*.b.com", "1.1.1.0/24"]


def test_machine_name_is_a_public_dns_label_with_no_attribution():
    name = machine_name()
    assert name.startswith("agentenv-")
    assert len(name) <= 63
    assert name == name.lower() and all(c.isalnum() or c == "-" for c in name)
    assert machine_name() != name


def test_from_config_takes_the_known_keys():
    provider = BoxdSandboxProvider.from_config(api_key="bxd_x", org="research", snapshot="agentenv-warm")
    assert (provider._api_key, provider._org, provider._base_url) == ("bxd_x", "research", DEFAULT_BASE_URL)
    assert provider._snapshot == "agentenv-warm"


@pytest.mark.parametrize(("config", "message"), [
    ({"api_key": "k", "region": "eu"}, "unknown key"),
    ({}, "requires a non-empty 'api_key'"),
    ({"api_key": "k", "org": " "}, "non-empty string"),
    ({"api_key": "k", "base_url": 9443}, "non-empty string"),
    ({"api_key": "k", "snapshot": ""}, "non-empty string"),
])
def test_from_config_refuses_bad_config(config, message):
    with pytest.raises(ConfigError, match=message):
        BoxdSandboxProvider.from_config(**config)


@pytest.mark.asyncio
async def test_create_vm_makes_an_isolated_machine_with_a_lifetime_and_a_proxy_per_port(client_cls):
    sandbox = await _provider(client_cls, org="research").create_vm(
        cpu=1.5, memory=4096, timeout=900, exposed_ports=[8080, 8080, 9000], setup_for_gateway=False,
        attribution={"run_id": "r1"},
    )
    client = client_cls.instances[0]
    assert client.kwargs == {"api_key": "bxd_test", "base_url": DEFAULT_BASE_URL}
    call = client.machines.create_calls[0]
    assert call["name"].startswith("agentenv-") and "r1" not in call["name"]
    assert (call["org"], call["isolated"], call["vcpu"], call["memory"]) == ("research", True, 2, "8G")
    assert (call["delete_after"], call["auto_suspend_timeout"]) == (900, 0)
    assert [(entry.name, entry.port) for entry in call["proxies"]] == [("p8080", 8080), ("p9000", 9000)]
    assert client.machines.egress_calls == []
    assert client.machines.ready_calls == ["m-1"]
    assert isinstance(sandbox, BoxdSandbox)
    assert (sandbox.type, sandbox.mode, sandbox.sandbox_id) == ("boxd", "vm", "m-1")
    assert sandbox.network_policy == NetworkPolicy()
    name = call["name"]
    assert sandbox.tunnel_urls == {8080: f"https://p8080.{name}.boxd.test", 9000: f"https://p9000.{name}.boxd.test"}


def _with_snapshot(client_cls, vcpu=2, memory_gib=8, proxy_ports=()):
    client_cls.on_init = lambda client: client.snapshots.snapshots.update(
        {"agentenv-warm": FakeSnapshot("agentenv-warm", vcpu=vcpu, memory_bytes=memory_gib << 30,
                                       proxy_ports=proxy_ports)})


@pytest.mark.asyncio
async def test_create_vm_restores_the_snapshot_and_adds_its_proxies(client_cls):
    """A machine restored from a snapshot keeps the snapshot's size and starts with no proxies of its own."""
    _with_snapshot(client_cls)
    sandbox = await _provider(client_cls, snapshot="agentenv-warm").create_vm(
        cpu=1, memory=4096, timeout=900, exposed_ports=[8080, 9000], setup_for_gateway=False)
    machines = _machines(client_cls)
    call = machines.create_calls[0]
    assert (call["from_snapshot"], call["isolated"], call["delete_after"], call["auto_suspend_timeout"]) == (
        "agentenv-warm", True, 900, 0)
    assert not {"vcpu", "memory", "proxies"} & set(call)
    assert [(name, port) for _, name, port in machines.proxies.create_calls] == [("p8080", 8080), ("p9000", 9000)]
    assert sandbox.tunnel_urls == {port: f"https://p{port}.{call['name']}.boxd.test" for port in (8080, 9000)}


@pytest.mark.asyncio
async def test_a_restored_machine_serves_only_the_ports_it_was_asked_for(client_cls):
    """boxd replays the snapshot's proxies onto the restored machine; the ones for other ports are removed."""
    _with_snapshot(client_cls, proxy_ports=(8080, 18765, 18767))
    provider = _provider(client_cls, snapshot="agentenv-warm")
    sandbox = await provider.create_vm(exposed_ports=[8080, 9000], setup_for_gateway=False)
    proxies = _machines(client_cls).proxies
    assert sorted(name for _, name in proxies.delete_calls) == ["p18765", "p18767"]
    assert [name for _, name, _ in proxies.create_calls] == ["p9000"]
    assert sorted(sandbox.tunnel_urls) == [8080, 9000]
    assert sorted((await provider.get_sandbox(sandbox.sandbox_id)).tunnel_urls) == [8080, 9000]


@pytest.mark.asyncio
async def test_a_request_bigger_than_the_snapshot_is_refused_before_creating(client_cls):
    _with_snapshot(client_cls, vcpu=2, memory_gib=8)
    with pytest.raises(ValueError, match="restores machines of 2 vCPU / 8 GiB"):
        await _provider(client_cls, snapshot="agentenv-warm").create_vm(cpu=4, memory=16384)
    assert _machines(client_cls).create_calls == []


@pytest.mark.asyncio
async def test_create_vm_applies_an_allowlist_with_the_boxd_hosts(client_cls):
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("203.0.113.0/24",))
    sandbox = await _provider(client_cls).create_vm(network_policy=policy, setup_for_gateway=False)
    [(machine_id, entries)] = _machines(client_cls).egress_calls
    assert machine_id == "m-1"
    assert entries[0] == "pypi.org" and entries[-1] == "203.0.113.0/24"
    assert "*.boxd.sh" in entries
    assert "*.boxd.sh" in sandbox.network_policy.allow_hosts


@pytest.mark.asyncio
async def test_an_allowlist_boxd_refuses_is_unenforceable_and_deletes_the_machine(client_cls):
    client_cls.on_init = lambda client: setattr(
        client.machines, "egress_error",
        APIStatusError("'*.amazonaws.com' is a shared-hosting domain", code="invalid_argument", grpc_code=3),
    )
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("*.amazonaws.com",))
    with pytest.raises(NetworkPolicyUnsupportedError, match="shared-hosting"):
        await _provider(client_cls).create_vm(network_policy=policy, setup_for_gateway=False)
    assert _machines(client_cls).deleted == ["m-1"]


@pytest.mark.asyncio
async def test_create_vm_refuses_an_unenforceable_policy_before_creating(client_cls):
    with pytest.raises(NetworkPolicyUnsupportedError):
        await _provider(client_cls).create_vm(
            network_policy=NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("10.0.0.0/8",)),
        )
    assert client_cls.instances == []


@pytest.mark.asyncio
async def test_an_allowlist_always_holds_the_boxd_hosts(client_cls, monkeypatch):
    """boxd reads an empty allowlist as unrestricted, so one is never sent, even with no other platform's hosts."""
    monkeypatch.setattr(sandbox_provider_module, "all_sandbox_egress_hosts", lambda: ())
    sandbox = await _provider(client_cls).create_vm(network_policy=NetworkPolicy(mode=ALLOWLIST), setup_for_gateway=False)
    assert _machines(client_cls).egress_calls == [("m-1", ["*.boxd.sh"])]
    assert sandbox.network_policy.allow_hosts == ("*.boxd.sh",)


@pytest.mark.asyncio
async def test_create_vm_refuses_a_per_call_image(client_cls):
    with pytest.raises(ValueError, match="per-call image"):
        await _provider(client_cls).create_vm(image="ubuntu:24.04")
    assert client_cls.instances == []


@pytest.mark.asyncio
async def test_create_vm_readies_docker_for_the_gateway(client_cls):
    await _provider(client_cls).create_vm(exposed_ports=[8080])
    machines = _machines(client_cls)
    assert machines.ready_calls == ["m-1"]
    assert [call["command"] for call in machines.exec_calls] == [
        _KEEPALIVE.format(interval=KEEPALIVE_SEC, line=f"sudo bash -c '{DOCKER_READY}' </dev/null"),
    ]


@pytest.mark.asyncio
async def test_a_setup_failure_deletes_the_machine(client_cls):
    client_cls.on_init = lambda client: client.machines.results.append(
        result(stderr=b"docker daemon did not become ready", exit_code=1))
    with pytest.raises(RuntimeError, match="did not become ready"):
        await _provider(client_cls).create_vm()
    assert _machines(client_cls).deleted == ["m-1"]


@pytest.mark.asyncio
async def test_a_machine_with_no_proxy_for_a_port_is_deleted(client_cls, monkeypatch):
    async def no_proxies(client, machine_id):
        return {}

    monkeypatch.setattr(provider_module, "_tunnel_urls", no_proxies)
    with pytest.raises(RuntimeError, match=r"no proxy for port\(s\) \[8080\]"):
        await _provider(client_cls).create_vm(exposed_ports=[8080], setup_for_gateway=False)
    assert _machines(client_cls).deleted == ["m-1"]


@pytest.mark.asyncio
async def test_a_create_cancelled_while_the_machine_is_made_deletes_it_once_it_answers(client_cls):
    client_cls.on_init = lambda client: setattr(client.machines, "create_delay", 0.05)
    task = asyncio.ensure_future(_provider(client_cls).create_vm(setup_for_gateway=False))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)
    assert _machines(client_cls).deleted == ["m-1"]


@pytest.mark.asyncio
async def test_a_create_cancelled_by_itself_still_reclaims_the_machine_it_made(client_cls):
    def cancelled_after_commit(client):
        client.machines.create_error = asyncio.CancelledError()
        client.machines.create_commits = True

    client_cls.on_init = cancelled_after_commit
    with pytest.raises(asyncio.CancelledError):
        await _provider(client_cls).create_vm(setup_for_gateway=False)
    await asyncio.sleep(0.05)
    assert _machines(client_cls).deleted == ["m-1"]


@pytest.mark.asyncio
async def test_a_create_whose_response_was_lost_deletes_the_machine_it_made(client_cls):
    def lost_response(client):
        client.machines.create_error = APIConnectionError("connection reset")
        client.machines.create_commits = True

    client_cls.on_init = lost_response
    with pytest.raises(APIConnectionError):
        await _provider(client_cls).create_vm(setup_for_gateway=False)
    assert _machines(client_cls).deleted == ["m-1"]


@pytest.mark.asyncio
async def test_a_create_the_sdk_retried_into_its_own_name_keeps_the_machine(client_cls):
    """The SDK resends a create whose connection dropped; the resend finds the name taken by the first attempt."""
    def first_attempt_made_it(client):
        client.machines.create_error = ConflictError("name already exists", code="already_exists")
        client.machines.create_commits = True

    client_cls.on_init = first_attempt_made_it
    sandbox = await _provider(client_cls).create_vm(setup_for_gateway=False)
    assert sandbox.sandbox_id == "m-1"
    assert _machines(client_cls).deleted == []


def test_asyncio_run_waits_for_the_delete_of_a_machine_whose_setup_it_cancelled(client_cls):
    """asyncio.run cancels what is left when its coroutine ends and waits for it; the delete is part of that wait."""
    def slow(client):
        client.machines.exec_delay = 5
        client.machines.delete_delay = 0.2

    client_cls.on_init = slow
    provider = _provider(client_cls)

    async def main():
        asyncio.ensure_future(provider.create_vm())
        while not client_cls.instances or not client_cls.instances[0].machines.exec_calls:
            await asyncio.sleep(0.01)

    asyncio.run(main())
    assert _machines(client_cls).deleted == ["m-1"]


@pytest.mark.asyncio
async def test_close_closes_this_loops_client(client_cls):
    provider = _provider(client_cls)
    await provider.create_vm(setup_for_gateway=False)
    await provider.close()
    await provider.close()
    assert client_cls.instances[0].closed == 1


@pytest.mark.asyncio
async def test_a_create_rejected_outright_leaves_nothing_to_reclaim(client_cls):
    client_cls.on_init = lambda client: setattr(
        client.machines, "create_error", RateLimitError("machine limit reached", code="resource_exhausted"))
    with pytest.raises(RateLimitError):
        await _provider(client_cls).create_vm(setup_for_gateway=False)
    assert _machines(client_cls).deleted == []


@pytest.mark.asyncio
async def test_the_reaper_retries_a_failed_delete(client_cls, monkeypatch):
    monkeypatch.setattr(provider_module.asyncio, "sleep", _no_sleep)
    client = client_cls()
    machine = await client.machines.create("agentenv-x")
    client.machines.delete_errors = [APIConnectionError("unavailable")]
    await provider_module._reap(client, machine.id)
    assert client.machines.deleted == [machine.id]


@pytest.mark.asyncio
async def test_create_sandbox_is_a_vm_with_its_port(client_cls):
    sandbox = await _provider(client_cls).create_sandbox(image_name="agent:1", port=8000, env={"K": "v"})
    assert list(sandbox.tunnel_urls) == [8000]
    assert sandbox.mode == "vm"


@pytest.mark.asyncio
async def test_get_sandbox_restores_ports_and_the_allowlist(client_cls):
    provider = _provider(client_cls)
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("203.0.113.0/24",))
    created = await provider.create_vm(exposed_ports=[8080], network_policy=policy, setup_for_gateway=False)
    reconnected = await provider.get_sandbox(created.sandbox_id)
    assert reconnected.tunnel_urls == created.tunnel_urls
    assert reconnected.network_policy.restricts_egress
    assert set(reconnected.network_policy.allow_hosts) == set(created.network_policy.allow_hosts)
    assert reconnected.network_policy.allow_cidrs == ("203.0.113.0/24",)


@pytest.mark.asyncio
async def test_get_sandbox_of_a_machine_boxd_no_longer_knows_is_not_found(client_cls):
    with pytest.raises(SandboxNotFoundError):
        await _provider(client_cls).get_sandbox("m-404")


@pytest.mark.asyncio
async def test_terminate_sandbox_deletes_by_id_and_counts_a_gone_machine_as_terminated(client_cls):
    provider = _provider(client_cls)
    created = await provider.create_vm(setup_for_gateway=False)
    await provider.terminate_sandbox(created.sandbox_id)
    await provider.terminate_sandbox(created.sandbox_id)
    assert _machines(client_cls).deleted == [created.sandbox_id]


@pytest.mark.asyncio
async def test_terminate_sandbox_raises_what_isnt_a_gone_machine(client_cls):
    provider = _provider(client_cls)
    created = await provider.create_vm(setup_for_gateway=False)
    _machines(client_cls).delete_errors = [ConflictError("machine is migrating", code="failed_precondition")]
    with pytest.raises(ConflictError):
        await provider.terminate_sandbox(created.sandbox_id)


@pytest.mark.asyncio
async def test_every_sandbox_of_a_provider_shares_one_client_per_event_loop(client_cls):
    provider = _provider(client_cls)
    created = await provider.create_vm(setup_for_gateway=False)
    await provider.get_sandbox(created.sandbox_id)
    await created.terminate()
    assert len(client_cls.instances) == 1

