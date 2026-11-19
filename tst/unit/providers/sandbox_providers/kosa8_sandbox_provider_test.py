"""Offline unit tests for the kosa8 sandbox provider."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.kosa8 import provider as provider_module
from agent_env.providers.sandbox_providers.kosa8 import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.kosa8.provider import (
    DEFAULT_IMAGE,
    Kosa8SandboxProvider,
    disk_gb,
    egress_entries,
    lifetime,
    sandbox_name,
    sandbox_size,
)
from agent_env.providers.sandbox_providers.kosa8.sandbox import DOCKER_READY, Kosa8Error, policy_from_enforced
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError
from tst.unit.providers.sandbox_providers.kosa8_fakes import API_KEY, FakeKosa8, command_of, error

ALLOWLIST = NetworkMode.ALLOWLIST


@pytest.fixture
def api():
    return FakeKosa8()


@pytest.fixture(autouse=True)
def _no_waits(monkeypatch):
    monkeypatch.setattr(sandbox_module, "RESEND_DELAYS", (0, 0, 0, 0, 0))
    monkeypatch.setattr(provider_module, "POLL_SEC", (0, 0))


def _provider(api: FakeKosa8, **kwargs) -> Kosa8SandboxProvider:
    return Kosa8SandboxProvider(api_key=API_KEY, transport=api.transport(), **kwargs)


@pytest.mark.parametrize(("cpu", "memory", "size"), [
    (1.0, 8192, "4x8"),
    (0.5, 1024, "2x4"),
    (2, 4096, "2x4"),
    (2.5, 2048, "4x8"),
    (1, 8193, "8x16"),
    (16, 32768, "16x32"),
])
def test_size_is_the_smallest_that_covers_cpu_and_memory(cpu, memory, size):
    assert sandbox_size(cpu, memory) == size


@pytest.mark.parametrize(("cpu", "memory"), [(17, 1024), (1, 32769)])
def test_size_refuses_rather_than_shrinks(cpu, memory):
    with pytest.raises(ValueError, match="kosa8 cannot size a sandbox"):
        sandbox_size(cpu, memory)


def test_disk_and_lifetime_stay_within_what_kosa8_takes():
    assert (disk_gb(10), disk_gb(32), disk_gb(100.5), disk_gb(200)) == (32, 32, 101, 200)
    assert (lifetime(1), lifetime(900)) == (60, 900)
    with pytest.raises(ValueError, match="200 GB"):
        disk_gb(201)
    with pytest.raises(ValueError, match="cannot keep a sandbox"):
        lifetime(91 * 86400)


def test_sandbox_names_are_unique_and_carry_no_attribution():
    name = sandbox_name()
    assert name.startswith("agentenv-") and len(name) == len("agentenv-") + 16
    assert sandbox_name() != name


@pytest.mark.parametrize("policy", [
    NetworkPolicy(),
    NetworkPolicy(mode=ALLOWLIST),
    NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org", "*.npmjs.org", "Files.PythonHosted.org", "203.0.113.7"),
                  allow_cidrs=("10.0.0.0/8",)),
])
def test_supports_allow_all_and_allow_lists_kosa8_takes(policy):
    assert Kosa8SandboxProvider.supports_network_policy(policy)


@pytest.mark.parametrize("entry", [
    "2001:db8::/32", "::1", "*.com", "example.com:443", "https://example.com", "example.com/path", "1.2.3",
])
def test_refuses_an_allow_list_entry_kosa8_refuses(entry):
    assert not Kosa8SandboxProvider.supports_network_policy(NetworkPolicy(mode=ALLOWLIST, allow_hosts=(entry,)))


def test_refuses_an_allow_list_longer_than_kosa8_keeps():
    hosts = tuple(f"h{index}.example.com" for index in range(257))
    assert not Kosa8SandboxProvider.supports_network_policy(NetworkPolicy(mode=ALLOWLIST, allow_hosts=hosts))


def test_egress_entries_are_hosts_then_addresses_without_repeats():
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("a.com", "*.b.com", "a.com"), allow_cidrs=("1.1.1.0/24",))
    assert egress_entries(policy) == ["a.com", "*.b.com", "1.1.1.0/24"]


def test_from_config_takes_the_known_keys():
    provider = Kosa8SandboxProvider.from_config(api_key="k8_x", region="eu", image="ghcr.io/acme/dind:1")
    assert (provider._region, provider._image) == ("eu", "ghcr.io/acme/dind:1")
    assert Kosa8SandboxProvider.from_config(api_key="k8_x")._image == DEFAULT_IMAGE


@pytest.mark.parametrize(("config", "message"), [
    ({"api_key": "k", "org": "x"}, "unknown key"),
    ({}, "requires a non-empty 'api_key'"),
    ({"api_key": "k", "region": " "}, "non-empty string"),
    ({"api_key": "k", "base_url": 443}, "non-empty string"),
    ({"api_key": "k", "image": ""}, "non-empty string"),
])
def test_from_config_refuses_bad_config(config, message):
    with pytest.raises(ConfigError, match=message):
        Kosa8SandboxProvider.from_config(**config)


@pytest.mark.asyncio
async def test_create_vm_boots_a_docker_sandbox_with_a_lifetime_and_a_public_url_per_port(api):
    sandbox = await _provider(api, region="eu").create_vm(
        cpu=1.5, memory=4096, timeout=900, exposed_ports=[8080, 8080, 9000], setup_for_gateway=False,
        attribution={"run_id": "r1"},
    )
    [body] = api.created
    assert body.pop("name").startswith("agentenv-")
    assert body == {"image": DEFAULT_IMAGE, "size": "2x4", "disk_gb": 32, "docker": True, "sleep_after_s": 0,
                    "ttl_s": 900, "region": "eu"}
    assert (sandbox.type, sandbox.mode, sandbox.sandbox_id) == ("kosa8", "vm", "sbx_00000001")
    assert sandbox.tunnel_urls == {8080: "https://8080-00000001.kosa8labs.com",
                                   9000: "https://9000-00000001.kosa8labs.com"}
    assert all(preview["public"] for preview in api.previews["sbx_00000001"].values())
    assert sandbox.network_policy == NetworkPolicy()
    assert api.execs == []


@pytest.mark.asyncio
async def test_create_vm_waits_for_a_cold_boot(api):
    api.boot_polls = 3
    await _provider(api).create_vm(setup_for_gateway=False)
    assert len(api.requests_to("GET", "/v1/sandboxes/sbx_00000001")) == 3


@pytest.mark.asyncio
async def test_create_vm_starts_docker_for_the_gateway(api):
    await _provider(api).create_vm(exposed_ports=[8000])
    [check] = api.execs
    assert command_of(check, api, "sbx_00000001") == ["bash", "-c", DOCKER_READY]


@pytest.mark.asyncio
async def test_create_vm_creates_the_sandbox_under_an_allow_list_with_kosa8s_hosts_and_reports_it(api):
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("203.0.113.0/24",))
    sandbox = await _provider(api).create_vm(network_policy=policy, setup_for_gateway=False)
    [body] = api.created
    allow = body["egress"]["allow"]
    assert api.requests_to("PUT", "/v1/sandboxes/sbx_00000001/egress") == []
    assert allow[0] == "pypi.org" and allow[-1] == "203.0.113.0/24"
    assert "*.kosa8labs.com" in allow and "*.boxd.sh" in allow
    assert sandbox.network_policy.restricts_egress
    assert sandbox.network_policy.allow_cidrs == ("203.0.113.0/24",)
    assert set(sandbox.network_policy.allow_hosts) == set(allow[:-1])


@pytest.mark.asyncio
async def test_an_org_default_policy_is_what_the_sandbox_reports(api):
    api.org_default = "allow:pypi.org;deny:gist.github.com"
    sandbox = await _provider(api).create_vm(setup_for_gateway=False)
    assert sandbox.network_policy == NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",))
    assert sandbox.denied_hosts == ("gist.github.com",)


@pytest.mark.asyncio
async def test_an_org_deny_list_survives_a_reconnect(api):
    api.org_default = "block:*.amazonaws.com"
    provider = _provider(api)
    created = await provider.create_vm(setup_for_gateway=False)
    reconnected = await _provider(api).get_sandbox(created.sandbox_id)
    for sandbox in (created, reconnected):
        assert sandbox.network_policy == NetworkPolicy()
        assert sandbox.denied_hosts == ("*.amazonaws.com",)
        assert not sandbox._can_download("https://bucket.s3.amazonaws.com/x?sig=1")


@pytest.mark.parametrize(("status", "code"), [(400, "invalid_egress"), (403, "egress_not_allowed"), (501, "unsupported")])
@pytest.mark.asyncio
async def test_a_create_kosa8_refuses_for_its_allow_list_is_unsupported(api, status, code):
    api.create_answer = error(status, code)
    with pytest.raises(NetworkPolicyUnsupportedError, match="refused the egress allow list"):
        await _provider(api).create_vm(network_policy=NetworkPolicy(mode=ALLOWLIST, allow_hosts=("a.com",)),
                                       setup_for_gateway=False)
    assert api.sandboxes == {} and api.deleted == []


@pytest.mark.parametrize("policy", [None, NetworkPolicy(mode=ALLOWLIST, allow_hosts=("a.com",))])
@pytest.mark.asyncio
async def test_a_create_refused_for_anything_else_is_raised_as_is(api, policy):
    api.create_answer = error(403, "quota_exceeded")
    with pytest.raises(Kosa8Error, match="quota_exceeded"):
        await _provider(api).create_vm(network_policy=policy, setup_for_gateway=False)


@pytest.mark.asyncio
async def test_create_vm_refuses_what_it_cannot_honour_before_creating_anything(api):
    provider = _provider(api)
    with pytest.raises(ValueError, match="per-call image is unsupported"):
        await provider.create_vm(image="ubuntu:24.04")
    with pytest.raises(NetworkPolicyUnsupportedError):
        await provider.create_vm(network_policy=NetworkPolicy(mode=ALLOWLIST, allow_hosts=("2001:db8::1",)))
    with pytest.raises(ValueError, match="cannot size"):
        await provider.create_vm(cpu=32)
    assert api.requests == []


@pytest.mark.asyncio
async def test_a_sandbox_whose_docker_never_readies_is_deleted(api):
    api.exec_result = lambda argv: {"exit": 1, "stderr": b"docker daemon did not become ready"}
    with pytest.raises(RuntimeError, match="Docker not ready on kosa8 sandbox sbx_00000001"):
        await _provider(api).create_vm(exposed_ports=[8000])
    assert api.deleted == ["sbx_00000001"]


@pytest.mark.asyncio
async def test_a_sandbox_that_fails_to_boot_is_deleted(api):
    api.boot_polls = 2
    provider = _provider(api)
    get = provider._api.get_sandbox

    async def failed(sandbox_id):
        sandbox = await get(sandbox_id)
        return {**sandbox, "state": "error", "last_error": "no image"}

    provider._api.get_sandbox = failed
    with pytest.raises(RuntimeError, match=r"did not start \(error\): no image"):
        await provider.create_vm(setup_for_gateway=False)
    assert api.deleted == ["sbx_00000001"]


@pytest.mark.asyncio
async def test_a_create_whose_answer_is_lost_deletes_the_sandbox_it_made_by_name(api):
    api.create_answer = httpx.ReadError("connection reset")
    with pytest.raises(httpx.ReadError):
        await _provider(api).create_vm(setup_for_gateway=False)
    assert api.deleted == ["sbx_00000001"]
    assert len(api.created) == 1


@pytest.mark.asyncio
async def test_a_refused_create_raises_the_reason_and_deletes_nothing(api):
    api.create_answer = httpx.Response(402, json={"error": "payment_required", "message": "Add a payment method."})
    with pytest.raises(Kosa8Error, match="402 payment_required: Add a payment method."):
        await _provider(api).create_vm(setup_for_gateway=False)
    assert api.sandboxes == {} and api.deleted == []


@pytest.mark.asyncio
async def test_a_create_refused_for_capacity_is_sent_again(api):
    answers = [error(503, "no_capacity")]
    provider = _provider(api)
    create = api._create

    def busy_once(body):
        return answers.pop() if answers else create(body)

    api._create = busy_once
    sandbox = await provider.create_vm(setup_for_gateway=False)
    assert sandbox.sandbox_id == "sbx_00000001"


@pytest.mark.asyncio
async def test_a_cancelled_create_deletes_the_sandbox_once_the_create_answers(api):
    gate = asyncio.Event()
    provider = _provider(api)
    create = provider._api.create_sandbox

    async def slow_create(body):
        await gate.wait()
        return await create(body)

    provider._api.create_sandbox = slow_create
    task = asyncio.ensure_future(provider.create_vm(setup_for_gateway=False))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert api.deleted == ["sbx_00000001"]


@pytest.mark.asyncio
async def test_create_sandbox_is_a_vm_exposing_the_agent_port(api):
    sandbox = await _provider(api).create_sandbox(image_name="agent:latest", port=8000, env={"A": "1"},
                                                  attribution={"run_id": "r2"})
    assert list(sandbox.tunnel_urls) == [8000]
    assert "env" not in api.created[0]


@pytest.mark.asyncio
async def test_get_sandbox_from_a_fresh_provider_restores_ports_and_policy(api):
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",))
    created = await _provider(api).create_vm(exposed_ports=[8080], network_policy=policy, setup_for_gateway=False)
    reconnected = await _provider(api).get_sandbox(created.sandbox_id)
    assert (reconnected.sandbox_id, reconnected.tunnel_urls) == (created.sandbox_id, created.tunnel_urls)
    assert reconnected.network_policy == created.network_policy


@pytest.mark.parametrize(("enforced", "policy"), [
    ("any", NetworkPolicy()),
    ("block:gist.github.com", NetworkPolicy()),
    ("deny", NetworkPolicy(mode=ALLOWLIST)),
    ("allow:pypi.org,*.npmjs.org,203.0.113.0/24;deny:x.com",
     NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org", "*.npmjs.org"), allow_cidrs=("203.0.113.0/24",))),
])
def test_policy_from_enforced_reads_kosa8s_compiled_policy(enforced, policy):
    assert policy_from_enforced(enforced) == policy


@pytest.mark.asyncio
async def test_terminate_is_idempotent_and_a_gone_sandbox_is_not_found(api):
    provider = _provider(api)
    sandbox = await provider.create_vm(setup_for_gateway=False)
    await sandbox.terminate()
    await sandbox.terminate()
    await provider.terminate_sandbox(sandbox.sandbox_id)
    with pytest.raises(SandboxNotFoundError):
        await provider.get_sandbox(sandbox.sandbox_id)
    with pytest.raises(SandboxNotFoundError):
        await provider.get_sandbox("sbx_missing")
