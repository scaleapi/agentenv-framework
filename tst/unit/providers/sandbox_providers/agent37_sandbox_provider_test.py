"""Offline unit tests for the Agent37 sandbox provider."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.agent37 import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.agent37.provider import (
    NAME_SUFFIX_LENGTH,
    TEMPLATE,
    Agent37SandboxProvider,
    instance_name,
    instance_resources,
)
from agent_env.providers.sandbox_providers.agent37.sandbox import Agent37Error, applied_policy
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError
from tst.unit.providers.sandbox_providers.agent37_fakes import API_KEY, FakeAgent37

ALLOWLIST = NetworkMode.ALLOWLIST


@pytest.fixture
def api():
    return FakeAgent37()


@pytest.fixture(autouse=True)
def _no_resend_delays(monkeypatch):
    monkeypatch.setattr(sandbox_module, "RESEND_DELAYS", (0, 0, 0, 0, 0))


def _provider(api: FakeAgent37, **kwargs) -> Agent37SandboxProvider:
    return Agent37SandboxProvider(api_key=API_KEY, transport=api.transport(), **kwargs)


@pytest.mark.parametrize(("cpu", "memory", "disk", "resources"), [
    (1.0, 8192, 10, {"cpu": 4, "memory": 8, "disk": 20}),
    (0.5, 1024, 10, {"cpu": 2, "memory": 4, "disk": 12}),
    (2, 4096, 12, {"cpu": 2, "memory": 4, "disk": 12}),
    (2, 4096, 13, {"cpu": 4, "memory": 8, "disk": 20}),
    (3, 2048, 10, {"cpu": 4, "memory": 8, "disk": 20}),
    (1, 8193, 10, {"cpu": 4, "memory": 16, "disk": 40}),
    (4, 32768, 80, {"cpu": 4, "memory": 32, "disk": 80}),
])
def test_resources_are_the_smallest_shape_that_covers_the_request(cpu, memory, disk, resources):
    assert instance_resources(cpu, memory, disk) == resources


@pytest.mark.parametrize(("cpu", "memory", "disk"), [(5, 1024, 10), (1, 32769, 10), (1, 1024, 81)])
def test_resources_refuse_rather_than_shrink(cpu, memory, disk):
    with pytest.raises(ValueError, match="Agent37 cannot size a VM"):
        instance_resources(cpu, memory, disk)


def test_instance_names_are_unique_and_carry_no_attribution():
    name = instance_name()
    assert name.startswith("agentenv-") and len(name) == len("agentenv-") + NAME_SUFFIX_LENGTH
    assert instance_name() != name


def test_from_config_takes_the_known_keys():
    provider = Agent37SandboxProvider.from_config(api_key="sk_live_x", region="eu")
    assert provider._region == "eu"
    assert Agent37SandboxProvider.from_config(api_key="sk_live_x")._region == "us"


@pytest.mark.parametrize(("config", "message"), [
    ({"api_key": "k", "org": "x"}, "unknown key"),
    ({}, "requires a non-empty 'api_key'"),
    ({"api_key": " "}, "requires a non-empty 'api_key'"),
    ({"api_key": "k", "region": "mars"}, "'region' must be one of"),
])
def test_from_config_refuses_bad_config(config, message):
    with pytest.raises(ConfigError, match=message):
        Agent37SandboxProvider.from_config(**config)


@pytest.mark.parametrize("policy", [
    NetworkPolicy(mode=ALLOWLIST),
    NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",)),
    NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("203.0.113.0/24",)),
])
def test_supports_only_unrestricted_egress(policy):
    assert Agent37SandboxProvider.supports_network_policy(NetworkPolicy())
    assert not Agent37SandboxProvider.supports_network_policy(policy)


@pytest.mark.asyncio
async def test_create_vm_makes_an_ubuntu_vm_with_a_public_url_per_port_and_its_attribution(api):
    sandbox = await _provider(api, region="ap-south").create_vm(
        cpu=1.5, memory=4096, exposed_ports=[8080, 8080, 9000], setup_for_gateway=False, attribution={"run_id": "r1"},
    )
    [body] = api.created
    assert {key: body[key] for key in ("template", "type", "region", "resources", "public_ports")} == {
        "template": TEMPLATE,
        "type": "vm",
        "region": "ap-south",
        "resources": {"cpu": 2, "memory": 4, "disk": 12},
        "public_ports": [{"port": 8080}, {"port": 9000}],
    }
    assert body["metadata"]["run_id"] == "r1"
    assert body["name"].startswith("agentenv-")
    assert (sandbox.type, sandbox.mode, sandbox.sandbox_id) == ("agent37", "vm", "i000000001")
    assert sandbox.tunnel_urls == {8080: "https://slug8080i000000001.agent37.app",
                                   9000: "https://slug9000i000000001.agent37.app"}
    assert sandbox.network_policy == NetworkPolicy()
    assert api.execs == []


@pytest.mark.asyncio
async def test_create_vm_readies_docker_for_the_gateway(api):
    await _provider(api).create_vm(exposed_ports=[8000])
    [check] = api.execs
    assert check["user"] == "root"
    assert check["command"] == f"bash -c {sandbox_module.shlex.quote(sandbox_module.DOCKER_READY)}"


@pytest.mark.asyncio
async def test_create_vm_refuses_what_it_cannot_honour_before_creating_anything(api):
    provider = _provider(api)
    with pytest.raises(ValueError, match="per-call image is unsupported"):
        await provider.create_vm(image="ubuntu:24.04")
    with pytest.raises(NetworkPolicyUnsupportedError):
        await provider.create_vm(network_policy=NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",)))
    with pytest.raises(ValueError, match="cannot size"):
        await provider.create_vm(cpu=16)
    assert api.requests == []


@pytest.mark.asyncio
async def test_a_vm_whose_docker_never_readies_is_deleted(api):
    api.exec_result = lambda body: {"exit_code": 1, "stdout": "", "stderr": "docker daemon did not become ready",
                                    "truncated": False}
    with pytest.raises(RuntimeError, match="Docker not ready on Agent37 instance i000000001"):
        await _provider(api).create_vm(exposed_ports=[8000])
    assert api.deleted == ["i000000001"]
    assert api.instances == {}


@pytest.mark.asyncio
async def test_a_create_whose_answer_is_lost_deletes_the_instance_it_made_by_name(api):
    api.create_answer = httpx.ReadError("connection reset")
    with pytest.raises(httpx.ReadError):
        await _provider(api).create_vm(setup_for_gateway=False)
    assert api.deleted == ["i000000001"]


@pytest.mark.asyncio
async def test_a_refused_create_raises_the_reason_and_deletes_nothing(api):
    api.create_answer = httpx.Response(402, json={"error": {"code": "insufficient_balance", "message": "Top up."}})
    with pytest.raises(Agent37Error, match="402 insufficient_balance: Top up."):
        await _provider(api).create_vm(setup_for_gateway=False)
    assert api.instances == {} and api.deleted == []
    assert api.requests_to("GET", "/v1/instances")


@pytest.mark.asyncio
async def test_a_cancelled_create_deletes_the_instance_once_the_create_answers(api):
    gate = asyncio.Event()
    provider = _provider(api)
    create = provider._api.create_instance

    async def slow_create(body):
        await gate.wait()
        return await create(body)

    provider._api.create_instance = slow_create
    task = asyncio.ensure_future(provider.create_vm(setup_for_gateway=False))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert api.deleted == ["i000000001"]


@pytest.mark.asyncio
async def test_create_sandbox_is_a_vm_exposing_the_agent_port(api):
    sandbox = await _provider(api).create_sandbox(image_name="agent:latest", port=8000, env={"A": "1"},
                                                  attribution={"run_id": "r2"})
    [body] = api.created
    assert body["public_ports"] == [{"port": 8000}]
    assert list(sandbox.tunnel_urls) == [8000]


@pytest.mark.asyncio
async def test_get_sandbox_from_a_fresh_provider_restores_ports_and_policy(api):
    created = await _provider(api).create_vm(exposed_ports=[8080], setup_for_gateway=False)
    reconnected = await _provider(api).get_sandbox(created.sandbox_id)
    assert (reconnected.sandbox_id, reconnected.tunnel_urls) == (created.sandbox_id, created.tunnel_urls)
    assert reconnected.network_policy == NetworkPolicy()


@pytest.mark.parametrize(("network", "policy"), [
    ("allow-all", NetworkPolicy()),
    ("deny-all", NetworkPolicy(mode=ALLOWLIST)),
    ({"allow": ["203.0.113.0/24"]}, NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("203.0.113.0/24",))),
])
def test_applied_policy_reads_the_instance_network_field(network, policy):
    assert applied_policy(network) == policy


@pytest.mark.asyncio
async def test_terminate_is_idempotent_and_a_gone_instance_is_not_found(api):
    provider = _provider(api)
    sandbox = await provider.create_vm(setup_for_gateway=False)
    await sandbox.terminate()
    await sandbox.terminate()
    await provider.terminate_sandbox(sandbox.sandbox_id)
    with pytest.raises(SandboxNotFoundError):
        await provider.get_sandbox(sandbox.sandbox_id)
