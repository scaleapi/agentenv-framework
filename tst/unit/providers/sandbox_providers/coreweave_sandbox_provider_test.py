"""Offline unit tests for the CoreWeave sandbox provider."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import cwsandbox
import pytest
from cwsandbox import CidrBlock, EgressRule, EndpointAuth, EndpointKind, SandboxFailedError, SandboxStatus
from cwsandbox import ServiceVisibility

from agent_env.config.errors import ConfigError
from agent_env.providers.sandbox_providers.coreweave import provider as provider_module
from agent_env.providers.sandbox_providers.coreweave.provider import (
    DEFAULT_IMAGE,
    DEFAULT_RUNTIME_CLASS,
    ENDPOINT_REQUEST_TIMEOUT_SEC,
    MIN_DOCKER_DISK_GB,
    SANDBOX_TAG,
    CoreWeaveSandboxProvider,
    auth_headers,
    docker_disk_gb,
    resources,
    sandbox_tags,
)
from agent_env.providers.sandbox_providers.coreweave.sandbox import CoreWeaveContainerSandbox, CoreWeaveSandbox
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, NetworkPolicyUnsupportedError
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER, SANDBOX_MODE_VM, Accepts
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError, _unimplemented
from agent_env.store.image_store.oci_registry_credentials import RegistryAuth
from tst.unit.providers.sandbox_providers.coreweave_fakes import Result, fake_sandbox_cls

ALLOWLIST = NetworkMode.ALLOWLIST


@pytest.fixture(autouse=True)
def _no_endpoint_parents(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(provider_module, "_endpoint_parents", set())


@pytest.fixture
def sandbox_cls():
    return fake_sandbox_cls()


def _provider(sandbox_cls, **kwargs) -> CoreWeaveSandboxProvider:
    return CoreWeaveSandboxProvider(api_key="cw_test", sandbox_cls=sandbox_cls, **kwargs)


async def _no_sleep(_seconds):
    return None


def test_the_provider_creates_vms_and_containers_and_implements_what_it_must():
    assert _unimplemented(CoreWeaveSandboxProvider) is None
    assert CoreWeaveSandboxProvider.SUPPORTED_SANDBOX_MODES == {SANDBOX_MODE_VM, SANDBOX_MODE_CONTAINER}


@pytest.mark.parametrize(("config", "message"), [
    ({}, "exactly one of 'api_key' and 'wandb_api_key'"),
    ({"api_key": "a", "wandb_api_key": "w"}, "exactly one of 'api_key' and 'wandb_api_key'"),
    ({"api_key": "a", "wandb_entity": "team"}, "only with 'wandb_api_key'"),
    ({"api_key": "a", "region": "us"}, "unknown key"),
    ({"api_key": "a", "image": ""}, "'image' must be a non-empty string"),
    ({"api_key": "a", "docker_storage": "vfs"}, "'docker_storage' must be 'loop' or 'memory'"),
    ({"api_key": "a", "runner_ids": "runner-a"}, "'runner_ids' must be a non-empty list"),
    ({"api_key": "a", "runner_ids": []}, "'runner_ids' must be a non-empty list"),
])
def test_from_config_refuses_a_bad_section(config, message):
    with pytest.raises(ConfigError, match=message):
        CoreWeaveSandboxProvider.from_config(**config)


def test_from_config_takes_every_documented_key(sandbox_cls, monkeypatch):
    monkeypatch.setattr(provider_module.cwsandbox, "Sandbox", sandbox_cls)
    provider = CoreWeaveSandboxProvider.from_config(
        wandb_api_key="wb", wandb_entity="team", wandb_project="evals", base_url="https://api.test",
        image="registry.test/host:1", docker_storage="memory", runtime_class="kata-clh", placement_mode="cks",
        runner_ids=["runner-a"],
    )
    assert provider._client.base_url == "https://api.test"
    assert (provider._image, provider._docker_storage, provider._runtime_class) == (
        "registry.test/host:1", "memory", "kata-clh")


def test_auth_headers_for_a_coreweave_key():
    headers = auth_headers({"api_key": "cw_secret"})
    assert headers.headers == {"Authorization": "Bearer cw_secret"}


def test_auth_headers_for_a_wandb_key_carry_its_entity_and_project():
    headers = auth_headers({"wandb_api_key": "wb_secret", "wandb_entity": "team", "wandb_project": "evals"})
    assert headers.headers == {"x-wandb-api-key": "wb_secret", "x-entity-id": "team", "x-project-name": "evals"}


def test_resources_request_and_limit_exactly_what_is_asked():
    shape = resources(1.5, 8192)
    assert shape.requests == shape.limits == {"cpu": "1500m", "memory": "8192Mi"}
    assert resources(0.0001, 1).requests["cpu"] == "1m"


@pytest.mark.parametrize(("cpu", "memory"), [(0, 1024), (float("nan"), 1024), (True, 1024), (1, 0), (1, 1.5)])
def test_resources_refuse_a_nonsense_shape(cpu, memory):
    with pytest.raises(ValueError):
        resources(cpu, memory)


def test_dockers_disk_has_a_floor():
    assert docker_disk_gb(10) == MIN_DOCKER_DISK_GB
    assert docker_disk_gb(40.2) == 41


def test_tags_are_slugged_into_coreweaves_alphabet_and_length():
    tags = sandbox_tags({"run_id": "run 1/2", "pipeline_step": "deploy:" + "x" * 80}, "agentenv-abc")
    assert tags[:3] == [SANDBOX_TAG, "agentenv-abc", "pipeline_step.deploy-" + "x" * 38]
    assert tags[3] == "run_id.run-1-2"
    assert all(len(tag) <= 59 for tag in tags)


def test_supports_allow_all_and_allowlists_of_valid_entries():
    assert CoreWeaveSandboxProvider.supports_network_policy(NetworkPolicy())
    assert CoreWeaveSandboxProvider.supports_network_policy(
        NetworkPolicy(mode=ALLOWLIST, allow_hosts=("a.com", "*.b.com", "10.0.0.1"), allow_cidrs=("1.0.0.0/8",)))
    assert not CoreWeaveSandboxProvider.supports_network_policy(NetworkPolicy(mode=ALLOWLIST, allow_hosts=("a b",)))


@pytest.mark.asyncio
async def test_create_vm_makes_a_privileged_docker_host_with_https_endpoints(sandbox_cls):
    sandbox = await _provider(sandbox_cls).create_vm(
        cpu=2, memory=4096, disk_size_gb=30, timeout=900, exposed_ports=[8080, 8080, 9000],
        attribution={"run_id": "r1"},
    )
    raw = sandbox_cls.server.created[0]
    kwargs = raw.kwargs
    assert (kwargs["command"], kwargs["args"][0]) == ("/bin/sh", "-c")
    assert "truncate -s 30G" in kwargs["args"][1]
    assert kwargs["container_image"] == DEFAULT_IMAGE
    assert kwargs["runtime_class"] == DEFAULT_RUNTIME_CLASS
    assert kwargs["security_context"].privileged is True
    assert kwargs["max_lifetime_seconds"] == 900
    assert kwargs["resources"].limits == {"cpu": "2000m", "memory": "4096Mi"}
    assert kwargs["network"] is None
    assert kwargs["auth"].headers == {"Authorization": "Bearer cw_test"}
    assert kwargs["tags"][0] == SANDBOX_TAG and kwargs["tags"][1].startswith(f"{SANDBOX_TAG}-")
    assert "run_id.r1" in kwargs["tags"]
    services = kwargs["services"]
    assert [service.port for service in services] == [8080, 9000]
    assert {(s.visibility, s.endpoint.kind, s.endpoint.auth, s.endpoint.request_timeout_seconds) for s in services} == {
        (ServiceVisibility.PUBLIC, EndpointKind.HTTPS, EndpointAuth.OPEN, ENDPOINT_REQUEST_TIMEOUT_SEC)}
    assert (sandbox.type, sandbox.mode, sandbox.sandbox_id, sandbox.command_timeout) == ("coreweave", "vm", "sb-1", 900)
    assert set(sandbox.tunnel_urls) == {8080, 9000}
    assert sandbox.network_policy == NetworkPolicy()
    assert raw.exec_calls[0]["argv"][:2] == ["sh", "-c"]  # the gateway setup waited for Docker


@pytest.mark.asyncio
async def test_create_vm_sends_and_stamps_an_allowlist(sandbox_cls):
    policy = NetworkPolicy(mode=ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("1.1.1.0/24",))
    sandbox = await _provider(sandbox_cls).create_vm(network_policy=policy, setup_for_gateway=False)
    rules = sandbox_cls.server.created[0].kwargs["network"].egress
    assert "pypi.org" in [rule.dns_name for rule in rules]
    assert sandbox.network_policy.restricts_egress
    assert "pypi.org" in sandbox.network_policy.allow_hosts
    assert "1.1.1.0/24" in sandbox.network_policy.allow_cidrs


@pytest.mark.asyncio
async def test_create_vm_refuses_a_per_call_image_and_an_unenforceable_policy(sandbox_cls):
    provider = _provider(sandbox_cls)
    with pytest.raises(ValueError, match="per-call image is unsupported"):
        await provider.create_vm(image="other")
    with pytest.raises(NetworkPolicyUnsupportedError):
        await provider.create_vm(network_policy=NetworkPolicy(mode=ALLOWLIST, allow_hosts=("not valid",)))
    assert sandbox_cls.server.created == []


@pytest.mark.asyncio
async def test_create_container_runs_the_image_itself_with_its_env_and_port(sandbox_cls):
    sandbox = await _provider(sandbox_cls).create_container(
        image_name="registry.example/agent:1", port=8000, env={"A2A_PORT": "8000"}, cpu=2, memory=4096, timeout=900,
        attribution={"run_id": "r1"},
    )
    raw = sandbox_cls.server.created[0]
    kwargs = raw.kwargs
    [container] = kwargs["containers"]
    assert (container.image, container.command, container.args) == ("registry.example/agent:1", None, None)
    assert container.environment_variables == {"A2A_PORT": "8000"}
    assert container.resources.limits == {"cpu": "2000m", "memory": "4096Mi"}
    assert not {"command", "args", "container_image", "resources", "security_context", "runtime_class"} & set(kwargs)
    assert kwargs["max_lifetime_seconds"] == 900
    assert "run_id.r1" in kwargs["tags"] and kwargs["network"] is None
    assert [service.port for service in kwargs["services"]] == [8000]
    assert type(sandbox) is CoreWeaveContainerSandbox
    assert (sandbox.type, sandbox.mode, sandbox.command_timeout) == ("coreweave", SANDBOX_MODE_CONTAINER, 900)
    assert list(sandbox.tunnel_urls) == [8000] and sandbox.network_policy == NetworkPolicy()
    assert raw.exec_calls == []  # no Docker host to ready


@pytest.mark.asyncio
async def test_the_sdk_sends_a_container_with_no_command_so_the_image_starts_as_it_was_built(sandbox_cls):
    """The SDK's single-container arguments fill an unset command with ``/bin/sh -c 'sleep infinity'``."""
    await _provider(sandbox_cls).create_container(image_name="registry.example/agent:1", port=8000, env={"A": "1"})
    real = cwsandbox.Sandbox(**sandbox_cls.server.created[0].kwargs)
    request = real._build_create_request(request_id="r", start_kwargs=dict(real._start_kwargs))
    [container] = request.sandbox.spec.containers
    assert (container.image, container.command, list(container.args)) == ("registry.example/agent:1", "", [])
    assert dict(container.environment_variables) == {"A": "1"}
    assert not container.HasField("security_context")


@pytest.mark.asyncio
async def test_create_sandbox_is_a_container_of_the_agents_image(sandbox_cls):
    sandbox = await _provider(sandbox_cls).create_sandbox(image_name="registry.example/agent:1", port=8000,
                                                          env={"A": "1"})
    assert sandbox.mode == SANDBOX_MODE_CONTAINER and list(sandbox.tunnel_urls) == [8000]
    assert sandbox_cls.server.created[0].kwargs["containers"][0].image == "registry.example/agent:1"


def test_containers_and_agents_run_images_by_name_and_vms_create_gateways():
    """Declared, since a provider implementing create_vm otherwise gives agents what its VMs load."""
    assert CoreWeaveSandboxProvider.SANDBOX_ACCEPTS is CoreWeaveSandboxProvider.CONTAINER_ACCEPTS is Accepts.NAME
    assert not CoreWeaveSandboxProvider.PRIVATE_NETWORK


@pytest.mark.asyncio
async def test_a_container_logs_its_disk_size_as_ignored(sandbox_cls, caplog):
    caplog.set_level("INFO", logger=provider_module.__name__)
    await _provider(sandbox_cls).create_container(image_name="registry.example/a:1", port=8000, env={},
                                                  disk_size_gb=40)
    assert "Ignoring disk_size_gb=40" in caplog.text


@pytest.mark.asyncio
async def test_a_container_on_a_private_network_is_refused(sandbox_cls):
    with pytest.raises(ValueError, match="share no private network"):
        await _provider(sandbox_cls).create_container(image_name="registry.example/a:1", port=1, env={},
                                                      private_network=True)
    assert sandbox_cls.server.created == []


@pytest.mark.asyncio
async def test_an_image_the_image_store_holds_credentials_for_is_refused(sandbox_cls, monkeypatch):
    store = SimpleNamespace(auth=lambda ref: RegistryAuth(registry="123.dkr.ecr.test", username="AWS", password="t"))
    monkeypatch.setattr(provider_module, "get_config", lambda: SimpleNamespace(get_image_store_at=lambda ref: store))
    with pytest.raises(ValueError, match="without the credentials agent-env's image store holds for 123.dkr.ecr.test"):
        await _provider(sandbox_cls).create_container(image_name="123.dkr.ecr.test/agent:1", port=1, env={})
    assert sandbox_cls.server.created == []


@pytest.mark.asyncio
async def test_a_container_whose_image_fails_to_start_is_deleted(sandbox_cls, monkeypatch):
    async def failing_wait(self):
        raise SandboxFailedError("ImagePullBackOff")

    monkeypatch.setattr(sandbox_cls, "__await__", lambda self: failing_wait(self).__await__())
    with pytest.raises(SandboxFailedError, match="ImagePullBackOff"):
        await _provider(sandbox_cls).create_container(image_name="registry.example/a:1", port=8000, env={})
    assert sandbox_cls.server.deleted == ["sb-1"]


@pytest.mark.asyncio
async def test_a_container_without_its_endpoint_is_deleted(sandbox_cls):
    original_init = sandbox_cls.__init__

    def init(self, **kwargs):
        original_init(self, **kwargs)
        self.service_endpoints = ()

    sandbox_cls.__init__ = init
    with pytest.raises(RuntimeError, match=r"no endpoint for port\(s\) \[8000\]"):
        await _provider(sandbox_cls).create_container(image_name="registry.example/a:1", port=8000, env={})
    assert sandbox_cls.server.deleted == ["sb-1"]


@pytest.mark.asyncio
async def test_a_failed_start_deletes_the_sandbox_it_made(sandbox_cls, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    async def failing_wait(self):
        raise SandboxFailedError("image pull failed")

    monkeypatch.setattr(sandbox_cls, "__await__", lambda self: failing_wait(self).__await__())
    with pytest.raises(SandboxFailedError, match="image pull failed"):
        await _provider(sandbox_cls).create_vm()
    assert sandbox_cls.server.deleted == ["sb-1"]


@pytest.mark.asyncio
async def test_a_start_whose_response_was_lost_is_found_by_its_unique_tag_and_deleted(sandbox_cls):
    sandbox_cls.server.start_error = ConnectionError("connection reset")
    sandbox_cls.server.start_commits = True
    with pytest.raises(ConnectionError):
        await _provider(sandbox_cls).create_vm()
    create_tag = sandbox_cls.server.created[0].kwargs["tags"][1]
    assert sandbox_cls.server.listed == [[create_tag]]
    assert sandbox_cls.server.deleted == ["sb-1"]


@pytest.mark.asyncio
async def test_a_cancelled_create_deletes_its_sandbox_once_it_exists(sandbox_cls):
    sandbox_cls.server.start_delay = 0.05
    create = asyncio.ensure_future(_provider(sandbox_cls).create_vm())
    await asyncio.sleep(0.01)
    create.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create
    assert sandbox_cls.server.deleted == ["sb-1"]


def _slow_round_trips(sandbox_cls) -> None:
    """List and delete take a network round trip, as the SDK's do."""
    listed, deleted = sandbox_cls.list.__func__, sandbox_cls.delete.__func__

    async def slow_list(cls, **kwargs):
        await asyncio.sleep(0.05)
        return await listed(cls, **kwargs)

    async def slow_delete(cls, sandbox_id, **kwargs):
        await asyncio.sleep(0.05)
        await deleted(cls, sandbox_id, **kwargs)

    sandbox_cls.list, sandbox_cls.delete = classmethod(slow_list), classmethod(slow_delete)


@pytest.mark.parametrize("create", ["create_vm", "create_container"])
def test_a_create_cancelled_by_the_loop_ending_deletes_the_sandbox_its_lost_response_made(sandbox_cls, create):
    """``asyncio.run`` cancels what is left when it ends, the SDK's start among it, and closes the loop once those are
    done: cleanup must finish inside the cancelled create."""

    async def start(self):
        sandbox_id = next(self.server.ids)
        self.server.sandboxes[sandbox_id] = self  # the server took the create; its response is still on its way
        await asyncio.sleep(1)
        self.sandbox_id = sandbox_id

    sandbox_cls.start = start
    _slow_round_trips(sandbox_cls)
    kwargs = {"setup_for_gateway": False} if create == "create_vm" else {
        "image_name": "registry.example/a:1", "port": 8000, "env": {}}

    async def main():
        asyncio.ensure_future(getattr(_provider(sandbox_cls), create)(**kwargs))
        await asyncio.sleep(0.01)

    asyncio.run(main())
    assert sandbox_cls.server.listed == [[sandbox_cls.server.created[0].kwargs["tags"][1]]]
    assert sandbox_cls.server.deleted == ["sb-1"] and sandbox_cls.server.sandboxes == {}


def test_a_create_cancelled_by_the_loop_ending_while_it_waits_to_run_deletes_its_sandbox(sandbox_cls):
    async def slow_running(self):
        await asyncio.sleep(1)
        return self

    sandbox_cls.__await__ = lambda self: slow_running(self).__await__()
    _slow_round_trips(sandbox_cls)

    async def main():
        asyncio.ensure_future(_provider(sandbox_cls).create_container(image_name="registry.example/a:1", port=8000,
                                                                      env={}))
        await asyncio.sleep(0.01)

    asyncio.run(main())
    assert sandbox_cls.server.deleted == ["sb-1"] and sandbox_cls.server.listed == []


@pytest.mark.asyncio
async def test_a_failed_docker_setup_deletes_the_sandbox_and_says_why(sandbox_cls):
    original_init = sandbox_cls.__init__

    def init(self, **kwargs):
        original_init(self, **kwargs)
        self.results = [Result(returncode=1, stderr_bytes=b"the image lacks bash curl e2fsprogs")]

    sandbox_cls.__init__ = init
    with pytest.raises(RuntimeError, match="the image lacks bash curl e2fsprogs"):
        await _provider(sandbox_cls).create_vm()
    assert sandbox_cls.server.deleted == ["sb-1"]


@pytest.mark.asyncio
async def test_a_port_without_an_endpoint_fails_the_create_and_deletes_the_sandbox(sandbox_cls):
    original_init = sandbox_cls.__init__

    def init(self, **kwargs):
        original_init(self, **kwargs)
        self.service_endpoints = ()

    sandbox_cls.__init__ = init
    with pytest.raises(RuntimeError, match=r"no endpoint for port\(s\) \[8080\]"):
        await _provider(sandbox_cls).create_vm(exposed_ports=[8080], setup_for_gateway=False)
    assert sandbox_cls.server.deleted == ["sb-1"]


@pytest.mark.asyncio
async def test_exposed_port_domains_join_later_allowlists(sandbox_cls):
    """A restricted agent must reach an environment another CoreWeave sandbox serves."""
    provider = _provider(sandbox_cls)
    await provider.create_vm(exposed_ports=[8080], setup_for_gateway=False)
    agent = await provider.create_vm(network_policy=NetworkPolicy(mode=ALLOWLIST), setup_for_gateway=False)
    assert "*.r1abc.ep1.cwsandbox.test" in agent.network_policy.allow_hosts


@pytest.mark.asyncio
async def test_get_sandbox_restores_endpoints_and_the_applied_policy(sandbox_cls):
    provider = _provider(sandbox_cls)
    policy = NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("1.1.1.0/24",))
    created = await provider.create_vm(exposed_ports=[8080], network_policy=policy, setup_for_gateway=False)
    reconnected = await provider.get_sandbox(created.sandbox_id)
    assert reconnected.tunnel_urls == created.tunnel_urls
    assert reconnected.network_policy == created.network_policy
    assert reconnected.command_timeout > 0
    assert sandbox_cls.server.calls[-1] == ("from_id", {"auth": provider._client.auth,
                                                        "base_url": provider._client.base_url})


@pytest.mark.asyncio
async def test_unrepresentable_applied_egress_is_an_unknown_policy_at_create_and_reconnect(sandbox_cls):
    """Never the requested policy: callers decide from it whether the sandbox can download for itself."""
    original_init = sandbox_cls.__init__

    def init(self, **kwargs):
        original_init(self, **kwargs)
        self.effective_egress = (EgressRule(cidr=CidrBlock(cidr="1.1.1.0/24"), ports=[443]),)

    sandbox_cls.__init__ = init
    policy = NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("1.1.1.0/24",))
    vm = await _provider(sandbox_cls).create_vm(network_policy=policy, setup_for_gateway=False)
    container = await _provider(sandbox_cls).create_container(image_name="registry.example/a:1", port=8000, env={},
                                                              network_policy=policy)
    fresh = _provider(sandbox_cls)
    assert [vm.network_policy, container.network_policy] == [None, None]
    assert [(await fresh.get_sandbox(s.sandbox_id)).network_policy for s in (vm, container)] == [None, None]


@pytest.mark.asyncio
async def test_a_fresh_provider_reconnects_each_sandbox_in_the_mode_it_was_created_in(sandbox_cls):
    policy = NetworkPolicy(mode=ALLOWLIST, allow_cidrs=("1.1.1.0/24",))
    vm = await _provider(sandbox_cls).create_vm(exposed_ports=[8080], network_policy=policy, setup_for_gateway=False)
    container = await _provider(sandbox_cls).create_container(image_name="registry.example/a:1", port=8000, env={},
                                                              network_policy=policy)
    fresh = _provider(sandbox_cls)
    vm_again, container_again = await fresh.get_sandbox(vm.sandbox_id), await fresh.get_sandbox(container.sandbox_id)
    assert (type(vm_again), vm_again.mode) == (CoreWeaveSandbox, SANDBOX_MODE_VM)
    assert (type(container_again), container_again.mode) == (CoreWeaveContainerSandbox, SANDBOX_MODE_CONTAINER)
    for created, again in ((vm, vm_again), (container, container_again)):
        assert (again.tunnel_urls, again.network_policy) == (created.tunnel_urls, created.network_policy)


@pytest.mark.asyncio
async def test_a_sandbox_whose_spec_is_not_echoed_is_not_guessed_at(sandbox_cls):
    created = await _provider(sandbox_cls).create_container(image_name="registry.example/a:1", port=8000, env={})
    sandbox_cls.server.sandboxes[created.sandbox_id].containers = ()
    with pytest.raises(RuntimeError, match="whether it is a container or a VM is unknown"):
        await _provider(sandbox_cls).get_sandbox(created.sandbox_id)


@pytest.mark.asyncio
async def test_get_sandbox_of_a_gone_sandbox_raises_not_found(sandbox_cls):
    with pytest.raises(SandboxNotFoundError, match="sb-gone"):
        await _provider(sandbox_cls).get_sandbox("sb-gone")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [SandboxStatus.TERMINATING, SandboxStatus.COMPLETED, SandboxStatus.FAILED,
                                    SandboxStatus.TERMINATED])
async def test_get_sandbox_of_a_sandbox_that_has_ended_raises_not_found(sandbox_cls, status):
    created = await _provider(sandbox_cls).create_container(image_name="registry.example/a:1", port=8000, env={})
    sandbox_cls.server.sandboxes[created.sandbox_id].status = status
    with pytest.raises(SandboxNotFoundError, match="has ended"):
        await _provider(sandbox_cls).get_sandbox(created.sandbox_id)


@pytest.mark.asyncio
async def test_terminate_sandbox_deletes_by_id_without_reconnecting(sandbox_cls):
    await _provider(sandbox_cls).terminate_sandbox("sb-gone")
    assert sandbox_cls.server.deleted == ["sb-gone"]
    assert [name for name, _ in sandbox_cls.server.calls] == ["delete"]


@pytest.mark.asyncio
async def test_terminating_a_container_twice_tolerates_it_being_gone(sandbox_cls):
    provider = _provider(sandbox_cls)
    container = await provider.create_container(image_name="registry.example/a:1", port=8000, env={})
    await container.terminate()
    await container.terminate()
    await provider.terminate_sandbox(container.sandbox_id)
    assert sandbox_cls.server.deleted == ["sb-1"] * 3
    assert all(kwargs["missing_ok"] for name, kwargs in sandbox_cls.server.calls if name == "delete")
