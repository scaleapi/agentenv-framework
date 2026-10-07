"""Unit tests for SandboxProvider, ChainedSandboxProvider, build_sandbox_provider."""

from __future__ import annotations

import asyncio

import pytest

from agent_env.config import get_config
from agent_env.config.loader import load_impl
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox, ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox, ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import Sandbox, VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import _BUILTIN_SANDBOX_PROVIDERS, SandboxProvider, _pull, build_sandbox_provider
from agent_env.store import ImageStore, RegistryAuth
from tst.util.exec_scripts import script_run


class _FakeSandbox(Sandbox):
    type = "fake"

    def __init__(self, sandbox_id: str = "fake-1"):
        self.sandbox_id = sandbox_id
        self.tunnel_urls = {}
        self.vnc_url = None
        self.mode = "container"

    async def terminate(self) -> None:
        pass


class _StubProvider(SandboxProvider):
    """Stub provider that records calls and returns a configurable sandbox or raises.

    `raises` applies to both create_sandbox and get_sandbox unless an
    independent `get_sandbox_raises` is provided. Pass `get_sandbox_raises=...`
    to simulate a provider that succeeds at create but fails at get (or vice
    versa via `raises=...` + `get_sandbox_raises=None`).
    """

    _UNSET: object = object()

    def __init__(
        self,
        *,
        raises: Exception | None = None,
        sandbox: Sandbox | None = None,
        get_sandbox_raises: Exception | None | object = _UNSET,
    ):
        self._raises = raises
        self._sandbox = sandbox or _FakeSandbox()
        self._get_sandbox_raises = raises if get_sandbox_raises is self._UNSET else get_sandbox_raises
        self.create_sandbox_calls = 0
        self.get_sandbox_calls = 0

    async def create_sandbox(self, **kwargs) -> Sandbox:
        self.create_sandbox_calls += 1
        if self._raises:
            raise self._raises
        return self._sandbox

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        self.get_sandbox_calls += 1
        if self._get_sandbox_raises:
            raise self._get_sandbox_raises
        return self._sandbox


@pytest.mark.parametrize("name", sorted(_BUILTIN_SANDBOX_PROVIDERS))
def test_each_builtin_impl_string_loads(name):
    # A stale string only fails when that backend is built: the registry skips it at DEBUG.
    assert issubclass(load_impl(_BUILTIN_SANDBOX_PROVIDERS[name], SandboxProvider), SandboxProvider)


def test_modal_sandbox_instance_mode_is_container():
    fake_sb = type("FakeSb", (), {"object_id": "modal-1"})()
    sandbox = ModalSandbox(fake_sb, tunnel_urls={})
    assert sandbox.mode == "container"


@pytest.mark.asyncio
async def test_default_create_container_mutates_mode_to_container():
    """The base SandboxProvider.create_container default impl should mutate the
    returned VmSandbox's mode to 'container' once docker is running on it."""

    class _FakeVm(VmSandbox):
        type = "fake-vm"

        def __init__(self):
            self.sandbox_id = "vm-fake"
            self.tunnel_urls = {}
            self.vnc_url = None
            self.mode = "vm"
            self.scripts: list[str] = []

        async def terminate(self) -> None:
            pass

        async def exec_script(self, script: str) -> str:
            self.scripts.append(script)
            return ""

    fake_vm = _FakeVm()

    class _VmStyleProvider(SandboxProvider):
        async def create_vm(self, **kwargs):
            return fake_vm

        async def create_sandbox(self, **kwargs):
            raise NotImplementedError

    provider = _VmStyleProvider()
    result = await provider.create_container(
        image_name="nginx:latest", port=8080, env={}, cpu=1.0, memory=512, disk_size_gb=10, timeout=300,
    )
    assert result is fake_vm
    assert result.mode == "container"


@pytest.mark.asyncio
async def test_create_container_mints_the_registry_login_off_the_event_loop():
    """A remote registry's login is a network round trip, an IAM token exchange for one."""
    minted_on_loop = []

    class _Registry(ImageStore):
        def image_ref(self, repository, tag):
            return f"registry.example/{repository}:{tag}"

        def auth(self, ref):
            try:
                asyncio.get_running_loop()
                minted_on_loop.append(True)
            except RuntimeError:
                minted_on_loop.append(False)
            return RegistryAuth("registry.example", "user", "token")

    class _Vm(VmSandbox):
        type = "fake-vm"

        def __init__(self):
            self.sandbox_id, self.tunnel_urls, self.vnc_url, self.mode = "vm-fake", {}, None, "vm"
            self.scripts: list[str] = []

        async def terminate(self) -> None:
            pass

        async def exec_script(self, script: str) -> str:
            self.scripts.append(script)
            return ""

    vm = _Vm()

    class _Provider(SandboxProvider):
        async def create_vm(self, **kwargs):
            return vm

        async def create_sandbox(self, **kwargs):
            raise NotImplementedError

    get_config().set_image_store(_Registry())
    await _Provider().create_container(image_name="registry.example/app:v1", port=8080, env={})
    assert minted_on_loop == [False]
    assert any("docker login" in script for script in vm.scripts)


@pytest.mark.asyncio
async def test_create_container_publishes_the_mapped_host_port():
    """Publish the mapped host port, not the container port, so a reader polling
    tunnel_urls[{port}] (= localhost:{host_port}) reaches the running container."""

    class _RemappingVm(VmSandbox):
        type = "fake-vm"

        def __init__(self):
            self.sandbox_id = "vm-fake"
            self.tunnel_urls = {8000: "http://localhost:57863"}
            self.vnc_url = None
            self.mode = "vm"
            self.scripts: list[str] = []

        def host_port(self, port: int) -> int:
            return 57863 if port == 8000 else port

        async def terminate(self) -> None:
            pass

        async def exec_script(self, script: str) -> str:
            self.scripts.append(script)
            return ""

    fake_vm = _RemappingVm()

    class _VmStyleProvider(SandboxProvider):
        async def create_vm(self, **kwargs):
            return fake_vm

        async def create_sandbox(self, **kwargs):
            raise NotImplementedError

    await _VmStyleProvider().create_container(
        image_name="img:v1", port=8000, env={}, cpu=1.0, memory=512, disk_size_gb=10, timeout=300,
    )
    run_cmd = next(s for s in fake_vm.scripts if "docker run" in s)
    assert "-p 57863:8000" in run_cmd          # host side = the mapped port tunnel_urls advertises
    assert "-p 8000:8000" not in run_cmd


@pytest.mark.asyncio
async def test_chain_create_sandbox_returns_first_success():
    second_sandbox = _FakeSandbox("second")
    first = _StubProvider(raises=RuntimeError("first failed"))
    second = _StubProvider(sandbox=second_sandbox)
    third = _StubProvider()

    chain = ChainedSandboxProvider([first, second, third])
    result = await chain.create_sandbox(
        image_name="img", port=8000, env={}, cpu=1.0, memory=512, disk_size_gb=10, timeout=300,
    )

    assert result is second_sandbox
    assert first.create_sandbox_calls == 1
    assert second.create_sandbox_calls == 1
    assert third.create_sandbox_calls == 0


@pytest.mark.asyncio
async def test_chain_create_sandbox_all_fail_raises_with_all_names():
    first = _StubProvider(raises=RuntimeError("first boom"))
    second = _StubProvider(raises=ValueError("second boom"))
    chain = ChainedSandboxProvider([first, second])

    with pytest.raises(RuntimeError) as exc_info:
        await chain.create_sandbox(
            image_name="img", port=8000, env={}, cpu=1.0, memory=512, disk_size_gb=10, timeout=300,
        )
    msg = str(exc_info.value)
    assert "_StubProvider" in msg
    assert "first boom" in msg
    assert "second boom" in msg


@pytest.mark.asyncio
async def test_chain_get_sandbox_returns_first_success():
    second_sandbox = _FakeSandbox("second")
    first = _StubProvider(get_sandbox_raises=RuntimeError("first failed"))
    second = _StubProvider(sandbox=second_sandbox)
    third = _StubProvider()

    chain = ChainedSandboxProvider([first, second, third])
    result = await chain.get_sandbox("sb-id")

    assert result is second_sandbox
    assert first.get_sandbox_calls == 1
    assert second.get_sandbox_calls == 1
    assert third.get_sandbox_calls == 0


@pytest.mark.asyncio
async def test_chain_get_sandbox_skips_not_implemented():
    """Providers that don't implement get_sandbox should be silently skipped."""
    expected = _FakeSandbox("via-second")

    class _NoGetSandbox(SandboxProvider):
        async def create_sandbox(self, **kwargs):
            raise NotImplementedError

    second = _StubProvider(sandbox=expected)
    chain = ChainedSandboxProvider([_NoGetSandbox(), second])

    result = await chain.get_sandbox("sb-id")
    assert result is expected
    assert second.get_sandbox_calls == 1


@pytest.mark.asyncio
async def test_chain_get_sandbox_all_fail_raises_with_all_names():
    first = _StubProvider(get_sandbox_raises=RuntimeError("first boom"))
    second = _StubProvider(get_sandbox_raises=ValueError("second boom"))
    chain = ChainedSandboxProvider([first, second])

    with pytest.raises(RuntimeError) as exc_info:
        await chain.get_sandbox("sb-id")
    msg = str(exc_info.value)
    assert "_StubProvider" in msg
    assert "first boom" in msg
    assert "second boom" in msg
    assert "sb-id" in msg


def test_chain_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        ChainedSandboxProvider([])


class _RecordingVm(VmSandbox):
    """VmSandbox that records the scripts passed to exec_script so the in-VM
    download commands can be inspected."""

    type = "recording-vm"

    def __init__(self):
        self.sandbox_id = "vm-rec"
        self.tunnel_urls = {}
        self.vnc_url = None
        self.mode = "vm"
        self.scripts: list[str] = []

    async def terminate(self) -> None:
        pass

    async def exec_script(self, script: str, *, max_retries: int = 0) -> str:
        self.scripts.append(script)
        return ""

    async def exec_with_output(self, *args):
        self.scripts.append(script_run(args))
        return 0, "", ""


@pytest.mark.asyncio
async def test_load_object_file_curl_retries_dns_failures():
    """The in-VM S3 download must use --retry-all-errors so a transient
    `curl: (6) Could not resolve host` is retried locally (curl does not retry
    exit 6 by default, and --retry-connrefused does not cover it)."""
    from agent_env.store import set_object_store
    from agent_env.config import reset_config

    class _Signing:
        def signed_get_url(self, url, expires_in=3600):
            return f"https://signed/{url.rsplit('/', 1)[-1]}"

    set_object_store(_Signing())
    try:
        vm = _RecordingVm()
        await vm.load_object_file("s3://artifact-bucket/foo/bar.tar", "/tmp/bar.tar")
    finally:
        reset_config()

    assert len(vm.scripts) == 1
    script = vm.scripts[0]
    assert "curl" in script
    assert "--retry-all-errors" in script
    assert "--retry " in script  # explicit retry count, not a bare flag


@pytest.mark.asyncio
async def test_write_file_from_url_curl_retries():
    """HTTP(S) downloads into the VM get the same resilient curl flags."""
    vm = _RecordingVm()
    await vm.write_file_from_url("https://example.com/x.bin", "/data/x.bin")

    download = next(s for s in vm.scripts if "curl" in s)
    assert "--retry-all-errors" in download


def test_factory_modal():
    assert isinstance(build_sandbox_provider(ModalSandbox.type), ModalSandboxProvider)


def test_factory_local():
    assert isinstance(build_sandbox_provider(LocalSandbox.type), LocalSandboxProvider)


def test_factory_chain():
    p = build_sandbox_provider(f"{ModalVmSandbox.type},{ModalSandbox.type}")
    assert isinstance(p, ChainedSandboxProvider)
    assert len(p._providers) == 2
    assert isinstance(p._providers[0], ModalVmSandboxProvider)
    assert isinstance(p._providers[1], ModalSandboxProvider)


def test_factory_chain_with_local():
    p = build_sandbox_provider(f"{ModalSandbox.type},{LocalSandbox.type}")
    assert isinstance(p, ChainedSandboxProvider)
    assert len(p._providers) == 2
    assert isinstance(p._providers[0], ModalSandboxProvider)
    assert isinstance(p._providers[1], LocalSandboxProvider)


def test_factory_modal_vm():
    assert isinstance(build_sandbox_provider(ModalVmSandbox.type), ModalVmSandboxProvider)


def test_modal_vm_is_vm_mode_and_not_container_provider():
    # The whole VM-mode routing hinges on this: ModalVmSandbox is a VmSandbox (mode "vm"),
    # and the provider is NOT a ModalSandboxProvider, so EnvironmentGatewayProvider.create_gateway
    # routes it to _deploy_via_vm (not _deploy_via_containers) and never forces i6pn.
    assert issubclass(ModalVmSandbox, VmSandbox)
    provider = build_sandbox_provider(ModalVmSandbox.type)
    assert not isinstance(provider, ModalSandboxProvider)


def test_factory_chain_with_modal_vm():
    p = build_sandbox_provider(f"{LocalSandbox.type},{ModalVmSandbox.type}")
    assert isinstance(p, ChainedSandboxProvider)
    assert len(p._providers) == 2
    assert isinstance(p._providers[0], LocalSandboxProvider)
    assert isinstance(p._providers[1], ModalVmSandboxProvider)


def test_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown sandbox backend"):
        build_sandbox_provider("notreal")


def test_factory_empty_raises():
    with pytest.raises(ValueError, match="must include at least one backend"):
        build_sandbox_provider("")


def test_cpu_floor_differs_between_container_and_vm_backends():
    """Modal's container minimum is 0.125; its VM runtime keeps half a core."""
    import inspect

    from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider

    def cpu_default(cls, method):
        return inspect.signature(getattr(cls, method)).parameters["cpu"].default

    assert cpu_default(ModalSandboxProvider, "create_sandbox") == 0.125
    assert cpu_default(ModalSandboxProvider, "create_container") == 0.125
    assert cpu_default(ModalVmSandboxProvider, "create_sandbox") == 0.5


class _PullingVm(VmSandbox):
    def __init__(self, first_pull_error: str | None):
        self.scripts: list[str] = []
        self._first_pull_error = first_pull_error

    async def terminate(self) -> None:
        pass

    async def exec_script(self, script: str, *, max_retries: int = 0) -> str:
        self.scripts.append(script)
        if self._first_pull_error and len(self.scripts) == 1:
            raise RuntimeError(f"Script failed (exit 1):\nstdout: \nstderr: {self._first_pull_error}")
        return ""


@pytest.mark.asyncio
async def test_an_image_with_nothing_for_this_platform_is_pulled_for_amd64():
    """An Apple Silicon host has no arm64 variant of an amd64-only image; its Docker runs the amd64 one emulated."""
    sandbox = _PullingVm("Error response from daemon: no matching manifest for linux/arm64/v8 in the manifest list entries")

    await _pull(sandbox, "registry/agent:v1")

    assert sandbox.scripts == ["docker pull registry/agent:v1", "docker pull --platform linux/amd64 registry/agent:v1"]


@pytest.mark.asyncio
async def test_any_other_pull_failure_is_raised_as_it_was():
    sandbox = _PullingVm("Error response from daemon: pull access denied for registry/agent")

    with pytest.raises(RuntimeError, match="pull access denied"):
        await _pull(sandbox, "registry/agent:v1")
    assert sandbox.scripts == ["docker pull registry/agent:v1"]
