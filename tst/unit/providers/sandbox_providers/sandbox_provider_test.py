"""Unit tests for SandboxProvider, ChainedSandboxProvider, build_sandbox_provider."""

from __future__ import annotations

import asyncio
import logging

import pytest

import agent_env.providers.sandbox_providers.chained_sandbox_provider as chained_module

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.config import get_config
from agent_env.config.loader import load_impl
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider, skip_refusing
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox, ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox, ModalVmSandboxProvider
from agent_env.providers.sandbox_providers import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.sandbox import Sandbox, VmSandbox, _pull
from agent_env.providers.sandbox_providers.sandbox_provider import _BUILTIN_SANDBOX_PROVIDERS, SandboxProvider, build_sandbox_provider
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
    if name == "smol_vm":
        pytest.importorskip("smol")
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
async def test_a_create_container_cancelled_during_the_pull_terminates_the_vm():
    pulling = asyncio.Event()

    class _HangingVm(VmSandbox):
        type = "fake-vm"

        def __init__(self):
            self.sandbox_id = "vm-fake"
            self.tunnel_urls = {}
            self.vnc_url = None
            self.mode = "vm"
            self.terminated = False

        async def terminate(self) -> None:
            self.terminated = True

        async def exec_script(self, script: str) -> str:
            pulling.set()
            await asyncio.Event().wait()
            return ""

    vm = _HangingVm()

    class _VmStyleProvider(SandboxProvider):
        async def create_vm(self, **kwargs):
            return vm

        async def create_sandbox(self, **kwargs):
            raise NotImplementedError

    task = asyncio.ensure_future(_VmStyleProvider().create_container(image_name="nginx:latest", port=8080, env={}))
    await pulling.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert vm.terminated


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


def test_skip_refusing_leaves_a_provider_none_of_whose_links_it_refuses():
    chain = ChainedSandboxProvider([_StubProvider(), _StubProvider()])

    assert skip_refusing(chain, lambda link: None, what="it") is chain


def test_skip_refusing_chains_the_links_it_doesnt_refuse_noting_each_it_skips(caplog):
    first, second, third = _StubProvider(), _StubProvider(), _StubProvider()

    with caplog.at_level(logging.INFO, logger=chained_module.__name__):
        running = skip_refusing(ChainedSandboxProvider([first, second, third]),
                                lambda link: "which is too far" if link is not third else None, what="it")

    assert running.links == (third,)
    warned, noted = [r for r in caplog.records if "sandbox_provider_skipped" in r.getMessage()]
    assert (warned.levelno, noted.levelno) == (logging.WARNING, logging.INFO)  # the first, preferred, is warned of
    assert warned.getMessage() == "sandbox_provider_skipped provider=_StubProvider: which is too far"


@pytest.mark.parametrize("provider, refusal", [
    (_StubProvider(), "with _StubProvider, which is too far"),
    (ChainedSandboxProvider([_StubProvider(), _StubProvider()]),
     "with _StubProvider, which is too far; nor with _StubProvider, which is too far"),
], ids=["one", "chain"])
def test_skip_refusing_names_each_refusal_when_it_refuses_every_link(provider, refusal):
    with pytest.raises(ValueError, match=f"^Can't deploy it {refusal}$"):
        skip_refusing(provider, lambda link: "which is too far", what="it")


class _Vm(VmSandbox):
    """A VM sandbox of ``platform``, recording whether it was terminated."""

    type = "fake"

    def __init__(self, sandbox_id: str, platform: str):
        self.sandbox_id, self.tunnel_urls, self.vnc_url = sandbox_id, {}, None
        self.mode, self.platform, self.terminated = "vm", platform, False

    async def terminate(self) -> None:
        self.terminated = True


@pytest.mark.asyncio
async def test_a_chain_closes_a_vm_that_cant_run_the_image_and_tries_the_next_provider(caplog):
    arm64 = DockerImageArtifact(id="img", version=1, description="d", image_name="img:v1", platform="linux/arm64",
                                tar_gz_object_url="s3://bucket/img.tar.gz")
    amd, arm = _Vm("amd", "linux/amd64"), _Vm("arm", "linux/arm64")
    chain = ChainedSandboxProvider([_StubProvider(sandbox=amd), _StubProvider(sandbox=arm)])
    await chain.prepare_image(arm64)

    assert await chain.create_sandbox(image_name="img:v1", port=8000, env={}) is arm
    assert amd.terminated and not arm.terminated
    assert "gave a VM that can't load the image: 'img' v1 is a linux/arm64 image" in caplog.text


@pytest.mark.asyncio
async def test_a_chain_none_of_whose_vms_runs_the_image_names_each_refusal():
    arm64 = DockerImageArtifact(id="img", version=1, description="d", image_name="img:v1", platform="linux/arm64",
                                tar_gz_object_url="s3://bucket/img.tar.gz")
    chain = ChainedSandboxProvider([_StubProvider(sandbox=_Vm("one", "linux/amd64")),
                                    _StubProvider(sandbox=_Vm("two", "linux/amd64"))])
    await chain.prepare_image(arm64)

    with pytest.raises(RuntimeError, match="All 2 providers failed: _StubProvider: ValueError.*linux/arm64 image.*; "
                                           "_StubProvider: ValueError"):
        await chain.create_sandbox(image_name="img:v1", port=8000, env={})


_CONTEXT_ONLY = DockerImageArtifact(id="solver-image", version=2, description="d", image_name="local/solver-0123456789ab:v2",
                                    build_context_object_url="s3://bucket/ctx.tar.gz", dockerfile_path="Dockerfile")


class _PreparingProvider(_StubProvider):
    """A stub whose prepare_image takes ``prepare_seconds``, or raises ``prepare_raises``, recording each image."""

    def __init__(self, *, prepare_seconds: float = 0, prepare_raises: Exception | None = None, **kwargs):
        super().__init__(**kwargs)
        self.prepare_seconds, self.prepare_raises, self.prepared, self.attributions = (
            prepare_seconds, prepare_raises, [], [])

    async def prepare_image(self, image, *, attribution=None):
        self.prepared.append(image.image_name)
        self.attributions.append(attribution)
        await asyncio.sleep(self.prepare_seconds)
        if self.prepare_raises:
            raise self.prepare_raises


async def _chain_create(chain: ChainedSandboxProvider) -> Sandbox:
    await chain.prepare_image(_CONTEXT_ONLY)
    return await chain.create_sandbox(image_name=_CONTEXT_ONLY.image_name, port=8000, env={})


@pytest.mark.asyncio
async def test_chain_prepares_a_provider_outside_the_deadline_on_creating_its_sandbox(monkeypatch):
    monkeypatch.setattr(chained_module, "_PROVISION_DEADLINE_SECONDS", 0.05)
    builds = _PreparingProvider(prepare_seconds=0.2)

    assert await _chain_create(ChainedSandboxProvider([builds])) is builds._sandbox
    assert builds.prepared == [_CONTEXT_ONLY.image_name]


@pytest.mark.asyncio
async def test_chain_prepares_only_the_providers_it_tries_and_falls_through_a_failed_prepare():
    fails = _PreparingProvider(prepare_raises=RuntimeError("the build failed"))
    second = _PreparingProvider(sandbox=_FakeSandbox("second"))
    third = _PreparingProvider()

    assert (await _chain_create(ChainedSandboxProvider([fails, second, third]))).sandbox_id == "second"
    assert (fails.create_sandbox_calls, fails.prepared) == (0, [_CONTEXT_ONLY.image_name])
    assert (second.prepared, third.prepared) == ([_CONTEXT_ONLY.image_name], [])


@pytest.mark.asyncio
async def test_chain_prepares_each_provider_for_the_deploy_its_sandbox_is_for():
    provider = _PreparingProvider()
    chain = ChainedSandboxProvider([provider])

    await chain.prepare_image(_CONTEXT_ONLY)
    await chain.create_sandbox(image_name=_CONTEXT_ONLY.image_name, port=8000, env={},
                               attribution={"project_id": "0123456789abcdef01234567"})

    assert provider.attributions == [{"project_id": "0123456789abcdef01234567"}]


@pytest.mark.asyncio
async def test_chain_prepares_nothing_for_an_image_it_wasnt_told_of():
    provider = _PreparingProvider()

    await ChainedSandboxProvider([provider]).create_sandbox(image_name="registry.example/solver:v2", port=8000, env={})
    assert provider.prepared == []


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
    """Its first ``failures`` pulls fail with ``pull_error``."""

    def __init__(self, pull_error: str | None, failures: int = 1):
        self.sandbox_id = "vm-test"
        self.scripts: list[str] = []
        self._pull_error = pull_error
        self._failures = failures

    async def terminate(self) -> None:
        pass

    async def exec_script(self, script: str, *, max_retries: int = 0) -> str:
        self.scripts.append(script)
        if self._pull_error and len(self.scripts) <= self._failures:
            raise RuntimeError(f"Script failed (exit 1):\nstdout: \nstderr: {self._pull_error}")
        return ""


@pytest.fixture
def no_pause(monkeypatch):
    monkeypatch.setattr(sandbox_module, "_REGISTRY_RETRY_SECONDS", 0)


@pytest.mark.asyncio
async def test_an_image_with_nothing_for_this_platform_is_pulled_for_amd64():
    """An Apple Silicon host has no arm64 variant of an amd64-only image; its Docker runs the amd64 one emulated."""
    sandbox = _PullingVm("Error response from daemon: no matching manifest for linux/arm64/v8 in the manifest list entries")

    await _pull(sandbox, "registry/agent:v1")

    assert sandbox.scripts == ["docker pull registry/agent:v1", "docker pull --platform linux/amd64 registry/agent:v1"]


@pytest.mark.asyncio
async def test_any_other_pull_failure_runs_once_more(no_pause):
    """As a registry's 404 for a blob it holds, which passes."""
    sandbox = _PullingVm("Error response from daemon: failed to copy: httpReadSeeker: failed open: not found")

    await _pull(sandbox, "registry/agent:v1")

    assert sandbox.scripts == ["docker pull registry/agent:v1"] * 2


@pytest.mark.asyncio
async def test_a_pull_that_fails_again_is_raised_as_it_was(no_pause):
    sandbox = _PullingVm("Error response from daemon: pull access denied for registry/agent", failures=2)

    with pytest.raises(RuntimeError, match="pull access denied"):
        await _pull(sandbox, "registry/agent:v1")
    assert sandbox.scripts == ["docker pull registry/agent:v1"] * 2
