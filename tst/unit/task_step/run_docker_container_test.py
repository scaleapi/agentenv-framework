"""Unit tests for RunDockerContainerTaskStep's keep_alive_with_base_command flag."""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_env.config import set_object_store
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.run_docker_container import (
    RunDockerContainerTaskStep as Step,
)
from tst.unit.store.fakes import FakeObjectStore


def _step(**kw):
    base = dict(
        id="s", version=None, sandbox_name="h",
        docker_context_artifact_id="u", docker_context_artifact_version=1,
        container_name="c",
    )
    base.update(kw)
    return Step(**base)


def test_keep_alive_and_command_override_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        _step(command_override="sleep infinity", keep_alive_with_base_command=True)


def test_keep_alive_roundtrips():
    step = _step(keep_alive_with_base_command=True)
    restored = Step.from_dict(step.to_dict())
    assert restored.keep_alive_with_base_command is True


def test_default_keep_alive_false():
    assert _step().keep_alive_with_base_command is False


@pytest.mark.parametrize("entrypoint,cmd,expected", [
    (["/bin/sh", "-c"], ["postgres -D /data"], ["/bin/sh", "-c", "postgres -D /data"]),
    (None, ["node", "server.js"], ["node", "server.js"]),
    (["/entry.sh"], None, ["/entry.sh"]),
    (None, None, []),
])
@pytest.mark.asyncio
async def test_inspect_base_command_combines_entrypoint_and_cmd(entrypoint, cmd, expected):
    import json

    class _FakeSandbox:
        async def exec_script(self, _script):
            ep = json.dumps(entrypoint) if entrypoint is not None else "null"
            cm = json.dumps(cmd) if cmd is not None else "null"
            return f"{ep}|{cm}\n"

    got = await _step()._inspect_base_command(_FakeSandbox(), "img:latest")
    assert got == expected


def test_network_and_ready_command_roundtrip():
    step = Step(
        id="s", version=None, sandbox_name="vm", docker_context_artifact_id="ctx",
        container_name="pg", network="task-net", ready_command="pg_isready -U agentenv",
    )
    restored = Step.from_dict(step.to_dict())
    assert restored.network == "task-net"
    assert restored.ready_command == "pg_isready -U agentenv"


def test_network_defaults_none():
    step = Step(
        id="s", version=None, sandbox_name="vm", docker_context_artifact_id="ctx",
    )
    assert step.network is None and step.ready_command is None
    assert Step.from_dict(step.to_dict()).network is None


def test_extra_networks_roundtrip():
    step = _step(network="dmz", extra_networks=["corp", "restricted"])
    restored = Step.from_dict(step.to_dict())
    assert restored.network == "dmz"
    assert restored.extra_networks == ["corp", "restricted"]


def test_extra_networks_defaults_empty():
    step = _step()
    assert step.extra_networks == []
    # None is normalized to an empty list on the way in and back out
    assert Step.from_dict(_step(extra_networks=None).to_dict()).extra_networks == []


def test_extra_networks_rejects_netns_modes():
    # host / none / container:<name> can't be attached with `docker network connect`
    for bad in ("host", "none", "container:estate"):
        with pytest.raises(ValueError, match="extra_networks"):
            _step(extra_networks=[bad])


def test_extra_networks_allows_bridge_and_user_bridges():
    step = _step(extra_networks=["dmz", "bridge"])
    assert step.extra_networks == ["dmz", "bridge"]


def test_device_and_cap_fields_roundtrip():
    step = _step(devices=["/dev/kvm", "/dev/net/tun"], cap_add=["NET_ADMIN"],
                 privileged=True, volumes=["/data:/storage"], shm_size="2g")
    restored = Step.from_dict(step.to_dict())
    assert restored.devices == ["/dev/kvm", "/dev/net/tun"]
    assert restored.cap_add == ["NET_ADMIN"]
    assert restored.privileged is True
    assert restored.volumes == ["/data:/storage"]
    assert restored.shm_size == "2g"


def test_device_fields_default_empty():
    step = _step()
    assert step.devices == [] and step.cap_add == [] and step.volumes == []
    assert step.privileged is False and step.shm_size is None
    # None normalizes to empty lists through a to_dict/from_dict roundtrip
    restored = Step.from_dict(_step(devices=None, cap_add=None, volumes=None).to_dict())
    assert restored.devices == [] and restored.cap_add == [] and restored.volumes == []


class _StagingSandbox:
    def __init__(self):
        self.loaded: list[tuple[str, str]] = []
        self.scripts: list[str] = []

    async def load_object_file(self, url, destination_path):
        self.loaded.append((url, destination_path))

    async def exec_script(self, script):
        self.scripts.append(script)
        return ""


@pytest.fixture
def fake_store():
    store = FakeObjectStore()
    set_object_store(store)
    return store


@pytest.mark.asyncio
async def test_a_context_zip_in_the_configured_store_loads_through_the_store(fake_store):
    """Recognised by the store, not by an s3:// prefix; a '#' in the key is kept."""
    url = fake_store.object_url("contexts/build#2.zip")
    sandbox = _StagingSandbox()

    await Step._stage_zip_from_url(sandbox, url, "/work")

    assert sandbox.loaded == [(url, "/work/_context.zip")]
    assert len(sandbox.scripts) == 1 and sandbox.scripts[0].startswith("unzip ")


@pytest.mark.asyncio
async def test_a_context_zip_from_a_non_s3_store_unpacks_in_a_real_local_sandbox(fake_store, tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("Dockerfile", "FROM scratch\n")
    url = fake_store.put("contexts/build#2.zip", buffer.getvalue())

    await Step._stage_zip_from_url(LocalSandbox(work_dir=tmp_path / "work"), url, "/app/ctx")

    assert (tmp_path / "work" / "ctx" / "Dockerfile").read_text() == "FROM scratch\n"
    assert not (tmp_path / "work" / "ctx" / "_context.zip").exists()


@pytest.mark.asyncio
async def test_an_https_context_zip_is_fetched_with_curl(fake_store):
    sandbox = _StagingSandbox()

    await Step._stage_zip_from_url(sandbox, "https://example.test/ctx.zip?sig=1", "/work")

    assert sandbox.loaded == []
    assert sandbox.scripts[0].startswith("curl ") and "https://example.test/ctx.zip?sig=1" in sandbox.scripts[0]


@pytest.mark.asyncio
async def test_a_context_url_outside_the_stores_own_root_is_still_read_through_the_store(fake_store):
    """The store reaches whatever its backend can, another bucket say; callers don't pre-filter."""
    sandbox = _StagingSandbox()

    await Step._stage_zip_from_url(sandbox, "fake://shared/ctx.zip", "/work")

    assert sandbox.loaded == [("fake://shared/ctx.zip", "/work/_context.zip")]


@pytest.mark.asyncio
async def test_a_context_url_without_a_scheme_is_refused(fake_store):
    sandbox = _StagingSandbox()

    with pytest.raises(ValueError, match="must be an object store url or an http"):
        await Step._stage_zip_from_url(sandbox, "contexts/ctx.zip", "/work")
    assert sandbox.loaded == [] and sandbox.scripts == []


@pytest.mark.asyncio
async def test_a_store_context_that_is_not_a_zip_is_refused(fake_store):
    sandbox = _StagingSandbox()

    with pytest.raises(NotImplementedError, match="tar archives"):
        await Step._stage_zip_from_url(sandbox, fake_store.object_url("ctx.tar.gz"), "/work")
    with pytest.raises(ValueError, match="Unsupported archive format"):
        await Step._stage_zip_from_url(sandbox, fake_store.object_url("ctx.bin"), "/work")
    assert sandbox.loaded == []


class _RunSandbox:
    sandbox_id = "local-1"

    def __init__(self, extra_hosts: tuple[str, ...] = ()):
        self.scripts: list[str] = []
        self.extra_hosts = extra_hosts

    async def exec_script(self, script):
        self.scripts.append(script)
        return ""


@pytest.mark.asyncio
async def test_what_it_starts_is_labeled_with_its_sandbox(monkeypatch):
    """A local sandbox shares the laptop's Docker, so it finds what to remove at teardown by this label."""
    sandbox = _RunSandbox()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda _type: SimpleNamespace(get_sandbox=AsyncMock(return_value=sandbox)))
    monkeypatch.setattr(Step, "_stage_from_universe", AsyncMock())
    context = TaskStepContext(deployed_sandboxes=[
        DeployedSandbox(sandbox_name="h", sandbox_id="local-1", sandbox_mode="vm", sandbox_type="local"),
    ])

    await _step(network="task-net").execute(context)

    for command in ("docker build", "docker run -d", "docker network create"):
        (script,) = [s for s in sandbox.scripts if command in s]
        assert "--label agentenv.sandbox=local-1" in script, command


@pytest.mark.asyncio
@pytest.mark.parametrize("extra_hosts", [(), ("host.docker.internal:host-gateway",)])
async def test_what_it_starts_maps_the_sandboxs_extra_hosts(monkeypatch, extra_hosts):
    """On Linux a local sandbox maps host.docker.internal, so the container reaches URLs on this machine."""
    sandbox = _RunSandbox(extra_hosts)
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda _type: SimpleNamespace(get_sandbox=AsyncMock(return_value=sandbox)))
    monkeypatch.setattr(Step, "_stage_from_universe", AsyncMock())
    context = TaskStepContext(deployed_sandboxes=[
        DeployedSandbox(sandbox_name="h", sandbox_id="local-1", sandbox_mode="vm", sandbox_type="local"),
    ])

    await _step().execute(context)

    (run,) = [s for s in sandbox.scripts if "docker run -d" in s]
    assert ("--add-host host.docker.internal:host-gateway" in run) is bool(extra_hosts)
