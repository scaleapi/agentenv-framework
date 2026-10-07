from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_env.providers.sandbox_providers.sandbox import Sandbox, VmSandbox, stage_files_into_container


_METHODS = (
    (Sandbox, "write_file_from_s3", "write_file_from_object"),
    (VmSandbox, "write_file_from_s3", "write_file_from_object"),
    (VmSandbox, "load_s3_file", "load_object_file"),
)


async def _terminate(self):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("base, legacy, neutral", _METHODS)
async def test_new_calls_reach_inherited_legacy_provider_overrides(base, legacy, neutral):
    calls = []

    async def implementation(self, s3_url, destination_path):
        calls.append((s3_url, destination_path))

    provider = type("LegacySandbox", (base,), {legacy: implementation, "terminate": _terminate})
    child = type("InheritedSandbox", (provider,), {})

    await getattr(child(), neutral)(object_url="file:///objects/input", destination_path="/app/input")

    assert calls == [("file:///objects/input", "/app/input")]


@pytest.mark.asyncio
@pytest.mark.parametrize("base, legacy, neutral", _METHODS)
async def test_old_calls_reach_modern_overrides_of_legacy_providers(base, legacy, neutral):
    calls = []

    async def old(self, s3_url, destination_path):
        raise AssertionError("the parent provider was bypassed by the modern override")

    async def new(self, object_url, destination_path):
        calls.append((object_url, destination_path))

    parent = type("LegacySandbox", (base,), {legacy: old, "terminate": _terminate})
    child = type("ModernSandbox", (parent,), {neutral: new})

    with pytest.warns(DeprecationWarning, match=f"use {neutral}"):
        await getattr(child(), legacy)(s3_url="file:///objects/input", destination_path="/app/input")

    assert calls == [("file:///objects/input", "/app/input")]


@pytest.mark.asyncio
async def test_artifact_staging_uses_an_older_container_provider(tmp_path):
    class LegacySandbox(Sandbox):
        async def terminate(self):
            pass

        async def exec(self, *command):
            Path(command[-1]).mkdir(parents=True, exist_ok=True)

        async def write_file_from_s3(self, s3_url, destination_path):
            Path(destination_path).write_bytes(b"artifact payload")

    artifacts = {"input.json": SimpleNamespace(object_url="file:///objects/input")}

    loaded = await stage_files_into_container(LegacySandbox(), artifacts, str(tmp_path))

    assert loaded == {"input.json": str(tmp_path / "input.json")}
    assert (tmp_path / "input.json").read_bytes() == b"artifact payload"


@pytest.mark.asyncio
async def test_legacy_vm_overrides_can_delegate_to_super(monkeypatch):
    calls = []

    class LegacyVm(VmSandbox):
        async def terminate(self):
            pass

        async def load_s3_file(self, s3_url, destination_path):
            calls.append("load override")
            await super().load_s3_file(s3_url, destination_path)

        async def write_file_from_s3(self, s3_url, destination_path):
            calls.append("write override")
            await super().write_file_from_s3(s3_url, destination_path)

        async def _download_object_to_vm(self, object_url, vm_path):
            calls.append((object_url, vm_path))

        async def _copy_into_container(self, vm_path, destination_path):
            calls.append((vm_path, destination_path))

        async def _remove_vm_temp_file(self, *vm_paths):
            calls.append(vm_paths)

    monkeypatch.setattr(LegacyVm, "_staging_path", staticmethod(lambda kind, dest: "/tmp/staged"))

    with pytest.warns(DeprecationWarning):
        await LegacyVm().write_file_from_object("file:///objects/input", "/app/input")

    assert calls == [
        "write override", "load override", ("file:///objects/input", "/tmp/staged"),
        ("/tmp/staged", "/app/input"), ("/tmp/staged",),
    ]
