"""A VM file reaches the object store: uploaded by the VM to a presigned URL, or copied off it when the store signs
none."""

import asyncio
import base64
import os
import re

import pytest

from agent_env.providers.sandbox_providers import sandbox as sandbox_module
from agent_env.providers.sandbox_providers.sandbox import read_vm_file, upload_vm_file
from agent_env.store.object_store.local import LocalFilesystemObjectStore


class _ShellVm:
    """Runs each script in a real shell on this machine, as the VM host would."""

    def __init__(self) -> None:
        self.scripts: list[str] = []

    async def exec_script(self, script: str) -> str:
        self.scripts.append(script)
        process = await asyncio.create_subprocess_shell(
            script, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await process.communicate()
        if process.returncode:
            raise RuntimeError(f"Script failed (exit {process.returncode}): {err.decode()}")
        return out.decode()


class _ShortRangeVm(_ShellVm):
    """Drops the last byte of every range after the first, as a truncating exec channel would."""

    async def exec_script(self, script: str) -> str:
        out = await super().exec_script(script)
        if (read := re.match(r"tail -c \+(\d+) ", script)) and read[1] != "1":
            return base64.b64encode(base64.b64decode(out)[:-1]).decode()
        return out


@pytest.fixture
def small_ranges(monkeypatch):
    monkeypatch.setattr(sandbox_module, "_READ_RANGE_BYTES", 1000)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [0, 999, 1000, 10_500], ids=["empty", "under-a-range", "one-range", "many-ranges"])
async def test_a_vm_file_is_copied_off_byte_for_byte(small_ranges, tmp_path, size):
    source, copy = tmp_path / "on the vm.bin", tmp_path / "copy.bin"
    source.write_bytes(os.urandom(size))

    await read_vm_file(_ShellVm(), str(source), str(copy))

    assert copy.read_bytes() == source.read_bytes()


@pytest.mark.asyncio
async def test_a_short_range_fails_the_copy(small_ranges, tmp_path):
    source = tmp_path / "image.tar.gz"
    source.write_bytes(os.urandom(5000))

    with pytest.raises(RuntimeError, match="refusing a truncated copy"):
        await read_vm_file(_ShortRangeVm(), str(source), str(tmp_path / "copy.bin"))


class _RecordingVm:
    def __init__(self) -> None:
        self.scripts: list[str] = []

    async def exec_script(self, script: str) -> str:
        self.scripts.append(script)
        return ""


class _SigningStore:
    def signed_put_url(self, object_url: str, expires_in: int = 3600) -> str:
        return f"https://put.example/{object_url.rsplit('/', 1)[-1]}?sig=1"

    def put_file_at(self, object_url, file_path, content_type=None):  # pragma: no cover - must not be reached
        raise AssertionError("a store that signs uploads is uploaded to by the VM")


@pytest.mark.asyncio
async def test_a_store_that_signs_uploads_is_uploaded_to_by_the_vm(tmp_path):
    vm = _RecordingVm()

    await upload_vm_file(vm, "/tmp/image.tar.gz", _SigningStore(), "s3://bucket/builds/image.tar.gz")

    assert vm.scripts == ['curl -fsSL -X PUT --upload-file /tmp/image.tar.gz "https://put.example/image.tar.gz?sig=1"']


@pytest.mark.asyncio
async def test_a_store_that_signs_nothing_gets_the_file_copied_off_the_vm(small_ranges, tmp_path):
    source = tmp_path / "vm" / "snapshot-image.tar.gz"
    source.parent.mkdir()
    source.write_bytes(os.urandom(4321))
    store = LocalFilesystemObjectStore(str(tmp_path / "objects"))
    object_url = store.object_url("env-snapshots/env/universe/image.tar.gz")
    vm = _ShellVm()

    await upload_vm_file(vm, str(source), store, object_url)

    assert store.get(object_url) == source.read_bytes()
    assert not any("curl" in script for script in vm.scripts)
