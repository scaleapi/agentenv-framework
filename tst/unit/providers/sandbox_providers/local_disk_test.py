from pathlib import Path

import pytest

from agent_env.providers.sandbox_providers.local_docker import disk
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider


@pytest.mark.parametrize("platform,expected", [
    ("linux/amd64", frozenset({"linux/amd64", "windows/amd64"})),
    ("linux/arm64", frozenset({"linux/arm64"})),
])
def test_docker_disk_platforms_do_not_require_client_device_permissions(monkeypatch, platform, expected):
    monkeypatch.setattr(Path, "is_char_device", lambda path: str(path) == disk.KVM)
    monkeypatch.setattr(disk.os, "access", lambda *args: False)
    monkeypatch.setattr(disk, "host_platform", lambda: platform)
    assert LocalSandboxProvider.disk_image_platforms() == expected


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_regular_kvm_path_does_not_advertise_disks(monkeypatch, tmp_path, exists):
    device = tmp_path / "kvm"
    if exists:
        device.touch()
    monkeypatch.setattr(disk, "KVM", str(device))
    assert LocalSandboxProvider.disk_image_platforms() == frozenset()
