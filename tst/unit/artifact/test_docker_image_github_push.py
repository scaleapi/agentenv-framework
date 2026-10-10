"""put_from_github pushes to the repository ``put`` would name, and reaches the stores off the event loop."""

from __future__ import annotations

import asyncio
import base64
import re
from types import SimpleNamespace

import pytest

import agent_env.artifact.artifacts.docker_image as docker_image
import agent_env.artifact.store as artifact_store
import agent_env.providers as providers
from agent_env.artifact import DockerImageArtifact
from agent_env.config import get_config
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.store import ImageStore, RegistryAuth, set_object_store
from agent_env.store.object_store.local import LocalFilesystemObjectStore


def _on_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class _ImageStore(ImageStore):
    def __init__(self) -> None:
        self.ensured: list[str] = []
        self.on_loop: list[bool] = []

    def image_ref(self, repository: str, tag: str) -> str:
        return f"registry.example/{repository}:{tag}"

    def ensure_repository(self, repository: str) -> None:
        self.on_loop.append(_on_loop())
        self.ensured.append(repository)

    def auth(self, ref: str) -> RegistryAuth:
        self.on_loop.append(_on_loop())
        return RegistryAuth("registry.example", "user", "token")


class _ObjectStore:
    def __init__(self) -> None:
        self.on_loop: list[bool] = []

    def object_url(self, key: str) -> str:
        return f"mem://bucket/{key}"

    def signed_put_url(self, object_url: str, expires_in: int = 3600) -> str:
        self.on_loop.append(_on_loop())
        return f"https://put.example/{object_url.rsplit('/', 1)[-1]}"


class _BuildVm:
    files = {"/tmp/image.tar.gz": b"the saved image", "/tmp/build-context.tar.gz": b"the build context"}

    def __init__(self) -> None:
        self.scripts: list[str] = []

    async def exec_script(self, script: str, **kwargs) -> str:
        self.scripts.append(script)
        if script.startswith("wc -c < "):
            return f"{len(self.files[script.removeprefix('wc -c < ')])}\n"
        if read := re.fullmatch(r"tail -c \+(\d+) (\S+) \| head -c (\d+) \| base64", script):
            start, length = int(read[1]) - 1, int(read[3])
            return base64.b64encode(self.files[read[2]][start:start + length]).decode()
        if script.startswith("cat "):
            return "FROM scratch\nCOPY app /app\n"
        return "0123abcd\n" if "rev-parse" in script else ""

    async def terminate(self) -> None:
        pass


@pytest.fixture
def build(monkeypatch):
    images, objects, vm = _ImageStore(), _ObjectStore(), _BuildVm()
    get_config().set_image_store(images)
    set_object_store(objects)

    async def create_vm(**kwargs):
        return vm

    monkeypatch.setattr(providers, "get_sandbox_provider", lambda: SimpleNamespace(create_vm=create_vm,
                                                                                    ON_THIS_MACHINE=False))
    monkeypatch.setattr(artifact_store, "get_artifact_store", lambda: SimpleNamespace(next_version=lambda id: 3))
    monkeypatch.setattr(
        DockerImageArtifact, "put_tar", classmethod(lambda cls, id, **kwargs: SimpleNamespace(id=id, version=3, **kwargs))
    )
    # An id whose repository is not the id itself, as an encoded one's is not.
    monkeypatch.setattr(docker_image, "image_repository", lambda entity_id: f"encoded/{entity_id}")
    return images, objects, vm


@pytest.mark.asyncio
async def test_the_image_is_pushed_to_the_repository_put_names(build):
    images, _, vm = build
    result = await DockerImageArtifact.put_from_github("server", "https://github.com/example/repo/blob/main/Dockerfile")
    assert images.ensured == ["encoded/server"]
    assert result.artifact.image_name == "registry.example/encoded/server:v3"
    assert any("docker push registry.example/encoded/server:v3" in script for script in vm.scripts)


@pytest.mark.asyncio
async def test_registry_login_and_upload_signing_run_off_the_event_loop(build):
    """Minting a registry login and signing an upload can each be a network call."""
    images, objects, _ = build
    await DockerImageArtifact.put_from_github("server", "https://github.com/example/repo/blob/main/Dockerfile")
    assert images.on_loop == [False, False]
    assert objects.on_loop == [False, False]


@pytest.mark.asyncio
async def test_the_build_tarballs_are_uploaded_under_the_fixture_prefix(build, monkeypatch):
    monkeypatch.setattr(get_config(), "fixture_prefix", "fx")
    result = await DockerImageArtifact.put_from_github("server", "https://github.com/example/repo/blob/main/Dockerfile")
    image, context = result.artifact.tar_gz_object_url, result.artifact.build_context_object_url
    assert image.startswith("mem://bucket/fx/github-builds/server/") and image.endswith(".tar.gz")
    assert context.startswith("mem://bucket/fx/github-builds/server/") and context.endswith("-context.tar.gz")


@pytest.mark.asyncio
@pytest.mark.parametrize("image_id", ["server", "s" * 238], ids=["short-id", "id-as-long-as-a-docker-name-allows"])
async def test_a_store_that_signs_nothing_gets_the_build_tarballs_copied_off_the_vm(build, tmp_path, image_id):
    _, _, vm = build
    store = LocalFilesystemObjectStore(str(tmp_path / "objects"))
    set_object_store(store)

    result = await DockerImageArtifact.put_from_github(image_id, "https://github.com/example/repo/blob/main/Dockerfile")

    assert store.get(result.artifact.tar_gz_object_url) == vm.files["/tmp/image.tar.gz"]
    assert store.get(result.artifact.build_context_object_url) == vm.files["/tmp/build-context.tar.gz"]
    assert not any("curl" in script for script in vm.scripts)


@pytest.mark.asyncio
async def test_a_registry_on_this_machine_is_refused_before_a_remote_build_vm_is_created(build, monkeypatch):
    images, _, vm = build
    monkeypatch.setattr(images, "image_ref", lambda repository, tag: f"127.0.0.1:5000/{repository}:{tag}")

    with pytest.raises(ValueError, match="is in a registry on this machine"):
        await DockerImageArtifact.put_from_github("server", "https://github.com/example/repo/blob/main/Dockerfile")

    assert vm.scripts == []


@pytest.mark.asyncio
async def test_a_registry_on_this_machine_takes_a_build_on_the_local_provider(build, monkeypatch):
    images, _, vm = build
    monkeypatch.setattr(images, "image_ref", lambda repository, tag: f"localhost:5000/{repository}:{tag}")
    local = object.__new__(LocalSandboxProvider)

    async def create_vm(**kwargs):
        return vm

    local.create_vm = create_vm
    monkeypatch.setattr(providers, "get_sandbox_provider", lambda: local)

    result = await DockerImageArtifact.put_from_github("server", "https://github.com/example/repo/blob/main/Dockerfile")

    assert result.artifact.image_name == "localhost:5000/encoded/server:v3"
