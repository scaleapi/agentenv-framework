"""The computer env tests' fixtures: this machine's Docker and registry faked for the composer, a base and a driver,
and stores a remote VM reaches."""

import urllib.parse

import pytest

from agent_env.artifact.artifacts import docker_image
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env.envs.computer.utils import builder, composer
from agent_env.providers.sandbox_providers.local_docker import provider as local_sandbox
from agent_env.store.image_store.image_store import OciRegistryImageStore
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from tst.util.oci_registry import UBUNTU_LAYER, Registry
from tst.unit.env.envs._computer import FakeDocker, _base, _driver


@pytest.fixture
def docker(local_stores, cli_routing, monkeypatch):
    registry = Registry().serve(monkeypatch)
    fake = FakeDocker(registry)
    monkeypatch.setattr(builder, "build_image", fake.build_image)
    monkeypatch.setattr(builder, "push_local_image", fake.push)
    monkeypatch.setattr(builder, "save_image", fake.save)
    monkeypatch.setattr(builder, "docker_login", lambda auth: None)
    monkeypatch.setattr(composer, "write_index", fake.joined)
    monkeypatch.setattr(composer, "pin_digest", registry.pin_digest)
    monkeypatch.setattr(docker_image, "pin_digest", registry.pin_digest)
    monkeypatch.setattr(builder, "host_platform", lambda: "linux/arm64")
    monkeypatch.setattr(local_sandbox, "host_platform", lambda: "linux/arm64")
    monkeypatch.setattr(composer, "_probe", fake.probe)
    monkeypatch.setattr(builder, "_emulated_platforms", lambda: {"linux/arm64", *fake.emulates})
    monkeypatch.setattr(builder.LocalBuilder, "installed", lambda self, tag, prefix: {"mss": "10.1.0"})
    monkeypatch.setattr(composer, "_protocol_source", lambda: None)
    monkeypatch.setattr(builder.LocalBuilder, "resolve", fake.resolving())
    monkeypatch.setattr(LocalRegistryImageStore, "ensure_repository", lambda self, repository: None)
    return fake


@pytest.fixture
def base(docker):
    index = docker.registry.index(amd64=docker.registry.image(UBUNTU_LAYER),
                                  arm64=docker.registry.image(UBUNTU_LAYER, arch="arm64"))
    image = get_artifact_store().put_document(DockerImageArtifact(id="base-image", description="",
                                                                  image_name=f"ghcr.io/acme/base:1@{index}"))
    return _base(image)


@pytest.fixture
def driver(docker, tmp_path):
    return _driver(tmp_path / "driver")


@pytest.fixture
def signing(local_stores, monkeypatch):
    """Stores a remote VM reaches: the object store signs URLs to its objects, and the registry isn't this machine's."""
    images = OciRegistryImageStore("registry.example")
    monkeypatch.setattr(type(local_stores), "get_image_store_for", lambda self, entity_id: images)
    monkeypatch.setattr(OciRegistryImageStore, "ensure_repository", lambda self, repository: None)
    monkeypatch.setattr(LocalFilesystemObjectStore, "signed_get_url", lambda self, url, expires_in=3600:
                        f"https://signed.example/get?{urllib.parse.quote(url)}")
    monkeypatch.setattr(LocalFilesystemObjectStore, "signed_put_url", lambda self, url, expires_in=3600:
                        f"https://signed.example/put?{urllib.parse.quote(url)}")
