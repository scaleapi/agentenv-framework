"""`DockerImageArtifact.put_ref` registers an image already in a registry: a document with no tar.gz, its tag pinned
to a digest, refused when no sandbox elsewhere could pull it. The registry API is patched out."""

import pytest

from agent_env.artifact.artifacts import docker_image
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.store import ImageStore, OciRegistryImageStore, RegistryAuth
from agent_env.store.base import NotFoundError
from agent_env.store.routing import namespace_routing

DIGEST = "sha256:" + "c" * 64


@pytest.fixture
def resolved(monkeypatch):
    """Each ref put_ref resolves and the credentials it resolves it with; every tag resolves to DIGEST."""
    calls = []

    def pin(ref, auth):
        calls.append((ref, auth))
        return ref if "@" in ref else f"{ref}@{DIGEST}"

    monkeypatch.setattr(docker_image, "pin_digest", pin)
    return calls


def test_a_ref_is_registered_with_its_tag_pinned_and_no_tarball(local_stores, resolved):
    image = DockerImageArtifact.put_ref("tool-image", description="d", image_name="ghcr.io/org/tool:v1")

    stored = DockerImageArtifact.get("tool-image", image.version)
    assert (stored.version, stored.image_name, stored.tar_gz_object_url) == (1, f"ghcr.io/org/tool:v1@{DIGEST}", None)
    assert stored.load_problem() is None
    assert DockerImageArtifact.put_ref("tool-image", description="d", image_name="ghcr.io/org/tool:v1").version == 2


def test_the_configured_store_lends_its_credentials_only_for_its_own_registry(local_stores, resolved):
    class _Registry(ImageStore):
        def image_ref(self, repository, tag):
            return f"registry.example/{repository}:{tag}"

        def auth(self, ref):
            return RegistryAuth("registry.example", "user", "pw") if ref.startswith("registry.example/") else None

    local_stores.set_image_store(_Registry())

    DockerImageArtifact.put_ref("own", description="d", image_name="registry.example/team/app:v1")
    DockerImageArtifact.put_ref("public", description="d", image_name="ghcr.io/org/tool:v1")

    assert resolved == [("registry.example/team/app:v1", RegistryAuth("registry.example", "user", "pw")),
                        ("ghcr.io/org/tool:v1", None)]


def test_a_name_without_its_registry_is_refused_before_any_request(local_stores, resolved):
    with pytest.raises(ValueError, match="'tool:v1' doesn't name its registry; spell it out, as in docker.io/library/tool:v1"):
        DockerImageArtifact.put_ref("tool-image", description="d", image_name="tool:v1")

    assert resolved == []
    with pytest.raises(NotFoundError):
        DockerImageArtifact.get("tool-image")


@pytest.mark.parametrize("ref", ["localhost:5000/team/img:v1", "127.0.0.1:5000/img", "host.docker.internal:5000/img",
                                 "0.0.0.0:5000/img"])
def test_a_registry_on_this_machine_is_refused_for_a_shared_store(local_stores, resolved, ref):
    local_stores.set_image_store(OciRegistryImageStore("registry.example"))

    with pytest.raises(ValueError, match="is in a registry on this machine"):
        DockerImageArtifact.put_ref("img", description="d", image_name=ref)
    assert resolved == []


def test_a_registry_on_this_machine_is_fine_for_an_image_in_the_local_registrys_store(local_stores, resolved):
    with namespace_routing():
        image = DockerImageArtifact.put_ref("@local/team/img", description="d", image_name="localhost:5000/team/img:v1")

    assert image.image_name == f"localhost:5000/team/img:v1@{DIGEST}"


def test_a_ref_the_registry_cant_serve_writes_nothing(local_stores, monkeypatch):
    def unresolvable(ref, auth):
        raise ValueError(f"{ref}: ghcr.io has no such image")

    monkeypatch.setattr(docker_image, "pin_digest", unresolvable)

    with pytest.raises(ValueError, match="ghcr.io has no such image"):
        DockerImageArtifact.put_ref("tool-image", description="d", image_name="ghcr.io/org/tool:v9")
    assert get_artifact_store().next_version("tool-image") == 1
