"""Unit tests for the generic OCI registry image store."""

import pytest

from agent_env.store import ImageStore, OciRegistryCredentials, OciRegistryImageStore, RegistryAuth
from agent_env.store.image_store import names_registry, registry_host_from_ref
from agent_env.store.image_store.oci_registry_credentials import is_loopback_host


class _Credentials(OciRegistryCredentials):
    def __init__(self):
        self.hosts = []

    def mint(self, host):
        self.hosts.append(host)
        return RegistryAuth(host, "user", "password")


def test_builds_prefixed_image_reference():
    store = OciRegistryImageStore("ghcr.io", "example/agent-env")

    assert store.image_ref("server", "v2") == "ghcr.io/example/agent-env/server:v2"
    assert store.registry_host == "ghcr.io"


def test_exact_host_match_is_required_before_minting_credentials():
    credentials = _Credentials()
    store = OciRegistryImageStore("ghcr.io", credentials=credentials)

    assert store.auth("ghcr.io/example/image:v1") == RegistryAuth(
        "ghcr.io", "user", "password"
    )
    assert store.auth("ghcr.io.attacker.example/example/image:v1") is None
    assert store.auth("quay.io/example/image:v1") is None
    assert credentials.hosts == ["ghcr.io"]


def test_port_is_part_of_registry_authority():
    credentials = _Credentials()
    store = OciRegistryImageStore("registry.example:5000", credentials=credentials)

    assert store.auth("registry.example:5000/team/image:v1") is not None
    assert store.auth("registry.example/team/image:v1") is None
    assert store.auth("registry.example:5001/team/image:v1") is None


def test_unqualified_references_resolve_to_docker_hub():
    credentials = _Credentials()
    store = OciRegistryImageStore("docker.io", credentials=credentials)

    assert store.auth("ubuntu:24.04") is not None
    assert store.auth("library/ubuntu:24.04") is not None
    assert registry_host_from_ref("LOCALHOST/image:v1") == "localhost"
    assert registry_host_from_ref("ghcr.io/example/image:v1") == "ghcr.io"
    assert registry_host_from_ref("/invalid:v1") is None


@pytest.mark.parametrize("ref, named", [
    ("ghcr.io/example/image:v1", True),
    ("registry.example:5000/image@sha256:" + "0" * 64, True),
    ("localhost/image:v1", True),
    ("LOCALHOST:5000/image", True),
    ("docker.io/library/ubuntu:24.04", True),
    ("ubuntu:24.04", False),
    ("library/ubuntu:24.04", False),
    ("team/image:v1", False),
    ("", False),
    ("https://ghcr.io/example/image:v1", False),
])
def test_a_reference_names_its_registry_only_as_docker_reads_one(ref, named):
    assert names_registry(ref) is named


@pytest.mark.parametrize("host, here", [
    ("localhost", True), ("localhost:5000", True), ("LOCALHOST:5000", True), ("127.0.0.2:5000", True),
    ("[::1]:5000", True), ("0.0.0.0:5000", True), ("host.docker.internal:5000", True), ("Host.Docker.Internal", True),
    ("registry.example", False), ("10.0.0.1:5000", False), ("ghcr.io", False), (None, False), ("", False),
])
def test_a_registry_host_on_this_machine_is_one_only_this_machine_can_reach(host, here):
    assert is_loopback_host(host) is here


@pytest.mark.parametrize("host", ["", "   ", "https://ghcr.io", "ghcr.io/team"])
def test_rejects_invalid_registry_host(host):
    with pytest.raises(ValueError, match="registry_host"):
        OciRegistryImageStore(host)


def test_from_config_builds_nested_credential_provider(monkeypatch):
    from agent_env.config import get_config
    from agent_env.store import LocalSecretStore, SecretStoreCredentials

    get_config().set_secret_store(
        LocalSecretStore(
            values={
                "REGISTRY_AUTHS": '{"auths":{"ghcr.io":{"username":"u","password":"p"}}}'
            },
            use_env=False,
        )
    )
    store = OciRegistryImageStore.from_config(
        registry_host="ghcr.io",
        credentials={
            "impl": "agent_env.store.image_store:SecretStoreCredentials",
            "secret_key": "REGISTRY_AUTHS",
        },
    )

    assert isinstance(store.credentials, SecretStoreCredentials)
    assert store.auth("ghcr.io/example/image:v1") == RegistryAuth("ghcr.io", "u", "p")


@pytest.mark.parametrize(
    ("ref", "owned"),
    [
        ("us-west1-docker.pkg.dev/example-project/agentenv/server:v2", True),
        ("us-west1-docker.pkg.dev/example-project/agentenv/team/server@sha256:" + "0" * 64, True),
        ("US-WEST1-DOCKER.PKG.DEV/example-project/agentenv/server", True),
        ("us-west1-docker.pkg.dev/other-project/agentenv/server:v2", True),
        ("us-east1-docker.pkg.dev/example-project/agentenv/server:v2", False),
        ("us-west1-docker.pkg.dev.attacker.example/example-project/agentenv/server:v2", False),
        ("example-project/agentenv/server:v2", False),
    ],
)
def test_owns_the_refs_its_registry_login_covers(ref, owned):
    """A registry login covers the whole host, other repositories on it included."""
    store = OciRegistryImageStore("us-west1-docker.pkg.dev", "example-project/agentenv", credentials=_Credentials())
    assert store.owns(store.image_ref("server", "v2"))
    assert store.owns(ref) is owned
    assert (store.auth(ref) is not None) is owned


def test_without_credentials_the_store_still_owns_its_registry():
    store = OciRegistryImageStore("localhost:5000")
    assert store.owns("localhost:5000/server:v1") and store.owns("localhost:5000/team/server")
    assert not store.owns("localhost:5001/server:v1") and not store.owns("server:v1")
    assert store.auth("localhost:5000/server:v1") is None


def test_widening_owns_does_not_send_the_login_to_another_registry():
    class _Mirroring(OciRegistryImageStore):
        def owns(self, ref):
            return super().owns(ref) or ref.startswith("mirror.example/")

    store = _Mirroring("registry.example", credentials=_Credentials())
    assert store.owns("mirror.example/app:v1")
    assert store.auth("mirror.example/app:v1") is None


def test_an_image_store_owns_nothing_unless_it_says_so():
    class _Minimal(ImageStore):
        def image_ref(self, repository, tag):
            return f"registry.example/{repository}:{tag}"

        def auth(self, ref):
            return None

    assert not _Minimal().owns("registry.example/server:v1")
