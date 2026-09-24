"""Unit tests for the generic OCI registry image store."""

import pytest

from agent_env.store import OciRegistryCredentials, OciRegistryImageStore, RegistryAuth
from agent_env.store.image_store import registry_host_from_ref


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
