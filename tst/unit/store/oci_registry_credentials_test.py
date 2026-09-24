"""Unit tests for OCI registry credential providers."""

import base64
import logging

import pytest

from agent_env.config import ConfigError, get_config
from agent_env.store import (
    LocalSecretStore,
    RegistryAuth,
    SecretStoreCredentials,
)


def _set_secret(value: object, *, key: str = "registry_auths") -> None:
    get_config().set_secret_store(LocalSecretStore(values={key: value}, use_env=False))


def test_reads_username_and_password_from_docker_config():
    _set_secret('{"auths":{"ghcr.io":{"username":"octocat","password":"pat"}}}')

    assert SecretStoreCredentials().mint("ghcr.io") == RegistryAuth(
        registry="ghcr.io", username="octocat", password="pat"
    )


def test_reads_base64_auth_from_bare_mapping():
    auth = base64.b64encode(b"alice:password:with:colons").decode()
    _set_secret(f'{{"registry.example":{{"auth":"{auth}"}}}}')

    assert SecretStoreCredentials().mint("registry.example") == RegistryAuth(
        registry="registry.example", username="alice", password="password:with:colons"
    )


@pytest.mark.parametrize(
    "configured_key,target",
    [
        ("https://index.docker.io/v1/", "docker.io"),
        ("registry-1.docker.io", "docker.io"),
        ("https://registry.example:5443/v1/", "registry.example:5443"),
    ],
)
def test_normalizes_docker_registry_keys(configured_key, target):
    _set_secret(
        '{"auths":{' + f'"{configured_key}":{{"username":"u","password":"p"}}' + "}}"
    )

    assert SecretStoreCredentials().mint(target) == RegistryAuth(target, "u", "p")


def test_returns_none_when_secret_or_host_is_absent():
    get_config().set_secret_store(LocalSecretStore(values={}, use_env=False))
    assert SecretStoreCredentials().mint("ghcr.io") is None

    _set_secret('{"auths":{"quay.io":{"username":"u","password":"p"}}}')
    assert SecretStoreCredentials().mint("ghcr.io") is None


@pytest.mark.parametrize(
    "value",
    [
        "not-json-secret-value",
        {"auths": {"ghcr.io": {"username": "u", "password": "secret-value"}}},
        '{"auths":[]}',
        '{"auths":{"ghcr.io":{"auth":"not-base64-secret-value"}}}',
    ],
)
def test_invalid_secret_is_rejected_without_leaking_value(value):
    _set_secret(value)

    with pytest.raises(ConfigError) as error:
        SecretStoreCredentials().mint("ghcr.io")

    assert "secret-value" not in str(error.value)


def test_credential_helper_settings_are_ignored_without_leaking_values(caplog):
    _set_secret(
        '{"auths":{"ghcr.io":{"username":"u","password":"private-password"}},'
        '"credsStore":"desktop","credHelpers":{"ghcr.io":"helper"}}'
    )

    with caplog.at_level(logging.DEBUG):
        auth = SecretStoreCredentials().mint("ghcr.io")

    assert auth is not None
    assert "private-password" not in caplog.text


def test_static_credentials_are_resolved_on_every_mint():
    secret = LocalSecretStore(
        values={
            "registry_auths": '{"auths":{"ghcr.io":{"username":"u","password":"one"}}}'
        },
        use_env=False,
    )
    get_config().set_secret_store(secret)
    credentials = SecretStoreCredentials()

    assert credentials.mint("ghcr.io").password == "one"  # type: ignore[union-attr]
    secret._values["registry_auths"] = (  # type: ignore[attr-defined]
        '{"auths":{"ghcr.io":{"username":"u","password":"two"}}}'
    )
    assert credentials.mint("ghcr.io").password == "two"  # type: ignore[union-attr]


def test_registry_auth_repr_redacts_password():
    auth = RegistryAuth("registry.example", "alice", "private-password")

    assert "private-password" not in repr(auth)
    assert "alice" in repr(auth)
