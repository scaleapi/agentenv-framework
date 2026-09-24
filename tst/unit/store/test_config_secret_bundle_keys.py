"""The bundle keys ``Config``'s getters actually read from the secret store.

These pin the *names*, not the values. The secret bundle used to be described by a
``_SecretValue`` TypedDict that had silently drifted from the code — it declared
``export_access_key``/``export_secret_key`` while ``get_export_credentials`` read
``agentenvhub_export_access_key``/``agentenvhub_export_secret_key``, and it omitted
the Meta keys entirely. The bundle is an open mapping (callers also
index it with names supplied by task metadata), so the contract can only be held by
tests over the getters. AWS-free: a ``LocalSecretStore`` supplies the bundle, so
nothing here touches Secrets Manager.

Sibling coverage: the ``litellm_api_key`` name now rides the [model] api_key
config (the platform plugin ships the ``secret:`` ref); the Mongo coordinates ride
[stores.document]. Neither is a bundle key core reads directly anymore.
"""

import pytest

from agent_env.config import Config
from agent_env.store import LocalSecretStore

_EXPORT_KEYS = {
    "agentenvhub_export_access_key": "AKIA-EXAMPLE",
    "agentenvhub_export_secret_key": "secret-example",
}


def _config_with(bundle: dict) -> Config:
    cfg = Config()
    cfg.set_secret_store(LocalSecretStore(values=bundle))
    return cfg


def test_export_credentials_read_the_agentenvhub_prefixed_keys():
    creds = _config_with(_EXPORT_KEYS).get_export_credentials()
    assert creds == {"access_key": "AKIA-EXAMPLE", "secret_key": "secret-example"}


def test_export_credentials_reject_the_stale_unprefixed_keys():
    # The names the retired _SecretValue TypedDict declared. If a bundle ever ships
    # only these, that is a missing key — not a silently-working fallback.
    cfg = _config_with({"export_access_key": "a", "export_secret_key": "b"})
    with pytest.raises(KeyError):
        cfg.get_export_credentials()


def test_modal_credentials_read_token_id_and_secret(monkeypatch, tmp_path):
    # Env + ~/.modal.toml both take precedence over the bundle, so clear them.
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = _config_with({"modal_token_id": "tid", "modal_token_secret": "tsecret"})
    assert cfg.get_modal_credentials() == ("tid", "tsecret")


