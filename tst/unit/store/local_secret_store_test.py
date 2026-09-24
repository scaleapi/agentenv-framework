"""LocalSecretStore runs the full SecretStore conformance suite + local specifics.

Fast tier — no AWS, no network (stdlib env/file), so the same assertions that a
Secrets Manager backend must satisfy also give quick backend-neutral coverage.
"""

import pytest

from agent_env.store import LocalSecretStore
from tst.store import secret_conformance


@pytest.fixture
def store():
    # use_env=False so the process environment can't shadow the fixture values.
    return LocalSecretStore(values=dict(secret_conformance.FIXTURE), use_env=False)


@pytest.mark.parametrize("case", secret_conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(case, store):
    case(store)


def test_env_var_takes_precedence_over_values(monkeypatch):
    monkeypatch.setenv("MY_SECRET", "from-env")
    store = LocalSecretStore(values={"MY_SECRET": "from-values"})
    assert store.get("MY_SECRET") == "from-env"


def test_env_ignored_when_use_env_false(monkeypatch):
    monkeypatch.setenv("MY_SECRET", "from-env")
    store = LocalSecretStore(values={"MY_SECRET": "from-values"}, use_env=False)
    assert store.get("MY_SECRET") == "from-values"


def test_loads_flat_mapping_from_file(tmp_path):
    secret_file = tmp_path / "secrets.yaml"
    secret_file.write_text("litellm_api_key: sk-local\nmodal_token_id: tok-123\n")
    store = LocalSecretStore(file_path=str(secret_file), use_env=False)
    assert store.get("litellm_api_key") == "sk-local"
    assert store.get("modal_token_id") == "tok-123"
    assert store.get("missing") is None


def test_values_override_file(tmp_path):
    secret_file = tmp_path / "secrets.yaml"
    secret_file.write_text("k: from-file\n")
    store = LocalSecretStore(values={"k": "from-values"}, file_path=str(secret_file), use_env=False)
    assert store.get("k") == "from-values"


def test_non_mapping_file_rejected(tmp_path):
    secret_file = tmp_path / "secrets.yaml"
    secret_file.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError):
        LocalSecretStore(file_path=str(secret_file))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        LocalSecretStore(file_path=str(tmp_path / "nope.yaml"))


def test_non_string_values_coerced_to_str():
    numeric = LocalSecretStore(values={"port": 5432}, use_env=False)
    assert numeric.get("port") == "5432"


def test_from_config_default_builds_from_kwargs(tmp_path):
    secret_file = tmp_path / "secrets.yaml"
    secret_file.write_text("k: v\n")
    store = LocalSecretStore.from_config(file_path=str(secret_file), use_env=False)
    assert isinstance(store, LocalSecretStore)
    assert store.get("k") == "v"
