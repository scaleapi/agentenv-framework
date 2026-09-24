"""A config.toml resolves to the backends it names, and says why when it cannot.

The OSS claim is that one file selects every backend and a bare install runs fully
local. That claim is only worth as much as the failure modes: a profile that silently
falls back to a default is worse than one that refuses, because the run looks fine and
writes to the wrong place. Each case here was first confirmed by hand against a live
`agent-env up`; this pins them.
"""

from __future__ import annotations

import pytest

from agent_env.config import configure, get_config, reset_config
from agent_env.config.errors import ConfigError


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Point at a config this test owns, so the developer's own file cannot leak in."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    yield
    reset_config()


def _write(tmp_path, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body)
    configure()
    return path


def test_bare_profile_resolves_every_store_to_a_local_backend(tmp_path):
    """The documented default: no coordinates, nothing external."""
    _write(tmp_path, "")
    config = get_config()
    assert type(config.get_document_store()).__name__ == "LocalSqliteDocumentStore"
    assert type(config.get_object_store()).__name__ == "LocalFilesystemObjectStore"


def test_a_profile_uses_the_paths_it_names_rather_than_the_defaults(tmp_path):
    """A mixed profile must actually write where it says, not quietly use the default.

    Literal TOML strings, so a Windows tmp path's backslashes are not read as escapes.
    """
    docs = tmp_path / "custom.db"
    objects = tmp_path / "custom-objects"
    _write(tmp_path, f"""
[stores.document]
impl = "agent_env.store.document_store:LocalSqliteDocumentStore"
[stores.document.config]
path = '{docs}'
[stores.object]
impl = "agent_env.store.object_store:LocalFilesystemObjectStore"
[stores.object.config]
root = '{objects}'
""")
    config = get_config()
    url = config.get_object_store().put("probe/x.txt", b"hi", content_type="text/plain")
    config.get_document_store().insert("probe", {"id": "p", "version": 1})

    assert str(objects) in url, f"object went somewhere else: {url}"
    assert docs.exists(), "the configured sqlite path was not used"


def test_an_env_var_overrides_the_file(tmp_path, monkeypatch):
    """Documented precedence: built-in default < config file < AGENT_ENV_* env var."""
    _write(tmp_path, '[stores]\ndocument = "mongo"\n')
    reset_config()
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    configure()
    assert type(get_config().get_document_store()).__name__ == "LocalSqliteDocumentStore"


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(
            '[stores.document]\nimpl = "agent_env.store.document_store:NoSuchStore"\n',
            "NoSuchStore",
            id="impl names a class that does not exist",
        ),
        pytest.param(
            '[stores.document]\nimpl = "totally.made.up:Thing"\n',
            "totally",
            id="impl names a module that does not exist",
        ),
        pytest.param("[stores\ndocument = broken\n", "Malformed", id="malformed toml"),
        pytest.param(
            '[stores.document]\n'
            'impl = "agent_env.store.document_store:MongoDocumentStore"\n'
            '[stores.document.config]\nuri = "secret:absent_key"\ndatabase = "x"\n',
            "absent_key",
            id="unresolvable secret reference",
        ),
    ],
)
def test_a_broken_profile_raises_config_error_naming_the_problem(tmp_path, body, expected):
    """A bad profile must fail loudly and say which part, never fall back to a default.

    Stores resolve lazily, so the error surfaces on first access rather than at
    `configure()`. That is the behaviour worth pinning: the failure must arrive before
    anything is read or written, not as a silent default.
    """
    with pytest.raises(ConfigError) as excinfo:
        _write(tmp_path, body)
        get_config().get_document_store()
    assert expected in str(excinfo.value)


def test_a_missing_config_file_is_refused_rather_than_ignored(tmp_path, monkeypatch):
    """Naming a file that isn't there must not silently run on the defaults."""
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "absent.toml"))
    with pytest.raises(ConfigError) as excinfo:
        configure()
    assert "does not point to an existing file" in str(excinfo.value)
