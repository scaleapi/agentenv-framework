"""Default infra env ids: one set in code (configure() or assignment) > [envs] in config.toml > the built-in id."""

import pytest

from agent_env.config.errors import ConfigError
from agent_env.config.runtime import configure, get_config, reset_config
from tst.util.config import config_with_document

BUILT_IN = {"default_gateway_env_id": "default", "default_service_db_env_id": "default-db",
            "default_website_browser_env_id": "playwright-mcp"}


def _config(envs=None):
    """A Config with its config.toml read stubbed: no file discovery, no AWS."""
    return config_with_document({} if envs is None else {"envs": envs})


@pytest.mark.parametrize("name", BUILT_IN)
def test_the_built_in_id_when_envs_names_none(name):
    assert getattr(_config(), name) == BUILT_IN[name]
    assert getattr(_config({"impls": []}), name) == BUILT_IN[name]


@pytest.mark.parametrize("name", BUILT_IN)
def test_an_envs_key_names_it_beside_the_registry_impls(name):
    assert getattr(_config({"impls": [], name: "from-file"}), name) == "from-file"


def test_an_envs_key_resolves_env_references():
    assert _config({"default_website_browser_env_id": "env:MISSING?browser-use"}).default_website_browser_env_id \
        == "browser-use"


@pytest.mark.parametrize("name", BUILT_IN)
def test_an_id_set_in_code_beats_the_envs_key(name):
    config = _config({name: "from-file"})
    setattr(config, name, "from-code")
    assert getattr(config, name) == "from-code"


def test_configure_sets_one_on_the_singleton_and_leaves_the_others():
    try:
        configure(default_website_browser_env_id="from-code")
        assert get_config().default_website_browser_env_id == "from-code"
        assert get_config().default_gateway_env_id == "default"
    finally:
        reset_config()


def test_an_empty_envs_key_is_rejected_naming_it():
    with pytest.raises(ConfigError, match=r"\[envs\] default_website_browser_env_id"):
        _config({"default_website_browser_env_id": ""}).default_website_browser_env_id
