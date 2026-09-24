"""Default A2A agent resolution: configure() > [agents] default_a2a_agent_id > built-in a2a-default."""

import pytest

from agent_env.config.errors import ConfigError
from agent_env.config.runtime import configure, get_config, reset_config
from tst.util.config import config_with_document


def _config(config_file=None, **kwargs):
    """A Config with its config.toml read stubbed — no file discovery, no AWS."""
    return config_with_document({} if config_file is None else config_file, **kwargs)


def test_built_in_default_when_the_section_is_absent():
    assert _config().get_default_a2a_agent_id() == "a2a-default"


def test_an_empty_agents_section_keeps_the_built_in_default():
    assert _config(config_file={"agents": {}}).get_default_a2a_agent_id() == "a2a-default"


def test_config_key_resolves():
    cfg = _config(config_file={"agents": {"default_a2a_agent_id": "claude-code-cli"}})
    assert cfg.get_default_a2a_agent_id() == "claude-code-cli"


def test_config_key_resolves_env_references():
    cfg = _config(config_file={"agents": {"default_a2a_agent_id": "env:MISSING?fallback-agent"}})
    assert cfg.get_default_a2a_agent_id() == "fallback-agent"


def test_explicit_value_beats_the_config_key():
    cfg = _config(config_file={"agents": {"default_a2a_agent_id": "from-file"}}, default_a2a_agent_id="from-code")
    assert cfg.get_default_a2a_agent_id() == "from-code"


def test_configure_accepts_the_override_and_it_reaches_the_singleton():
    try:
        configure(default_a2a_agent_id="from-code")
        assert get_config().get_default_a2a_agent_id() == "from-code"
    finally:
        reset_config()


def test_an_empty_value_is_rejected_naming_the_key():
    with pytest.raises(ConfigError, match=r"\[agents\] default_a2a_agent_id"):
        _config(config_file={"agents": {"default_a2a_agent_id": ""}}).get_default_a2a_agent_id()


def test_an_unknown_key_is_rejected_naming_the_allowed_one():
    with pytest.raises(ConfigError, match=r"unknown keys \['default_agent'\].*default_a2a_agent_id"):
        _config(config_file={"agents": {"default_agent": "x"}}).get_default_a2a_agent_id()


def test_a_non_table_section_is_rejected():
    with pytest.raises(ConfigError):
        _config(config_file={"agents": "claude-code-cli"}).get_default_a2a_agent_id()
