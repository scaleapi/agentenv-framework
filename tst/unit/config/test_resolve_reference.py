"""Config.resolve_reference: the resolver a step applies to a per-run secret given by reference."""

from agent_env.config import get_config


def test_an_env_reference_resolves_and_anything_else_passes_through(monkeypatch):
    monkeypatch.setenv("PER_RUN_KEY", "sk-from-env")
    config = get_config()
    assert config.resolve_reference("env:PER_RUN_KEY") == "sk-from-env"
    assert config.resolve_reference("sk-plain") == "sk-plain"
    assert config.resolve_reference(None) is None
