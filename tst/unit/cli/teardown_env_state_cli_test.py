"""Unit tests for `agent-env env teardown-env-state` (CLI wiring). No DB: the state store +
provider are patched. The idempotent drop itself is a backend concern, covered where it lives."""

from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from agent_env.cli.env.state.teardown import teardown_env_state
from agent_env.providers.state import EnvStateInstance

# Any config.toml-registered external backend; core ships none, so the tag is just a string here.
PERSISTENT_REMOTE_POSTGRES_STATE_TYPE = "persistent_remote_postgres"
REMOTE_POSTGRES_STATE_TYPE = "remote_postgres"
from agent_env.store.base import NotFoundError


def _invoke_persistent(metadata):
    instance = EnvStateInstance(
        state_type=PERSISTENT_REMOTE_POSTGRES_STATE_TYPE, instance_id="esi-p", metadata=metadata,
    )
    store = MagicMock()
    store.get.return_value = instance
    provider = MagicMock()
    provider.teardown = AsyncMock(return_value=None)
    with patch("agent_env.providers.state.get_env_state_instance_store", return_value=store), \
         patch("agent_env.providers.state.build_state_provider", return_value=provider):
        return CliRunner().invoke(teardown_env_state, ["--instance-id", "esi-p"]), provider


def test_teardown_cli_drops_and_retires():
    instance = EnvStateInstance(state_type=REMOTE_POSTGRES_STATE_TYPE, instance_id="esi-abc")
    store = MagicMock()
    store.get.return_value = instance
    provider = MagicMock()
    provider.teardown = AsyncMock(return_value=None)
    with patch("agent_env.providers.state.get_env_state_instance_store", return_value=store), \
         patch("agent_env.providers.state.build_state_provider", return_value=provider) as build:
        res = CliRunner().invoke(teardown_env_state, ["--instance-id", "esi-abc"])
    assert res.exit_code == 0, res.output
    build.assert_called_once_with(REMOTE_POSTGRES_STATE_TYPE)
    provider.teardown.assert_awaited_once_with(instance)
    assert "esi-abc" in res.output


def test_teardown_cli_unknown_instance_is_noop():
    """No record = already gone: exit 0, friendly message, no provider built (reaper/re-run safe)."""
    store = MagicMock()
    store.get.side_effect = NotFoundError("nope")
    with patch("agent_env.providers.state.get_env_state_instance_store", return_value=store), \
         patch("agent_env.providers.state.build_state_provider") as build:
        res = CliRunner().invoke(teardown_env_state, ["--instance-id", "esi-missing"])
    assert res.exit_code == 0, res.output
    assert "nothing to tear down" in res.output
    build.assert_not_called()


def test_teardown_cli_requires_instance_id():
    res = CliRunner().invoke(teardown_env_state, [])
    assert res.exit_code != 0
    assert "instance-id" in res.output.lower()


def test_teardown_cli_base_warns_about_orphaning():
    """Tearing down a persistent BASE drops the shared DB → warn it orphans referencing overlays."""
    res, provider = _invoke_persistent({"kind": "base", "dbname": "base_universe_x"})
    assert res.exit_code == 0, res.output
    assert "BASE" in res.output and "orphan" in res.output.lower()
    assert "base_universe_x" in res.output
    provider.teardown.assert_awaited_once()


def test_teardown_cli_run_overlay_notes_base_preserved():
    """Tearing down a RUN_OVERLAY drops only overlay+role → note the base is preserved (no orphaning)."""
    res, provider = _invoke_persistent(
        {"kind": "run_overlay", "dbname": "base_universe_x", "base_env_state_instance_id": "esi-base"}
    )
    assert res.exit_code == 0, res.output
    assert "RUN_OVERLAY" in res.output and "preserved" in res.output.lower()
    assert "esi-base" in res.output
    provider.teardown.assert_awaited_once()
