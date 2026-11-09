"""Unit tests for `agent-env env init-env-state` (CLI wiring).

No DB / no RDS: ``Env.get`` and the ``acquire_state_for_deploy`` seam are patched, so these
cover the CLI surface — environment-name resolution, env-state-type mapping, arg validation, and that the
returned instance id is printed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from agent_env.cli.env.state.init import DEFAULT_TTL_SECONDS, init_env_state
from agent_env.env import Env, MultiEnv
from agent_env.providers.env_state.env_state_provider import EnvStateInstance

# Any config.toml-registered external backend; core ships none, so the tag is just a string here.
REMOTE_POSTGRES_STATE_TYPE = "remote_postgres"
PERSISTENT_REMOTE_POSTGRES_STATE_TYPE = "persistent_remote_postgres"


def _fake_multi_env(mcp=("slack", "email"), websites=()):
    env = MagicMock(spec=MultiEnv)  # isinstance(env, MultiEnv) is True
    env.id, env.version, env.type = "multi-slack-email", 3, "multi_env"
    env.mcp_server_envs = [MagicMock(environment_name=s) for s in mcp]
    env.website_envs = [MagicMock(environment_name=s) for s in websites]
    return env


def _fake_instance():
    return EnvStateInstance(
        state_type=REMOTE_POSTGRES_STATE_TYPE,
        instance_id="esi-test1234",
        metadata={"host": "rds.example", "dbname": "run_multi_slack_email_abcd1234"},
        created_at_utc="2026-07-12 00:00 UTC",
        expires_at_utc="2026-08-11 00:00 UTC",
    )


def _invoke(args):
    acquire = AsyncMock(return_value=_fake_instance())
    # The ephemeral path is acquire (stage 1) + provider.prepare (stage 2); patch build_state_provider
    # so prepare is a no-op mock rather than a real RDS-touching provider.
    provider = MagicMock()
    provider.prepare = AsyncMock()
    with patch.object(Env, "get", return_value=_fake_multi_env()) as get, \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", acquire), \
         patch("agent_env.providers.env_state.build_state_provider", return_value=provider):
        res = CliRunner().invoke(init_env_state, args)
    return res, acquire, provider, get


def test_happy_path_resolves_environments_and_prints_instance_id():
    res, acquire, provider, _get = _invoke(
        ["--id", "multi-slack-email", "--env-state-type", "remote_postgres"]
    )
    assert res.exit_code == 0, res.output
    acquire.assert_awaited_once_with(
        env_state_type=REMOTE_POSTGRES_STATE_TYPE,
        ttl_seconds=DEFAULT_TTL_SECONDS,
        name_hint="multi-slack-email",
    )
    # stage 2: schemas created via prepare with the resolved environment list
    provider.prepare.assert_awaited_once()
    assert provider.prepare.await_args.args[0] == ["slack", "email"]
    assert "esi-test1234" in res.output


def test_custom_ttl_is_forwarded():
    _res, acquire, _provider, _get = _invoke(
        ["--id", "e", "--env-state-type", "remote_postgres", "--ttl-seconds", "604800"]
    )
    assert acquire.await_args.kwargs["ttl_seconds"] == 604800


def test_missing_env_state_type_errors():
    res = CliRunner().invoke(init_env_state, ["--id", "e"])
    assert res.exit_code != 0
    assert "an env state type must be supplied" in res.output


def test_unregistered_env_state_type_is_rejected():
    """No hardcoded choice list any more — the registry is the source of truth, and an unknown tag
    fails before any Mongo/AWS work with the registered names in the message."""
    res = CliRunner().invoke(init_env_state, ["--id", "e", "--env-state-type", "mysql"])
    assert res.exit_code != 0
    assert "mysql" in res.output and "local_postgres" in res.output


def test_ttl_below_min_rejected():
    res = CliRunner().invoke(
        init_env_state, ["--id", "e", "--env-state-type", "remote_postgres", "--ttl-seconds", "60"]
    )
    assert res.exit_code != 0


def test_persistent_backend_materializes_base_via_acquire_then_prepare():
    """The persistent backend stands up its base through the unified acquire + prepare path (same as
    the ephemeral type): acquire_state_for_deploy makes the empty base, then prepare builds its
    identity schemas from the env's environment list."""
    base = EnvStateInstance(
        state_type=PERSISTENT_REMOTE_POSTGRES_STATE_TYPE, instance_id="esi-base9",
        metadata={"kind": "base", "host": "h", "dbname": "base_x"},
    )
    provider = MagicMock()
    provider.prepare = AsyncMock()
    with patch.object(Env, "get", return_value=_fake_multi_env()), \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", AsyncMock(return_value=base)) as acquire, \
         patch("agent_env.providers.env_state.build_state_provider", return_value=provider):
        res = CliRunner().invoke(
            init_env_state,
            ["--id", "multi-slack-email", "--env-state-type", "persistent_remote_postgres"],
        )
    assert res.exit_code == 0, res.output
    assert acquire.await_args.kwargs["env_state_type"] == PERSISTENT_REMOTE_POSTGRES_STATE_TYPE
    assert acquire.await_args.kwargs["name_hint"] == "multi-slack-email"
    # schemas are built by prepare, from the env's declared environments
    provider.prepare.assert_awaited_once()
    assert provider.prepare.await_args.args[0] == ["slack", "email"]
    assert provider.prepare.await_args.kwargs["instance"] is base
    assert "esi-base9" in res.output


def test_website_env_has_no_schema_for_the_browser_the_gateway_adds():
    """The gateway adds a website browser to an env with websites but gives it no schema at deploy, so the
    pre-initialized store carries the env's own MCP servers and websites alone, as the deploy's ``prepare`` does."""
    provider = MagicMock()
    provider.prepare = AsyncMock()
    with patch.object(Env, "get", return_value=_fake_multi_env(mcp=("slack",), websites=("shop",))), \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", AsyncMock(return_value=_fake_instance())), \
         patch("agent_env.providers.env_state.build_state_provider", return_value=provider):
        res = CliRunner().invoke(
            init_env_state, ["--id", "multi-shop", "--env-state-type", "remote_postgres"]
        )
    assert res.exit_code == 0, res.output
    assert provider.prepare.await_args.args[0] == ["slack", "shop"]


def test_environment_names_are_deduped():
    """``prepare`` receives one schema per name — the gateway dedupes its list, so this must too."""
    provider = MagicMock()
    provider.prepare = AsyncMock()
    with patch.object(Env, "get", return_value=_fake_multi_env(mcp=("slack", "slack", "email"))), \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", AsyncMock(return_value=_fake_instance())), \
         patch("agent_env.providers.env_state.build_state_provider", return_value=provider):
        res = CliRunner().invoke(
            init_env_state, ["--id", "e", "--env-state-type", "remote_postgres"]
        )
    assert res.exit_code == 0, res.output
    assert provider.prepare.await_args.args[0] == ["slack", "email"]
