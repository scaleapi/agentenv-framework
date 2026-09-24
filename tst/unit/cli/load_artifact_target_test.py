"""`load-environment-artifact` must work on every sandbox backend.

Only some gateway URLs carry the sandbox id; Modal's carries a tunnel id, so
parsing one died with `IndexError` on every modal_vm deploy. The fix routes
through `Env.from_instance_id`, which reconnects via the instance's
`sandbox_type` and restores `_instance_id`.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import AsyncMock, MagicMock, patch

from agent_env.cli.utils import deployed_env_from_instance


def test_instance_id_goes_through_the_env_rehydration_api():
    env = MagicMock(id="env-1")
    with patch("agent_env.env.Env.from_instance_id",
               new=AsyncMock(return_value=env)) as from_instance:
        assert deployed_env_from_instance(None, "inst-1") is env
    from_instance.assert_awaited_once_with("inst-1")


def test_the_cli_cannot_fall_back_to_the_default_sandbox_provider():
    """The provider must come from the instance's sandbox_type, via
    Env.from_instance_id — a default provider 404s on a modal sandbox id."""
    import agent_env.cli.utils as utils

    assert not hasattr(utils, "get_env_sandbox_provider")
    assert not hasattr(utils, "extract_sandbox_id")


def test_mismatched_env_id_is_rejected():
    from click import UsageError

    env = MagicMock(id="env-1")
    with patch("agent_env.env.Env.from_instance_id", new=AsyncMock(return_value=env)):
        with pytest.raises(UsageError) as e:
            deployed_env_from_instance("other-env", "inst-1")
    assert "does not match" in str(e.value)


def test_missing_instance_is_a_usage_error():
    from click import UsageError
    from agent_env.store.base import NotFoundError

    with patch("agent_env.env.Env.from_instance_id",
               new=AsyncMock(side_effect=NotFoundError("nope"))):
        with pytest.raises(UsageError) as e:
            deployed_env_from_instance(None, "inst-1")
    assert "not found" in str(e.value)





@pytest.mark.parametrize("group_name", ["mcp_server", "website"])
def test_both_clis_offer_instance_id_and_no_longer_force_mcp_url(group_name):
    import importlib

    mod = importlib.import_module(f"agent_env.cli.env.{group_name}")
    out = CliRunner().invoke(getattr(mod, group_name),
                             ["load-environment-artifact", "--help"]).output
    assert "--instance-id" in out
    assert "--mcp-url" not in out
