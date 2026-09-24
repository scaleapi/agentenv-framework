"""``env deploy`` must be able to set vCPUs, not just memory and disk.

The provider default is 1 vCPU, which is fine for a bare env but starves a universe load:
every service unzips and inserts into one Postgres concurrently, so a small service can blow
its 600s data-plane timeout purely from contention while a much larger one succeeds. Sizing
the VM up was impossible from the CLI — `--memory-mb` and `--disk-size-gb` existed, `--cpu`
did not — so the only way to load a large universe was through the Task path.

Only the `env.deploy` seam is mocked; the real click command runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from agent_env.cli.env import env as env_cli
from agent_env.env.env import DeployedEnv


def _deploy_kwargs(extra_args: list[str]) -> dict:
    captured: dict = {}

    async def fake_deploy(**kwargs):
        captured.update(kwargs)
        return DeployedEnv(
            env_id="e1", env_version=1, gateway_url="http://gw", mcp_url="http://mcp",
            db_web_url=None, sandbox_id="s1",
        )

    fake_env = MagicMock(id="e1", version=1, type="multi")
    fake_env.deploy = fake_deploy
    with patch("agent_env.cli.env.deploy.Env") as Env, \
         patch("agent_env.providers.build_sandbox_provider"):
        Env.get.return_value = fake_env
        res = CliRunner().invoke(env_cli, ["deploy", "--id", "e1", *extra_args])
    assert res.exit_code == 0, res.output
    return captured


def test_cpu_reaches_env_deploy():
    assert _deploy_kwargs(["--cpu", "8"])["cpu"] == 8.0


def test_cpu_accepts_a_fractional_value():
    assert _deploy_kwargs(["--cpu", "0.5"])["cpu"] == 0.5


def test_cpu_is_omitted_when_not_given():
    """Absent, not 1.0: an unset flag must leave the provider default in force rather than
    pinning a value the CLI invented."""
    assert "cpu" not in _deploy_kwargs([])


def test_cpu_composes_with_memory_and_disk():
    kwargs = _deploy_kwargs(["--cpu", "8", "--memory-mb", "32768", "--disk-size-gb", "100"])
    assert kwargs["cpu"] == 8.0
    assert kwargs["memory_mb"] == 32768
    assert kwargs["disk_size_gb"] == 100.0
