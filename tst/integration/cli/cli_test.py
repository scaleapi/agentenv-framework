import os
import subprocess
import uuid

import pytest
from pathlib import Path
from click.testing import CliRunner
from agent_env.cli import cli
from agent_env.cli.up import _bootstrap_envs

REPO_ROOT = Path(__file__).resolve().parents[3]
TST_DATA_DIR = REPO_ROOT / "tst" / "data"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run agent-env CLI via subprocess."""
    return subprocess.run(
        ["agent-env", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "AGENT_ENV_ENVIRONMENT": "dev"},
    )


@pytest.fixture(scope="class")
def test_run_id():
    """Unique ID for this test run to avoid artifact conflicts."""
    return uuid.uuid4().hex[:8]


class TestCliHelp:

    def test_env_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', '--help'])
        assert result.exit_code == 0
        assert 'deploy' in result.output
        assert 'gateway' in result.output
        assert 'mcp-server' in result.output
        assert 'multi' in result.output
        assert 'website' in result.output

    def test_deploy_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'deploy', '--help'])
        assert result.exit_code == 0
        assert '--id' in result.output
        assert '--ttl-seconds' in result.output

    def test_mcp_server_put_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'mcp-server', 'put', '--help'])
        assert result.exit_code == 0
        assert '--id' in result.output
        assert '--dockerfile' in result.output
        assert '--context' in result.output
        assert '--environment-name' in result.output
        assert '--service-name' not in result.output
        assert '--service-version' in result.output

    def test_mcp_server_load_environment_artifact_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'mcp-server', 'load-environment-artifact', '--help'])
        assert result.exit_code == 0
        assert '--id' in result.output
        assert '--environment-artifact-id' in result.output
        # The command addresses a deployment by instance id, not by URL: it has never
        # taken a --mcp-url, so asserting one was asserting the command's own absence.
        assert '--instance-id' in result.output
        assert '--mcp-url' not in result.output
        assert '--service-artifact-id' not in result.output

    def test_artifact_environment_put_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['artifact', 'environment', 'put', '--help'])
        assert result.exit_code == 0
        assert '--id' in result.output
        assert '--description' in result.output
        assert '--environment-name' in result.output
        assert '--service-name' not in result.output

    def test_artifact_environment_universe_put_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['artifact', 'environment-universe', 'put', '--help'])
        assert result.exit_code == 0
        assert '--id' in result.output
        assert '--environment-artifact' in result.output
        assert '--service-artifact' not in result.output

    def test_multi_put_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'multi', 'put', '--help'])
        assert result.exit_code == 0
        assert '--id' in result.output
        assert '--mcp-server' in result.output
        assert '--website' in result.output

    def test_multi_load_environment_artifact_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'multi', 'load-environment-artifact', '--help'])
        assert result.exit_code == 0
        assert '--env-instance-id' in result.output
        assert '--environment-artifact-id' in result.output
        assert '--service-artifact-id' not in result.output

    def test_multi_load_environment_universe_artifact_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'multi', 'load-environment-universe-artifact', '--help'])
        assert result.exit_code == 0
        assert '--env-instance-id' in result.output
        assert '--environment-universe-artifact-id' in result.output
        assert '--service-universe-artifact-id' not in result.output

    def test_website_put_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'website', 'put', '--help'])
        assert result.exit_code == 0
        assert '--id' in result.output
        assert '--backend-dockerfile' in result.output
        assert '--backend-docker-context' in result.output
        assert '--frontend-dockerfile' in result.output
        assert '--frontend-docker-context' in result.output
        assert '--environment-name' in result.output
        assert '--service-name' not in result.output
        assert '--service-version' in result.output

    def test_website_load_environment_artifact_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'website', 'load-environment-artifact', '--help'])
        assert result.exit_code == 0
        assert '--id' in result.output
        assert '--environment-artifact-id' in result.output
        # As above: instance id, never a URL.
        assert '--instance-id' in result.output
        assert '--mcp-url' not in result.output
        assert '--service-artifact-id' not in result.output


class TestCliValidation:

    def test_deploy_missing_id(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'deploy'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output or '--id' in result.output

    def test_mcp_server_put_missing_dockerfile(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'mcp-server', 'put', '--id', 'test'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output or '--dockerfile' in result.output

    def test_multi_put_missing_both_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'multi', 'put', '--id', 'test'])
        assert result.exit_code != 0
        assert 'At least one' in result.output

    def test_website_put_missing_backend_dockerfile(self):
        runner = CliRunner()
        result = runner.invoke(cli, ['env', 'website', 'put', '--id', 'test'])
        assert result.exit_code != 0
        assert 'Missing option' in result.output or '--backend-dockerfile' in result.output


@pytest.mark.integration
@pytest.mark.int_test_slow
class TestCliIntegration:

    @pytest.fixture(scope="class", autouse=True)
    def default_envs(self):
        """The gateway and service-db envs that `env deploy` resolves by id, built when the store lacks them."""
        _bootstrap_envs()

    def test_mcp_server_put_email(self, test_run_id):
        result = run_cli(
            'env', 'mcp-server', 'put',
            '--id', f'cli-test-email-{test_run_id}',
            '--environment-name', 'email',
            '--service-version', '1',
            '--dockerfile', str(TST_DATA_DIR / 'email_mcp' / 'Dockerfile'),
            '--context', str(TST_DATA_DIR),
            '--skip-validation',
        )
        assert result.returncode == 0, f"Command failed: {result.stdout}\n{result.stderr}"
        assert 'Created MCPServerEnv' in result.stdout

    def test_mcp_server_put_slack(self, test_run_id):
        result = run_cli(
            'env', 'mcp-server', 'put',
            '--id', f'cli-test-slack-{test_run_id}',
            '--environment-name', 'slack',
            '--service-version', '1',
            '--dockerfile', str(TST_DATA_DIR / 'slack_mcp' / 'Dockerfile'),
            '--context', str(TST_DATA_DIR),
            '--skip-validation',
        )
        assert result.returncode == 0, f"Command failed: {result.stdout}\n{result.stderr}"
        assert 'Created MCPServerEnv' in result.stdout

    def test_multi_put(self, test_run_id):
        result = run_cli(
            'env', 'multi', 'put',
            '--id', f'cli-test-multi-{test_run_id}',
            '--mcp-server', f'cli-test-email-{test_run_id}',
            '--mcp-server', f'cli-test-slack-{test_run_id}',
        )
        assert result.returncode == 0, f"Command failed: {result.stdout}\n{result.stderr}"
        assert 'Created MultiEnv' in result.stdout

    def test_deploy(self, test_run_id):
        result = run_cli(
            'env', 'deploy',
            '--id', f'cli-test-multi-{test_run_id}',
        )
        assert result.returncode == 0, f"Command failed: {result.stdout}\n{result.stderr}"
        assert 'Deployed!' in result.stdout
        assert 'Env Gateway Url:' in result.stdout
        assert 'Env MCP Url:' in result.stdout
