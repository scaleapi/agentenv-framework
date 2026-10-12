"""``agent-env -v`` logs agent-env's own DEBUG messages, and only warnings from other libraries, whose debug logs can
carry credentials: hpack logs every HTTP/2 header Modal's client sends, its token among them."""

import logging
import subprocess
import sys

from click.testing import CliRunner

from agent_env.cli import cli

# In a fresh interpreter, as the CLI runs: pytest's own handlers on the root logger would make basicConfig a no-op here.
_LOGS = """
import logging
from agent_env.cli import _verbose_logging
with _verbose_logging():
    logging.getLogger("agent_env.providers.sandbox_providers.modal_sandbox").debug("creating a sandbox")
    logging.getLogger("agentenv_protocol.client").debug("reading a card")
    logging.getLogger("hpack.hpack").debug("Adding b'x-modal-token-secret'=b'as-0000' to the header table")
    logging.getLogger("hpack.hpack").warning("a header table warning")
"""


def test_verbose_logs_agent_envs_debug_messages_and_only_warnings_from_other_libraries():
    done = subprocess.run([sys.executable, "-c", _LOGS], capture_output=True, text=True, timeout=120)

    assert done.returncode == 0, done.stderr
    assert "creating a sandbox" in done.stderr and "reading a card" in done.stderr
    assert "a header table warning" in done.stderr
    assert "x-modal-token-secret" not in done.stderr


def test_a_verbose_command_run_in_process_leaves_the_callers_logging_as_it_was():
    """Run in a caller's process, as the tests and any embedding code run it, -v lasts only for its command."""
    loggers = [logging.getLogger(name) for name in ("agent_env", "agentenv_protocol")]
    before = [logger.level for logger in loggers]

    result = CliRunner().invoke(cli, ["-v", "config", "--help"])

    assert result.exit_code == 0, result.output
    assert [logger.level for logger in loggers] == before
