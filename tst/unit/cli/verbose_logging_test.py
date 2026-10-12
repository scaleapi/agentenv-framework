"""``agent-env -v`` logs agent-env's own DEBUG messages, and only warnings from other libraries, whose debug logs can
carry credentials: hpack logs every HTTP/2 header Modal's client sends, its token among them."""

import subprocess
import sys

# In a fresh interpreter, as the CLI runs: pytest's own handlers on the root logger would make basicConfig a no-op here.
_LOGS = """
import logging
from agent_env.cli import _verbose_logging
_verbose_logging()
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
