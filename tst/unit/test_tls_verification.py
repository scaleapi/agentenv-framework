"""Every HTTP client in agent-env verifies TLS. The sandbox providers terminate TLS at their
public tunnel edge with publicly trusted certificates, and every deploy already passes a
default-verifying probe against that edge, so a call site that opts out only weakens the hop.
"""

import pathlib
import re

from agent_env.task_step.task_steps.mcp_cli_builder import generate_cli_script

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "agent_env"
DISABLED = re.compile(r"verify\s*=\s*False")


def _offenders(pattern: re.Pattern[str]) -> list[str]:
    return sorted(str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if pattern.search(p.read_text()))


def test_no_module_disables_tls_verification():
    assert _offenders(DISABLED) == []


def test_no_module_reads_an_insecure_tls_switch():
    assert _offenders(re.compile("INSECURE_TLS")) == []


def test_generated_cli_verifies_tls():
    assert not DISABLED.search(generate_cli_script("svc"))
