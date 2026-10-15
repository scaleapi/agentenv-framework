"""The files ``.github/workflows/publish-pypi.yml`` rewrites to pin agentenv-framework-protocol on a release: each must
hold exactly one line its sed matches, or the release fails after the protocol is already on PyPI."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = (REPO / ".github" / "workflows" / "publish-pypi.yml").read_text()


def test_the_pyproject_holds_the_one_quoted_protocol_requirement_the_release_pins():
    assert len(re.findall(r'"agentenv-framework-protocol[^"]*"', (REPO / "pyproject.toml").read_text())) == 1


def test_the_echo_mcp_requirements_the_release_pins_hold_one_protocol_line():
    (path,) = re.findall(r"^\s*echo_requirements=(\S+)$", WORKFLOW, re.MULTILINE)
    lines = (REPO / path).read_text().splitlines()

    assert len([line for line in lines if line.startswith("agentenv-framework-protocol")]) == 1
