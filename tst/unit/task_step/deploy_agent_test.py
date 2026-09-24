"""Unit tests for the _choose_mcp_url dispatch matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent_env.task_step.task_steps.deploy_agent import _choose_mcp_url


@dataclass
class _Agent:
    sandbox_type: Optional[str]


@dataclass
class _Env:
    sandbox_type: Optional[str]
    mcp_url: str


def test_modal_to_modal_uses_public_url():
    agent = _Agent(sandbox_type="modal")
    env = _Env(sandbox_type="modal", mcp_url="https://pub/mcp")
    assert _choose_mcp_url(agent, env) == "https://pub/mcp"


def test_remote_agent_modal_env_uses_public():
    agent = _Agent(sandbox_type="remote")
    env = _Env(sandbox_type="modal", mcp_url="https://pub/mcp")
    assert _choose_mcp_url(agent, env) == "https://pub/mcp"


def test_remote_to_remote_uses_env_mcp_url():
    agent = _Agent(sandbox_type="remote")
    env = _Env(sandbox_type="remote", mcp_url="https://env-pub/mcp")
    assert _choose_mcp_url(agent, env) == "https://env-pub/mcp"
