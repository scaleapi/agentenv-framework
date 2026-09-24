"""Shared universe-filename validation.

Every loader that stages a universe into a destination directory funnels through
this, so a hostile or malformed filename can't escape via posixpath.join.
"""

from __future__ import annotations

import pytest

from agent_env.utils.paths import validate_relative_filename


@pytest.mark.parametrize("filename", ["a.txt", "sub/dir/deep.bin", "a..b/c.txt"])
def test_accepts_relative_paths(filename):
    validate_relative_filename(filename)


@pytest.mark.parametrize(
    "filename, match",
    [
        ("", "empty filename"),
        ("/etc/passwd", "must be relative"),
        ("a/../../escape.txt", "'\\.\\.' segments"),
        ("a\\..\\..\\escape.txt", "'\\.\\.' segments"),  # windows separators normalised
    ],
)
def test_rejects_unsafe_paths(filename, match):
    with pytest.raises(ValueError, match=match):
        validate_relative_filename(filename)


@pytest.mark.asyncio
async def test_agent_loader_rejects_absolute_filename(monkeypatch):
    """The agent loader was the one staging path with no validation — an
    absolute filename silently escaped `destination` via posixpath.join."""
    from agent_env.a2a_agent.a2a_agent import A2AAgent, DeployedA2AAgent
    from agent_env.providers import sandbox_provider

    class _Sandbox:
        async def exec(self, *a):
            raise AssertionError("must not reach the sandbox")

        async def write_file_from_s3(self, *a):
            raise AssertionError("must not write anything")

    class _Provider:
        async def get_sandbox(self, sandbox_id):
            return _Sandbox()

    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda _t: _Provider())
    monkeypatch.setattr(sandbox_provider, "get_agent_sandbox_provider", lambda: _Provider())

    universe = type("U", (), {
        "id": "u", "version": 1,
        "get_file_artifacts": lambda self: {"/etc/cron.d/evil": object()},
    })()
    deployed = DeployedA2AAgent(
        agent_id="judge", agent_version=1, a2a_url="http://a",
        sandbox_id="sb", agent_card={},
    )

    with pytest.raises(ValueError, match="must be relative"):
        await A2AAgent.load_file_artifact_universe(deployed, universe, "/tmp/dest")
