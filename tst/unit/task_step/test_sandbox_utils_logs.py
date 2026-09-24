"""fetch_container_logs must surface the agent's real error for BOTH agent shapes.

Regression cover for a production failure: three openclaw runs died with
``error_type=cli_error`` and "no container logs were retrievable", while the
actual cause — a LiteLLM 429 (``Current limit: 500000, Remaining: 0``) — sat in
``/tmp/openclaw-gateway.log`` on the sandbox. The openclaw sidecar is the
sandbox's main process, so ``docker ps`` raised, the single enclosing try
swallowed it, and the fetch returned None before any log file was read.
"""

from __future__ import annotations

import types

import pytest

from agent_env.task_step.task_steps.sandbox_utils import sandbox_utils
from agent_env.task_step.task_steps.sandbox_utils.sandbox_utils import (
    _salient_log_lines,
    fetch_container_logs,
)

# Verbatim tail from the sandbox of the reproduced failure, trimmed to the
# lines that matter plus enough boot noise to prove the filter earns its keep.
GATEWAY_LOG = """2026-08-19T22:00:35.935+00:00 [gateway] ready
2026-08-19T22:00:38.705+00:00 [gateway] update available (latest): v2026.7.1-2
2026-08-19T22:03:08.943+00:00 [diagnostic] liveness warning: reasons=event_loop_delay,cpu
2026-08-19T22:03:21.672+00:00 [agent/embedded] embedded run agent end: isError=true \
model=anthropic/claude-fable-5 provider=litellm rawError=429 {"error":{"message":"429: \
Rate limit exceeded for api_key: team-key. Limit type: tokens. Current limit: \
500000, Remaining: 0. Limit resets at: 2026-08-19 22:04:21 UTC"}}
2026-08-19T22:03:41.419+00:00 [agent/embedded] embedded run failover decision: \
stage=assistant decision=surface_error reason=rate_limit
"""


class FakeSandbox:
    def __init__(self, docker_ok: bool, files: dict[str, str]):
        self.docker_ok, self.files = docker_ok, files

    async def exec_with_output(self, *args):
        if args[:2] == ("sudo", "docker"):
            if not self.docker_ok:
                raise RuntimeError("docker: command not found")
            return 0, "", ""
        if args[0] == "sh":
            for path, body in self.files.items():
                if path in args[2]:
                    return 0, body, ""
            return 1, "", ""
        return 0, "", ""


@pytest.fixture
def fake_provider(monkeypatch):
    def _install(docker_ok: bool, files: dict[str, str]):
        sandbox = FakeSandbox(docker_ok, files)

        class _Provider:
            async def get_sandbox(self, _sandbox_id):
                return sandbox

        import agent_env.providers.sandbox_provider as sp

        monkeypatch.setattr(sp, "build_sandbox_provider", lambda _t: _Provider())
        monkeypatch.setattr(sp, "get_sandbox_provider", lambda: _Provider())
        return types.SimpleNamespace(sandbox_id="sb-01TEST", sandbox_type="modal")

    return _install


@pytest.mark.asyncio
async def test_reads_log_file_when_agent_is_not_containerized(fake_provider):
    """The openclaw shape: `docker ps` raises, the gateway log still surfaces."""
    agent = fake_provider(docker_ok=False, files={"/tmp/openclaw-gateway.log": GATEWAY_LOG})
    out = await fetch_container_logs(agent)
    assert out is not None, "docker failure must not abort the whole fetch"
    assert "/tmp/openclaw-gateway.log" in out
    assert "Rate limit exceeded for api_key: team-key" in out
    assert "reason=rate_limit" in out


@pytest.mark.asyncio
async def test_drops_boot_noise(fake_provider):
    agent = fake_provider(docker_ok=False, files={"/tmp/openclaw-gateway.log": GATEWAY_LOG})
    out = await fetch_container_logs(agent)
    assert "update available" not in out
    assert "[gateway] ready" not in out


@pytest.mark.asyncio
async def test_returns_none_when_nothing_to_report(fake_provider):
    """Contract preserved: the caller's own fallback message depends on None."""
    agent = fake_provider(docker_ok=False, files={})
    assert await fetch_container_logs(agent) is None


@pytest.mark.asyncio
async def test_no_sandbox_id_is_none():
    assert await fetch_container_logs(types.SimpleNamespace()) is None


def test_salient_lines_fall_back_to_tail_when_no_marker_matches():
    """An unrecognized failure shape must still yield something, not nothing."""
    assert _salient_log_lines("alpha\nbeta\ngamma\n") == "alpha\nbeta\ngamma"


@pytest.mark.parametrize("blank", ["", "   \n\n  "])
def test_salient_lines_empty_input(blank):
    assert _salient_log_lines(blank) == ""


def test_salient_lines_are_bounded():
    long_log = "\n".join(f"line {i} ERROR" for i in range(200))
    out = _salient_log_lines(long_log)
    assert len(out) <= 2000
    assert len(out.splitlines()) <= 8
    assert "line 199 ERROR" in out, "must keep the newest lines"


def test_gateway_log_path_is_covered():
    assert "/tmp/openclaw-gateway.log" in sandbox_utils._AGENT_LOG_PATHS
