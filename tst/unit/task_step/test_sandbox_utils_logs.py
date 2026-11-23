"""fetch_container_logs tails the agent's own container, agent_error_text bounds the agent's report, and
ended_turn_error says why a turn ended without completing."""

from __future__ import annotations

import logging
import types

import pytest

from agent_env.task_step.task_steps.sandbox_utils import sandbox_utils
from agent_env.task_step.task_steps.sandbox_utils.sandbox_utils import (
    agent_error_text,
    ended_turn_error,
    fetch_container_logs,
)
from agent_env.a2a_agent.protocol import TerminalResponse


class FakeSandbox:
    def __init__(self, rows=(), logs=None, docker_ok=True, container_name="agent-api"):
        self.rows, self.logs, self.docker_ok = rows, logs or {}, docker_ok
        self.container_name = container_name
        self.calls = []

    async def exec_with_output(self, *args):
        self.calls.append(args)
        if not self.docker_ok:
            raise RuntimeError("docker: command not found")
        if args[:3] == ("sudo", "docker", "ps"):
            return 0, "\n".join(f"{cid}|{name}" for cid, name in self.rows), ""
        if args[:3] == ("sudo", "docker", "logs"):
            out, err = self.logs.get(args[-1], ("", ""))
            return 0, out, err
        raise AssertionError(f"unexpected exec: {args}")


@pytest.fixture
def install(monkeypatch):
    def _install(sandbox=None, get_error=None):
        class _Provider:
            async def get_sandbox(self, _sandbox_id):
                if get_error:
                    raise get_error
                return sandbox

        monkeypatch.setattr(sandbox_utils, "build_sandbox_provider", lambda _t: _Provider())
        monkeypatch.setattr(sandbox_utils, "get_sandbox_provider", lambda: _Provider())
        return types.SimpleNamespace(sandbox_id="sb-01TEST", sandbox_type="beta")

    return _install


@pytest.mark.asyncio
async def test_tails_only_the_agents_own_container(install):
    sandbox = FakeSandbox(
        rows=[("c1", "agent-api"), ("c2", "agent-dind"), ("c3", "agent-other")],
        logs={cid: ("", f"{cid} failed") for cid in ("c1", "c2", "c3")},
    )

    out = await fetch_container_logs(install(sandbox))

    assert out == "[container c1]\n[stderr]\nc1 failed"


@pytest.mark.asyncio
async def test_falls_back_to_every_container_up_to_the_limit(install):
    rows = [(f"c{i}", f"env-{i}") for i in range(1, 6)]
    sandbox = FakeSandbox(rows=rows, logs={cid: (f"{cid} out", "") for cid, _ in rows})

    out = await fetch_container_logs(install(sandbox))

    assert [c for c in ("c1", "c2", "c3", "c4", "c5") if f"[container {c}]" in out] == ["c1", "c2", "c3"]


@pytest.mark.asyncio
async def test_fallback_puts_agent_named_containers_first(install):
    rows = [("c1", "env-db"), ("c2", "env-mcp"), ("c3", "env-web"), ("c4", "agent-worker")]
    sandbox = FakeSandbox(rows=rows, logs={cid: (f"{cid} out", "") for cid, _ in rows})

    out = await fetch_container_logs(install(sandbox))

    assert [c for c in ("c1", "c2", "c3", "c4") if f"[container {c}]" in out] == ["c1", "c2", "c4"]
    assert out.index("[container c4]") < out.index("[container c1]")


@pytest.mark.asyncio
async def test_long_stdout_does_not_push_out_stderr(install):
    sandbox = FakeSandbox(rows=[("c1", "agent-api")], logs={"c1": ("x" * 5000, "model alias not found")})

    out = await fetch_container_logs(install(sandbox))

    assert "model alias not found" in out
    assert out.count("x") == 500


@pytest.mark.asyncio
async def test_no_docker_is_none_after_one_exec(install):
    sandbox = FakeSandbox(docker_ok=False)

    assert await fetch_container_logs(install(sandbox)) is None
    assert len(sandbox.calls) == 1


@pytest.mark.asyncio
async def test_nothing_to_report_is_none(install):
    assert await fetch_container_logs(install(FakeSandbox(rows=[("c1", "agent-api")]))) is None


@pytest.mark.asyncio
async def test_sandbox_lookup_failure_is_none_and_warns(install, caplog):
    agent = install(get_error=RuntimeError("sandbox gone"))

    with caplog.at_level(logging.WARNING, logger=sandbox_utils.__name__):
        assert await fetch_container_logs(agent) is None

    assert "Could not fetch agent container logs from sandbox sb-01TEST" in caplog.text


@pytest.mark.asyncio
async def test_no_sandbox_id_is_none():
    assert await fetch_container_logs(types.SimpleNamespace()) is None


def test_agent_error_text_prefers_the_error_message():
    assert agent_error_text("budget exceeded", "partial answer") == "budget exceeded"
    assert agent_error_text(None, "partial answer") == "partial answer"
    assert agent_error_text(None, None) == ""


def test_agent_error_text_is_bounded():
    assert len(agent_error_text("e" * 2000, None)) == 2000
    assert len(agent_error_text("e" * 2001, None)) == 2000


@pytest.mark.parametrize("reply,expected", [
    (TerminalResponse(response_text="", error_type="api_error", error_code="400", error_class="BadRequestError",
                      error_message="the model was refused"),
     "state failed: error_type=api_error, error_code=400, exception=BadRequestError: the model was refused"),
    (TerminalResponse(response_text="it broke"), "state failed: it broke"),
    (TerminalResponse(response_text=""), "state failed"),
], ids=["full", "text-only", "nothing"])
def test_a_turn_that_ended_without_completing_keeps_the_agents_own_error(reply, expected):
    assert ended_turn_error("failed", reply) == expected
