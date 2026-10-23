"""Pins which codex_bridge.CodexResult becomes a failed task, and the terminal message it carries.

codex_bridge decides what counts as an error; the `failed` vs `completed` outcome
lives in `run()`, so it needs its own coverage. Driven with canned CodexResults
through the A2A endpoint — no CLI, no LiteLLM.
"""
import asyncio
import tomllib

import pytest

from tst.unit.agents.codex.agent import a2a_server, codex_bridge, codex_config


def _result(**kw):
    base = dict(response="answer", tool_call_count=7, session_id="tid-1")
    base.update(kw)
    return codex_bridge.CodexResult(**base)


@pytest.fixture
def drive(codex, monkeypatch):
    """run(result) -> (state, text, the agent's data part, the framework's data part)."""

    def _run(result: codex_bridge.CodexResult, prompt: str = "do the thing"):
        async def fake_run_codex(**kwargs):
            return result

        monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

        async def exercise():
            async with codex.client() as client:
                return await codex.send(client, prompt, context_id="ctx-1")

        state, text, (own, framework) = codex.terminal(asyncio.run(exercise()))
        return state, text, own, framework

    return _run


@pytest.mark.parametrize("kwargs, outcome, has_warnings", [
    ({}, "completed", False),
    ({"warnings": ["Model metadata for `x` not found"]}, "completed", True),
    ({"error": "stream disconnected"}, "failed", False),
    ({"error": "turn.failed", "warnings": ["retry 1/2"]}, "failed", True),
])
def test_only_error_fails_the_task(drive, kwargs, outcome, has_warnings):
    got, _, data, framework = drive(_result(**kwargs))
    # `tool_call_count` is the key name the a2a validator reads.
    assert (got, "warnings" in data, "error_type" in framework, data["tool_call_count"]) == (
        outcome, has_warnings, outcome == "failed", 7)


def test_warnings_are_capped_and_truncated(drive):
    _, _, data, _ = drive(_result(warnings=["x" * 900] * 25))
    assert (len(data["warnings"]), max(map(len, data["warnings"]))) == (10, 500)


def test_failed_turn_surfaces_error_as_text_and_attempt_count(drive):
    outcome, text, data, _ = drive(
        _result(response="", error="proxy stream disconnected", attempts=3)
    )
    assert (outcome, text, data["cli_attempts"]) == (
        "failed",
        "proxy stream disconnected",
        3,
    )


def test_incremental_usage_is_emitted(drive):
    _, _, data, _ = drive(
        _result(
            token_usage={
                "input_tokens": 80,
                "cached_input_tokens": 70,
                "cache_write_input_tokens": 30,
                "output_tokens": 5,
            }
        )
    )
    assert data["token_usage"] == {
        "input_tokens": 80,
        "cached_input_tokens": 70,
        "cache_write_input_tokens": 30,
        "output_tokens": 5,
    }


def test_session_id_is_bound_to_the_context(drive, codex):
    drive(_result(session_id="tid-99"))
    assert codex.agent.session_ref_for_context("ctx-1") == "tid-99"


def test_independent_contexts_run_concurrently(codex, monkeypatch):
    active = 0
    max_active = 0

    async def fake_run_codex(**kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _result(session_id=f"thread-{kwargs['prompt']}")

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.client() as client:
            await asyncio.gather(
                codex.send(client, "a", context_id="context-a"),
                codex.send(client, "b", context_id="context-b"),
            )

    asyncio.run(exercise())
    assert max_active == 2


def test_workspace_barrier_prioritizes_snapshot_without_leaking_waiters():
    async def exercise():
        barrier = a2a_server._WorkspaceBarrier()
        order = []
        await barrier.acquire_shared()

        async def writer():
            await barrier.acquire_exclusive()
            order.append("writer")
            await asyncio.sleep(0)
            await barrier.release_exclusive()

        async def later_reader():
            await barrier.acquire_shared()
            order.append("reader")
            await barrier.release_shared()

        writer_task = asyncio.create_task(writer())
        for _ in range(100):
            if barrier._waiting_writers:
                break
            await asyncio.sleep(0)
        reader_task = asyncio.create_task(later_reader())
        await asyncio.sleep(0)
        await barrier.release_shared()
        await asyncio.gather(writer_task, reader_task)
        assert order == ["writer", "reader"]

        await barrier.acquire_shared()
        cancelled_writer = asyncio.create_task(
            barrier.acquire_exclusive()
        )
        for _ in range(100):
            if barrier._waiting_writers:
                break
            await asyncio.sleep(0)
        cancelled_writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_writer
        await asyncio.wait_for(barrier.acquire_shared(), timeout=0.1)
        await barrier.release_shared()
        await barrier.release_shared()

    asyncio.run(exercise())


def test_concurrent_same_context_turns_share_one_session(codex, monkeypatch):
    seen_thread_ids = []

    async def fake_run_codex(**kwargs):
        seen_thread_ids.append(kwargs["thread_id"])
        await asyncio.sleep(0.01)
        return _result(session_id="shared-thread")

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.client() as client:
            await asyncio.gather(
                codex.send(client, "a", context_id="same-context"),
                codex.send(client, "b", context_id="same-context"),
            )

    asyncio.run(exercise())
    assert seen_thread_ids == [None, "shared-thread"]


def test_every_server_is_required_and_reaches_mcp_remote_with_the_trust_env(monkeypatch, tmp_path):
    """Codex drops an optional server that misses its 1s grace, which mcp-remote always does; and codex hands a stdio
    server only a few of its variables, so the CA ones are written in."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", "/etc/agentenv/ca.pem")
    monkeypatch.delenv("NODE_TLS_REJECT_UNAUTHORIZED", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    asyncio.run(codex_config.rewrite_mcp_servers_role(
        {"envmcp": {"url": "https://gw/mcp", "headers": None},
         "peer_agents": {"url": "http://127.0.0.1:8000/ext/peer-agents/mcp", "headers": None}}, "viewer"))

    servers = tomllib.loads((tmp_path / "config.toml").read_text())["mcp_servers"]
    assert {name: (server["required"], server["env"]) for name, server in servers.items()} == {
        "envmcp": (True, {"NODE_EXTRA_CA_CERTS": "/etc/agentenv/ca.pem"}),
        "peer_agents": (True, {"NODE_EXTRA_CA_CERTS": "/etc/agentenv/ca.pem"}),
    }
