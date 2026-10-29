"""Transient terminal failures resume the persisted Codex thread safely."""
import asyncio
import json

import pytest

from tst.unit.agents.codex.agent import codex_bridge


def _line(event: dict) -> bytes:
    return (json.dumps(event) + "\n").encode()


THREAD_ID = "019feca9-6ad2-7753-bbc4-0f4b402b3af3"
THREAD_STARTED = {"type": "thread.started", "thread_id": THREAD_ID}
TURN_STARTED = {"type": "turn.started"}
TURN_COMPLETED = {"type": "turn.completed"}


def _failed(message: str, **error_fields) -> bytes:
    return b"".join(
        [
            _line(THREAD_STARTED),
            _line(TURN_STARTED),
            _line(
                {
                    "type": "item.completed",
                    "item": {"id": "tool-1", "type": "command_execution"},
                }
            ),
            _line(
                {
                    "type": "turn.failed",
                    "error": {"message": message, **error_fields},
                }
            ),
        ]
    )


def _success(text: str = "done") -> bytes:
    return b"".join(
        [
            _line(TURN_STARTED),
            _line(
                {
                    "type": "item.completed",
                    "item": {"id": "answer", "type": "agent_message", "text": text},
                }
            ),
            _line(TURN_COMPLETED),
        ]
    )


class _FakeStream:
    def __init__(self, data: bytes):
        self.data = data

    async def read(self, n: int = -1) -> bytes:
        if n < 0:
            data, self.data = self.data, b""
            return data
        chunk, self.data = self.data[:n], self.data[n:]
        return chunk


class _FakeStdin:
    def __init__(self):
        self.data = b""

    def write(self, data: bytes):
        self.data += data

    async def drain(self):
        return None

    def close(self):
        return None


class _FakeProc:
    def __init__(self, stdout: bytes, returncode: int):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(b"")
        self.returncode = returncode

    async def wait(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


class _StallingStream(_FakeStream):
    async def read(self, n: int = -1) -> bytes:
        if self.data:
            return await super().read(n)
        await asyncio.sleep(3600)
        return b""


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    queue: list[tuple[bytes, int]] = []
    calls: list[dict] = []

    async def _fake_exec(*cmd, **kwargs):
        stdout, returncode = queue.pop(0)
        proc = _FakeProc(stdout, returncode)
        calls.append({"cmd": list(cmd), "proc": proc, "kwargs": kwargs})
        return proc

    rollout = tmp_path / f"rollout-{THREAD_ID}.jsonl"
    rollout.write_text(json.dumps({"type": "session_meta"}) + "\n")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(codex_bridge, "find_rollout_path", lambda _: rollout)
    monkeypatch.setattr(codex_bridge, "_RETRY_BACKOFF_S", (0, 0))
    monkeypatch.delenv("CODEX_API_ERROR_RETRIES", raising=False)
    monkeypatch.chdir(tmp_path)
    return {"queue": queue, "calls": calls}


def test_transient_failure_resumes_without_replaying_prompt(scripted):
    scripted["queue"].extend(
        [
            (_failed("stream disconnected before completion"), 1),
            (_success("recovered"), 0),
        ]
    )

    result = asyncio.run(
        codex_bridge.run_codex(prompt="original prompt", thread_id=None)
    )

    assert result.error is None
    assert result.response == "recovered"
    assert result.attempts == 2
    assert result.tool_call_count == 1
    assert "resume" in scripted["calls"][1]["cmd"]
    assert THREAD_ID in scripted["calls"][1]["cmd"]
    assert scripted["calls"][0]["proc"].stdin.data == b"original prompt"
    assert scripted["calls"][1]["proc"].stdin.data == codex_bridge._RESUME_PROMPT.encode()
    assert sum(
        event.get("type") == "turn.failed" for event in result.trajectory
    ) == 1


def test_a_recovered_turn_reports_exactly_its_trajectory(scripted):
    scripted["queue"].extend(
        [
            (_failed("stream disconnected before completion"), 1),
            (_success("recovered"), 0),
        ]
    )
    reported: list[dict] = []

    result = asyncio.run(
        codex_bridge.run_codex(prompt="original prompt", thread_id=None, event_callback=reported.append)
    )

    assert result.attempts == 2
    assert reported == result.trajectory


@pytest.mark.parametrize(
    "message",
    [
        "authentication failed",
        "HTTP 400 invalid request",
        "prompt is too long",
        "quota exhausted",
        "invalid configuration",
        "thread not found",
        "connection refused",
        "something unknown",
    ],
)
def test_permanent_or_unknown_failure_is_not_retried(scripted, message):
    scripted["queue"].append((_failed(message), 1))
    result = asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None))
    assert result.attempts == 1
    assert result.error is not None
    assert len(scripted["calls"]) == 1


def test_missing_rollout_blocks_resume(scripted, monkeypatch):
    monkeypatch.setattr(codex_bridge, "find_rollout_path", lambda _: None)
    scripted["queue"].append((_failed("HTTP 503 service unavailable"), 1))
    result = asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None))
    assert result.attempts == 1


def test_nested_client_status_vetoes_transient_words(scripted):
    nested = json.dumps(
        {"error": {"message": "internal server error", "code": "400"}}
    )
    scripted["queue"].append((_failed(nested), 1))
    result = asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None))
    assert result.attempts == 1


def test_structured_server_status_is_retried(scripted):
    scripted["queue"].extend(
        [
            (_failed("provider failure", status_code=503), 1),
            (_success(), 0),
        ]
    )
    result = asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None))
    assert result.attempts == 2
    assert result.error is None


@pytest.mark.parametrize(
    "failure",
    [
        _failed("HTTP 429 rate limited"),
        _failed("provider overloaded", status_code=429),
    ],
)
def test_rate_limits_are_retried(scripted, failure):
    scripted["queue"].extend([(failure, 1), (_success(), 0)])
    result = asyncio.run(
        codex_bridge.run_codex(prompt="p", thread_id=None)
    )
    assert result.attempts == 2
    assert result.error is None


@pytest.mark.parametrize(
    "message",
    [
        "HTTP 429 quota exhausted",
        "HTTP 429 spend cap reached",
    ],
)
def test_permanent_rate_limits_are_not_retried(scripted, message):
    scripted["queue"].append((_failed(message), 1))
    result = asyncio.run(
        codex_bridge.run_codex(prompt="p", thread_id=None)
    )
    assert result.attempts == 1
    assert result.error is not None


def test_zero_exit_requires_terminal_success(scripted):
    scripted["queue"].append(
        (
            _line(THREAD_STARTED)
            + _line(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "partial"},
                }
            ),
            0,
        )
    )
    result = asyncio.run(
        codex_bridge.run_codex(prompt="p", thread_id=None)
    )
    assert result.response == "partial"
    assert result.error == "codex CLI exited without turn.completed"


def test_completed_turn_survives_lingering_process(monkeypatch, tmp_path):
    stopped = asyncio.Event()

    class LingeringProc(_FakeProc):
        async def wait(self):
            await stopped.wait()
            return self.returncode

        def kill(self):
            self.returncode = -9
            stopped.set()

    proc = LingeringProc(_success(), 0)

    async def _fake_exec(*cmd, **kwargs):
        return proc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    result = asyncio.run(
        codex_bridge.run_codex(
            prompt="p", thread_id=None, timeout_s=0.01
        )
    )
    assert result.error is None
    assert result.response == "done"
    assert proc.returncode == -9


@pytest.mark.parametrize("split_terminal_record", [False, True])
def test_unterminated_completion_at_deadline_wins_over_timeout(
    monkeypatch, split_terminal_record
):
    output = _success().rstrip(b"\n")
    proc = _FakeProc(output, 0)
    if split_terminal_record:
        split_at = output.rfind(b'"turn.completed"') + 8

        class SplitStream:
            def __init__(self):
                self.chunks = [output[:split_at], output[split_at:]]

            async def read(self, n=-1):
                return self.chunks.pop(0) if self.chunks else b""

        proc.stdout = SplitStream()

    class DeadlineLoop:
        calls = 0

        def time(self):
            self.calls += 1
            return 0 if self.calls <= 2 else 2

    monkeypatch.setattr(
        asyncio, "get_event_loop", lambda: DeadlineLoop()
    )

    async def exercise():
        return await codex_bridge._read_with_watchdog(
            proc, 1, b"prompt"
        )

    events, _, timed_out, stalled = asyncio.run(exercise())
    assert events[-1] == TURN_COMPLETED
    assert not timed_out and not stalled


def test_initial_attempt_receives_full_deadline(monkeypatch):
    seen = []

    async def fake_once(**kwargs):
        seen.append(kwargs["timeout_s"])
        return codex_bridge.CodexResult(
            response="ok",
            tool_call_count=0,
            session_id=THREAD_ID,
            trajectory=[TURN_COMPLETED],
        )

    monkeypatch.setattr(codex_bridge, "_run_codex_once", fake_once)
    result = asyncio.run(
        codex_bridge.run_codex(
            prompt="p", thread_id=None, timeout_s=60
        )
    )
    assert result.error is None
    assert seen[0] > 59


def test_retry_reserves_headroom_and_cleanup(monkeypatch, tmp_path):
    seen = []
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("{}\n")

    async def fake_once(**kwargs):
        seen.append(kwargs["timeout_s"])
        if len(seen) == 1:
            return codex_bridge.CodexResult(
                response="",
                tool_call_count=0,
                session_id=THREAD_ID,
                trajectory=[
                    {
                        "type": "turn.failed",
                        "error": {"message": "HTTP 503"},
                    }
                ],
                error="HTTP 503",
            )
        return codex_bridge.CodexResult(
            response="ok",
            tool_call_count=0,
            session_id=THREAD_ID,
            trajectory=[TURN_COMPLETED],
        )

    monkeypatch.setattr(codex_bridge, "_run_codex_once", fake_once)
    monkeypatch.setattr(
        codex_bridge, "find_rollout_path", lambda _: rollout
    )
    monkeypatch.setattr(codex_bridge, "_RETRY_BACKOFF_S", (0, 0))
    result = asyncio.run(
        codex_bridge.run_codex(
            prompt="p", thread_id=None, timeout_s=600
        )
    )
    assert result.error is None
    assert seen[1] + codex_bridge._DEADLINE_HEADROOM_S <= 600


def test_retries_are_bounded(scripted, monkeypatch):
    monkeypatch.setenv("CODEX_API_ERROR_RETRIES", "1")
    scripted["queue"].extend(
        [
            (_failed("HTTP 503 service unavailable"), 1),
            (_failed("HTTP 503 service unavailable"), 1),
        ]
    )
    result = asyncio.run(codex_bridge.run_codex(prompt="p", thread_id=None))
    assert result.attempts == 2
    assert "HTTP 503" in (result.error or "")


def test_exhausted_budget_blocks_resume(scripted, monkeypatch):
    monkeypatch.setattr(codex_bridge, "_MIN_RETRY_BUDGET_S", 999)
    scripted["queue"].append((_failed("HTTP 503 service unavailable"), 1))
    result = asyncio.run(
        codex_bridge.run_codex(prompt="p", thread_id=None, timeout_s=1)
    )
    assert result.attempts == 1


def test_retry_admission_preserves_minimum_budget(scripted, monkeypatch):
    monkeypatch.setattr(codex_bridge, "_RETRY_BACKOFF_S", (0, 0))
    scripted["queue"].append(
        (_failed("HTTP 503 service unavailable"), 1)
    )
    result = asyncio.run(
        codex_bridge.run_codex(
            prompt="p", thread_id=None, timeout_s=149
        )
    )
    assert result.attempts == 1


def test_stall_preserves_partial_thread_and_trajectory(monkeypatch, tmp_path):
    proc = _FakeProc(_line(THREAD_STARTED) + _line(TURN_STARTED), 0)
    proc.stdout = _StallingStream(proc.stdout.data)

    async def _fake_exec(*cmd, **kwargs):
        return proc

    rollout = tmp_path / f"rollout-{THREAD_ID}.jsonl"
    rollout.write_text("{}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(codex_bridge, "find_rollout_path", lambda _: rollout)
    monkeypatch.setattr(codex_bridge, "_INACTIVITY_TIMEOUT_S", 0.01)
    monkeypatch.setenv("CODEX_API_ERROR_RETRIES", "2")
    seen = []

    result = asyncio.run(
        codex_bridge.run_codex(
            prompt="p",
            thread_id=None,
            timeout_s=60,
            event_callback=seen.append,
        )
    )
    assert result.session_id == THREAD_ID
    assert result.attempts == 1
    assert [event["type"] for event in result.trajectory] == [
        "thread.started",
        "turn.started",
    ]
    assert [event["type"] for event in seen] == [
        "thread.started",
        "turn.started",
    ]
    assert "no output" in (result.error or "")


def test_stdin_and_stdout_are_drained_concurrently(monkeypatch, tmp_path):
    gate = asyncio.Event()
    proc = _FakeProc(_line(THREAD_STARTED) + _line(TURN_COMPLETED), 0)

    class CoordinatedStdin(_FakeStdin):
        async def drain(self):
            await gate.wait()

    class CoordinatedStdout(_FakeStream):
        async def read(self, n=-1):
            gate.set()
            return await super().read(n)

    proc.stdin = CoordinatedStdin()
    proc.stdout = CoordinatedStdout(proc.stdout.data)

    async def _fake_exec(*cmd, **kwargs):
        return proc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    result = asyncio.run(
        codex_bridge.run_codex(prompt="x" * 200_000, thread_id=None)
    )
    assert result.error is None
    assert len(proc.stdin.data) == 200_000


def test_cancellation_kills_codex_process(monkeypatch, tmp_path):
    proc = _FakeProc(b"", 0)
    proc.stdout = _StallingStream(b"")
    started = asyncio.Event()

    async def _fake_exec(*cmd, **kwargs):
        started.set()
        return proc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    async def exercise():
        task = asyncio.create_task(
            codex_bridge.run_codex(prompt="p", thread_id=None)
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())
    assert proc.returncode == -9


def test_deadline_bounds_process_wait_after_stdout_eof(monkeypatch, tmp_path):
    stopped = asyncio.Event()

    class WaitingProc(_FakeProc):
        async def wait(self):
            await stopped.wait()
            return self.returncode

        def kill(self):
            self.returncode = -9
            stopped.set()

    proc = WaitingProc(b"", 0)

    async def _fake_exec(*cmd, **kwargs):
        return proc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    result = asyncio.run(
        codex_bridge.run_codex(prompt="p", thread_id=None, timeout_s=0.05)
    )
    assert "timed out" in (result.error or "")
    assert proc.returncode == -9


def test_cleanup_wait_is_bounded_after_kill(monkeypatch, tmp_path):
    class UnreapableProc(_FakeProc):
        async def wait(self):
            await asyncio.sleep(3600)

    proc = UnreapableProc(b"", 0)

    async def _fake_exec(*cmd, **kwargs):
        return proc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(codex_bridge, "_PROCESS_CLEANUP_TIMEOUT_S", 0.01)
    result = asyncio.run(
        codex_bridge.run_codex(prompt="p", thread_id=None, timeout_s=0.01)
    )
    assert "timed out" in (result.error or "")
    assert proc.returncode == -9


def test_process_group_cleanup_is_idempotent(monkeypatch):
    class Gone:
        pid = 123

        def kill(self):
            raise ProcessLookupError

    def gone(*_):
        raise ProcessLookupError

    monkeypatch.setattr(codex_bridge.os, "killpg", gone)
    codex_bridge._kill_process_group(Gone())


def test_stderr_is_drained_with_bounded_retention(monkeypatch):
    proc = _FakeProc(_line(TURN_COMPLETED), 0)
    proc.stderr = _FakeStream(b"x" * 100)
    monkeypatch.setattr(codex_bridge, "_MAX_STDERR_BYTES", 10)

    async def exercise():
        return await codex_bridge._read_with_watchdog(
            proc, 1, b"prompt"
        )

    _, stderr, timed_out, stalled = asyncio.run(exercise())
    assert not timed_out and not stalled
    assert stderr == b"x" * 10 + b"\n[stderr truncated]"


def test_oversized_event_kills_codex_process(monkeypatch, tmp_path):
    proc = _FakeProc(b"xx", 0)

    async def _fake_exec(*cmd, **kwargs):
        return proc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    monkeypatch.setattr(codex_bridge, "_MAX_EVENT_BYTES", 1)
    with pytest.raises(RuntimeError, match="event limit"):
        asyncio.run(
            codex_bridge.run_codex(prompt="p", thread_id=None)
        )
    assert proc.returncode == -9


def test_resume_preserves_schema_and_model_configuration(scripted):
    scripted["queue"].extend(
        [
            (_failed("HTTP 503 service unavailable"), 1),
            (_success(), 0),
        ]
    )
    asyncio.run(
        codex_bridge.run_codex(
            prompt="p",
            thread_id=None,
            model="gpt-5.6-sol",
            effort="high",
            output_format={
                "type": "json_schema",
                "schema": {"type": "object"},
            },
        )
    )
    second = scripted["calls"][1]["cmd"]
    assert 'model="gpt-5.6-sol"' in second
    assert 'model_reasoning_effort="high"' in second
    assert "--output-schema" in second
