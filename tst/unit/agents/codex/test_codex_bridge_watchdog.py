"""The inactivity watchdog scales with reasoning effort.

A `max`/`xhigh` turn is silent for as long as the model thinks, routinely past
five minutes before the first token. 25 of 61 gpt-6-astra `max` runs on
abra-sota-template v21 were killed as "stalled" mid-thought under the flat
300 s default. These tests pin the rule: deep efforts get 900 s, everything
else keeps 300 s, and an explicit AGENT_CLI_INACTIVITY_TIMEOUT_S pin beats both.
"""
import asyncio

import pytest

from tst.unit.agents.codex.agent import codex_bridge


@pytest.fixture
def unpinned(monkeypatch):
    monkeypatch.delenv("AGENT_CLI_INACTIVITY_TIMEOUT_S", raising=False)
    monkeypatch.setattr(codex_bridge, "_INACTIVITY_TIMEOUT_S", 300)
    monkeypatch.setattr(codex_bridge, "_DEEP_EFFORT_INACTIVITY_TIMEOUT_S", 900)


@pytest.mark.parametrize("effort", [None, "", "minimal", "low", "medium", "high"])
def test_ordinary_efforts_keep_the_default_window(unpinned, effort):
    assert codex_bridge.inactivity_timeout_s(effort) == 300


@pytest.mark.parametrize("effort", ["xhigh", "max", "MAX", " max "])
def test_deep_efforts_get_the_longer_window(unpinned, effort):
    assert codex_bridge.inactivity_timeout_s(effort) == 900


def test_an_explicit_pin_wins_over_effort(monkeypatch):
    # The env var is read at import into _INACTIVITY_TIMEOUT_S; a pinned
    # container behaves as if that value were the only rule.
    monkeypatch.setenv("AGENT_CLI_INACTIVITY_TIMEOUT_S", "120")
    monkeypatch.setattr(codex_bridge, "_INACTIVITY_TIMEOUT_S", 120)
    assert codex_bridge.inactivity_timeout_s("max") == 120
    assert codex_bridge.inactivity_timeout_s("medium") == 120


def test_a_pin_above_the_deep_window_is_not_lowered(monkeypatch):
    monkeypatch.setenv("AGENT_CLI_INACTIVITY_TIMEOUT_S", "1200")
    monkeypatch.setattr(codex_bridge, "_INACTIVITY_TIMEOUT_S", 1200)
    assert codex_bridge.inactivity_timeout_s("max") == 1200


# --- the watchdog honours the window it is handed -----------------------------

class _Stdin:
    def write(self, data: bytes) -> None: ...
    async def drain(self) -> None: ...
    def close(self) -> None: ...


class _Silent:
    """A CLI that never says anything."""
    async def read(self, n: int = -1) -> bytes:
        await asyncio.sleep(3600)
        return b""


class _Empty:
    async def read(self, n: int = -1) -> bytes:
        return b""


class _SilentProc:
    pid = None

    def __init__(self):
        self.stdin = _Stdin()
        self.stdout = _Silent()
        self.stderr = _Empty()
        self.returncode = None

    async def wait(self):
        self.returncode = -9
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_watchdog_uses_the_window_it_is_given(monkeypatch):
    monkeypatch.setattr(codex_bridge, "_INACTIVITY_TIMEOUT_S", 3600)
    loop = asyncio.new_event_loop()
    try:
        started = loop.time()
        _, _, timed_out, stalled = loop.run_until_complete(
            codex_bridge._read_with_watchdog(
                _SilentProc(), 30, b"prompt", None, inactivity_s=0.05
            )
        )
    finally:
        loop.close()
    assert stalled and not timed_out
    assert started is not None  # ran to a verdict rather than the 30 s deadline


def test_watchdog_falls_back_to_the_module_default(monkeypatch):
    monkeypatch.setattr(codex_bridge, "_INACTIVITY_TIMEOUT_S", 0.05)
    _, _, timed_out, stalled = asyncio.run(
        codex_bridge._read_with_watchdog(_SilentProc(), 30, b"prompt")
    )
    assert stalled and not timed_out


def test_stall_error_names_the_effective_window(monkeypatch, tmp_path):
    """End to end through _run_codex_once: a `max` turn that stays silent is
    reported against the deep-effort window, not the flat default."""
    monkeypatch.delenv("AGENT_CLI_INACTIVITY_TIMEOUT_S", raising=False)
    monkeypatch.setattr(codex_bridge, "_INACTIVITY_TIMEOUT_S", 0.02)
    monkeypatch.setattr(codex_bridge, "_DEEP_EFFORT_INACTIVITY_TIMEOUT_S", 0.05)
    seen: dict = {}

    async def fake_exec(*cmd, **kwargs):
        seen["cmd"] = list(cmd)
        return _SilentProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(codex_bridge, "find_rollout_path", lambda _: None)
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        codex_bridge._run_codex_once("think hard", None, model="gpt-6-astra",
                                     effort="max", timeout_s=30)
    )
    assert result.error and "no output for 0.05s" in result.error, result.error
    assert 'model_reasoning_effort="max"' in " ".join(seen["cmd"])
