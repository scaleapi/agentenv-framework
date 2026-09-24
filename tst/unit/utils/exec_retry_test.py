"""Unit tests for the sandbox exec retry helper."""

from __future__ import annotations

import asyncio

import pytest

from agent_env.utils.exec_retry import is_transient_exec_error, retry_transient


@pytest.mark.asyncio
async def test_returns_value_when_first_attempt_succeeds():
    calls = 0

    async def op():
        nonlocal calls
        calls += 1
        return "ok"

    result = await retry_transient(op, attempts=5, initial_backoff=0.0, desc="t")
    assert result == "ok"
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_then_succeeds_on_transient():
    calls = 0

    async def op():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise TimeoutError("timed out during opening handshake")
        return "ok"

    result = await retry_transient(op, attempts=5, initial_backoff=0.0, desc="t")
    assert result == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_does_not_retry_non_transient():
    calls = 0

    async def op():
        nonlocal calls
        calls += 1
        raise ValueError("application bug")

    with pytest.raises(ValueError):
        await retry_transient(op, attempts=5, initial_backoff=0.0, desc="t")
    assert calls == 1


@pytest.mark.asyncio
async def test_raises_last_transient_after_exhausting_attempts():
    calls = 0

    async def op():
        nonlocal calls
        calls += 1
        raise TimeoutError(f"attempt-{calls}")

    with pytest.raises(TimeoutError, match="attempt-3"):
        await retry_transient(op, attempts=3, initial_backoff=0.0, desc="t")
    assert calls == 3


def test_is_transient_classifier_basic():
    assert is_transient_exec_error(TimeoutError("x"))
    assert is_transient_exec_error(asyncio.TimeoutError())
    assert is_transient_exec_error(ConnectionResetError())
    assert is_transient_exec_error(ConnectionRefusedError())
    assert not is_transient_exec_error(ValueError("x"))
    assert not is_transient_exec_error(RuntimeError("x"))


def test_is_transient_classifier_websocket_5xx():
    # Import the submodule directly: `websockets` resolves `.exceptions` lazily,
    # so referencing it off the top-level package only works if it was already
    # imported elsewhere (flaky under parallel/test-isolated runs).
    ws_exceptions = pytest.importorskip("websockets.exceptions")

    class FakeResponse:
        status_code = 503

    exc = ws_exceptions.InvalidStatus(FakeResponse())
    assert is_transient_exec_error(exc)


def test_is_transient_classifier_websocket_4xx_not_retried():
    ws_exceptions = pytest.importorskip("websockets.exceptions")

    class FakeResponse:
        status_code = 401

    exc = ws_exceptions.InvalidStatus(FakeResponse())
    assert not is_transient_exec_error(exc)


def test_is_transient_classifier_httpx_connect_error():
    httpx = pytest.importorskip("httpx")
    assert is_transient_exec_error(httpx.ConnectError("[SSL] record layer failure"))
    assert is_transient_exec_error(httpx.ReadTimeout("read timed out"))
    assert is_transient_exec_error(httpx.RemoteProtocolError("server closed"))
    assert is_transient_exec_error(httpx.PoolTimeout("pool exhausted"))
    assert is_transient_exec_error(httpx.ConnectTimeout("connect timed out"))


def test_is_transient_classifier_httpx_status_5xx_and_throttling():
    httpx = pytest.importorskip("httpx")

    class FakeResp:
        def __init__(self, status: int):
            self.status_code = status

    def make(status: int) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError("boom", request=None, response=FakeResp(status))  # type: ignore[arg-type]

    assert is_transient_exec_error(make(500))
    assert is_transient_exec_error(make(502))
    assert is_transient_exec_error(make(503))
    assert is_transient_exec_error(make(408))
    assert is_transient_exec_error(make(429))
    assert not is_transient_exec_error(make(400))
    assert not is_transient_exec_error(make(401))
    assert not is_transient_exec_error(make(404))


def test_is_transient_classifier_ssl_error():
    import ssl
    assert is_transient_exec_error(ssl.SSLError("[SSL] record layer failure (_ssl.c:1016)"))
