"""``agent_env.training._openenv`` opening a rollout's session: tried again while the server is at its limit or too busy
to answer its handshake in time, until the patience runs out, and given up at once when nothing listens at its
address."""

import asyncio

import pytest

from agent_env.training import _openenv
from agent_env.training._openenv import Failed, reopened


@pytest.mark.parametrize(("cause", "opens", "attempts"), [
    (TimeoutError("timed out"), True, 3),
    (ConnectionRefusedError(61, "Connection refused"), False, 1),
], ids=["busy", "refused"])
def test_a_session_is_tried_again_while_its_server_is_busy_and_given_up_at_once_when_none_listens(monkeypatch, cause,
                                                                                                  opens, attempts):
    tried = []

    async def flaky(url, timeout, *, token=None):
        tried.append(url)
        if len(tried) < 3:
            client_error = ConnectionError(f"Failed to connect to {url}/ws: {cause}")
            client_error.__cause__ = cause
            raise ConnectionError(f"couldn't open a session on the server at {url}") from client_error
        return "session"

    monkeypatch.setattr(_openenv, "opened", flaky)
    monkeypatch.setattr(_openenv, "REOPEN_SECONDS", 0.01)

    if opens:
        assert asyncio.run(reopened("http://busy", 5, dropped=False, patience=5)) == "session"
    else:
        with pytest.raises(Failed, match="couldn't open a session on the server at http://busy"):
            asyncio.run(reopened("http://busy", 5, dropped=False, patience=5))
    assert len(tried) == attempts
