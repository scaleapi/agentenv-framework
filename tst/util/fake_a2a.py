"""A2A agents answered in-process, for unit tests of the steps that message them.

Every httpx client a test makes once ``serve`` runs reaches them through a MockTransport. Each origin
answers ``message/send`` with a new task (``task-1``, ``task-2``, ...) and ``tasks/get`` with its reply,
completed; ``sent`` lists the parts each origin was sent, per message. A request to any other path goes
to ``other``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

import httpx
import pytest


class FakeA2AAgents:
    def __init__(
        self,
        replies: Mapping[str, list[dict]],
        *,
        other: Callable[[httpx.Request], httpx.Response] | None = None,
    ) -> None:
        self.replies = dict(replies)
        self.other = other
        self.sent: dict[str, list[list[dict]]] = {origin: [] for origin in self.replies}

    def serve(self, monkeypatch: pytest.MonkeyPatch) -> FakeA2AAgents:
        real_client = httpx.AsyncClient
        transport = httpx.MockTransport(self._answer)
        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=transport, **kw))
        return self

    def _answer(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/a2a":
            if self.other is None:
                raise AssertionError(f"unexpected request to {request.url}")
            return self.other(request)
        origin = f"{request.url.scheme}://{request.url.host}"
        body = json.loads(request.content)
        if body["method"] == "message/send":
            self.sent[origin].append(body["params"]["message"]["parts"])
            result = {"id": f"task-{len(self.sent[origin])}", "contextId": "ctx"}
        else:
            result = {"status": {"state": "completed", "message": {"parts": self.replies[origin]}}}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
