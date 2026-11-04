"""An in-memory Agent37 API behind an ``httpx.MockTransport``, for the offline provider and sandbox tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

API_KEY = "sk_live_test_secret"

ExecAnswer = httpx.Response | Exception | dict[str, Any]


def _error(status: int, code: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": code}})


class FakeAgent37:
    """Instances on ``api.agent37.com`` and exec on ``sandbox.agent37.app``. Each exec answers with the next of
    ``exec_answers`` (a result body, a response, or an exception to raise), then with ``exec_result``. A create answers
    ``create_answer`` when it is set: a response refuses it, an exception is raised after the instance is made."""

    def __init__(self) -> None:
        self.instances: dict[str, dict[str, Any]] = {}
        self.requests: list[httpx.Request] = []
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.execs: list[dict[str, Any]] = []
        self.exec_answers: list[ExecAnswer] = []
        self.exec_result: Callable[[dict[str, Any]], dict[str, Any]] = (
            lambda body: {"exit_code": 0, "stdout": "", "stderr": "", "truncated": False})
        self.create_answer: httpx.Response | Exception | None = None

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def requests_to(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        parts = request.url.path.strip("/").split("/")
        if request.url.host == "sandbox.agent37.app":
            assert request.headers["X-Agent37-Key"] == API_KEY
            return self._exec(parts[2], json.loads(request.content))
        assert request.url.host == "api.agent37.com"
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        if parts == ["v1", "instances"] and request.method == "POST":
            return self._create(json.loads(request.content))
        if parts == ["v1", "instances"]:
            return httpx.Response(200, json={"data": list(self.instances.values())})
        instance = self.instances.get(parts[2])
        if instance is None:
            return _error(404, "not_found")
        if request.method == "DELETE":
            del self.instances[parts[2]]
            self.deleted.append(parts[2])
            return httpx.Response(200, json={"id": parts[2], "deleted": True})
        return httpx.Response(200, json=instance)

    def _create(self, body: dict[str, Any]) -> httpx.Response:
        self.created.append(body)
        if isinstance(self.create_answer, httpx.Response):
            return self.create_answer
        instance_id = f"i{len(self.created):09d}"
        self.instances[instance_id] = {
            "id": instance_id,
            "status": "running",
            "name": body.get("name"),
            "metadata": body.get("metadata"),
            "network": "allow-all",
            "public_ports": [
                {"port": entry["port"], "url": f"https://slug{entry['port']}{instance_id}.agent37.app"}
                for entry in body.get("public_ports", [])
            ],
        }
        if self.create_answer is not None:
            raise self.create_answer
        return httpx.Response(201, json=self.instances[instance_id])

    def _exec(self, instance_id: str, body: dict[str, Any]) -> httpx.Response:
        self.execs.append(body)
        if instance_id not in self.instances:
            return httpx.Response(404, json={"error": "not_found"})
        answer = self.exec_answers.pop(0) if self.exec_answers else self.exec_result(body)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        return httpx.Response(200, json=answer)
