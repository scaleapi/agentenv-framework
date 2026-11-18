"""An in-memory kosa8 Cloud API behind an ``httpx.MockTransport``, for the offline provider and sandbox tests."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import httpx

API_KEY = "k8_live_test_secret"
BASE_URL = "https://api.kosa8.com"

ExecAnswer = httpx.Response | Exception | dict[str, Any] | list[dict[str, Any]]


def error(status: int, code: str) -> httpx.Response:
    return httpx.Response(status, json={"error": code, "message": code})


def frames(stdout: bytes = b"", stderr: bytes = b"", exit_code: int | None = 0, error_text: str | None = None,
           heartbeats: int = 0) -> list[dict[str, Any]]:
    """The NDJSON frames of a streamed exec: heartbeats, output, then its exit (or an error with no exit)."""
    out: list[dict[str, Any]] = [{"stream": "stdout", "data": "", "heartbeat": True} for _ in range(heartbeats)]
    if stdout:
        out.append({"stream": "stdout", "data": base64.b64encode(stdout).decode()})
    if stderr:
        out.append({"stream": "stderr", "data": base64.b64encode(stderr).decode()})
    if error_text is not None:
        out.append({"done": True, "error": error_text})
    elif exit_code is not None:
        out.append({"done": True, "exit": exit_code})
    return out


class FakeKosa8:
    """Sandboxes, egress, previews, files and streamed exec on ``api.kosa8.com``. A create answers 202 ``creating`` and
    the sandbox is ``running`` from the ``boot_polls``-th GET on. Each exec answers with the next of ``exec_answers`` (a
    result dict, a list of frames, a response, or an exception to raise), then with ``exec_result``. A create answers
    ``create_answer`` when it is set: a response refuses it, an exception is raised after the sandbox is made."""

    def __init__(self) -> None:
        self.sandboxes: dict[str, dict[str, Any]] = {}
        self.egress: dict[str, str] = {}
        self.previews: dict[str, dict[int, dict[str, Any]]] = {}
        self.files: dict[tuple[str, str], bytes] = {}
        self.dirs: list[tuple[str, str]] = []
        self.requests: list[httpx.Request] = []
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.execs: list[list[str]] = []
        self.exec_answers: list[ExecAnswer] = []
        self.exec_result: Callable[[list[str]], ExecAnswer] = lambda argv: {"exit": 0}
        self.create_answer: httpx.Response | Exception | None = None
        self.org_default = "any"
        self.boot_polls = 1

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def requests_to(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert f"{request.url.scheme}://{request.url.host}" == BASE_URL
        assert request.headers["Authorization"] == f"Bearer {API_KEY}"
        parts = request.url.path.strip("/").split("/")
        if parts == ["v1", "sandboxes"] and request.method == "POST":
            return self._create(json.loads(request.content))
        if parts == ["v1", "sandboxes"]:
            name = request.url.params.get("name")
            return httpx.Response(200, json={"sandboxes": [s for s in self.sandboxes.values()
                                                           if name is None or s["name"] == name],
                                             "next_cursor": None})
        sandbox = self.sandboxes.get(parts[2])
        if sandbox is None or sandbox["state"] == "deleted":
            return error(404, "not_found")
        route = parts[3:]
        if not route:
            return self._sandbox(request, sandbox)
        if route == ["egress"]:
            return self._egress(request, sandbox)
        if route == ["previews"]:
            return self._previews(request, sandbox)
        if route == ["files", "mkdir"]:
            self.dirs.append((sandbox["id"], json.loads(request.content)["path"]))
            return httpx.Response(200, json={"file": {"path": json.loads(request.content)["path"], "type": "dir"}})
        if route == ["files"] and request.method == "PUT":
            path = request.url.params["path"]
            assert path.startswith("/workspace/")
            assert len(request.content) <= 8 * 1024 * 1024
            self.files[(sandbox["id"], path)] = request.content
            return httpx.Response(200, json={"file": {"path": path, "size": len(request.content)}})
        if route == ["exec"]:
            return self._exec(json.loads(request.content))
        return error(404, "not_found")

    def _create(self, body: dict[str, Any]) -> httpx.Response:
        self.created.append(body)
        if isinstance(self.create_answer, httpx.Response):
            return self.create_answer
        sandbox_id = f"sbx_{len(self.created):08d}"
        self.sandboxes[sandbox_id] = {"id": sandbox_id, "state": "creating", "name": body.get("name"),
                                      "template": "base", "polls": 0}
        if "egress" in body:
            self.egress[sandbox_id] = f"allow:{','.join(body['egress']['allow'])}"
        if self.create_answer is not None:
            raise self.create_answer
        return httpx.Response(202, json={"sandbox": self._view(self.sandboxes[sandbox_id]), "host_ms": 4})

    @staticmethod
    def _view(sandbox: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in sandbox.items() if key != "polls"}

    def _sandbox(self, request: httpx.Request, sandbox: dict[str, Any]) -> httpx.Response:
        if request.method == "DELETE":
            sandbox["state"] = "deleted"
            self.deleted.append(sandbox["id"])
            return httpx.Response(200, json={"deleted": sandbox["id"]})
        sandbox["polls"] += 1
        if sandbox["state"] == "creating" and sandbox["polls"] >= self.boot_polls:
            sandbox["state"] = "running"
        return httpx.Response(200, json={"sandbox": self._view(sandbox)})

    def _egress(self, request: httpx.Request, sandbox: dict[str, Any]) -> httpx.Response:
        enforced = self.egress.get(sandbox["id"], self.org_default)
        return httpx.Response(200, json={"egress": {"enforced": enforced}})

    def _previews(self, request: httpx.Request, sandbox: dict[str, Any]) -> httpx.Response:
        open_ports = self.previews.setdefault(sandbox["id"], {})
        if request.method == "GET":
            return httpx.Response(200, json={"previews": list(open_ports.values()), "configured": True})
        body = json.loads(request.content)
        port = body["port"]
        preview = {"port": port, "public": body.get("public") is True,
                   "url": f"https://{port}-{sandbox['id'][-8:]}.kosa8labs.com/"}
        status = 200 if port in open_ports else 201
        open_ports[port] = preview
        return httpx.Response(status, json={"preview": preview})

    def _exec(self, body: dict[str, Any]) -> httpx.Response:
        argv = body["cmd"]
        assert body.get("stream") is True
        if len(argv) > 256 or any(len(arg) > 8192 for arg in argv):
            return error(400, "invalid_cmd")
        self.execs.append(argv)
        answer = self.exec_answers.pop(0) if self.exec_answers else self.exec_result(argv)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, httpx.Response):
            return answer
        if isinstance(answer, dict):
            answer = frames(answer.get("stdout", b""), answer.get("stderr", b""), answer.get("exit", 0))
        content = "".join(json.dumps(frame) + "\n" for frame in answer).encode()
        return httpx.Response(200, content=content, headers={"Content-Type": "application/x-ndjson"})


def command_of(argv: list[str], fake: FakeKosa8, sandbox_id: str) -> list[str]:
    """The argv a wrapped exec runs: inline after ``bash -c … kosa8-exec``, or read back from its spilled file."""
    assert argv[:2] == ["bash", "-c"] and argv[3] == "kosa8-exec"
    if "mapfile" not in argv[2]:
        return argv[4:]
    data = fake.files[(sandbox_id, argv[4])]
    return [part.decode() for part in data.split(b"\0")[:-1]]
