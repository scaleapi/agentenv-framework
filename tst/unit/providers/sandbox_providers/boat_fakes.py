"""In-memory stand-ins for the parts of boat's synchronous SDK the provider uses: each records what it is sent and
answers from state a test sets, with no network."""

from __future__ import annotations

import itertools
import json
import shlex
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from boat_sdk.exceptions import ApiException

from agent_env.providers.sandbox_providers.boat.config import Settings

# (exit_code, stdout, stderr) for one command, or an exception its output stream raises.
Outcome = tuple[int, str, str] | BaseException
Handler = Callable[[list[str]], Outcome]

SETTINGS = Settings(api_key="boat_test")


def api_error(status: int, code: str, **body: Any) -> ApiException:
    return ApiException(status=status, reason=code, body=json.dumps({"code": code, **body}))


def refused(error: BaseException | None) -> bool:
    """Whether boat answers ``error`` to a create having made nothing: a 4xx, or no capacity."""
    return isinstance(error, ApiException) and error.status is not None and (error.status < 500 or error.status == 503)


@dataclass
class FakeBoatApi:
    """Sandboxes, hosted ports and commands kept in memory. ``create_errors`` are raised by the next creates in turn;
    with ``create_gate`` set, a create waits for it after setting ``create_entered``."""

    states: list[str] = field(default_factory=lambda: ["ready"])
    create_errors: list[BaseException] = field(default_factory=list)
    hosted_port_errors: set[int] = field(default_factory=set)
    create_gate: threading.Event | None = None
    create_entered: threading.Event = field(default_factory=threading.Event)
    sandboxes_by_id: dict[str, SimpleNamespace] = field(default_factory=dict)
    creates: list[dict[str, Any]] = field(default_factory=list)
    deletes: list[str] = field(default_factory=list)
    hosts: list[tuple[str, int, bool]] = field(default_factory=list)
    label_queries: list[list[str]] = field(default_factory=list)
    stopped: list[tuple[str, int]] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)
    # The SDK's ApiClient, as far as close reaches: its REST client's urllib3 pool.
    api_client: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(rest_client=SimpleNamespace(pool_manager=SimpleNamespace(cleared=False))))
    _ids: Iterator[int] = field(default_factory=itertools.count)

    def __post_init__(self) -> None:
        pool = self.api_client.rest_client.pool_manager
        pool.clear = lambda: setattr(pool, "cleared", True)

    def url(self, sandbox_id: str, port: int) -> str:
        return f"https://box-{sandbox_id}-{port}.on.boat.dev"

    def create(self, idempotency_key: str | None = None, create_sandbox_request: Any = None) -> SimpleNamespace:
        self.creates.append({"key": idempotency_key, "request": create_sandbox_request})
        self.create_entered.set()
        if self.create_gate is not None:
            assert self.create_gate.wait(5)
        request = create_sandbox_request
        error = self.create_errors.pop(0) if self.create_errors else None
        if refused(error):
            raise error
        # Any other failure stands for an answer lost after boat made the sandbox.
        sandbox_id = f"bx_{next(self._ids)}"
        self.sandboxes_by_id[sandbox_id] = SimpleNamespace(id=sandbox_id, state="provisioned",
                                                          labels=dict(request.labels or {}), states=list(self.states))
        if error is not None:
            raise error
        hosted = [
            SimpleNamespace(port=entry.port, url=None if entry.port in self.hosted_port_errors else self.url(sandbox_id, entry.port),
                            success=entry.port not in self.hosted_port_errors)
            for entry in request.hosted_ports or []
        ]
        return SimpleNamespace(sandbox=SimpleNamespace(id=sandbox_id, state="provisioned"), hosted_ports=hosted)

    def get(self, sandbox_id: str) -> SimpleNamespace:
        if sandbox_id not in self.sandboxes_by_id:
            raise api_error(404, "not_found")
        raw = self.sandboxes_by_id[sandbox_id]
        if raw.states:
            raw.state = raw.states.pop(0)
        return SimpleNamespace(sandbox=SimpleNamespace(id=sandbox_id, state=raw.state, labels=dict(raw.labels)))

    def delete_sandbox(self, x_ascii_confirm_delete: str, sandbox_id: str) -> SimpleNamespace:
        assert x_ascii_confirm_delete == sandbox_id
        self.deletes.append(sandbox_id)
        self.sandboxes_by_id.pop(sandbox_id, None)
        return SimpleNamespace(ok=True)

    def host_port(self, sandbox_id: str, host_port_request: Any) -> SimpleNamespace:
        self.hosts.append((sandbox_id, host_port_request.port, bool(host_port_request.public)))
        return SimpleNamespace(url=self.url(sandbox_id, host_port_request.port))

    def sandboxes(self, label: list[str] | None = None) -> SimpleNamespace:
        self.label_queries.append(list(label or []))
        wanted = dict(entry.split("=", 1) for entry in label or [])
        found = [raw for raw in self.sandboxes_by_id.values()
                 if all(raw.labels.get(key) == value for key, value in wanted.items())]
        return SimpleNamespace(sandboxes=[SimpleNamespace(id=raw.id) for raw in found])

    def stop_command(self, sandbox_id: str, process_id: int) -> SimpleNamespace:
        self.stopped.append((sandbox_id, process_id))
        return SimpleNamespace(ok=True)

    def create_upload_url(self, sandbox_id: str, file_url_request: Any) -> SimpleNamespace:
        self.uploads.append(file_url_request.path)
        return SimpleNamespace(url=f"https://box-{sandbox_id}.on.boat.dev/file-transfer?path={file_url_request.path}",
                               method="PUT", path=file_url_request.path)


@dataclass
class FakeCommands:
    """``boat_sdk.run_command`` over ``handler``, which gets each command's argv; by default exit 0 with no output.
    With ``gate`` set, a command waits for it after its start, having set ``started``."""

    handler: Handler = lambda argv: (0, "", "")
    commands: list[list[str]] = field(default_factory=list)
    gate: threading.Event | None = None
    started: threading.Event = field(default_factory=threading.Event)
    _process_ids: Iterator[int] = field(default_factory=lambda: itertools.count(100))

    def __call__(self, api: Any, sandbox_id: str, command: str, cwd: str | None = None) -> Iterator[dict[str, Any]]:
        argv = shlex.split(command)
        self.commands.append(argv)
        yield {"type": "started", "processId": next(self._process_ids)}
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(5)
        outcome = self.handler(argv)
        if isinstance(outcome, BaseException):
            raise outcome
        code, stdout, stderr = outcome
        if stdout:
            yield {"type": "stdout", "data": stdout, "nextOffset": len(stdout)}
        if stderr:
            yield {"type": "stderr", "data": stderr, "nextOffset": len(stderr)}
        yield {"type": "exit", "exitCode": code}
