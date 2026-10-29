"""In-memory stand-ins for the Tenki SDK's AsyncClient, AsyncSandbox and AsyncProcess."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from tenki_sandbox.models import CommandResult, ExposedPort, SessionEgressPolicy


@dataclass
class FakeInfo:
    id: str
    name: str = ""
    state: str = "CREATING"
    outbound_enabled: bool = True
    egress: SessionEgressPolicy = field(default_factory=SessionEgressPolicy)
    timeout_at: datetime | None = None


class FakeProcess:
    def __init__(self, result: CommandResult):
        self._result = result
        self.stdin: list[bytes] = []
        self.stdin_closed = False
        self.killed = False
        self.calls: list[str] = []
        self.wait_error: BaseException | None = None

    async def write_stdin(self, data: bytes) -> None:
        self.stdin.append(data)

    async def close_stdin(self) -> None:
        self.calls.append("close_stdin")
        self.stdin_closed = True

    async def kill(self) -> None:
        self.calls.append("kill")
        self.killed = True

    async def wait(self, timeout: float | None = None) -> CommandResult:
        self.calls.append("wait")
        if self.wait_error is not None and len([c for c in self.calls if c == "wait"]) == 1:
            raise self.wait_error
        return self._result


class FakeSession:
    """Records every call; ``results`` answers exec and start in order, raising an exception found there."""

    def __init__(self, session_id: str = "sess-1", **info: Any):
        self.id = session_id
        self.info = FakeInfo(id=session_id, **info)
        self.exec_calls: list[dict[str, Any]] = []
        self.results: list[CommandResult | BaseException] = []
        self.processes: list[FakeProcess] = []
        self.exposed: dict[int, str] = {}
        self.closed = 0
        self.close_error: BaseException | None = None
        self.ready_delay = 0.0
        self.close_delay = 0.0
        self.preview_parent = "sb.tenki.sh"

    @property
    def state(self) -> str:
        return self.info.state

    async def wait_ready(self, timeout: float = 180) -> None:
        if self.ready_delay:
            await asyncio.sleep(self.ready_delay)
        self.info.state = "RUNNING"

    def _next(self, argv: tuple[str, ...], timeout: Any, privileged: bool) -> CommandResult:
        self.exec_calls.append({"argv": list(argv), "timeout": timeout, "privileged": privileged})
        outcome = self.results.pop(0) if self.results else CommandResult(argv=list(argv), exit_code=0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def exec(self, *argv: str, timeout: Any = None, privileged: bool = False) -> CommandResult:
        return self._next(argv, timeout, privileged)

    async def start(self, *argv: str, timeout: Any = None, privileged: bool = False) -> FakeProcess:
        process = FakeProcess(self._next(argv, timeout, privileged))
        self.processes.append(process)
        return process

    async def expose_port(self, port: int) -> ExposedPort:
        url = f"p{port}-{self.id}.{self.preview_parent}"
        self.exposed[port] = url
        return ExposedPort(port=port, url=url)

    async def list_exposed_ports(self) -> list[ExposedPort]:
        return [ExposedPort(port=port, url=f"https://{url}") for port, url in self.exposed.items()]

    async def refresh(self) -> FakeInfo:
        return self.info

    async def close_if_open(self) -> None:
        if self.state in {"TERMINATING", "TERMINATED"}:
            return
        if self.close_delay:
            await asyncio.sleep(self.close_delay)
        if self.close_error is not None:
            raise self.close_error
        self.closed += 1
        self.info.state = "TERMINATED"


class FakeClient:
    """Each instance records itself on its class's ``instances``; ``on_init`` stages what a new one returns."""

    instances: list[FakeClient]
    on_init: Any = None

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.create_calls: list[dict[str, Any]] = []
        self.sessions: dict[str, FakeSession] = {}
        self.closed = False
        self.create_delay = 0.0
        self.create_error: BaseException | None = None
        # The server made the VM but the response was lost: create records the session, then raises create_error.
        self.create_commits = False
        self.next_session: FakeSession | None = None
        type(self).instances.append(self)
        if type(self).on_init is not None:
            type(self).on_init(self)

    async def create(self, **kwargs: Any) -> FakeSession:
        self.create_calls.append(kwargs)
        if self.create_delay:
            await asyncio.sleep(self.create_delay)
        session = self.next_session or FakeSession()
        session.info.name = kwargs.get("name", "")
        if self.create_error is not None and not self.create_commits:
            raise self.create_error
        self.sessions[session.id] = session
        if self.create_error is not None:
            raise self.create_error
        return session

    async def list(self, tags: list[str] | None = None) -> list[FakeSession]:
        return list(self.sessions.values())

    async def get(self, sandbox_id: str) -> FakeSession:
        return self.sessions[sandbox_id]

    async def close(self) -> None:
        self.closed = True


def fake_client_cls() -> type[FakeClient]:
    """A fresh ``FakeClient`` class, so no test sees another's clients."""
    return type("FakeClient", (FakeClient,), {"instances": [], "on_init": None})


def result(stdout: bytes = b"", stderr: bytes = b"", exit_code: int = 0, **extra: Any) -> CommandResult:
    return CommandResult(argv=[], exit_code=exit_code, stdout=stdout, stderr=stderr, **extra)
