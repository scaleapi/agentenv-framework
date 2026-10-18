"""In-memory stand-ins for the cwsandbox SDK's Sandbox, Process and stdin writer."""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from cwsandbox import CidrBlock, Container, EgressRule, HttpsEndpointStatus, SandboxExecutionError, SandboxNotFoundError
from cwsandbox import EndpointAuth, EndpointKind

# A runner's default egress: the public IPv4 internet.
DEFAULT_EGRESS = (EgressRule(cidr=CidrBlock(cidr="0.0.0.0/0", except_cidrs=("10.0.0.0/8", "192.168.0.0/16"))),)


@dataclass
class Result:
    returncode: int = 0
    stdout_bytes: bytes = b""
    stderr_bytes: bytes = b""


def _done(value: Any = None) -> asyncio.Future:
    """An operation that has completed, as an SDK ``OperationRef`` the caller may or may not await."""
    future = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


class FakeStdin:
    def __init__(self, process: FakeProcess):
        self._process = process
        self.pieces: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> Any:
        if self._process.finished.done():
            raise SandboxExecutionError("Cannot write to stdin: stream has failed")
        if self._process.read_limit is not None and len(self.pieces) >= self._process.read_limit:
            return asyncio.get_running_loop().create_future()  # the command stopped reading: the queue stays full
        self.pieces.append(data)
        return _done()

    def close(self) -> Any:
        self.closed = True
        if not self._process.finished.done() and self._process.outcome_on_close:
            self._process.finish()
        return _done()


class FakeProcess:
    """Resolves to ``outcome`` (a Result, or an exception raised) when awaited, or, with stdin, once stdin closes or
    the command ends by itself after reading ``read_limit`` pieces."""

    def __init__(self, outcome: Result | BaseException, *, stdin: bool, read_limit: int | None = None):
        self.outcome = outcome
        self.finished: asyncio.Future = asyncio.get_running_loop().create_future()
        self.read_limit = read_limit
        self.outcome_on_close = stdin
        self.stdin = FakeStdin(self) if stdin else None
        if not stdin:
            self.finish()

    def finish(self) -> None:
        if isinstance(self.outcome, BaseException):
            self.finished.set_exception(self.outcome)
        else:
            self.finished.set_result(self.outcome)

    def __await__(self):
        return asyncio.shield(self.finished).__await__()


@dataclass
class FakeServer:
    """What the Sandbox API holds and was asked for, shared by one fake ``Sandbox`` class and its instances."""

    sandboxes: dict[str, FakeSandbox] = field(default_factory=dict)
    created: list[FakeSandbox] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    listed: list[list[str]] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    start_error: BaseException | None = None
    # The server made the sandbox but the start's response was lost: the sandbox exists, then start_error is raised.
    start_commits: bool = False
    start_delay: float = 0.0
    ids: Any = field(default_factory=lambda: (f"sb-{n}" for n in itertools.count(1)))


class FakeSandbox:
    """Records the create kwargs; ``results`` answers exec in order, ``exec_calls`` records each call, and ``files``
    holds what ``write_file`` wrote. ``containers`` echoes the container spec, as the Sandbox API does."""

    server: FakeServer

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.sandbox_id: str | None = None
        self.runner_id = "runner-a"
        self.effective_egress = DEFAULT_EGRESS if kwargs.get("network") is None else self._egress(kwargs["network"])
        self.service_endpoints = tuple(
            HttpsEndpointStatus(port=service.port, name="", kind=EndpointKind.HTTPS, auth=EndpointAuth.OPEN,
                                url=f"https://{service.port}-x.r1abc.ep1.cwsandbox.test", request_timeout_seconds=900)
            for service in kwargs.get("services") or ()
        )
        self.containers = tuple(kwargs.get("containers") or (
            (Container(image=kwargs["container_image"], command=kwargs.get("command"), args=kwargs.get("args"),
                       primary=True),) if kwargs.get("container_image") else ()))
        self.results: list[Result | BaseException | FakeProcess] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.processes: list[FakeProcess] = []
        self.files: dict[str, bytes] = {}
        self.server.created.append(self)

    @staticmethod
    def _egress(network: Any) -> tuple[EgressRule, ...]:
        return () if network.deny_egress else tuple(network.egress)

    async def start(self) -> None:
        server = self.server
        if server.start_delay:
            await asyncio.sleep(server.start_delay)
        if server.start_error is not None and not server.start_commits:
            raise server.start_error
        self.sandbox_id = next(server.ids)
        server.sandboxes[self.sandbox_id] = self
        if server.start_error is not None:
            self.sandbox_id = None
            raise server.start_error

    def __await__(self):
        return _done(self).__await__()

    def exec(self, command: list[str], *, timeout_seconds: float | None = None, stdin: bool = False) -> FakeProcess:
        self.exec_calls.append({"argv": list(command), "timeout": timeout_seconds, "stdin": stdin})
        outcome = self.results.pop(0) if self.results else Result()
        if isinstance(outcome, FakeProcess):
            process = outcome
        else:
            process = FakeProcess(outcome, stdin=stdin)
        self.processes.append(process)
        return process

    async def write_file(self, filepath: str, contents: bytes) -> None:
        self.files[filepath] = contents

    @classmethod
    async def from_id(cls, sandbox_id: str, **kwargs: Any) -> FakeSandbox:
        cls.server.calls.append(("from_id", kwargs))
        if sandbox_id not in cls.server.sandboxes:
            raise SandboxNotFoundError(f"sandbox {sandbox_id} not found")
        return cls.server.sandboxes[sandbox_id]

    @classmethod
    async def list(cls, *, tags: list[str], **kwargs: Any) -> list[SimpleNamespace]:
        cls.server.listed.append(tags)
        return [SimpleNamespace(sandbox_id=sandbox_id) for sandbox_id, sb in cls.server.sandboxes.items()
                if set(tags) <= set(sb.kwargs.get("tags") or ())]

    @classmethod
    async def delete(cls, sandbox_id: str, *, missing_ok: bool = False, **kwargs: Any) -> None:
        cls.server.calls.append(("delete", {"missing_ok": missing_ok, **kwargs}))
        cls.server.deleted.append(sandbox_id)
        cls.server.sandboxes.pop(sandbox_id, None)


def fake_sandbox_cls() -> type[FakeSandbox]:
    """A fresh ``FakeSandbox`` class with a server of its own, so no test sees another's sandboxes."""
    return type("FakeSandbox", (FakeSandbox,), {"server": FakeServer()})
