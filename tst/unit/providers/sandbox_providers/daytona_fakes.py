"""In-memory stand-ins for the parts of the Daytona async SDK the provider uses: each records what it is sent and
answers from state a test sets, with no network."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from daytona import DaytonaNotFoundError

from agent_env.providers.sandbox_providers.daytona.config import Settings

# (exit_code, stdout, stderr), or an exception to raise, for one session command.
Outcome = tuple[int | None, str | None, str | None] | BaseException
Handler = Callable[[list[str]], Outcome]

PROXY_DOMAIN = "daytonaproxy01.net"


def argv_of(command: str) -> list[str]:
    """The argv a ``bash -c <quoted>`` session command runs."""
    words = shlex.split(command)
    assert words[:2] == ["bash", "-c"], command
    return shlex.split(words[2])


@dataclass
class _Response:
    exit_code: int | None
    stdout: str | None
    stderr: str | None


@dataclass
class FakeProcess:
    """Sessions run ``handler`` on each command's argv; by default exit 0 with no output. With ``gate`` set, a command
    waits for it after setting ``entered``."""

    handler: Handler = lambda argv: (0, "", "")
    commands: list[list[str]] = field(default_factory=list)
    timeouts: list[int | None] = field(default_factory=list)
    open_sessions: set[str] = field(default_factory=set)
    created_sessions: list[str] = field(default_factory=list)
    deleted_sessions: list[str] = field(default_factory=list)
    gate: asyncio.Event | None = None
    entered: asyncio.Event = field(default_factory=asyncio.Event)

    async def create_session(self, session_id: str, request_timeout: float | None = None) -> None:
        self.open_sessions.add(session_id)
        self.created_sessions.append(session_id)

    async def execute_session_command(self, session_id: str, req: Any, timeout: int | None = None) -> _Response:
        assert session_id in self.open_sessions and req.run_async is False
        argv = argv_of(req.command)
        self.commands.append(argv)
        self.timeouts.append(timeout)
        if self.gate is not None:
            self.entered.set()
            await self.gate.wait()
        outcome = self.handler(argv)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(*outcome)

    async def delete_session(self, session_id: str, request_timeout: float | None = None) -> None:
        self.open_sessions.discard(session_id)
        self.deleted_sessions.append(session_id)


@dataclass
class FakeFs:
    files: dict[str, bytes] = field(default_factory=dict)
    fail_on: Callable[[str], BaseException | None] = lambda path: None

    async def upload_file(self, src: bytes, dst: str, timeout: int = 1800) -> None:
        if (error := self.fail_on(dst)) is not None:
            raise error
        self.files[dst] = src


_ids = itertools.count(1)


@dataclass
class FakeRaw:
    """A Daytona ``AsyncSandbox``."""

    id: str = field(default_factory=lambda: f"sb-{next(_ids)}")
    name: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    state: str = "started"
    desired_state: str = "started"
    domain_allow_list: str | None = None
    network_allow_list: str | None = None
    network_block_all: bool = False
    process: FakeProcess = field(default_factory=FakeProcess)
    fs: FakeFs = field(default_factory=FakeFs)
    deletes: int = 0
    previewed: list[int] = field(default_factory=list)
    network_updates: list[dict[str, Any]] = field(default_factory=list)
    delete_error: BaseException | None = None
    # Raised by the next deletes, one each, before they succeed.
    delete_failures: list[BaseException] = field(default_factory=list)
    # When set, a delete waits for it before doing anything.
    delete_gate: asyncio.Event | None = None

    async def delete(self, timeout: float | None = 60, wait: bool = False) -> None:
        if self.delete_gate is not None:
            await self.delete_gate.wait()
        self.deletes += 1
        if self.delete_failures:
            raise self.delete_failures.pop(0)
        if self.delete_error is not None:
            raise self.delete_error
        self.state = self.desired_state = "destroyed"

    async def get_preview_link(self, port: int, request_timeout: float | None = None) -> _Preview:
        self.previewed.append(port)
        return _Preview(url=f"https://{port}-{self.id}.{PROXY_DOMAIN}", token="tok")

    async def update_network_settings(self, **fields: Any) -> None:
        self.network_updates.append(fields)

    async def refresh_data(self, request_timeout: float | None = None) -> None:
        return None


@dataclass
class _Preview:
    url: str
    token: str


@dataclass
class FakeSnapshot:
    """A Daytona ``Snapshot``: ``cpu`` in vCPU, ``mem`` and ``disk`` in GiB."""

    name: str
    cpu: int = 2
    mem: int = 8
    disk: int = 10


@dataclass
class FakeSnapshots:
    snapshots: dict[str, FakeSnapshot] = field(default_factory=dict)
    gets: list[str] = field(default_factory=list)

    def add(self, snapshot: FakeSnapshot) -> None:
        self.snapshots[snapshot.name] = snapshot

    async def get(self, name: str) -> FakeSnapshot:
        self.gets.append(name)
        if name not in self.snapshots:
            raise DaytonaNotFoundError(f"Failed to get snapshot: Snapshot {name} not found")
        return self.snapshots[name]


@dataclass
class FakeDaytona:
    """An ``AsyncDaytona``. ``create_hook`` may raise, sleep or mutate before a create returns."""

    sandboxes: dict[str, FakeRaw] = field(default_factory=dict)
    snapshot: FakeSnapshots = field(default_factory=FakeSnapshots)
    created_params: list[Any] = field(default_factory=list)
    create_timeouts: list[float] = field(default_factory=list)
    create_hook: Callable[[Any, FakeRaw], Any] | None = None
    closed: int = 0
    lookups: list[str] = field(default_factory=list)
    make_raw: Callable[[], FakeRaw] = FakeRaw

    async def create(self, params: Any = None, *, timeout: float = 60, on_snapshot_create_logs: Any = None) -> FakeRaw:
        self.created_params.append(params)
        self.create_timeouts.append(timeout)
        raw = self.make_raw()
        raw.name = params.name or ""
        raw.labels = dict(params.labels or {})
        raw.domain_allow_list = params.domain_allow_list
        raw.network_allow_list = params.network_allow_list
        raw.network_block_all = bool(params.network_block_all)
        if self.create_hook is not None:
            result = self.create_hook(params, raw)
            if asyncio.iscoroutine(result):
                await result
        self.sandboxes[raw.id] = raw
        return raw

    async def get(self, sandbox_id_or_name: str, request_timeout: float | None = None) -> FakeRaw:
        self.lookups.append(sandbox_id_or_name)
        for raw in self.sandboxes.values():
            if sandbox_id_or_name in (raw.id, raw.name) and raw.state != "destroyed":
                return raw
        raise DaytonaNotFoundError(f"Sandbox with ID or name {sandbox_id_or_name} not found")

    async def close(self) -> None:
        self.closed += 1


class FakeClients:
    """A ``client_factory`` making a new FakeDaytona per call, all over one set of sandboxes and snapshots; ``made``
    holds them in order and ``settings`` what each was made with."""

    def __init__(self):
        self.sandboxes: dict[str, FakeRaw] = {}
        self.snapshot = FakeSnapshots()
        self.made: list[FakeDaytona] = []
        self.settings: list[Settings] = []

    def __call__(self, settings: Settings) -> FakeDaytona:
        self.settings.append(settings)
        self.made.append(FakeDaytona(sandboxes=self.sandboxes, snapshot=self.snapshot))
        return self.made[-1]


def joining_handler(raw: FakeRaw, *, corrupt: bool = False, otherwise: Handler = lambda argv: (0, "", "")) -> Handler:
    """Answers an upload's join script with the sha256 of the parts it names, in order, as ``sha256sum`` would."""

    def handle(argv: list[str]) -> Outcome:
        script = argv[-1]
        if "sha256sum" not in script:
            return otherwise(argv)
        if script.startswith(": >"):
            data = b""
        else:
            stem = re.search(r'cat "(\S+)\.\$i"', script).group(1)
            count = int(re.search(r"seq 0 (\d+)", script).group(1)) + 1
            data = b"".join(raw.fs.files[f"{stem}.{i}"] for i in range(count))
        digest = hashlib.sha256(data + (b"x" if corrupt else b"")).hexdigest()
        return (0, f"{digest}  /target\n", "")

    return handle
