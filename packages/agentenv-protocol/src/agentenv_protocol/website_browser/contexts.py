"""The contexts a website browser leases beside its default one: each a browser context with its own cookies, local
storage, tabs and downloads folder, named by a label, granted to the agents a task or the relay names, and ended by its
release, a reset or the browser's relaunch, which no context outlives. A context an agent makes over CDP is leased to
that agent as ``agent-<user>-<n>``; one made unseen, ``agent-unknown-<n>``, to nobody. Grants are recorded, not
enforced: the CDP addresses agents hold still see every context."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from ..computer.errors import InvalidArgument, TargetUnavailable
from .control import DOWNLOADS
from .devtools import DevToolsError

if TYPE_CHECKING:
    from .control import BrowserControl
    from .devtools import DevToolsPipe

logger = logging.getLogger(__name__)

#: Where each lease's files are under the session user's home, a folder per label.
CONTEXTS = "Contexts"
#: Who made a lease: the task, through ``contexts/lease``; else the agent that made it over CDP, by its user, or
#: ``UNKNOWN`` for one made where the relay didn't see.
TASK, UNKNOWN = "task", "unknown"
#: How the labels of contexts agents make start, which a task's labels may not.
AGENT_PREFIX = "agent-"
#: The longest label, so ``chromium-<label>`` stays a short name.
LABEL_CHARS = 40
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?")


def label_problem(label: object, *, by_task: bool = True) -> Optional[str]:
    """Why ``label`` can't name a lease, or None: lowercase letters, digits and ``-``, not at either end, at most
    ``LABEL_CHARS`` of them; a task's may not start ``agent-``, which names the contexts agents make."""
    if not isinstance(label, str) or len(label) > LABEL_CHARS or not _LABEL.fullmatch(label):
        return (f"a context's label is lowercase letters, digits and '-', not at either end, at most {LABEL_CHARS} "
                f"characters, not {label!r}")
    if by_task and label.startswith(AGENT_PREFIX):
        return f"a label starting {AGENT_PREFIX!r} names a context an agent made, so a task's can't: {label!r}"
    return None


@dataclass
class Lease:
    """A leased context: its ``label``, the browser's ``id`` for it, who made it and the agents it's granted to."""

    label: str
    id: str
    made_by: str = TASK
    grants: list[str] = field(default_factory=list)

    def folder(self, home: Path) -> Path:
        return home / CONTEXTS / self.label

    def under(self) -> str:
        """Its downloads folder's path under the home."""
        return f"{CONTEXTS}/{self.label}/{DOWNLOADS}"

    def downloads(self, home: Path) -> Path:
        return home / self.under()

    def record(self) -> dict:
        return {"context": self.label, "id": self.id, "made_by": self.made_by, "grants": list(self.grants)}


class Contexts:
    """A browser's leases by label, in memory, since a server that restarts relaunches the browser and so ends them
    all, and what leasing, releasing and recording them does over its pipe ``devtools``. ``control`` follows each
    context's tabs, storage and downloads; ``home`` is the session user's home."""

    def __init__(self, devtools: DevToolsPipe, control: BrowserControl,
                 home: Callable[[], Path] = Path.home) -> None:
        self.devtools, self.control, self.home = devtools, control, home
        self.leases: dict[str, Lease] = {}
        self._made: dict[str, int] = {}
        self._tasks: set[asyncio.Future] = set()
        self._lock = asyncio.Lock()

    def __iter__(self):
        return iter(list(self.leases.values()))

    def get(self, label: str) -> Optional[Lease]:
        return self.leases.get(label)

    def of(self, context_id: Optional[str]) -> Optional[Lease]:
        return next((lease for lease in self.leases.values() if lease.id == context_id), None) \
            if context_id else None

    def id_of(self, label: Optional[str]) -> Optional[str]:
        """The browser's id for the lease ``label``, None for the default context; TargetUnavailable for a label that
        isn't leased."""
        if label is None:
            return None
        if (lease := self.leases.get(label)) is None:
            raise TargetUnavailable(f"no context {label!r} is leased; contexts/list names the leased ones")
        return lease.id

    async def lease(self, label: str, *, made_by: str = TASK, grants: tuple[str, ...] = ()) -> Lease:
        """The lease ``label``: the one leased under it before, as a retried step finds it, else a new context, its
        downloads going to its own folder. A task's label is checked; one ``made_by`` an agent may start ``agent-``,
        as a load of a snapshot restores such leases."""
        async with self._lock:
            if (lease := self.leases.get(label)) is None:
                if problem := label_problem(label, by_task=made_by == TASK):
                    raise InvalidArgument(problem)
                context_id = (await self.devtools.call("Target.createBrowserContext", {"disposeOnDetach": False}))[
                    "browserContextId"]
                lease = self._record(Lease(label, context_id, made_by))
                await self._send_downloads(lease)
        for user in grants:
            self.grant(label, user)
        return lease

    def grant(self, label: str, user: str) -> Lease:
        """The lease ``label`` granted to ``user`` too, recorded; granting a user twice changes nothing."""
        if not isinstance(user, str) or not user:
            raise InvalidArgument("a grant names the agent's user, a non-empty string")
        if (lease := self.leases.get(label)) is None:
            raise TargetUnavailable(f"no context {label!r} is leased; contexts/list names the leased ones")
        if user not in lease.grants:
            lease.grants.append(user)
        return lease

    async def release(self, label: str) -> bool:
        """The lease ``label`` ended: its context disposed, which closes its tabs, and its folder deleted. Whether
        there was one: releasing a label that isn't leased does nothing, so a retried step is harmless."""
        async with self._lock:
            if (lease := self.leases.pop(label, None)) is None:
                return False
            self.control.closing_context(lease.id)
            with contextlib.suppress(DevToolsError):
                await self.devtools.call("Target.disposeBrowserContext", {"browserContextId": lease.id})
            await asyncio.to_thread(shutil.rmtree, lease.folder(self.home()), True)
        return True

    def made_by_agent(self, user: str, context_id: str) -> Lease:
        """The context ``context_id`` an agent made over CDP as ``user``, leased to it as ``agent-<user>-<n>``,
        starting empty, its downloads sent to its own folder."""
        if (known := self.of(context_id)) is not None:
            return known
        lease = self._record(Lease(self._agent_label(user), context_id, user, [user]))
        self._spawn(self._send_downloads(lease))
        return lease

    async def find_unseen(self) -> list[Lease]:
        """Each context the browser has that no lease names, made where the relay didn't see, leased to nobody as
        ``agent-unknown-<n>``; the leases it made."""
        try:
            ids = (await self.devtools.call("Target.getBrowserContexts"))["browserContextIds"]
        except (ConnectionError, DevToolsError, KeyError):
            return []
        found = []
        for context_id in ids:
            if self.of(context_id) is None:
                found.append(self._record(Lease(self._agent_label(UNKNOWN), context_id, UNKNOWN)))
                await self._send_downloads(found[-1])
        return found

    def may_dispose(self, user: str, context_id: Optional[str]) -> Optional[str]:
        """Why an agent ``user`` may not dispose the context ``context_id``, or None: only a context it made ends
        that way; a task's lease ends by its release, and another agent's isn't its to end."""
        lease = self.of(context_id)
        if lease is None or lease.made_by == user:
            return None
        return (f"the context {lease.label!r} is the task's, which ends by its release" if lease.made_by == TASK else
                f"the context {lease.label!r} is another agent's")

    def disposed(self, context_id: Optional[str]) -> None:
        """The context ``context_id`` an agent disposed: its lease ended and its folder moved aside at once, then
        deleted, so a lease of its label after has a folder of its own."""
        if (lease := self.of(context_id)) is not None:
            self.leases.pop(lease.label, None)
            folder, gone = lease.folder(self.home()), self.home() / CONTEXTS / f".gone-{lease.id}"
            with contextlib.suppress(OSError):
                folder.rename(gone)
                self._spawn(asyncio.to_thread(shutil.rmtree, gone, True))

    def download_path(self, context_id: Optional[str]) -> Optional[str]:
        """Where the lease of the context ``context_id`` saves downloads, None for the default context or one that
        isn't leased."""
        lease = self.of(context_id)
        return str(lease.downloads(self.home())) if lease is not None else None

    def holding(self, path: str) -> Optional[str]:
        """The browser's id for the lease whose downloads folder holds ``path``, None for any other path."""
        home = self.home()
        return next((lease.id for lease in self.leases.values() if Path(path).is_relative_to(lease.downloads(home))),
                    None)

    def resend_downloads(self) -> None:
        """Each lease's downloads sent to its own folder again, as an agent's session that set them may leave them
        set otherwise as it ends."""
        for lease in self:
            self._spawn(self._send_downloads(lease))

    def ended(self) -> None:
        """The browser has gone, and every context with it: no lease is left, and their folders are deleted."""
        self.leases.clear()
        self._made.clear()
        shutil.rmtree(self.home() / CONTEXTS, ignore_errors=True)

    def _record(self, lease: Lease) -> Lease:
        self.leases[lease.label] = lease
        self.control.other_contexts[lease.id] = lease.label
        lease.downloads(self.home()).mkdir(parents=True, exist_ok=True)
        return lease

    def _agent_label(self, user: str) -> str:
        """The label of the next context ``user`` makes: ``agent-<user>-<n>``, its user lowercased, each run of
        other characters a ``-``, and cut to fit."""
        name = re.sub(r"[^a-z0-9]+", "-", user.lower()).strip("-") or UNKNOWN
        while True:
            self._made[name] = self._made.get(name, 0) + 1
            tail = f"-{self._made[name]}"
            label = f"{AGENT_PREFIX}{name[:LABEL_CHARS - len(AGENT_PREFIX) - len(tail)].rstrip('-')}{tail}"
            if label not in self.leases:
                return label

    async def _send_downloads(self, lease: Lease) -> None:
        """The lease's downloads sent to its own folder, which reports them on agentenv's connection."""
        with contextlib.suppress(ConnectionError, DevToolsError):
            await self.devtools.call("Browser.setDownloadBehavior", {
                "behavior": "allow", "browserContextId": lease.id, "downloadPath": str(lease.downloads(self.home())),
                "eventsEnabled": True})

    def _spawn(self, work) -> None:
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Future) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.warning(f"the browser's contexts: {type(error).__name__}: {error}")
