"""The gateway's MCP client sessions to its child servers, one per child, shared by the MCP proxy, /step and the trigger
engine.

A child's session is in one of three places at any moment, never two:

- **current**: the one a new call to that child gets (``current[url]``);
- **draining**: replaced, because the child rejected its session id (it restarted, or expired it), and still
  answering the calls already sent on it; it closes once the last of them ends;
- **gone**: closed, or being closed.

A call that never reached the child (the child rejected the session, or its stream had closed) is retried once on
a fresh session. A call the transport lost in flight may already have written, so its session is retired: every call
still on it fails with CONNECTION_CLOSED at once, it is closed in the background, and the call is not retried. A
session whose ``close`` is set is never handed to a new call. Nothing here touches another child's session.

Each session is held open by an owner task of its own: mcp's client contexts are anyio task groups, exited only by
the task that entered them. A child's lock covers opening, replacing and closing its session; an open holds it through
its connection attempts, so failing a session's calls never waits for it."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio
import httpx
from anyio import ClosedResourceError
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, CallToolResult, ErrorData

from .child_transport import Report, child_http_client
from .constants import CARD_FETCH_TIMEOUT_S

if TYPE_CHECKING:
    from . import InternalMCPServer

logger = logging.getLogger(__name__)


@dataclass(eq=False)
class ChildSession:
    """One child's MCP session and the task that holds it open until ``close`` is set."""

    url: str
    close: asyncio.Event = field(default_factory=asyncio.Event)
    session: ClientSession | None = None
    owner: asyncio.Task | None = None
    in_flight: int = 0
    replaced: bool = False
    retiring: bool = False


class ChildSessions:
    """Every child's session by its MCP URL (see the module docstring for the rules they follow)."""

    CLOSE_TIMEOUT_S = 3.0  # wait for an owner task to exit a closed session; past it, cancel it

    def __init__(self, servers: list[InternalMCPServer]):
        self.servers = servers
        self.current: dict[str, ChildSession] = {}
        self.draining: set[ChildSession] = set()
        self.retirements: set[asyncio.Task] = set()  # sessions being closed in the background
        self.closing = False  # set at process shutdown; no session is opened or retired past it
        self._locks: dict[str, asyncio.Lock] = {}

    def lock(self, url: str) -> asyncio.Lock:
        return self._locks.setdefault(url, asyncio.Lock())

    async def sessions(self) -> dict[str, ClientSession]:
        """Every child's session by its MCP URL, each opened on first need, as discovery reads them."""
        return {server.mcp_url: (await self.ensure(server.mcp_url)).session for server in self.servers}

    async def call(self, url: str, tool_name: str, arguments: dict, meta: dict | None, *,
                   timeout: float) -> CallToolResult:
        """``tool_name`` called on the child at ``url``, retried once on a fresh session when it never reached the
        child."""
        child = await self.ensure(url)
        try:
            return await asyncio.wait_for(self._call_on(child, tool_name, arguments, meta), timeout=timeout)
        except Exception as e:
            if not is_dead_session(e):
                raise
            if not call_never_reached_child(e):
                self.retire_soon(child)
                logger.warning(f"child session to {url} died with {tool_name} in flight ({e!r}); not retried")
                raise
            await self.replace(child)
            logger.warning(f"child session to {url} is dead ({e!r}), reconnecting...")
            child = await self.ensure(url)
            return await asyncio.wait_for(self._call_on(child, tool_name, arguments, meta), timeout=timeout)

    async def ensure(self, url: str) -> ChildSession:
        """The current session to the child at ``url``, opened on first need, and again once the last one was retired
        or replaced. Liveness is its owner task's, and a session being retired has none."""
        child = self.current.get(url)
        if child is not None and self._open(child):
            return child
        async with self.lock(url):
            child = self.current.get(url)
            if child is not None and self._open(child):
                return child
            if self.closing:
                raise RuntimeError("gateway is shutting down")
            if child is not None:
                await self._close(child)  # one that died on its own, or is being retired
            server = next((s for s in self.servers if s.mcp_url == url), None)
            if server is None:
                raise KeyError(url)
            child = ChildSession(url)
            loop = asyncio.get_running_loop()
            ready: asyncio.Future = loop.create_future()
            child.owner = loop.create_task(self._own(server, child, ready), name="gateway-child-session")
            child.owner.add_done_callback(_log_owner_exit)
            try:
                child.session = await ready
            except BaseException:
                # Nobody will use this session: an open still in progress is abandoned (its stack unwinds what it
                # entered), not finished in the background to hold a connection nobody waits on.
                child.close.set()
                child.owner.cancel()
                raise
            self.current[url] = child
            return child

    async def replace(self, child: ChildSession) -> None:
        """Have the next call to ``child``'s child open a fresh session, as the child rejected this one's id: the
        calls already sent on it each get their own answer, and it closes once they end."""
        async with self.lock(child.url):
            if self.current.get(child.url) is not child:
                return  # already replaced or retired
            del self.current[child.url]
            if not child.in_flight:
                await self._close(child)
                return
            child.replaced = True
            self.draining.add(child)
            child.owner.add_done_callback(lambda _: self.draining.discard(child))

    def retire_soon(self, child: ChildSession) -> None:
        """Retire ``child``: its calls in flight fail with CONNECTION_CLOSED now, and it is closed in the background
        under its child's lock. However many ask, it is retired once."""
        if self.closing or child.retiring or not self._live(child):
            return
        child.retiring = True
        child.close.set()
        task = asyncio.get_running_loop().create_task(self._retire(child), name="gateway-child-retire")
        self.retirements.add(task)
        task.add_done_callback(self.retirements.discard)

    def transport_failed(self, child: ChildSession, exc: httpx.TransportError) -> None:
        """A request on ``child`` failed at the transport. mcp's client would keep the request pending, so the session
        is retired now: every call it has in flight fails with CONNECTION_CLOSED instead of waiting out its timeout,
        and the next call reopens. Called from inside the client's own task; a report for a session already retired,
        or still opening, is nothing, as every request in flight on a dying session reports."""
        if self.closing or child.retiring or not self._live(child):
            return
        logger.warning(f"child transport to {child.url} failed ({exc!r}); retiring its session")
        self.retire_soon(child)

    async def close(self) -> None:
        """Close every session, those draining too: the process lifespan, after the last request."""
        for child in [*self.current.values(), *self.draining]:
            await self._close(child)

    async def open_session(self, server: InternalMCPServer, exit_stack: contextlib.AsyncExitStack, report: Report,
                           max_retries: int = 10, retry_delay: float = 1.0) -> ClientSession:
        """Open a persistent MCP client session to ``server``, over a transport that hands ``report`` a failed request
        so the session is retired at once (see child_transport).

        Each attempt opens its contexts on a stack of its own. mcp's client runs its requests in an anyio task group
        whose cancel scope belongs to the task that entered it, this one: a request that fails outright (the child
        refusing the connection) cancels whatever this task awaits, and keeps cancelling it until that group is
        exited. So a failed attempt is exited before anything else is awaited, and reads as a failure to retry
        rather than as this task being cancelled; only a cancellation that is still pending once the attempt's
        group is gone came from outside."""
        for attempt in range(max_retries):
            attempt_stack = contextlib.AsyncExitStack()
            try:
                client = await attempt_stack.enter_async_context(child_http_client(report))
                read_stream, write_stream, _ = await attempt_stack.enter_async_context(
                    streamable_http_client(server.mcp_url, http_client=client))
                session = await attempt_stack.enter_async_context(ClientSession(read_stream, write_stream))
                # a child that accepts the connection but never answers initialize is a failed attempt, not a hang
                await asyncio.wait_for(session.initialize(), timeout=CARD_FETCH_TIMEOUT_S)
            except BaseException as e:
                failure: BaseException = e
                try:
                    await attempt_stack.aclose()
                except BaseException as unwound:  # the client's task group names what failed
                    if isinstance(e, asyncio.CancelledError):
                        failure = unwound
                if isinstance(e, asyncio.CancelledError) and asyncio.current_task().cancelling():
                    raise
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to connect to {server.name} (attempt {attempt + 1}/{max_retries}): {failure!r}")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to {server.name} after {max_retries} attempts: {failure!r}")
                    if failure is e:
                        raise
                    raise RuntimeError(f"could not open a session to {server.name}: {failure!r}") from failure
            else:
                exit_stack.push_async_callback(attempt_stack.aclose)
                logger.info(f"Opened persistent session to {server.name}")
                return session
        raise RuntimeError(f"could not open a session to {server.name}: no attempt was made")

    @staticmethod
    def _open(child: ChildSession) -> bool:
        return not child.owner.done() and not child.close.is_set()

    def _live(self, child: ChildSession) -> bool:
        """Whether ``child`` is current, or draining."""
        return self.current.get(child.url) is child or child in self.draining

    async def _own(self, server: InternalMCPServer, child: ChildSession, ready: asyncio.Future) -> None:
        """Holds ``child`` open: opens its client contexts, reports through ``ready``, parks until ``close`` is set (or
        it is cancelled) and exits them. Returns rather than raising on a failed open, which ``ready`` carries.
        ``close`` is set on the way out whatever the reason, so a call racing on it fails now, not at its timeout."""
        try:
            async with contextlib.AsyncExitStack() as stack:
                try:
                    session = await self.open_session(server, stack, lambda exc: self.transport_failed(child, exc))
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if not ready.done():
                        ready.set_exception(e)
                    return
                if ready.cancelled():
                    return  # nobody is waiting any more; the stack closes what was opened
                ready.set_result(session)
                await child.close.wait()
        except BaseExceptionGroup as e:
            # a request failing outright under a live session cancels this task through mcp's client task group
            # (see open_session); the stack has unwound, and the transport failure retires the session
            logger.warning(f"child session to {server.name} ended: {e!r}")
        finally:
            child.close.set()

    async def _close(self, child: ChildSession) -> None:
        """Take ``child`` out of current and draining, close it and wait (bounded) for its owner to exit it. Callers
        hold its child's lock, but the process lifespan, which runs after the last request."""
        if self.current.get(child.url) is child:
            del self.current[child.url]
        self.draining.discard(child)
        child.close.set()
        if child.owner is not None and not child.owner.done():
            try:
                await asyncio.wait_for(child.owner, timeout=self.CLOSE_TIMEOUT_S)
            except Exception:
                logger.warning(f"the child session to {child.url} did not close cleanly", exc_info=True)

    async def _retire(self, child: ChildSession) -> None:
        async with self.lock(child.url):
            if self._live(child):
                await self._close(child)

    async def _call_on(self, child: ChildSession, tool_name: str, arguments: dict, meta: dict | None) -> CallToolResult:
        """``session.call_tool`` on ``child``, counted among its calls in flight, unless the session closes first:
        then the call fails with CONNECTION_CLOSED the way a closed transport would fail it. mcp's client keeps a
        request whose transport failed pending, and exiting the session under it cancels the receive loop before that
        loop can fail the request, so this fails it. A draining session closes once the last of its calls ends."""
        child.in_flight += 1
        try:
            call = asyncio.ensure_future(child.session.call_tool(tool_name, arguments, meta=meta))
            closed = asyncio.ensure_future(child.close.wait())
            try:
                done, _ = await asyncio.wait({call, closed}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                closed.cancel()
                if not call.done():
                    call.cancel()
                await asyncio.gather(call, closed, return_exceptions=True)  # the loser's cleanup ends first
            if call in done:
                return call.result()
            raise McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))
        finally:
            child.in_flight -= 1
            if child.replaced and not child.in_flight:
                child.close.set()


def _log_owner_exit(owner: asyncio.Task) -> None:
    """Retrieve a session's exception when it dies on its own (a transport crash), so it is logged now rather than as
    'exception never retrieved' at garbage collection."""
    if not owner.cancelled() and owner.exception() is not None:
        logger.warning("child session ended", exc_info=owner.exception())


def is_dead_session(exc: BaseException) -> bool:
    """Whether a child call failed because its session is gone, so the session is replaced or retired: the stream
    closed under it, the child answered 404 for the session id (idle expiry or a restart, which the client reports as
    code 32600), or the transport died with the call in flight (CONNECTION_CLOSED)."""
    if isinstance(exc, (ClosedResourceError, anyio.BrokenResourceError)):
        return True
    return isinstance(exc, McpError) and exc.error.code in (32600, CONNECTION_CLOSED)


def call_never_reached_child(exc: BaseException) -> bool:
    """Whether a dead-session failure may be retried: the stream was closed before the request was written, or the
    child rejected the session id at the transport. Not CONNECTION_CLOSED: the call was in flight when the transport
    died, and the child may already have performed its write."""
    if isinstance(exc, (ClosedResourceError, anyio.BrokenResourceError)):
        return True
    return isinstance(exc, McpError) and exc.error.code == 32600
