"""A CDP connection over a Chromium's debugging pipe."""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
from typing import Any, Callable, ClassVar, Optional

logger = logging.getLogger(__name__)

#: The fds a Chromium on ``--remote-debugging-pipe`` reads its CDP messages from and writes its answers to.
PIPE_IN, PIPE_OUT = 3, 4
#: The largest CDP message the pipe takes, such as a screenshot's.
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


class DevToolsError(RuntimeError):
    """A CDP call the browser answered with an error."""


class DevToolsPipe:
    """A CDP connection over a Chromium's debugging pipe, each message JSON ended by a NUL. It connects anew at every
    launch; a call with no browser connected, or whose browser exits before answering, raises ConnectionError. Each
    event, a message with no ``id``, goes to every listener, and each closer hears when the browser closes the pipe."""

    #: How long a call waits for the browser's answer when it names no wait of its own.
    call_timeout_s: ClassVar[float] = 30.0

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self._waiting: dict[int, asyncio.Future] = {}
        self._answered: dict[int, Callable[[dict], None]] = {}
        self._writer: Optional[asyncio.WriteTransport] = None
        self._reading: Optional[asyncio.Task] = None
        self._listeners: list[Callable[[dict], None]] = []
        self._closers: list[Callable[[], None]] = []

    def listen(self, listener: Callable[[dict], None]) -> None:
        """Hand ``listener`` each event: its ``method``, ``params`` and, on a page's session, ``sessionId``."""
        self._listeners.append(listener)

    def on_close(self, closer: Callable[[], None]) -> None:
        """Call ``closer`` when the browser closes its end of the pipe, not when this end closes it."""
        self._closers.append(closer)

    async def connect(self, to_browser: int, from_browser: int) -> None:
        """Take over this end of a new pipe, ending any connection to a browser launched before."""
        self.close()
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=MAX_MESSAGE_BYTES)
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(from_browser, "rb", 0))
        self._writer, _ = await loop.connect_write_pipe(asyncio.Protocol, os.fdopen(to_browser, "wb", 0))
        self._reading = asyncio.ensure_future(self._read(reader))

    async def call(self, method: str, params: Optional[dict] = None, *, session_id: Optional[str] = None,
                   timeout_s: Optional[float] = None) -> dict:
        """``method``'s result, or DevToolsError with the browser's message."""
        answered = await self._answer(method, params, session_id,
                                      self.call_timeout_s if timeout_s is None else timeout_s)
        if "error" in answered:
            raise DevToolsError(f"{method}: {answered['error'].get('message', answered['error'])}")
        return answered.get("result", {})

    async def request(self, method: str, params: Optional[dict] = None, *, session_id: Optional[str] = None,
                      answered: Optional[Callable[[dict], None]] = None) -> dict:
        """The browser's whole answer to ``method``, its ``result`` or its ``error``, however long it takes: what a
        relay passes on as it is. ``answered`` hears it as the pipe reads it, in order with the events before and after
        it, which the caller, resumed only later, may not be."""
        return await self._answer(method, params, session_id, None, answered)

    async def _answer(self, method: str, params: Optional[dict], session_id: Optional[str],
                      timeout_s: Optional[float], answered: Optional[Callable[[dict], None]] = None) -> dict:
        if self._writer is None or self._writer.is_closing():
            raise ConnectionError("no browser is connected over its debugging pipe")
        message_id = next(self._ids)
        answer = asyncio.get_running_loop().create_future()
        self._waiting[message_id] = answer
        if answered is not None:
            self._answered[message_id] = answered
        message = {"id": message_id, "method": method, "params": params or {},
                   **({"sessionId": session_id} if session_id else {})}
        self._writer.write(json.dumps(message).encode() + b"\0")
        try:
            return await asyncio.wait_for(answer, timeout_s)
        finally:
            self._waiting.pop(message_id, None)
            self._answered.pop(message_id, None)

    def close(self) -> None:
        if self._reading is not None:
            self._reading.cancel()
        if self._writer is not None:
            self._writer.close()
        self._fail_waiting()
        self._writer = self._reading = None

    async def _read(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                message = json.loads((await reader.readuntil(b"\0"))[:-1])
                if "id" not in message:
                    for listener in self._listeners:
                        _heard(listener, message)
                    continue
                answer = self._waiting.get(message.get("id"))
                if answer is not None and not answer.done():
                    if (answered := self._answered.pop(message["id"], None)) is not None:
                        _heard(answered, message)
                    answer.set_result(message)
        except (asyncio.IncompleteReadError, ConnectionError):
            if self._writer is not None:
                self._writer.close()
            self._fail_waiting()
            for closer in self._closers:
                _heard(closer)

    def _fail_waiting(self) -> None:
        for answer in self._waiting.values():
            if not answer.done():
                answer.set_exception(ConnectionError("the browser closed its debugging pipe"))


def _heard(callback: Callable, *args: Any) -> None:
    try:
        callback(*args)
    except Exception:
        logger.exception("a DevToolsPipe listener failed")
