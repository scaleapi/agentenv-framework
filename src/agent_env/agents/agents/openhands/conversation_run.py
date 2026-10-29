"""One ``arun()`` of a conversation, on a thread and an event loop of its own. The SDK does blocking work between its
awaits (tools, persistence), which on the server's loop would stall every other task and the agent's own HTTP
endpoints, the peer-agents MCP server among them; and a run on the loop's default executor would hold one of the few
threads the server's other blocking calls share for as long as it runs."""

import asyncio
import threading
from contextlib import suppress
from typing import Protocol


class Conversation(Protocol):
    async def arun(self) -> None: ...

    def interrupt(self) -> None: ...

    def close(self) -> None: ...


class ConversationRun:
    def __init__(self, conversation: Conversation) -> None:
        self.conversation = conversation
        self.stopped = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def stop(self) -> None:
        """Interrupt the run, or keep it from starting, and close its tool executors: the terminal tool does not heed
        the interrupt, and a command it is running would otherwise hold the run until the command ends."""
        with self._lock:
            self.stopped.set()
            loop = self._loop
        if loop is not None:
            # A run that ends meanwhile closes its loop, and then has nothing left to interrupt.
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self.conversation.interrupt)
        self.conversation.close()

    async def to_its_end(self) -> None:
        """Run to the end; cancelled, stop the run, wait until it has stopped and re-raise."""
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()

        def settle(error: BaseException | None) -> None:
            if done.done():
                return
            if error is None:
                done.set_result(None)
            else:
                done.set_exception(error)

        def drive() -> None:
            error: BaseException | None = None
            try:
                self._drive()
            except BaseException as exc:
                error = exc
            # The server's loop is closed once it has stopped, and then nothing waits for the run.
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(settle, error)

        threading.Thread(target=drive, name="openhands-run").start()
        try:
            await asyncio.shield(done)
        except asyncio.CancelledError:
            if not done.done():
                await asyncio.to_thread(self.stop)
                await asyncio.wait([done])
            if not done.cancelled():
                done.exception()  # the caller was cancelled, so how the run ended is no longer its to see
            raise

    def _drive(self) -> None:
        # Made on this thread, so the server's thread keeps its own event loop. Closing the runner cancels what the
        # run left on its loop, LiteLLM's logging worker among them.
        with asyncio.Runner() as runner:
            with self._lock:
                if self.stopped.is_set():
                    return
                self._loop = runner.get_loop()
            runner.run(self._arun())

    async def _arun(self) -> None:
        # A stop sets the flag before it sends the interrupt here, so one that arrives before arun() has recorded the
        # task it would cancel, and so only pauses, which arun() undoes, keeps the run from starting.
        if not self.stopped.is_set():
            await self.conversation.arun()
