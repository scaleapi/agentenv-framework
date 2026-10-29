import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tst.unit.agents.openhands.agent import conversation_run

ConversationRun = conversation_run.ConversationRun


class FakeConversation:
    """Interrupts as the SDK's conversation does: it cancels the task ``arun()`` runs in once ``arun()`` has recorded
    it, and before that only pauses, which ``arun()`` undoes as it starts. A run lasts until ``finish`` is set."""

    def __init__(self) -> None:
        self.finish = threading.Event()
        self.started = threading.Event()
        self.task: asyncio.Task | None = None
        self.thread: str | None = None
        self.paused = self.cancelled = False
        self.closed = 0

    async def arun(self) -> None:
        self.task = asyncio.current_task()
        self.paused = False
        self.thread = threading.current_thread().name
        self.started.set()
        try:
            while not self.finish.is_set():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self.cancelled = True

    def interrupt(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.get_loop().call_soon_threadsafe(self.task.cancel)
        else:
            self.paused = True

    def close(self) -> None:
        self.closed += 1


def test_a_run_goes_to_its_end_on_a_thread_of_its_own() -> None:
    conversation = FakeConversation()
    conversation.finish.set()

    asyncio.run(ConversationRun(conversation).to_its_end())

    assert conversation.started.is_set() and conversation.thread == "openhands-run"


def test_a_stop_before_the_run_starts_keeps_it_from_starting() -> None:
    conversation = FakeConversation()
    run = ConversationRun(conversation)

    run.stop()
    asyncio.run(run.to_its_end())

    assert not conversation.started.is_set() and conversation.closed == 1


def test_a_stop_while_it_runs_interrupts_the_run() -> None:
    conversation = FakeConversation()
    run = ConversationRun(conversation)

    async def main() -> None:
        running = asyncio.ensure_future(run.to_its_end())
        await asyncio.to_thread(conversation.started.wait)
        run.stop()
        await asyncio.wait_for(running, 5)

    asyncio.run(main())

    assert conversation.cancelled and conversation.closed == 1


def test_a_cancelled_caller_stops_the_run_and_waits_for_it_to_stop() -> None:
    conversation = FakeConversation()
    run = ConversationRun(conversation)

    async def main() -> None:
        running = asyncio.ensure_future(run.to_its_end())
        await asyncio.to_thread(conversation.started.wait)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(main())

    assert conversation.cancelled and conversation.closed == 1


def test_running_turns_leave_the_default_executor_free() -> None:
    conversations = [FakeConversation() for _ in range(3)]

    async def main() -> str:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        runs = [asyncio.ensure_future(ConversationRun(c).to_its_end()) for c in conversations]
        for conversation in conversations:
            await asyncio.to_thread(conversation.started.wait)
        free = await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 5)
        for conversation in conversations:
            conversation.finish.set()
        await asyncio.gather(*runs)
        return free

    assert asyncio.run(main()) == "free"

