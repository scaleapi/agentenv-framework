"""A task's Codex events are readable while it runs, through the framework's cursor read, and are
its final trajectory once it ends."""
import asyncio
import uuid

from tst.unit.agents.codex.agent import a2a_server, codex_bridge

EVENTS = [{"type": "thread.started", "thread_id": "t-1"}, {"type": "turn.started"}, {"type": "turn.completed"}]


def test_a_task_is_readable_while_it_runs_and_its_log_is_its_trajectory(codex, monkeypatch):
    release = asyncio.Event()

    async def fake_run_codex(**kwargs):
        kwargs["event_callback"](EVENTS[0])
        await release.wait()
        for event in EVENTS[1:]:
            kwargs["event_callback"](event)
        return codex_bridge.CodexResult(response="done", tool_call_count=0, session_id="t-1", trajectory=EVENTS)

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.client() as client:
            sent = await client.post("/a2a", json={
                "jsonrpc": "2.0", "id": 1, "method": "message/send",
                "params": {
                    "message": {
                        "kind": "message", "role": "user", "messageId": uuid.uuid4().hex,
                        "parts": [{"kind": "text", "text": "go"}],
                    },
                    "configuration": {"blocking": False},
                },
            })
            task_id = sent.json()["result"]["id"]

            async def read(after: int) -> dict:
                return (await client.post("/ext/trajectory", json={"task_id": task_id, "after": after})).json()

            async with asyncio.timeout(5):
                while (running := await read(0))["next"] < 1:
                    await asyncio.sleep(0.01)
            release.set()
            async with asyncio.timeout(5):
                while (ended := await read(1))["state"] != "completed":
                    await asyncio.sleep(0.01)
            final = (await client.post("/ext/trajectory", json={"task_id": task_id})).json()
            return running, ended, final

    running, ended, final = asyncio.run(exercise())

    assert (running["state"], running["format"], running["events"]) == (
        "running", a2a_server.TRAJECTORY_FORMAT, EVENTS[:1],
    )
    assert (ended["events"], ended["next"], ended["has_more"]) == (EVENTS[1:], 3, False)
    assert final == {"trajectory": EVENTS}


def test_a_canceled_task_keeps_the_events_it_emitted(codex, monkeypatch):
    async def fake_run_codex(**kwargs):
        kwargs["event_callback"](EVENTS[0])
        await asyncio.Event().wait()

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.client() as client:
            sent = await client.post("/a2a", json={
                "jsonrpc": "2.0", "id": 1, "method": "message/send",
                "params": {
                    "message": {
                        "kind": "message", "role": "user", "messageId": uuid.uuid4().hex,
                        "parts": [{"kind": "text", "text": "go"}],
                    },
                    "configuration": {"blocking": False},
                },
            })
            task_id = sent.json()["result"]["id"]

            async def read() -> dict:
                return (await client.post("/ext/trajectory", json={"task_id": task_id, "after": 0})).json()

            async with asyncio.timeout(5):
                while (await read())["next"] < 1:
                    await asyncio.sleep(0.01)
            await client.post("/a2a", json={"jsonrpc": "2.0", "id": 2, "method": "tasks/cancel", "params": {"id": task_id}})
            async with asyncio.timeout(5):
                while (canceled := await read())["state"] != "canceled":
                    await asyncio.sleep(0.01)
            final = (await client.post("/ext/trajectory", json={"task_id": task_id})).json()
            return canceled, final

    canceled, final = asyncio.run(exercise())

    assert canceled["events"] == EVENTS[:1]
    assert final == {"trajectory": EVENTS[:1]}
