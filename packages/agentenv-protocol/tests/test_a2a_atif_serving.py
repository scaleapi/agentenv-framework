"""The SDK serves atif/1 beside the native log: an agent names its ATIF adapter in ``enable()``, and the SDK builds one
document per A2A context as the agent writes its log, serving it on task, context, cursor and final reads."""
from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import pytest
from atif_cases import Toy
from starlette.testclient import TestClient

from agentenv_protocol.a2a_agent import (
    TRAJECTORY_V1,
    AgentEnvAgent,
    AgentIdentity,
    ContextTrajectoryRequest,
    TaskRequest,
    TaskResult,
    TrajectoryLog,
    Uploaded,
    a2a_agent,
    enable,
    extension,
    framework,
)
from agentenv_protocol.atif import (
    FORMAT,
    PROFILE_VERSION,
    SCHEMA_VERSION,
    RecordedTask,
    atif_adapter,
    aux,
    errors_in,
    replay,
    validate,
    validate_live,
)

NATIVE = "toy-events/1"
_IDENTITY = AgentIdentity(name="toy-agent", description="test", version="2.0")


@atif_adapter(native_format=NATIVE, version="0.1.0", input_includes_cache=False)
class Hooks(Toy):
    """The toy adapter, with a hook log the agent's own code feeds."""

    @aux("hooks", type="note")
    def note(self, record: dict, turn: Any) -> None:
        # Takes the text out of the record it's handed, which must leave the served aux log whole.
        turn.message(step=record["step"], text=record.pop("text"))


def _usage(step: str, prompt: int, out: int) -> dict[str, Any]:
    return {"type": "usage", "step": step, "in": prompt, "out": out}


def _work(tag: str) -> list[dict[str, Any]]:
    return [
        {"type": "say", "step": f"{tag}-1", "text": f"{tag}: listing"},
        {"type": "call", "step": f"{tag}-1", "id": f"{tag}-c1", "name": "Bash", "args": {"command": "ls"}},
        _usage(f"{tag}-1", 10, 5),
        {"type": "result", "id": f"{tag}-c1", "out": "a.txt b.txt"},
        {"type": "say", "step": f"{tag}-2", "text": f"{tag}: two files"},
        _usage(f"{tag}-2", 12, 3),
        {"type": "total", "cost": 0.25},
    ]


class _Hold:
    """Where a test holds a turn: ``arrivals`` counts the holds it reached, and each goes on once ``resume`` is set."""

    def __init__(self) -> None:
        self.arrivals = 0
        self.resume = threading.Event()

    async def wait(self) -> None:
        self.arrivals += 1
        while not self.resume.is_set():
            await asyncio.sleep(0.01)
        self.resume.clear()

    def reached(self, count: int = 1) -> None:
        deadline = time.monotonic() + 10
        while self.arrivals < count:
            assert time.monotonic() < deadline, "the turn didn't reach its hold"
            time.sleep(0.01)


#: What a toy agent's turn does, by the scenario its message names first; the message's second word is a tag unique in
#: the context, so the turn's call ids and step keys are too.
SCENARIOS: dict[str, Callable[[TrajectoryLog, str, _Hold], Awaitable[TaskResult]]] = {}


def _scenario(name: str) -> Callable[[Callable], Callable]:
    def register(play: Callable) -> Callable:
        SCENARIOS[name] = play
        return play
    return register


def _append(log: TrajectoryLog, records: list[dict[str, Any]]) -> None:
    for record in records:
        log.append(record)


def _spawning(tag: str) -> list[dict[str, Any]]:
    """A Task call that spawns a sub-agent, and the sub-agent's first step."""
    return [{"type": "call", "step": f"{tag}-1", "id": f"{tag}-task", "name": "Task", "args": {"goal": "ls"}},
            {"type": "spawn", "id": f"{tag}-task", "agent": f"{tag}-sub", "name": "explorer"},
            {"type": "say", "step": f"{tag}-s1", "text": "sub: listing", "agent": f"{tag}-sub"}]


@_scenario("work")
async def _works(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Lists the files and answers with its last message."""
    _append(log, _work(tag))
    return TaskResult.text(f"{tag}: two files")


@_scenario("hooks")
async def _hooks(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Feeds a hook's note after its first event, and a note no handler takes at the end."""
    records = _work(tag)
    log.append(records[0])
    log.aux("hooks", {"type": "note", "step": f"{tag}-h", "text": "the pre-tool hook allowed ls"})
    _append(log, records[1:])
    log.aux("hooks", {"type": "other", "step": f"{tag}-x", "text": "no handler takes this one"})
    return TaskResult.text(f"{tag}: two files")


@_scenario("late")
async def _late(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Moves on before its call's result comes, holding there, and holds again once it has."""
    records = _work(tag)
    _append(log, records[:3])
    log.append({"type": "say", "step": f"{tag}-w", "text": f"{tag}: waiting on ls"})
    await hold.wait()
    _append(log, [records[3], {"type": "say", "step": f"{tag}-3", "text": f"{tag}: got it"},
                  {"type": "say", "step": f"{tag}-4", "text": f"{tag}: two files"}])
    await hold.wait()
    return TaskResult.text(f"{tag}: two files")


@_scenario("failed")
async def _fails(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Works, then fails the task."""
    _append(log, _work(tag))
    return TaskResult.failure("toy.failed", "ls found nothing to fix")


@_scenario("raised")
async def _raises(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Raises with its call unanswered."""
    _append(log, _work(tag)[:2])
    raise RuntimeError("the toy CLI crashed")


@_scenario("canceled")
async def _waits(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Holds after its first step, for a test to cancel it or let it answer."""
    _append(log, _work(tag)[:3])
    await hold.wait()
    return TaskResult.text("never reached")


@_scenario("data")
async def _answers_with_data(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Answers with a structured output and the harness's diagnostics beside its text."""
    _append(log, _work(tag))
    return (TaskResult.builder().succeeded().add_text(f"{tag}: two files")
            .add_structured_output({"files": ["a.txt", "b.txt"]})
            .add_data({"tool_call_count": 1, "cli_attempts": 2}).build())


@_scenario("returned")
async def _returns(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Appends nothing and returns its native trajectory at the end."""
    return (TaskResult.builder().succeeded().add_text(f"{tag}: two files")
            .native_trajectory(format=NATIVE, payload=_work(tag)).build())


@_scenario("both")
async def _appends_and_returns(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Appends an event and returns a native trajectory too."""
    records = _work(tag)
    log.append(records[0])
    return TaskResult.builder().succeeded().add_text("done").native_trajectory(format=NATIVE, payload=records).build()


@_scenario("broken")
async def _breaks_its_adapter(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Appends a result for a call it never made, which its adapter refuses."""
    records = _work(tag)
    _append(log, [records[0], {"type": "result", "id": "no-such-call", "out": "x"}, records[1]])
    return TaskResult.text("done anyway")


@_scenario("spawn")
async def _spawns(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Spawns a sub-agent that works and answers."""
    _append(log, [*_spawning(tag),
                  {"type": "say", "step": f"{tag}-s2", "text": "sub: two files", "agent": f"{tag}-sub"},
                  {"type": "result", "id": f"{tag}-task", "out": "two files"},
                  {"type": "say", "step": f"{tag}-2", "text": f"{tag}: the explorer found two files"}])
    return TaskResult.text(f"{tag}: the explorer found two files")


@_scenario("grow")
async def _sub_agent_grows(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Spawns a sub-agent that goes on writing steps after its call's result, holding in between."""
    _append(log, [*_spawning(tag),
                  {"type": "say", "step": f"{tag}-s2", "text": "sub: still listing", "agent": f"{tag}-sub"},
                  {"type": "result", "id": f"{tag}-task", "out": "listing"},
                  {"type": "say", "step": f"{tag}-2", "text": f"{tag}: the explorer is listing"},
                  {"type": "say", "step": f"{tag}-3", "text": f"{tag}: waiting"}])
    await hold.wait()
    _append(log, [{"type": "say", "step": f"{tag}-s3", "text": "sub: two files", "agent": f"{tag}-sub"},
                  {"type": "say", "step": f"{tag}-4", "text": f"{tag}: two files"}])
    return TaskResult.text(f"{tag}: two files")


@_scenario("orphan")
async def _spawn_never_answered(log: TrajectoryLog, tag: str, hold: _Hold) -> TaskResult:
    """Spawns a sub-agent and holds before its call's result, for a test to cancel it."""
    _append(log, _spawning(tag))
    await hold.wait()
    return TaskResult.text("never reached")


async def _turn(request: TaskRequest, hold: _Hold) -> TaskResult:
    scenario, tag = request.parts[0].text.split()
    return await SCENARIOS[scenario](request.trajectory, tag, hold)


def _toy(hold: _Hold | None = None, *, live: bool = True, atif: Any = Hooks, bind_context: bool = False) -> AgentEnvAgent:
    hold = hold or _Hold()
    extensions = (enable(TRAJECTORY_V1, live=live, native=NATIVE, **({"atif": atif} if atif else {})),)

    if bind_context:
        @a2a_agent(identity=_IDENTITY, extensions=extensions)
        class Bound(AgentEnvAgent):
            async def run(self, request: TaskRequest) -> TaskResult:
                return await _turn(request, hold)

            @extension(TRAJECTORY_V1.get.context)
            async def own_context(self, request: ContextTrajectoryRequest) -> dict[str, Any]:
                return {"trajectory": ["the agent's own shape"], "format": "toy-session/1"}

        return Bound()

    @a2a_agent(identity=_IDENTITY, extensions=extensions)
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return await _turn(request, hold)

    return Agent()


def _send(client: TestClient, text: str, message_id: str, *, context_id: str = "ctx-1",
          extra_parts: list[dict[str, Any]] | None = None) -> str:
    message = {"kind": "message", "messageId": message_id, "role": "user", "contextId": context_id,
               "parts": [{"kind": "text", "text": text}, *(extra_parts or [])]}
    response = client.post("/a2a", json={"jsonrpc": "2.0", "id": message_id, "method": "message/send",
                                         "params": {"message": message, "configuration": {"blocking": False}}})
    return response.json()["result"]["id"]


def _state(client: TestClient, task_id: str) -> str:
    response = client.post("/a2a", json={"jsonrpc": "2.0", "id": "get", "method": "tasks/get",
                                         "params": {"id": task_id}})
    return response.json()["result"]["status"]["state"]


def _ended(client: TestClient, task_id: str, state: str = "completed") -> None:
    deadline = time.monotonic() + 10
    while _state(client, task_id) not in ("completed", "failed", "canceled"):
        assert time.monotonic() < deadline, f"task {task_id} didn't end"
        time.sleep(0.01)
    assert _state(client, task_id) == state


def _read(client: TestClient, **payload: Any) -> Any:
    return client.post("/ext/trajectory", json=payload)


def _atif(client: TestClient, **payload: Any) -> dict[str, Any]:
    response = _read(client, format=FORMAT, **payload)
    assert response.status_code == 200, response.text
    return response.json()


def _grant() -> dict[str, Any]:
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {"media_type": "application/json", "max_bytes": 1_000_000,
            "write": {"kind": "http-put", "url": "https://objects.example.test/write", "expires_at": expires}}


def _rebuild(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """A document from cursor pages: the last page's header, every page's steps, and each sub-agent's latest copy."""
    document = dict(pages[-1]["header"])
    document["steps"] = [step for page in pages for step in page["events"]]
    subagents: dict[str, Any] = {}
    for page in pages:
        subagents.update({sub["trajectory_id"]: sub for sub in page["subagents"]})
    if subagents:
        document["subagent_trajectories"] = list(subagents.values())
    return document


def test_the_card_offers_atif_beside_the_native_format() -> None:
    card = _toy().create_app().state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True)
    (trajectory,) = [e for e in card["capabilities"]["extensions"] if e["uri"] == TRAJECTORY_V1.uri]
    assert {key: trajectory["params"][key] for key in ("formats", "native", "default")} == {
        "formats": [NATIVE, FORMAT], "native": NATIVE, "default": NATIVE}


@pytest.mark.parametrize("arguments, error", [
    ({"native": NATIVE, "atif": Toy.__mro__[1]}, TypeError),
    ({"native": NATIVE, "atif": dict}, TypeError),
    ({"atif": Hooks}, ValueError),
    ({"native": "other-events/1", "atif": Hooks}, ValueError),
])
def test_enable_refuses_an_adapter_that_doesnt_fit(arguments: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        enable(TRAJECTORY_V1, **arguments)


def test_every_read_serves_both_formats_and_names_its_format(monkeypatch: pytest.MonkeyPatch) -> None:
    uploaded: list[bytes] = []

    async def capture(_target: Any, body: bytes) -> Uploaded:
        uploaded.append(body)
        return Uploaded(size_bytes=len(body))

    monkeypatch.setattr(framework, "upload", capture)
    native_bodies = []
    for agent in (_toy(atif=None), _toy()):
        with TestClient(agent.create_app()) as client:
            task_id = _send(client, "work t1", "m-1")
            _ended(client, task_id)
            bodies = [_read(client, **read).json() for read in (
                {"task_id": task_id}, {"context_id": "ctx-1"}, {"task_id": task_id, "after": 0},
                {"context_id": "ctx-1", "after": 0}, {"task_id": task_id, "objects": {"trajectory": _grant()}})]
            native_bodies.append(json.loads(json.dumps(bodies).replace(task_id, "TASK")))
    with TestClient(_toy().create_app()) as client:
        task_id = _send(client, "work t1", "m-1")
        _ended(client, task_id)
        task = _atif(client, task_id=task_id)
        context = _atif(client, context_id="ctx-1")
        cursor = _atif(client, task_id=task_id, after=0)
        context_cursor = _atif(client, context_id="ctx-1", after=0)
        objects = _atif(client, task_id=task_id, objects={"trajectory": _grant()})

    # A read naming no format keeps today's body exactly, adapter or not.
    assert native_bodies[0] == native_bodies[1]
    for answer in (task, context, cursor, context_cursor, objects):
        assert (answer["format"], answer["schema_version"], answer["profile_version"]) == (
            FORMAT, SCHEMA_VERSION, PROFILE_VERSION)
    document = task["trajectory"]
    assert errors_in(validate(document)) == []
    assert context["trajectory"] == document
    assert json.loads(uploaded[-1]) == document
    assert [step["step_id"] for step in cursor["events"]] == [1, 2, 3]
    assert cursor["next"] == 3 and cursor["has_more"] is False and cursor["state"] == "completed"
    # The task's cost is on no step, only in final_metrics, so the header has to carry it.
    assert context_cursor["header"]["final_metrics"]["total_cost_usd"] == 0.25
    assert _rebuild([context_cursor]) == document
    assert document["session_id"] == "ctx-1" and document["trajectory_id"]
    assert document["agent"] == {"name": "toy-agent", "version": "2.0"}
    user = document["steps"][0]
    assert user["message"] == "work t1"
    assert user["extra"]["agentenv"] == {"context_id": "ctx-1", "task_id": task_id, "message_id": "m-1", "turn": 0}
    assert document["steps"][-1]["extra"]["agentenv"]["state"] == "completed"


@pytest.mark.parametrize("scenario, state", [
    ("work", "completed"), ("failed", "failed"), ("raised", "failed"), ("canceled", "canceled")])
def test_the_copy_validates_on_every_terminal_path(scenario: str, state: str) -> None:
    hold = _Hold()
    with TestClient(_toy(hold).create_app()) as client:
        task_id = _send(client, f"{scenario} t1", "m-1")
        if scenario == "canceled":
            hold.reached()
            client.post("/a2a", json={"jsonrpc": "2.0", "id": "c", "method": "tasks/cancel",
                                      "params": {"id": task_id}})
        _ended(client, task_id, state)
        document = _atif(client, task_id=task_id)["trajectory"]
    assert errors_in(validate(document)) == []
    closing = document["steps"][-1]["extra"]["agentenv"]
    assert closing["state"] == state
    if scenario == "failed":
        assert closing["error"] == {"kind": "agent_error", "message": "ls found nothing to fix", "code": "toy.failed"}
    if scenario == "raised":
        assert closing["error"]["kind"] == "infra_error"
        assert closing["error"]["code"] == "framework.unhandled_exception"


def test_two_tasks_in_one_context_make_one_document() -> None:
    with TestClient(_toy().create_app()) as client:
        first = _send(client, "work t1", "m-1")
        _ended(client, first)
        second = _send(client, "work t2", "m-2")
        _ended(client, second)
        through_first = _atif(client, task_id=first)["trajectory"]
        through_second = _atif(client, task_id=second)["trajectory"]
    assert errors_in(validate(through_second)) == []
    users = [step for step in through_second["steps"] if step["source"] == "user"]
    assert [user["extra"]["agentenv"]["turn"] for user in users] == [0, 1]
    assert users[1]["extra"]["agentenv"]["task_id"] == second
    assert [step["step_id"] for step in through_second["steps"]] == list(range(1, len(through_second["steps"]) + 1))
    # The first task's read is the document as it stood then: a prefix of the second's.
    assert through_second["steps"][:len(through_first["steps"])] == through_first["steps"]
    assert through_first["final_metrics"]["total_steps"] == len(through_first["steps"])
    assert through_first["trajectory_id"] == through_second["trajectory_id"]


def test_a_step_waiting_on_its_result_holds_back_the_steps_after_it() -> None:
    hold = _Hold()
    pages = []
    with TestClient(_toy(hold).create_app()) as client:
        task_id = _send(client, "late t1", "m-1")
        hold.reached(1)
        pages.append(_atif(client, context_id="ctx-1")["trajectory"])
        waiting = _atif(client, task_id=task_id, after=0)
        hold.resume.set()
        hold.reached(2)
        pages.append(_atif(client, context_id="ctx-1")["trajectory"])
        hold.resume.set()
        _ended(client, task_id)
        final = _atif(client, task_id=task_id)["trajectory"]
    # The call's step is unanswered, so only the user step is served, and the steps after it wait.
    assert [step["source"] for step in pages[0]["steps"]] == ["user"]
    assert [step["step_id"] for step in waiting["events"]] == [1] and waiting["state"] == "running"
    assert len(pages[1]["steps"]) == 4
    assert errors_in(validate_live(pages, final)) == []


def test_aux_records_reach_the_adapter_and_replay_gives_the_served_document(monkeypatch: pytest.MonkeyPatch) -> None:
    uploaded: dict[str, bytes] = {}

    async def capture(target: Any, body: bytes) -> Uploaded:
        uploaded[target.write.url.rsplit("/", 1)[-1]] = body
        return Uploaded(size_bytes=len(body))

    monkeypatch.setattr(framework, "upload", capture)
    grants = {name: {**_grant(), "write": {**_grant()["write"], "url": f"https://objects.example.test/{name}"}}
              for name in ("trajectory", "aux", "turn_context")}
    with TestClient(_toy().create_app()) as client:
        tasks = []
        for tag in ("t1", "t2"):
            tasks.append(_send(client, f"hooks {tag}", f"m-{tag}"))
            _ended(client, tasks[-1])
        final = _atif(client, task_id=tasks[1])
        natives = [_read(client, task_id=task).json()["trajectory"] for task in tasks]
        served = [_atif(client, task_id=task) for task in tasks]
        objects = _atif(client, task_id=tasks[1], objects=grants)
        refused = _read(client, task_id=tasks[1], objects=grants)
    document = final["trajectory"]
    assert [step["message"] for step in document["steps"] if step["message"].endswith("allowed ls")] == [
        "the pre-tool hook allowed ls"] * 2
    assert final["aux"][0] == {"name": "hooks", "record": {"type": "note", "step": "t2-h",
                                                          "text": "the pre-tool hook allowed ls"}, "after": 1}
    turn = final["turn_context"]
    assert (turn["index"], turn["task_id"], turn["state"], turn["final"]) == (1, tasks[1], "completed",
                                                                              ["t2: two files"])
    replayed = replay(Hooks, [RecordedTask.from_served(answer["turn_context"], native, answer["aux"])
                              for answer, native in zip(served, natives)], trajectory_id=document["trajectory_id"])
    assert replayed == document
    assert set(objects["objects"]) == {"trajectory", "aux", "turn_context"}
    assert json.loads(uploaded["aux"]) == final["aux"]
    assert json.loads(uploaded["turn_context"]) == turn
    assert json.loads(uploaded["trajectory"]) == document
    # The aux log and turn context ride an atif/1 task read only.
    assert refused.status_code == 400


def test_an_atif_context_read_goes_to_the_sdk_even_when_the_agent_binds_its_own() -> None:
    with TestClient(_toy(bind_context=True).create_app()) as client:
        task_id = _send(client, "work t1", "m-1")
        _ended(client, task_id)
        own = _read(client, context_id="ctx-1").json()
        atif = _atif(client, context_id="ctx-1")
    assert own == {"trajectory": ["the agent's own shape"], "format": "toy-session/1"}
    assert atif["format"] == FORMAT and errors_in(validate(atif["trajectory"])) == []


def test_data_parts_are_written_as_the_conversation_or_the_profile() -> None:
    image = base64.b64encode(b"\x89PNG fake").decode()
    with TestClient(_toy().create_app()) as client:
        task_id = _send(client, "data t1", "m-1", extra_parts=[
            {"kind": "data", "data": {"ticket": 42, "priority": "high"}},
            {"kind": "file", "file": {"name": "shot.png", "mimeType": "image/png", "bytes": image}},
            {"kind": "file", "file": {"name": "log.txt", "mimeType": "text/plain",
                                      "uri": "https://objects.example.test/log.txt?X-Signature=secret"}}])
        _ended(client, task_id)
        document = _atif(client, task_id=task_id)["trajectory"]
    assert errors_in(validate(document)) == []
    user = document["steps"][0]
    texts = [part["text"] for part in user["message"] if part["type"] == "text"]
    assert texts[:2] == ["data t1", '{"ticket":42,"priority":"high"}']
    assert "secret" not in json.dumps(user) and texts[2] == "[file: log.txt, text/plain, https://objects.example.test/log.txt]"
    assert [part["type"] for part in user["message"]] == ["text", "text", "image", "text"]
    last = document["steps"][-1]
    assert last["message"] == 't1: two files\n\n{"files":["a.txt","b.txt"]}'
    assert last["extra"]["agentenv"]["data"] == [{"data": {"tool_call_count": 1, "cli_attempts": 2}}]
    assert "usage" not in json.dumps(last["extra"])


def test_an_adapter_agent_that_appends_and_returns_fails_the_turn() -> None:
    with TestClient(_toy().create_app()) as client:
        task_id = _send(client, "both t1", "m-1")
        _ended(client, task_id, "failed")
        answer = client.post("/a2a", json={"jsonrpc": "2.0", "id": "g", "method": "tasks/get",
                                           "params": {"id": task_id}}).json()["result"]
        document = _atif(client, task_id=task_id)["trajectory"]
    assert "framework.trajectory_appended_and_returned" in json.dumps(answer["status"])
    assert document["steps"][-1]["extra"]["agentenv"]["error"]["code"] == "framework.trajectory_appended_and_returned"


def test_a_returned_native_trajectory_runs_through_the_adapter_when_the_task_ends() -> None:
    with TestClient(_toy(live=False).create_app()) as client:
        task_id = _send(client, "returned t1", "m-1")
        _ended(client, task_id)
        document = _atif(client, task_id=task_id)["trajectory"]
        native = _read(client, task_id=task_id).json()
    assert native == {"trajectory": _work("t1"), "format": NATIVE}
    assert errors_in(validate(document)) == []
    assert [step["message"] for step in document["steps"]] == ["returned t1", "t1: listing", "t1: two files"]


def test_an_adapter_error_stops_the_copy_not_the_turn() -> None:
    with TestClient(_toy().create_app()) as client:
        task_id = _send(client, "broken t1", "m-1")
        _ended(client, task_id)
        native = _read(client, task_id=task_id)
        refused = _read(client, task_id=task_id, format=FORMAT)
        later = _send(client, "work t2", "m-2")
        _ended(client, later)
    assert native.status_code == 200 and len(native.json()["trajectory"]) == 3
    assert refused.status_code == 409
    error = refused.json()["error"]
    assert error["code"] == "atif_unavailable" and "no-such-call" in error["message"]


def test_cursor_pages_carry_the_header_and_their_tasks_sub_agents() -> None:
    with TestClient(_toy().create_app()) as client:
        task_id = _send(client, "spawn t1", "m-1")
        _ended(client, task_id)
        document = _atif(client, task_id=task_id)["trajectory"]
        pages = []
        after = 0
        while True:
            page = _atif(client, context_id="ctx-1", after=after, limit=1)
            pages.append(page)
            after = page["next"]
            if not page["has_more"]:
                break
    assert errors_in(validate(document)) == []
    assert [len(page["events"]) for page in pages] == [1] * len(document["steps"])
    spawning = next(page for page in pages if any(
        "subagent_trajectory_ref" in result for result in page["events"][0].get("observation", {}).get("results", [])))
    assert [sub["trajectory_id"] for sub in spawning["subagents"]] == ["t1-sub"]
    assert _rebuild(pages) == document


def test_a_reader_following_the_cursor_gets_the_steps_a_sub_agent_writes_after_its_result() -> None:
    hold = _Hold()
    pages = []
    with TestClient(_toy(hold).create_app()) as client:
        task_id = _send(client, "grow t1", "m-1")
        hold.reached()
        while not pages or pages[-1]["has_more"]:
            pages.append(_atif(client, context_id="ctx-1", after=pages[-1]["next"] if pages else 0))
        early = [sub["steps"] for page in pages for sub in page["subagents"]]
        hold.resume.set()
        _ended(client, task_id)
        while pages[-1]["has_more"] or pages[-1]["state"] != "completed":
            pages.append(_atif(client, context_id="ctx-1", after=pages[-1]["next"]))
        document = _atif(client, task_id=task_id)["trajectory"]
    # While the task ran the sub-agent had one closed step; the page that served the task's last step had all three.
    assert [len(steps) for steps in early][-1] == 1
    assert [len(sub["steps"]) for sub in pages[-1]["subagents"]] == [3]
    assert _rebuild(pages) == document


def test_a_spawn_its_task_ended_without_answering_keeps_its_sub_agent_referenced() -> None:
    hold = _Hold()
    with TestClient(_toy(hold).create_app()) as client:
        task_id = _send(client, "orphan t1", "m-1")
        hold.reached()
        client.post("/a2a", json={"jsonrpc": "2.0", "id": "c", "method": "tasks/cancel", "params": {"id": task_id}})
        _ended(client, task_id, "canceled")
        document = _atif(client, task_id=task_id)["trajectory"]
    assert errors_in(validate(document)) == []
    (result,) = document["steps"][1]["observation"]["results"]
    assert result["subagent_trajectory_ref"] == [{"trajectory_id": "t1-sub"}]
    assert result["extra"]["agentenv"] == {"status": "canceled"}


def test_an_unknown_or_unfinished_read_is_not_found() -> None:
    hold = _Hold()
    with TestClient(_toy(hold).create_app()) as client:
        task_id = _send(client, "canceled t1", "m-1")
        hold.reached()
        unfinished = _read(client, task_id=task_id, format=FORMAT)
        unknown = _read(client, context_id="nowhere", format=FORMAT)
        past = _read(client, context_id="ctx-1", after=99, format=FORMAT)
        hold.resume.set()
        _ended(client, task_id)
    assert (unfinished.status_code, unknown.status_code, past.status_code) == (404, 404, 400)
