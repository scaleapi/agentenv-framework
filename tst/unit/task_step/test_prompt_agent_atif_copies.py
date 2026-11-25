"""A prompt step stores each turn's atif/1 copy when the run prescribes atif/1: read inline from the agent once the turn
ends, stored as ``<trajectory_id>.json`` in the run's folder for the agent with its aux log, turn context and media,
checked, and recorded per turn beside the native copy, off the turn's path. Against a live SDK agent served in-process,
whose uploads land in a granting local store."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote

import httpx
import pytest
from agentenv_protocol.a2a_agent import (
    TRAJECTORY_V1,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    a2a_agent,
    enable,
)
from agentenv_protocol.atif import FORMAT as ATIF, Adapter, atif_adapter, errors_in, media_paths, native, tasks, validate

from agent_env.config import configure
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.snapshot_utils import atif_copies
from agent_env.task_step.task_steps import prompt_agent as pa
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.unit.task_step.prompt_agent_rig import card_of, deployed, quiet_conversations, route_clients
from tst.util.granting_object_store import GRANT_ORIGIN, GrantingObjectStore
from tst.util.images import png

NATIVE = "shots/1"
#: How long the agent holds a gated turn open for its ``gate`` at most, so a test that never sets it still ends.
_GATE_SECONDS = 30
_PREFIX = "prompt_agent_trajectories/prompt_id=solve/"


@atif_adapter(native_format=NATIVE, version="2.1.0", input_includes_cache=False)
class _Shots(Adapter):
    """Each native shot a step: its caption and its screenshot."""

    @native(type="shot")
    def shot(self, record: dict, turn) -> None:
        image = turn.media(base64.b64decode(record["png"]), "image/png")
        turn.message(step=f"s{record['n']}", content=[f"shot {record['n']}", image])
        turn.step(key=f"s{record['n']}", llm_call_count=1)


def _app(*, atif: bool = True, live: bool = False, gate: asyncio.Event | None = None):
    """An agent that, each turn, appends one shot, a new screenshot each time, and answers; with ``gate``, a second
    shot, then a third once ``gate`` is set (at most ``_GATE_SECONDS`` on), so a live reader sees the first closed
    while the turn runs (a step closes when the next begins)."""
    shots = iter(range(1, 1000))

    @a2a_agent(
        identity=AgentIdentity(name="solver", description="test", version="1"),
        extensions=(enable(TRAJECTORY_V1, live=live, native=NATIVE, **({"atif": _Shots} if atif else {})),),
    )
    class Agent(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            request.trajectory.set_format(NATIVE)
            n = next(shots)
            request.trajectory.append({"type": "shot", "n": n, "png": base64.b64encode(png(n, 1)).decode()})
            if gate is not None:
                for held in (False, True):
                    if held:
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(gate.wait(), _GATE_SECONDS)
                    n = next(shots)
                    request.trajectory.append({"type": "shot", "n": n, "png": base64.b64encode(png(n, 1)).decode()})
            return TaskResult.text(f"shot {n}")

    return Agent().create_app()


class _Wire(httpx.AsyncBaseTransport):
    """The agent at ``http://agent.test``, served in-process, and the store's grant URLs, which write into the store.
    A test can hold the agent's atif/1 reads until it lets them go, rewrite their answers, or rewrite any answer."""

    def __init__(self, store: GrantingObjectStore, *, atif: bool, live: bool = False,
                 gate: asyncio.Event | None = None) -> None:
        self.store = store
        self._atif, self._live, self._gate = atif, live, gate
        self.restart()
        self.atif_reads: list[dict] = []
        self.native_reads: list[dict] = []
        self.media_requests: list[dict] = []
        self.prescriptions: list[dict] = []
        self.hold: asyncio.Event | None = None
        #: Set once an atif/1 read has arrived, whether or not it's held.
        self.read_arrived = asyncio.Event()
        self.answered_atif_reads = 0
        self.rewrite_atif: Callable[[dict], dict] | None = None
        self.answer: Callable[[httpx.Request, str, dict, httpx.Response], httpx.Response | None] | None = None
        #: Paths the agent never receives a request for: each is answered 503.
        self.unreachable: set[str] = set()

    def restart(self) -> None:
        self.app = _app(atif=self._atif, live=self._live, gate=self._gate)
        self._agent = httpx.ASGITransport(app=self.app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        if request.url.host == httpx.URL(GRANT_ORIGIN).host:
            self.store.put(unquote(request.url.path.lstrip("/")), body, content_type=request.headers["content-type"],
                           allow_overwrite=True)
            return httpx.Response(200, request=request)
        sent = json.loads(body) if body and request.method == "POST" else {}
        path = request.url.path
        if path in self.unreachable:
            return httpx.Response(503, request=request)
        atif_read = path == "/ext/trajectory" and sent.get("format") == ATIF and "after" not in sent
        if path == "/ext/trajectory" and not atif_read and "after" not in sent:
            self.native_reads.append(sent)
        if atif_read:
            self.atif_reads.append(sent)
            self.read_arrived.set()
            if self.hold is not None:
                await self.hold.wait()
        elif path == "/ext/trajectory/media":
            self.media_requests.append(sent)
        elif path == "/ext/trajectory/prescribe":
            self.prescriptions.append(sent)
        response = await self._agent.handle_async_request(request)
        self.answered_atif_reads += atif_read
        if atif_read and self.rewrite_atif is not None and response.status_code == 200:
            await response.aread()
            response = httpx.Response(200, json=self.rewrite_atif(json.loads(response.content)), request=request)
        if self.answer is not None:
            await response.aread()
            response = self.answer(request, path, sent, response) or response
        return response


class _Run:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, atif: bool = True, live: bool = False,
                 gate: asyncio.Event | None = None) -> None:
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        self.root = tmp_path / "store"
        self.store = GrantingObjectStore(str(self.root), reaches=True)
        configure(object_store=self.store)
        quiet_conversations(monkeypatch)
        self.wire = _Wire(self.store, atif=atif, live=live, gate=gate)
        self.asked: list[int] = []
        route_clients(monkeypatch, self.wire)

        async def ask(user_side, number: int, reply: str, parts: list[dict]) -> dict:
            self.asked.append(number)
            return {"message": [{"kind": "text", "text": f"again {number}"}]}

        monkeypatch.setattr(pa._UserSide, "ask", ask)

    @property
    def card(self) -> dict:
        return card_of(self.wire.app)

    def context(self, formats: list[str] | None = (ATIF,), **kwargs) -> TaskStepContext:
        context = TaskStepContext(instance_id="ti-1", **kwargs)
        if formats is not None:
            context.metadata["user_overrides"] = {"trajectory_formats": list(formats)}
        return context

    async def execute(self, context: TaskStepContext, *, card: dict | None = None, **step) -> PromptResponse:
        context.deployed_agents[:] = [deployed(card or self.card)]
        step.setdefault("max_conversation_turns", 2)
        step.setdefault("user_a2a_url", "http://user.test")  # whose answers ``ask`` stands in for
        await PromptAgentTaskStep(id="solve", version=None, prompt="go", agent_name="solver", poll_interval_seconds=0,
                                  trajectory_output_prefix=self.store.object_url(_PREFIX), **step).execute(context)
        return context.prompt_responses[-1]

    def read(self, url: str) -> Any:
        return json.loads(self.store.get(url))

    def folder(self, context: TaskStepContext) -> Path:
        return self.root / self.store.get_object_key(context.metadata["atif_copies"]["agents"]["solver"]["folder"])


def _atifs(response: PromptResponse) -> list[dict | None]:
    return [turn["atif"] for turn in response.target_agent_per_turn_trajectories]


@pytest.mark.asyncio
async def test_a_run_that_prescribes_atif_has_each_turns_copy_as_its_trajectory_and_reads_no_native(
        monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    context = run.context()

    response = await run.execute(context)

    first, second = _atifs(response)
    folder = context.metadata["atif_copies"]["agents"]["solver"]["folder"]
    assert context.metadata["atif_copies"]["run"] == "ti-1"
    assert folder == f"{run.store.object_url('atif/instance_id=ti-1/solver')}/"
    document = run.read(second["url"])
    assert first["url"] == second["url"] == f"{folder}{document['trajectory_id']}.json"
    assert response.target_agent_per_turn_trajectory_object_urls == [first["url"], second["url"]]
    assert response.agent_trajectory_object_url == second["url"]
    assert [turn["format"] for turn in response.target_agent_per_turn_trajectories] == [ATIF, ATIF]
    assert [first["task_id"], second["task_id"]] == [task.task_id for task in tasks(document)]
    assert [first["status"], second["status"]] == ["valid", "valid"]
    assert first["rules"] == second["rules"] == [] and first["reason"] is None
    assert first["adapter_version"] == "2.1.0" and first["validator_version"] == atif_copies._VALIDATOR_VERSION
    assert first["through_step"] < second["through_step"] == document["steps"][-1]["step_id"]
    assert document["session_id"] == response.a2a_context_id
    assert len(media_paths(document)) == 2
    assert errors_in(validate(document, media=run.folder(context))) == []
    for record in (first, second):
        assert record["aux_url"].startswith(f"{folder}turns/") and run.read(record["aux_url"]) == []
        assert run.read(record["turn_context_url"])["state"] == "completed"
    assert [read["format"] for read in run.wire.atif_reads] == [ATIF, ATIF]
    assert run.wire.native_reads == [] and not run.store.list(_PREFIX)


@pytest.mark.asyncio
async def test_a_run_that_prescribes_nothing_reads_and_stores_native_only(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    context = run.context(formats=None)

    response = await run.execute(context)

    assert _atifs(response) == [None, None]
    assert run.wire.atif_reads == [] and run.wire.media_requests == []
    assert "atif_copies" not in context.metadata and not run.store.list("atif/")


@pytest.mark.asyncio
async def test_an_agent_that_doesnt_offer_atif_keeps_its_native_trajectory_as_ever(monkeypatch, tmp_path):
    """Known before the first turn, so the step reads native on each turn's path, as a run that prescribes nothing."""
    run = _Run(monkeypatch, tmp_path, atif=False)

    response = await run.execute(run.context())

    assert [(atif["status"], atif["reason"]) for atif in _atifs(response)] == [
        ("unavailable", "format_not_offered")] * 2
    assert run.wire.atif_reads == [] and len(run.wire.native_reads) == 2
    assert [turn["format"] for turn in response.target_agent_per_turn_trajectories] == [NATIVE, NATIVE]
    assert [run.read(url)[0]["type"] for url in response.target_agent_per_turn_trajectory_object_urls] == ["shot"] * 2


def _natives(run: _Run, response: PromptResponse) -> list[str | None]:
    """The format of what each turn's url holds, read back."""
    return [None if url is None else ("shot" if isinstance(run.read(url), list) else ATIF)
            for url in response.target_agent_per_turn_trajectory_object_urls]


@pytest.mark.asyncio
async def test_a_turn_whose_copy_the_agent_refuses_has_its_native_trajectory_read_in_its_place(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    run.wire.unreachable.add("/ext/trajectory/prescribe")

    response = await run.execute(run.context())

    assert [(atif["status"], atif["reason"]) for atif in _atifs(response)] == [
        ("unavailable", "format_not_prescribed")] * 2
    assert [turn["format"] for turn in response.target_agent_per_turn_trajectories] == [NATIVE, NATIVE]
    assert _natives(run, response) == ["shot", "shot"] and len(run.wire.native_reads) == 2
    assert response.error_type is None


@pytest.mark.asyncio
async def test_one_turns_failed_copy_falls_back_to_native_and_the_others_keep_theirs(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    failed: list[str] = []

    def first_read_fails(request: httpx.Request, path: str, sent: dict,
                         response: httpx.Response) -> httpx.Response | None:
        if path == "/ext/trajectory" and sent.get("format") == ATIF and not failed:
            failed.append(sent["task_id"])
            return httpx.Response(500, request=request)
        return None

    run.wire.answer = first_read_fails

    response = await run.execute(run.context())

    assert [turn["format"] for turn in response.target_agent_per_turn_trajectories] == [NATIVE, ATIF]
    assert [atif["status"] for atif in _atifs(response)] == ["unavailable", "valid"]
    assert _natives(run, response) == ["shot", ATIF] and [read["task_id"] for read in run.wire.native_reads] == failed
    assert response.agent_trajectory_object_url == _atifs(response)[1]["url"]


@pytest.mark.asyncio
async def test_a_turn_whose_copy_and_native_trajectory_both_fail_has_none(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    run.wire.answer = lambda request, path, sent, response: (
        httpx.Response(500, request=request) if path == "/ext/trajectory" else None)

    response = await run.execute(run.context(), max_conversation_turns=1)

    assert response.target_agent_per_turn_trajectory_object_urls == [None]
    assert response.target_agent_per_turn_trajectories == [{"format": None, "atif": _atifs(response)[0]}]
    assert response.agent_trajectory_object_url is None and response.error_type is None


@pytest.mark.asyncio
async def test_a_copy_that_fails_validation_is_stored_and_recorded_invalid_with_its_rules(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)

    def stray_key(answer: dict) -> dict:
        answer["trajectory"]["extra"]["elsewhere"] = {"x": 1}
        return answer

    run.wire.rewrite_atif = stray_key

    response = await run.execute(run.context())

    [first, second] = _atifs(response)
    assert first["status"] == second["status"] == "invalid"
    assert first["rules"] == ["outside-key"] and "outside-key" in first["reason"]
    assert run.read(second["url"])["extra"]["elsewhere"] == {"x": 1}
    assert response.error_type is None


@pytest.mark.asyncio
@pytest.mark.parametrize("field, version", [("schema_version", "ATIF-v1.9"), ("profile", "1.2")])
async def test_a_copy_newer_than_the_worker_is_stored_and_recorded_newer(monkeypatch, tmp_path, field, version):
    run = _Run(monkeypatch, tmp_path)

    def newer(answer: dict) -> dict:
        if field == "schema_version":
            answer["trajectory"]["schema_version"] = version
        else:
            answer["trajectory"]["extra"]["agentenv"]["version"] = version
        return answer

    run.wire.rewrite_atif = newer

    response = await run.execute(run.context())

    assert [atif["status"] for atif in _atifs(response)] == ["newer", "newer"]
    assert run.read(_atifs(response)[-1]["url"])["schema_version"] == (
        version if field == "schema_version" else "ATIF-v1.8")


@pytest.mark.asyncio
async def test_a_newer_copys_missing_media_are_in_its_rules(monkeypatch, tmp_path):
    """ATIF's models stop at a newer schema, and the copy's media are still checked."""
    run = _Run(monkeypatch, tmp_path)
    card = run.card
    trajectory = next(e for e in card["capabilities"]["extensions"] if e["uri"] == TRAJECTORY_V1.uri)
    del trajectory["params"]["methods"]["media"]

    def newer(answer: dict) -> dict:
        answer["trajectory"]["schema_version"] = "ATIF-v1.9"
        return answer

    run.wire.rewrite_atif = newer

    response = await run.execute(run.context(), card=card, max_conversation_turns=1)

    [atif] = _atifs(response)
    assert (atif["status"], atif["rules"]) == ("newer", ["atif", "media-bytes"])
    assert response.target_agent_per_turn_trajectories[0]["format"] == NATIVE


@pytest.mark.asyncio
async def test_a_copy_whose_media_didnt_arrive_is_recorded_and_its_turn_reads_native_in_its_place(monkeypatch, tmp_path):
    """Its judges would read the turn without those files, which the native trajectory holds."""
    run = _Run(monkeypatch, tmp_path)
    response = await run.execute(run.context(), card=_without_media(run.card))

    assert [(atif["status"], atif["rules"]) for atif in _atifs(response)] == [("invalid", ["media-bytes"])] * 2
    assert run.wire.media_requests == []
    assert [turn["format"] for turn in response.target_agent_per_turn_trajectories] == [NATIVE, NATIVE]
    assert _natives(run, response) == ["shot", "shot"] and len(run.wire.native_reads) == 2
    assert all(run.read(atif["url"])["steps"] for atif in _atifs(response))


@pytest.mark.asyncio
async def test_a_copy_whose_media_didnt_arrive_stays_the_trajectory_when_native_cant_be_read(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    run.wire.answer = lambda request, path, sent, response: (
        httpx.Response(500, request=request) if path == "/ext/trajectory" and sent.get("format") != ATIF else None)

    response = await run.execute(run.context(), card=_without_media(run.card), max_conversation_turns=1)

    [atif] = _atifs(response)
    assert (atif["status"], atif["rules"]) == ("invalid", ["media-bytes"])
    assert response.target_agent_per_turn_trajectories[0]["format"] == ATIF
    assert response.target_agent_per_turn_trajectory_object_urls == [atif["url"]]


def _without_media(card: dict) -> dict:
    """The card with no media method, so the worker can't ask for a copy's media."""
    trajectory = next(e for e in card["capabilities"]["extensions"] if e["uri"] == TRAJECTORY_V1.uri)
    del trajectory["params"]["methods"]["media"]
    return card


@pytest.mark.asyncio
async def test_a_file_whose_push_answer_was_lost_is_found_in_the_store_and_not_asked_for_again(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    lost: list[str] = []

    def lose_first_push(request: httpx.Request, path: str, sent: dict,
                        response: httpx.Response) -> httpx.Response | None:
        if path == "/ext/trajectory/media" and not lost:
            lost.extend(sent["objects"])
            raise httpx.ReadTimeout("the answer was lost", request=request)
        return None

    run.wire.answer = lose_first_push
    context = run.context()

    response = await run.execute(context)

    assert [atif["status"] for atif in _atifs(response)] == ["valid", "valid"]
    [first_shot] = lost
    assert [list(request["objects"]) for request in run.wire.media_requests] == [
        [first_shot], [path for path in media_paths(run.read(_atifs(response)[-1]["url"])) if path != first_shot]]

    run.wire.media_requests.clear()
    await run.execute(context, max_conversation_turns=1, context_id=None)
    assert all(first_shot not in request["objects"] for request in run.wire.media_requests)


@pytest.mark.asyncio
async def test_a_restart_between_steps_links_the_sessions_earlier_document_to_the_next(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    context = run.context()
    before = await run.execute(context, context_id="talk", max_conversation_turns=1)
    run.wire.restart()

    after = await run.execute(context, context_id="talk", max_conversation_turns=1)

    earlier, later = run.read(_atifs(before)[0]["url"]), run.read(_atifs(after)[0]["url"])
    assert earlier["session_id"] == later["session_id"] == "talk"
    assert earlier["trajectory_id"] != later["trajectory_id"]
    assert earlier["continued_trajectory_ref"] == f"{later['trajectory_id']}.json"
    assert "continued_trajectory_ref" not in later
    assert errors_in(validate(earlier, media=run.folder(context))) == []
    assert context.metadata["atif_copies"]["agents"]["solver"]["sessions"] == {"talk": later["trajectory_id"]}


@pytest.mark.asyncio
async def test_the_user_sides_next_message_goes_out_before_the_turns_atif_copy_is_read(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    run.wire.hold = asyncio.Event()
    real_ask = pa._UserSide.ask

    async def ask(user_side, number: int, reply: str, parts: list[dict]) -> dict:
        await asyncio.wait_for(run.wire.read_arrived.wait(), 10)
        assert run.wire.answered_atif_reads == 0  # the turn's read is still held, and the user side is asked anyway
        run.wire.hold.set()
        return await real_ask(user_side, number, reply, parts)

    monkeypatch.setattr(pa._UserSide, "ask", ask)

    response = await asyncio.wait_for(run.execute(run.context()), 30)

    assert run.asked == [1] and [atif["status"] for atif in _atifs(response)] == ["valid", "valid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("trajectory_id", ["../elsewhere", "a/b", "", None])
async def test_a_trajectory_id_that_isnt_a_plain_file_name_is_refused(monkeypatch, tmp_path, trajectory_id):
    run = _Run(monkeypatch, tmp_path)

    def renamed(answer: dict) -> dict:
        answer["trajectory"]["trajectory_id"] = trajectory_id
        return answer

    run.wire.rewrite_atif = renamed

    response = await run.execute(run.context(), max_conversation_turns=1)

    assert [(atif["status"], atif["reason"]) for atif in _atifs(response)] == [
        ("unavailable", "invalid_trajectory_id")]
    assert not run.store.list("atif/")


def test_a_run_without_an_instance_id_keeps_one_folder_run_for_all_its_agents(tmp_path):
    configure(object_store=GrantingObjectStore(str(tmp_path)))
    context = TaskStepContext()

    solver, judge = atif_copies.atif_folder(context, "solver"), atif_copies.atif_folder(context, "judge")

    run = context.metadata["atif_copies"]["run"]
    assert run.startswith("adhoc-") and len(run) == len("adhoc-") + 12
    assert solver.endswith(f"atif/instance_id={run}/solver/") and judge.endswith(f"atif/instance_id={run}/judge/")
    assert atif_copies.atif_folder(context, "solver") == solver


@pytest.mark.asyncio
async def test_a_link_that_fails_is_made_by_a_later_turn_and_its_turn_keeps_its_copy(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)
    context = run.context()
    before = await run.execute(context, context_id="talk", max_conversation_turns=1)
    earlier_url = _atifs(before)[0]["url"]
    run.wire.restart()
    real_read, failed = run.store.read, []

    def read_failing_once(key: str) -> bytes:
        if key.endswith(earlier_url.rsplit("/", 1)[1]) and not failed:
            failed.append(key)
            raise OSError("the store blinked")
        return real_read(key)

    monkeypatch.setattr(run.store, "read", read_failing_once)

    after = await run.execute(context, context_id="talk", max_conversation_turns=2)

    assert failed and [atif["status"] for atif in _atifs(after)] == ["valid", "valid"]
    later = run.read(_atifs(after)[-1]["url"])
    assert run.read(earlier_url)["continued_trajectory_ref"] == f"{later['trajectory_id']}.json"


@pytest.mark.asyncio
async def test_a_copy_that_doesnt_read_as_atif_is_stored_and_recorded_invalid(monkeypatch, tmp_path):
    run = _Run(monkeypatch, tmp_path)

    def broken(answer: dict) -> dict:
        answer["trajectory"]["steps"] = [None]
        return answer

    run.wire.rewrite_atif = broken

    response = await run.execute(run.context(), max_conversation_turns=1)

    [atif] = _atifs(response)
    assert (atif["status"], atif["rules"], atif["through_step"]) == ("invalid", ["atif"], None)
    assert run.read(atif["url"])["steps"] == [None] and run.wire.media_requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["bytes", "sha256"])
async def test_a_profile_that_misdescribes_a_stored_file_makes_the_copy_invalid(monkeypatch, tmp_path, field):
    run = _Run(monkeypatch, tmp_path)

    def misdescribed(answer: dict) -> dict:
        for step in answer["trajectory"]["steps"]:
            for info in (step.get("extra") or {}).get("agentenv", {}).get("media", {}).values():
                info[field] = info[field] + 1 if field == "bytes" else "0" * 64
        return answer

    run.wire.rewrite_atif = misdescribed

    response = await run.execute(run.context(), max_conversation_turns=1)

    [atif] = _atifs(response)
    assert (atif["status"], atif["rules"]) == ("invalid", ["media-bytes"])
    assert ("bytes" if field == "bytes" else "sha256") in atif["reason"]
    assert _natives(run, response) == ["shot"]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["x" * 5000, "Not A Code", 7])
async def test_an_error_code_the_sdk_wouldnt_send_isnt_kept_as_the_reason(monkeypatch, tmp_path, code):
    run = _Run(monkeypatch, tmp_path)
    run.wire.answer = lambda request, path, sent, response: (
        httpx.Response(409, json={"error": {"code": code}}, request=request)
        if path == "/ext/trajectory" and sent.get("format") == ATIF else None)

    response = await run.execute(run.context(), max_conversation_turns=1)

    assert [(atif["status"], atif["reason"]) for atif in _atifs(response)] == [("unavailable", "http_409")]


@pytest.mark.asyncio
async def test_post_turn_hooks_read_each_turns_native_trajectory_while_the_runs_copies_wait(monkeypatch, tmp_path):
    from tst.unit.task_step.test_prompt_agent_post_turn_hooks import _hook
    run = _Run(monkeypatch, tmp_path)
    context = run.context()
    hook = _hook("reads", """
        async def post_turn(turn, params):
            texts = [await turn.read_trajectory(n) for n in range(1, len(turn.trajectory_urls) + 1)]
            turn.record({"urls": list(turn.trajectory_urls), "read": [t is not None for t in texts]})
            return None
        """)

    response = await run.execute(context, post_turn_hooks=[{"file_artifact_id": hook, "params": {}}])

    records = context.metadata["post_turn_hooks"]["solve"]
    assert [r["read"] for r in records] == [[True], [True, True]]  # this turn's work, while the conversation runs
    assert all(url is not None and isinstance(run.read(url), list) for url in records[-1]["urls"])  # native shots
    # The step's own record is still the run's copies.
    assert [turn["format"] for turn in response.target_agent_per_turn_trajectories] == [ATIF, ATIF]
    assert response.target_agent_per_turn_trajectory_object_urls == [t["atif"]["url"] for t in
                                                                     response.target_agent_per_turn_trajectories]
