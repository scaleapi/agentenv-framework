"""An agent builds the trajectory formats beside native that the run prescribes (``TRAJECTORY_V1.prescribe``), and only
those: with none prescribed it builds no atif/1 copy, spools no media and counts no turns, and its atif/1 reads are
refused. Each conversation keeps the formats it started with, so no copy misses its first turns."""
from __future__ import annotations

import base64

import pytest
from starlette.testclient import TestClient
from test_a2a_atif_media import SHOT, Shots, _file, _path, _shooting
from test_a2a_atif_serving import _ended, _read, _send, prescribe

from agentenv_protocol.a2a_agent import TextPart, TrajectoryState
from agentenv_protocol.a2a_agent import _turn_index
from agentenv_protocol.a2a_agent._atif_copy import TurnStart
from agentenv_protocol.a2a_agent._trajectory_log import NotPrescribed, TaskTrajectories
from agentenv_protocol.atif import FORMAT

NATIVE = "shots/1"


def _services(client: TestClient):
    return client.app.state.agentenv_a2a.services


def _atif_reads(client: TestClient, task_id: str, context_id: str) -> list:
    """Every atif/1 read of a task and its context: the task read, the context read and both cursor pages."""
    return [_read(client, format=FORMAT, task_id=task_id), _read(client, format=FORMAT, context_id=context_id),
            _read(client, format=FORMAT, task_id=task_id, after=0), _read(client, format=FORMAT, context_id=context_id,
                                                                           after=0)]


def test_an_agent_prescribed_nothing_builds_only_native_and_refuses_atif_reads() -> None:
    with TestClient(_shooting().create_app()) as client:
        task_id = _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png")])
        _ended(client, task_id)
        reads = _atif_reads(client, task_id, "ctx-1")
        native = _read(client, task_id=task_id).json()
        services = _services(client)
        assert services.task_trajectories._copies == {} and services.task_trajectories.media.folder is None
    assert [read.status_code for read in reads] == [409] * 4
    assert {read.json()["error"]["code"] for read in reads} == {"format_not_prescribed"}
    assert native["format"] == NATIVE and native["trajectory"][0]["type"] == "shot"
    assert not (_turn_index.turns_folder()).exists() or not any(_turn_index.turns_folder().iterdir())


def test_a_conversation_keeps_the_formats_it_started_with_and_a_later_call_sets_the_next_ones() -> None:
    with TestClient(_shooting().create_app()) as client:
        before = _send(client, "shoot", "m-1", context_id="before")
        _ended(client, before)
        assert prescribe(client, FORMAT).json() == {"formats": [NATIVE, FORMAT]}
        again = _send(client, "again", "m-2", context_id="before")
        after = _send(client, "shoot", "m-3", context_id="after", extra_parts=[_file(SHOT, "image/png")])
        _ended(client, again)
        _ended(client, after)
        assert prescribe(client).json() == {"formats": [NATIVE]}
        later = _send(client, "shoot", "m-4", context_id="later")
        _ended(client, later)
        statuses = {name: _read(client, format=FORMAT, task_id=task).status_code
                    for name, task in (("before", again), ("after", after), ("later", later))}
        held = _services(client).task_trajectories.media.held("after", _path(SHOT, ".png"))
    assert statuses == {"before": 409, "after": 200, "later": 409}
    assert held is not None


def test_a_prescription_names_only_formats_the_card_offers_and_may_name_native() -> None:
    with TestClient(_shooting().create_app()) as client:
        refused = prescribe(client, "elsewhere/2")
        native_named = prescribe(client, NATIVE, FORMAT).json()
    with TestClient(_shooting(atif=False).create_app()) as client:
        unlisted = prescribe(client, FORMAT)
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "format_not_offered"
    assert native_named == {"formats": [NATIVE, FORMAT]}
    assert unlisted.status_code == 404


def test_a_read_that_names_no_format_gets_native_whatever_is_prescribed() -> None:
    with TestClient(_shooting().create_app()) as client:
        prescribe(client, FORMAT)
        task_id = _send(client, "shoot", "m-1", extra_parts=[_file(SHOT, "image/png")])
        _ended(client, task_id)
        unnamed = _read(client, task_id=task_id).json()
    assert unnamed["format"] == NATIVE


def test_a_conversation_restored_over_takes_the_prescription_in_effect_at_its_next_task() -> None:
    trajectories = TaskTrajectories(native=NATIVE, atif=Shots)

    def play(task_id: str) -> None:
        log = trajectories.register(task_id, "ctx")
        trajectories.start(task_id, TurnStart(message_id=None, parts=(TextPart(text="hi"),), agent_name="toy",
                                              agent_version="1", agent_model=None))
        log.append({"type": "shot", "step": "s", "data": base64.b64encode(SHOT).decode(), "media_type": "image/png"})
        trajectories.seal(task_id, TrajectoryState.COMPLETED, final=(TextPart(text="done"),))

    play("t1")
    with pytest.raises(NotPrescribed):
        trajectories.atif_task("t1")
    trajectories.prescribe([FORMAT])
    trajectories.restored("ctx", 1)
    play("t2")
    assert trajectories.atif_task("t2") is not None
    trajectories.media.close()


def test_a_task_says_whether_it_feeds_an_atif_copy() -> None:
    trajectories = TaskTrajectories(native=NATIVE, atif=Shots)
    turn = TurnStart(message_id=None, parts=(TextPart(text="hi"),), agent_name="toy", agent_version="1",
                     agent_model=None)
    unprescribed = trajectories.register("t1", "before")
    trajectories.start("t1", turn)
    trajectories.prescribe([FORMAT])
    prescribed = trajectories.register("t2", "after")
    waiting = prescribed.builds_atif
    trajectories.start("t2", turn)
    assert (unprescribed.builds_atif, waiting, prescribed.builds_atif) == (False, False, True)
    assert TaskTrajectories(native=NATIVE).register("t3", "ctx").builds_atif is False


def test_a_task_says_whether_it_opens_a_new_document_of_its_conversation() -> None:
    trajectories = TaskTrajectories(native=NATIVE, atif=Shots)
    trajectories.prescribe([FORMAT])
    turn = TurnStart(message_id=None, parts=(TextPart(text="hi"),), agent_name="toy", agent_version="1",
                     agent_model=None)
    opened = []
    for task_id in ("t1", "t2", "t3"):
        log = trajectories.register(task_id, "ctx")
        trajectories.start(task_id, turn)
        opened.append(log.starts_document)
        trajectories.seal(task_id, TrajectoryState.COMPLETED, final=(TextPart(text="done"),))
        if task_id == "t2":
            trajectories.restored("ctx", 2)
    assert opened == [True, False, True]
