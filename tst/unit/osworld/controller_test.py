"""The controller over HTTP: a call's answer, an HTTP error as ControllerCallError with its status, /evaluate started
and polled, the blocking /evaluate on a controller without the async routes, a score as a float, and a task ref signed
only when it's an object under the object store's artifacts."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from agent_env.osworld import controller
from tst.unit.osworld.helpers import CONTROLLER, Controller, route_httpx


@pytest.fixture
def ctl(monkeypatch):
    stand_in = Controller()
    route_httpx(monkeypatch, controller, stand_in)
    monkeypatch.setattr(controller.asyncio, "sleep", AsyncMock())
    return stand_in


@pytest.mark.asyncio
async def test_a_call_returns_its_answer_and_an_http_error_raises_with_its_status(ctl):
    ctl.answers["/setup"] = {"success": True}
    assert await controller.post(CONTROLLER, "/setup", {"config": []}) == {"success": True}
    ctl.answers["/setup"] = httpx.Response(500, text="boom")
    with pytest.raises(controller.ControllerCallError) as raised:
        await controller.post(CONTROLLER, "/setup", {})
    assert raised.value.status == 500 and raised.value.body == "boom"


@pytest.mark.asyncio
async def test_a_bare_answer_is_empty_when_json_isnt_expected(ctl):
    ctl.answers["/reset"] = httpx.Response(200, text="ok")
    assert await controller.post(CONTROLLER, "/reset", {}, expect_json=False) == {}
    with pytest.raises(controller.ControllerCallError, match="non-JSON"):
        await controller.post(CONTROLLER, "/reset", {})


@pytest.mark.asyncio
async def test_a_transport_failure_raises(ctl):
    ctl.answers["/setup"] = httpx.ConnectError("refused")
    with pytest.raises(controller.ControllerCallError, match="transport failure"):
        await controller.post(CONTROLLER, "/setup", {})


@pytest.mark.asyncio
async def test_evaluate_is_started_and_polled_until_done(ctl):
    ctl.answers["/evaluate_async"] = {"job_id": "j1"}
    ctl.answers["/evaluate_result"] = [httpx.ConnectError("blip"), {"status": "running"},
                                       httpx.Response(200, text="<html>"), {"status": "done", "result": {"score": 1}}]
    assert await controller.evaluate(CONTROLLER, {"func": "compare_table"}) == {"score": 1}
    assert ctl.bodies("/evaluate_async") == [{"func": "compare_table"}]
    assert {r.url.params["job_id"] for r in ctl.seen if r.url.path == "/evaluate_result"} == {"j1"}


@pytest.mark.asyncio
async def test_evaluate_falls_back_to_the_blocking_route(ctl):
    ctl.answers["/evaluate_async"] = httpx.Response(404)
    ctl.answers["/evaluate"] = {"score": 0.5}
    assert await controller.evaluate(CONTROLLER, {"func": "x"}) == {"score": 0.5}
    assert ctl.paths() == ["/evaluate_async", "/evaluate"]


@pytest.mark.asyncio
async def test_a_failed_job_raises(ctl):
    ctl.answers["/evaluate_async"] = {"job_id": "j1"}
    ctl.answers["/evaluate_result"] = {"status": "error", "message": "judge died"}
    with pytest.raises(controller.ControllerCallError, match="judge died"):
        await controller.evaluate(CONTROLLER, {})


def test_scores_normalize_to_floats():
    assert [controller.normalize_score(s) for s in (True, False, 1, 0.25, {"a": 1, "b": 0, "c": "x"}, {}, None)] == \
        [1.0, 0.0, 1.0, 0.25, 0.5, 0.0, 0.0]


class _Store:
    def __init__(self, signs=True):
        self.signs = signs

    def owns(self, url):
        return url.startswith("s3://bucket/")

    def get_object_key(self, url):
        return url.removeprefix("s3://bucket/")

    def signed_get_url(self, url, expires_in):
        return f"https://signed.example/{self.get_object_key(url)}?ttl={expires_in}" if self.signs else None


def _config(monkeypatch, store):
    monkeypatch.setattr(controller, "get_config", lambda: SimpleNamespace(
        get_object_store_at=lambda url: store, get_artifact_key_prefix=lambda: "pre/"))


def test_a_task_ref_is_signed_only_under_the_artifacts(monkeypatch):
    _config(monkeypatch, _Store())
    assert controller.resolve_task_ref("https://host/task.py") == "https://host/task.py"
    assert controller.resolve_task_ref("s3://bucket/pre/artifacts/t.py") == \
        f"https://signed.example/pre/artifacts/t.py?ttl={controller.TASK_REF_TTL_S}"
    for outside in ("s3://bucket/pre/other/t.py", "s3://elsewhere/pre/artifacts/t.py"):
        with pytest.raises(ValueError, match="refusing to sign"):
            controller.resolve_task_ref(outside)


def test_a_store_that_cant_sign_is_refused(monkeypatch):
    _config(monkeypatch, _Store(signs=False))
    with pytest.raises(ValueError, match="can't sign"):
        controller.resolve_task_ref("s3://bucket/pre/artifacts/t.py")
