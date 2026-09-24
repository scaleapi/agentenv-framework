"""Unit tests for RubricsVerifierTaskStep._fetch_judge_trajectory."""

import httpx
import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep


def _verifier() -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(
        id="verify",
        version=1,
        criteria=[{"id": "c1", "title": "Test c1"}],
        prompt_id="ask",
        use_agent_judge=True,
        verifier_id="verifier-test",
    )


def _card_with_trajectory_ext() -> dict:
    return {"capabilities": {"extensions": [
        {"uri": A2AAgent.EXT_TRAJECTORY, "params": {"endpoint": "/ext/trajectory"}},
    ]}}


@pytest.mark.asyncio
async def test_no_extension_on_card_skips_the_fetch_entirely(monkeypatch):
    verifier = _verifier()

    async def boom(self, method, url, **kwargs):
        raise AssertionError("should never make a request without the extension")

    monkeypatch.setattr(httpx.AsyncClient, "request", boom)

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example", judge_agent_card={}, a2a_server_task_id="t1",
    )
    assert uri is None


@pytest.mark.asyncio
async def test_extension_without_an_endpoint_skips_the_fetch_without_guessing_one(monkeypatch):
    verifier = _verifier()

    async def boom(self, method, url, **kwargs):
        raise AssertionError("should never make a request without a declared endpoint")

    monkeypatch.setattr(httpx.AsyncClient, "request", boom)

    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_TRAJECTORY, "params": {}}]}}
    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example", judge_agent_card=card, a2a_server_task_id="t1",
    )
    assert uri is None


@pytest.mark.asyncio
async def test_inline_trajectory_gets_uploaded_and_its_url_returned(monkeypatch):
    verifier = _verifier()
    captured = {}

    async def fake_request(self, method, url, **kwargs):
        assert url == "http://judge.example/ext/trajectory"
        assert kwargs["json"] == {"task_id": "t1"}
        return httpx.Response(
            200, json={"trajectory": [{"span": "x"}], "is_live": False},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    def fake_upload(trajectory, prefix, *, name=None):
        captured.update(trajectory=trajectory, prefix=prefix, name=name)
        return "s3://bucket/judge_trajectories/verifier_id=verifier-test/trajectory-t1.json"

    monkeypatch.setattr(
        "agent_env.task_step.task_steps.verifiers.rubrics_verifier.upload_trajectory", fake_upload,
    )

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_trajectory_ext(),
        a2a_server_task_id="t1",
    )

    assert uri == "s3://bucket/judge_trajectories/verifier_id=verifier-test/trajectory-t1.json"
    assert captured["trajectory"] == [{"span": "x"}]
    assert captured["name"] == "t1"
    assert "/judge_trajectories/verifier_id=verifier-test" in captured["prefix"]


@pytest.mark.asyncio
async def test_server_side_s3_prefix_is_listed_for_the_object_url(monkeypatch):
    verifier = _verifier()

    async def fake_request(self, method, url, **kwargs):
        return httpx.Response(
            200, json={"trajectory_s3_prefix": "s3://bucket/pre/", "is_live": False},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    class FakeStore:
        def list_at(self, prefix):
            assert prefix == "s3://bucket/pre/"
            return ["s3://bucket/pre/trajectory-t1.json"]

    class FakeConfig:
        def get_object_store(self):
            return FakeStore()

    monkeypatch.setattr(
        "agent_env.task_step.task_steps.verifiers.rubrics_verifier.get_config", lambda: FakeConfig(),
    )

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_trajectory_ext(),
        a2a_server_task_id="t1",
    )
    assert uri == "s3://bucket/pre/trajectory-t1.json"


@pytest.mark.asyncio
async def test_a_transport_failure_is_retried_then_swallowed_not_raised(monkeypatch):
    verifier = _verifier()
    attempts = 0

    async def boom(self, method, url, **kwargs):
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx.AsyncClient, "request", boom)

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_trajectory_ext(),
        a2a_server_task_id="t1",
    )
    assert uri is None
    assert attempts == RubricsVerifierTaskStep.DEFAULT_MAX_RETRIES


@pytest.mark.asyncio
async def test_a_transient_failure_is_retried_until_it_succeeds(monkeypatch):
    verifier = _verifier()
    attempts = 0

    async def flaky(self, method, url, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise httpx.ConnectError("no route")
        return httpx.Response(
            200, json={"trajectory": [{"span": "x"}], "is_live": False},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", flaky)
    monkeypatch.setattr(
        "agent_env.task_step.task_steps.verifiers.rubrics_verifier.upload_trajectory",
        lambda trajectory, prefix, *, name=None: "s3://bucket/trajectory.json",
    )

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_trajectory_ext(),
        a2a_server_task_id="t1",
    )
    assert uri == "s3://bucket/trajectory.json"
    assert attempts == 2


@pytest.mark.asyncio
async def test_invoke_judge_a2a_wires_the_captured_trajectory_onto_the_result(monkeypatch):
    verifier = _verifier()

    from agent_env.a2a_agent import protocol

    async def fake_send(*args, **kwargs):
        return "task-1", "ctx-1"

    async def fake_poll(*args, **kwargs):
        return {"status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": "ok"}]}}}

    monkeypatch.setattr(protocol, "send_a2a_message", fake_send)
    monkeypatch.setattr(protocol, "poll_a2a_task", fake_poll)

    async def fake_fetch(self, *, judge_a2a_url, judge_agent_card, a2a_server_task_id):
        assert judge_a2a_url == "http://judge.example"
        assert a2a_server_task_id == "task-1"
        return "s3://bucket/judge_trajectories/verifier_id=verifier-test/trajectory-task-1.json"

    monkeypatch.setattr(RubricsVerifierTaskStep, "_fetch_judge_trajectory", fake_fetch)

    result = await verifier._invoke_judge_a2a(
        eval_prompt="evaluate",
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_trajectory_ext(),
        judge_agent=None,
    )

    assert result == {
        "response": "ok",
        "trajectory_s3_uri": "s3://bucket/judge_trajectories/verifier_id=verifier-test/trajectory-task-1.json",
    }
