"""Unit tests for RubricsVerifierTaskStep._fetch_judge_trajectory."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from agentenv_protocol.transfers import HttpPutGrant

from agent_env.a2a_agent import A2AAgent
from agent_env.store import GrantUnavailableError
from agent_env.a2a_agent.object_transfer import DEFAULT_TRAJECTORY_MAX_BYTES
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

RV = "agent_env.task_step.task_steps.verifiers.rubrics_verifier"


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


def _card_with_object_trajectory_ext() -> dict:
    return {
        "capabilities": {
            "extensions": [
                {
                    "uri": A2AAgent.EXT_TRAJECTORY,
                    "params": {
                        "endpoint": "/ext/trajectory",
                        "methods": {
                            "get": {
                                "request": {
                                    "oneOf": [
                                        {"required": ["task_id"]},
                                        {"required": ["task_id", "objects"]},
                                    ]
                                }
                            }
                        },
                    },
                }
            ]
        }
    }


class GrantingStore:
    supports_transfer_grants = True
    max_single_upload_bytes = None
    grant_error: Exception | None = None

    def __init__(self):
        self.write_grants: list[tuple[str, str, int]] = []
        self.puts: list[str] = []

    def object_url(self, key):
        return f"s3://bucket/{key}"

    def grants_reach(self, sandbox_type):
        return True

    def get_object_key(self, object_url):
        return object_url.removeprefix("s3://bucket/")

    def put(self, key, body, content_type=None):
        self.puts.append(key)
        return self.object_url(key)

    def issue_write_grant(self, object_url, *, media_type, max_bytes, expires_in=None):
        self.write_grants.append((object_url, media_type, max_bytes))
        if self.grant_error is not None:
            raise self.grant_error
        return HttpPutGrant(
            kind="http-put",
            url="https://objects.example.test/write?secret=signed",
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )


def _use_store(monkeypatch, store, *, key_prefix=""):
    config = type(
        "Config", (), {"get_object_store": lambda self: store, "get_artifact_key_prefix": lambda self: key_prefix}
    )()
    monkeypatch.setattr(f"{RV}.get_config", lambda: config)
    monkeypatch.setattr(
        "agent_env.task_step.snapshot_utils.agent_state_capture.get_config", lambda: config
    )


@pytest.mark.asyncio
async def test_no_extension_on_card_skips_the_fetch_entirely(monkeypatch):
    verifier = _verifier()

    async def boom(self, method, url, **kwargs):
        raise AssertionError("should never make a request without the extension")

    monkeypatch.setattr(httpx.AsyncClient, "request", boom)

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example", judge_agent_card={}, a2a_server_task_id="t1",
        sandbox_type="local",
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
        sandbox_type="local",
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
        "agent_env.task_step.snapshot_utils.agent_state_capture.upload_trajectory", fake_upload,
    )

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_trajectory_ext(),
        a2a_server_task_id="t1",
        sandbox_type="local",
    )

    assert uri == "s3://bucket/judge_trajectories/verifier_id=verifier-test/trajectory-t1.json"
    assert captured["trajectory"] == [{"span": "x"}]
    assert "/judge_trajectories/verifier_id=verifier-test" in captured["prefix"]


@pytest.mark.asyncio
async def test_a_judge_naming_its_own_trajectory_prefix_yields_no_trajectory(monkeypatch):
    verifier = _verifier()

    async def fake_request(self, method, url, **kwargs):
        return httpx.Response(
            200, json={"trajectory_s3_prefix": "s3://bucket/pre/", "is_live": False},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    class FakeStore:
        supports_transfer_grants = False

        def object_url(self, key):
            return f"s3://bucket/{key}"

        def list_at(self, prefix):
            raise AssertionError("an agent-named prefix is never listed")

    _use_store(monkeypatch, FakeStore())

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_trajectory_ext(),
        a2a_server_task_id="t1",
        sandbox_type="local",
    )
    assert uri is None


@pytest.mark.asyncio
async def test_advertised_object_trajectory_writes_to_the_judge_prefix(monkeypatch):
    verifier = _verifier()
    sent: list[dict] = []

    async def fake_request(self, method, url, **kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(
            200,
            json={
                "objects": {
                    "trajectory": {"size_bytes": 42}
                }
            },
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    store = GrantingStore()
    _use_store(monkeypatch, store)

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_object_trajectory_ext(),
        a2a_server_task_id="t1",
        sandbox_type="local",
    )

    assert store.write_grants == [(uri, "application/json", DEFAULT_TRAJECTORY_MAX_BYTES)]
    assert uri.startswith("s3://bucket/judge_trajectories/verifier_id=verifier-test/trajectory-")
    assert sent[0]["task_id"] == "t1"
    assert sent[0]["objects"]["trajectory"]["write"]["kind"] == "http-put"


@pytest.mark.asyncio
async def test_the_judge_prefix_is_under_the_fixture_prefix(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return httpx.Response(200, json={"objects": {"trajectory": {"size_bytes": 42}}}, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    _use_store(monkeypatch, GrantingStore(), key_prefix="fx/")

    uri = await _verifier()._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_object_trajectory_ext(),
        a2a_server_task_id="t1",
        sandbox_type="local",
    )

    assert uri.startswith("s3://bucket/fx/judge_trajectories/verifier_id=verifier-test/trajectory-")


@pytest.mark.parametrize(
    "card, body",
    [
        (_card_with_trajectory_ext(), {"trajectory": [{"span": "x"}]}),
        (_card_with_object_trajectory_ext(), {"objects": {"trajectory": {"size_bytes": 42}}}),
    ],
    ids=["inline", "objects"],
)
@pytest.mark.asyncio
async def test_each_call_names_its_trajectory_apart_from_the_judge_task_id(monkeypatch, card, body):
    verifier = _verifier()

    async def fake_request(self, method, url, **kwargs):
        return httpx.Response(200, json=body, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    store = GrantingStore()
    _use_store(monkeypatch, store)

    uris = [
        await verifier._fetch_judge_trajectory(
            judge_a2a_url="http://judge.example", judge_agent_card=card, a2a_server_task_id="t1",
            sandbox_type="local",
        )
        for _ in range(2)
    ]

    assert uris[0] != uris[1]
    assert not any(uri.endswith("/trajectory-t1.json") for uri in uris)


@pytest.mark.asyncio
async def test_a_store_without_grants_asks_an_object_capable_judge_inline(monkeypatch):
    verifier = _verifier()
    sent: list[dict] = []

    async def fake_request(self, method, url, **kwargs):
        sent.append(kwargs["json"])
        return httpx.Response(
            200, json={"trajectory": [{"span": "x"}]}, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    store = GrantingStore()
    store.supports_transfer_grants = False
    _use_store(monkeypatch, store)

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_object_trajectory_ext(),
        a2a_server_task_id="t1",
        sandbox_type="local",
    )

    assert sent == [{"task_id": "t1"}]
    assert store.write_grants == []
    assert uri == f"s3://bucket/{store.puts[0]}"


@pytest.mark.asyncio
async def test_a_grant_that_cannot_be_issued_is_not_retried(monkeypatch):
    verifier = _verifier()

    async def boom(self, method, url, **kwargs):
        raise AssertionError("no request without a grant")

    monkeypatch.setattr(httpx.AsyncClient, "request", boom)
    store = GrantingStore()
    store.grant_error = GrantUnavailableError("credentials expire first")
    _use_store(monkeypatch, store)

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_object_trajectory_ext(),
        a2a_server_task_id="t1",
        sandbox_type="local",
    )

    assert uri is None
    assert len(store.write_grants) == 1


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
        sandbox_type="local",
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
        "agent_env.task_step.snapshot_utils.agent_state_capture.upload_trajectory",
        lambda trajectory, prefix, *, name=None: "s3://bucket/trajectory.json",
    )

    uri = await verifier._fetch_judge_trajectory(
        judge_a2a_url="http://judge.example",
        judge_agent_card=_card_with_trajectory_ext(),
        a2a_server_task_id="t1",
        sandbox_type="local",
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

    async def fake_fetch(
        self, *, judge_a2a_url, judge_agent_card, a2a_server_task_id, sandbox_type
    ):
        assert judge_a2a_url == "http://judge.example"
        assert a2a_server_task_id == "task-1"
        assert sandbox_type is None
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
