"""prompt_agent: generic capture of an agent's StructuredOutput from its terminal response.

The agent harness flattens structured_output to a JSON string in the response text, so the
capture is a tolerant parse. It must be a no-op (None) for free-text replies, so the many
other prompt_agent users are unaffected."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from agentenv_protocol.a2a_agent import (
    TRAJECTORY_V1,
    TaskObjectTrajectoryRequest,
    TaskTrajectoryRequest,
    build_registry,
    enable,
)
from agentenv_protocol.transfers import HttpPutGrant

from agent_env.a2a_agent.object_transfer import DEFAULT_TRAJECTORY_MAX_BYTES
from agent_env.config import configure
from agent_env.store.object_store import LocalFilesystemObjectStore
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.snapshot_utils import agent_state_capture
from agent_env.task_step.task_steps import prompt_agent as pa
from agent_env.task_step.task_steps.prompt_agent import (
    PromptAgentTaskStep,
    _duplicates_prompt_text,
    _parse_structured_output,
)


def test_parses_pure_json_object():
    out = _parse_structured_output(
        '{"status":"success","artifacts":{"gold_patch":"/app/patches/gold_patch.diff"}}'
    )
    assert out["status"] == "success"
    assert out["artifacts"]["gold_patch"].endswith("gold_patch.diff")


def test_strips_json_code_fence():
    assert _parse_structured_output('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_structured_output('```json{"a": 1}```') == {"a": 1}


def test_none_for_non_object_responses():
    # free text, a JSON array (not an object), and empty/whitespace all yield None -> no capture
    assert _parse_structured_output("I refactored the repo and zipped the harbor.") is None
    assert _parse_structured_output("[1, 2, 3]") is None
    assert _parse_structured_output("") is None
    assert _parse_structured_output("   \n  ") is None


def test_tolerates_surrounding_whitespace():
    assert _parse_structured_output('  \n {"x": true}\n ') == {"x": True}


def test_extracts_object_embedded_in_prose():
    # the harness may emit the structured_output alongside other text blocks
    assert _parse_structured_output('Done. Here is the result:\n{"status": "success"}') == {"status": "success"}
    assert _parse_structured_output('{"status": "success"}\n\nLet me know if you need anything else.') == {"status": "success"}
    # nested objects survive (raw_decode balances braces)
    assert _parse_structured_output('prefix {"a": {"b": 1}} suffix') == {"a": {"b": 1}}
    # an earlier JSON-looking snippet in prose must NOT win over the real (last) output
    assert _parse_structured_output(
        'considered {"foo": 1} but produced:\n{"status": "ok"}'
    ) == {"status": "ok"}


# _duplicates_prompt_text: turn-0 dedup of source_agent_per_turn_prompt_parts vs
# prompt_text. Only the exact prompt-mode shape — a single text part whose
# text equals prompt_text — is a duplicate; anything else must be stored verbatim.


def test_duplicate_when_single_text_part_equals_prompt_text():
    prompt = "Fix the bug in /inputs/main.py\n\ndata:image/png;base64,iVBORw0KGgo..."
    assert _duplicates_prompt_text([{"kind": "text", "text": prompt}], prompt) is True


def test_not_duplicate_without_prompt_text():
    # parts-mode steps persist prompt_text=None — nothing to dedup against
    assert _duplicates_prompt_text([{"kind": "text", "text": "hello"}], None) is False


def test_not_duplicate_for_different_text():
    assert _duplicates_prompt_text([{"kind": "text", "text": "hello"}], "other") is False


def test_not_duplicate_for_multi_part_or_non_text_parts():
    prompt = "hello"
    multi = [
        {"kind": "text", "text": prompt},
        {"kind": "file", "file": {"uri": "s3://bucket/a.png"}},
    ]
    assert _duplicates_prompt_text(multi, prompt) is False
    assert _duplicates_prompt_text([{"kind": "file", "file": {"uri": "s3://b/a.png"}}], prompt) is False
    assert _duplicates_prompt_text([], prompt) is False


def test_not_duplicate_when_part_carries_extra_keys():
    # extra metadata on the part means it is NOT the materialized prompt shape
    assert (
        _duplicates_prompt_text([{"kind": "text", "text": "hello", "metadata": {"a": 1}}], "hello")
        is False
    )


def _on_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def test_fetch_trajectory_object_named_by_target_a2a_task_id(monkeypatch):
    """Inline trajectory is named by `target_a2a_task_id` (the client message id, recorded on the
    conversation as `a2a_task_id`), not `a2a_server_task_id` — so it's resolvable from a conversation."""
    client = MagicMock()
    client.__aenter__.return_value.post = AsyncMock(
        return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"trajectory": {}})
    )
    monkeypatch.setattr(pa.httpx, "AsyncClient", lambda *a, **k: client)
    puts_on_loop = []

    def put(key, *args, **kwargs):
        puts_on_loop.append(_on_loop())
        return f"s3://b/{key}"

    store = SimpleNamespace(get_object_key=lambda p: "traj/", put=put)
    config = SimpleNamespace(get_object_store=lambda: store)
    monkeypatch.setattr(pa, "get_config", lambda: config)
    monkeypatch.setattr(agent_state_capture, "get_config", lambda: config)

    # unbound call: the inline branch never touches `self`, so we skip AWS-touching construction.
    uri = asyncio.run(PromptAgentTaskStep._fetch_trajectory(
        object(), "http://a", {"params": {"endpoint": "/t"}}, "SERVER-ID", "s3://b/traj/", "CLIENT-ID"))
    assert uri == "s3://b/traj/trajectory-CLIENT-ID.json"
    assert puts_on_loop == [False]


def test_fetch_trajectory_prefers_advertised_object_mode(monkeypatch):
    sent = {}
    grants_on_loop = []
    client = MagicMock()

    async def post(url, *, json, timeout):
        sent.update(url=url, json=json, timeout=timeout)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "objects": {
                    "trajectory": {"size_bytes": 42}
                }
            },
        )

    client.__aenter__.return_value.post = post
    monkeypatch.setattr(pa.httpx, "AsyncClient", lambda *a, **k: client)

    class Store:
        supports_transfer_grants = True
        max_single_upload_bytes = None

        def get_object_key(self, prefix):
            assert prefix == "s3://b/traj/"
            return "traj/"

        def object_url(self, key):
            return f"s3://b/{key}"

        def issue_write_grant(self, object_url, *, media_type, max_bytes, expires_in):
            grants_on_loop.append(_on_loop())
            assert object_url == "s3://b/traj/trajectory-CLIENT-ID.json"
            assert media_type == "application/json"
            assert max_bytes == DEFAULT_TRAJECTORY_MAX_BYTES
            return HttpPutGrant(
                kind="http-put",
                url="https://objects.example.test/write?secret=signed",
                expires_at=datetime.now(UTC) + timedelta(minutes=5),
            )

    monkeypatch.setattr(
        pa, "get_config", lambda: SimpleNamespace(get_object_store=lambda: Store())
    )
    extension = {
        "params": {
            "endpoint": "/t",
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
        }
    }
    uri = asyncio.run(
        PromptAgentTaskStep._fetch_trajectory(
            object(),
            "https://agent.example.test",
            extension,
            "SERVER-ID",
            "s3://b/traj/",
            "CLIENT-ID",
        )
    )

    assert uri == "s3://b/traj/trajectory-CLIENT-ID.json"
    assert grants_on_loop == [False]
    assert sent["url"] == "https://agent.example.test/t"
    assert sent["json"]["task_id"] == "SERVER-ID"
    descriptor = sent["json"]["objects"]["trajectory"]
    assert descriptor["media_type"] == "application/json"
    assert descriptor["max_bytes"] == DEFAULT_TRAJECTORY_MAX_BYTES
    assert descriptor["write"]["kind"] == "http-put"


@pytest.mark.asyncio
async def test_sdk_agent_on_the_local_store_still_records_its_trajectory(monkeypatch, tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path))
    configure(object_store=store)
    for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task",
               "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa.conversation_store, fn, lambda *a, **kw: None)

    async def sdk_get(request: TaskTrajectoryRequest | TaskObjectTrajectoryRequest):
        return {"trajectory": []}

    extensions = build_registry(
        object(), [enable(TRAJECTORY_V1)], sdk_handlers={(TRAJECTORY_V1.uri, "get"): sdk_get}
    ).card_extensions()
    trajectory_requests: list[dict] = []

    def agent(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/ext/trajectory":
            trajectory_requests.append(body)
            return httpx.Response(200, json={"trajectory": [{"type": "echo", "output": "Echo: hi"}]})
        if body["method"] == "message/send":
            result = {"id": "server-task", "contextId": "ctx"}
        else:
            result = {"status": {"state": "completed",
                                 "message": {"parts": [{"kind": "text", "text": "Echo: hi"}]}}}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    transport = httpx.MockTransport(agent)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_client(*a, transport=transport, **kw))
    context = TaskStepContext(instance_id="ti-1")
    context.deployed_agents.append(DeployedAgent(
        agent_name="solver", api_url="http://agent.test", a2a_url="http://agent.test",
        a2a_card={"capabilities": {"extensions": extensions}},
    ))
    step = PromptAgentTaskStep(
        id="solve", version=None, prompt="hi", agent_name="solver", poll_interval_seconds=0,
        trajectory_output_prefix=store.object_url("prompt_agent_trajectories"),
    )

    result = await step.execute(context)

    uri = result.prompt_responses[-1].agent_trajectory_s3_uri
    assert trajectory_requests == [{"task_id": "server-task"}]
    assert store.get_object_key(uri).startswith("prompt_agent_trajectories/trajectory-")
    assert json.loads(store.get(uri)) == [{"type": "echo", "output": "Echo: hi"}]


