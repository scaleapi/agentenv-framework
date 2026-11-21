"""A slot's door token rides the env's grant to the slot's own addresses, and every caller that sends no credential of
its own sends it there; the default stack's grant and calls are what they always were."""

from __future__ import annotations

import functools
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.env.gateway.constants import SLOT_TOKEN_HEADER
from agent_env.explorer.routers import triggers
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_providers.stack_layout import DEFAULT_STACK_LAYOUT, StackLayout
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.mcp_cli_builder.build_mcp_cli import _resolve_manifest_endpoint
from agent_env.task_step.task_steps.mcp_cli_builder.codegen import generate_cli_script
from agent_env.task_step.task_steps.mcp_env_validator.verify_spec_conformance import VerifySpecConformanceTaskStep
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep

GATEWAY = "https://s1-gw.modal.host"
PGWEB = "https://s1-pgweb.modal.host/"
DB_MCP = "https://s1-dbmcp.modal.host/mcp"
DOOR = {SLOT_TOKEN_HEADER: "door-tok"}
CONTROL = {"Authorization": "Bearer ctl"}


def _record(door_token: str | None) -> DeployedGatewayEnv:
    return DeployedGatewayEnv(env_id="crm", env_version=1, sandbox_id="vm", gateway_url=GATEWAY, db_web_url=PGWEB,
                              db_mcp_url=DB_MCP, environment_card_url=f"{GATEWAY}/.well-known/agent-env.json",
                              control_token="ctl", door_token=door_token)


def test_the_default_stacks_grant_is_its_control_token_on_its_address_alone():
    record = _record(None)
    assert record.grants == {GATEWAY: CONTROL}
    assert record.control_headers == CONTROL
    for url in (GATEWAY, PGWEB, DB_MCP):
        assert record.door_headers_for(url) == {}


def test_a_slots_grant_carries_the_door_header_to_each_of_its_addresses_and_nowhere_else():
    record = _record("door-tok")
    assert record.grants == {GATEWAY: {**DOOR, **CONTROL}, "https://s1-pgweb.modal.host": DOOR,
                             "https://s1-dbmcp.modal.host": DOOR}
    assert record.control_headers == {**DOOR, **CONTROL}
    assert record.control_headers_for(f"{GATEWAY}/svc/mcp-slack/api/reset") == {**DOOR, **CONTROL}
    assert record.control_headers_for("https://s1-dbmcp.modal.host/") == DOOR
    assert record.control_headers_for(f"{PGWEB}api/query") == DOOR
    for elsewhere in ("https://s2-gw.modal.host", "https://s1-gw.modal.host.evil.example", "https://example.com"):
        assert record.control_headers_for(elsewhere) == {}
    assert record.door_headers_for(f"{GATEWAY}/mcp") == DOOR


def test_the_door_token_survives_the_record_round_trip():
    record = DeployedEnv.from_dict({"env_id": "crm", "env_version": 1, "sandbox_id": "vm", "door_token": "door-tok",
                                    "gateway_url": GATEWAY})
    assert isinstance(record, DeployedGatewayEnv) and record.door_token == "door-tok"
    assert DeployedEnv.from_dict({"env_id": "crm", "env_version": 1, "door_token": "t"}).door_token == "t"


@pytest.fixture
def seen(monkeypatch) -> list[httpx.Request]:
    """Every request any httpx client sends, answered 200 with a card-shaped body."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"name": "crm", "ok": True})

    monkeypatch.setattr(httpx, "AsyncClient", functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)))
    return requests


async def _legacy_calls(base: str) -> None:
    await legacy_protocol.reset_via_rest(base)
    await legacy_protocol.add_via_rest(base, "/data/x.json")
    await legacy_protocol._export_state_at(base, timeout=5, verify=True)
    await legacy_protocol.v1_base_url(None, GATEWAY, "slack")
    await _resolve_manifest_endpoint(base, "slack")


@pytest.mark.asyncio
@pytest.mark.parametrize(("door_token", "sent"), [(None, {}), ("door-tok", DOOR)])
async def test_legacy_calls_under_a_grant_send_the_door_header_alone(seen, door_token, sent):
    with protocol_v1.authorized(_record(door_token).grants):
        await _legacy_calls(f"{GATEWAY}/svc/mcp-slack")
    assert len(seen) == 5
    for request in seen:
        assert {k: v for k, v in request.headers.items() if k in (SLOT_TOKEN_HEADER.lower(), "authorization")} == \
            {k.lower(): v for k, v in sent.items()}


@pytest.mark.asyncio
async def test_a_card_read_sends_only_what_it_is_given(seen):
    await protocol_v1.get_card(GATEWAY)
    await protocol_v1.get_card(GATEWAY, headers=DOOR)
    assert SLOT_TOKEN_HEADER.lower() not in seen[0].headers
    assert seen[1].headers[SLOT_TOKEN_HEADER] == "door-tok" and "authorization" not in seen[1].headers


@pytest.mark.asyncio
@pytest.mark.parametrize(("door_token", "sent"), [(None, None), ("door-tok", "door-tok")])
async def test_the_explorer_reads_live_trigger_state_with_the_records_door_token(seen, door_token, sent):
    record = {"instance_id": "i1", "gateway_url": GATEWAY, **({"door_token": door_token} if door_token else {})}
    with patch.object(triggers, "docs", return_value=SimpleNamespace(query=lambda *a, **k: [record])):
        await triggers._live_trigger_state([{"env_id": "crm", "instance_id": "i1"}])
    assert [r.url.path for r in seen] == ["/triggers/state", "/clock/state"]
    assert all(r.headers.get(SLOT_TOKEN_HEADER) == sent for r in seen)


@pytest.mark.parametrize(("token", "sent"), [(None, None), ("door-tok", "door-tok")])
def test_a_generated_cli_sends_the_slot_token_it_was_installed_with(monkeypatch, token, sent):
    namespace: dict = {"__name__": "generated_cli", "__file__": "/nonexistent/bin/crm"}
    exec(compile(generate_cli_script("crm"), "cli", "exec"), namespace)
    monkeypatch.delenv(namespace["SLOT_TOKEN_ENV"], raising=False)
    if token:
        monkeypatch.setenv(namespace["SLOT_TOKEN_ENV"], token)
    assert namespace["_headers"]().get(SLOT_TOKEN_HEADER) == sent
    assert namespace["ROLE_HEADER"] in namespace["_headers"]()


def _mcp_answer(request: httpx.Request) -> httpx.Response:
    """A streamable-HTTP MCP server's answer to ``request``, with one tool, ``crm_reset``."""
    if request.method != "POST":
        return httpx.Response(405 if request.method == "GET" else 200)
    message = json.loads(request.content)
    if "id" not in message:
        return httpx.Response(202)
    results = {
        "initialize": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                       "serverInfo": {"name": "crm", "version": "1"}},
        "tools/list": {"tools": [{"name": "crm_reset", "inputSchema": {"type": "object"}}]},
        "tools/call": {"content": [{"type": "text", "text": "reset"}], "isError": False},
    }
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": results[message["method"]]})


@pytest.fixture
def served(monkeypatch) -> list[httpx.Request]:
    """Every request any httpx client sends, answered as an MCP server at ``/mcp``, a card-less service elsewhere (404
    for a card, a JSON body for ``/export-state``)."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/mcp"):
            return _mcp_answer(request)
        if request.url.path.endswith("/export-state"):
            return httpx.Response(200, json={"rows": []})
        return httpx.Response(404)

    monkeypatch.setattr(httpx, "AsyncClient", functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)))
    return requests


def _door_sent(requests: list[httpx.Request]) -> set[str | None]:
    return {request.headers.get(SLOT_TOKEN_HEADER) for request in requests}


@pytest.mark.asyncio
@pytest.mark.parametrize(("door_token", "sent"), [(None, None), ("door-tok", "door-tok")])
async def test_the_legacy_mcp_reset_sends_the_door_header_under_a_slots_grant(served, door_token, sent):
    with protocol_v1.authorized(_record(door_token).grants):
        assert await legacy_protocol.offers_tool(f"{GATEWAY}/mcp", "crm_reset")
        assert not (await legacy_protocol.reset_via_mcp_tool(f"{GATEWAY}/mcp", "crm", max_retries=1)).isError
    assert {request.url.path for request in served} == {"/mcp"}
    assert _door_sent(served) == {sent}


@pytest.mark.asyncio
@pytest.mark.parametrize(("door_token", "sent"), [(None, None), ("door-tok", "door-tok")])
async def test_the_legacy_export_state_stream_sends_the_door_header_under_a_slots_grant(served, tmp_path, door_token, sent):
    with protocol_v1.authorized(_record(door_token).grants):
        suffix = await SnapshotEnvTaskStep._export_environment_to_file(GATEWAY, "slack", str(tmp_path / "out"), 30)
    assert suffix == ".json"
    assert served[-1].url.path == "/svc/mcp-slack/export-state"
    assert _door_sent(served) == {sent}


@pytest.mark.asyncio
@pytest.mark.parametrize(("door_token", "sent"), [(None, None), ("door-tok", "door-tok")])
async def test_the_spec_conformance_check_reads_live_tools_with_the_door_header(monkeypatch, door_token, sent):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/openapi.yaml"):
            return httpx.Response(200, text="paths:\n  /reset:\n    post:\n      operationId: reset\n")
        return _mcp_answer(request)

    monkeypatch.setattr(httpx, "AsyncClient", functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)))
    record = _record(door_token)
    record.mcp_url = f"{GATEWAY}/mcp"
    record.environment_card = {"name": "crm", "url": "/agentenv",
                               "children_environments": [{"name": "crm", "url": "/svc/mcp-crm/agentenv"}]}
    env = SimpleNamespace(environment_name="crm", merge_metadata=lambda _: None)
    with patch("agent_env.env.env.Env.get", return_value=env):
        await VerifySpecConformanceTaskStep(id="s", version=None, env_id="crm").execute(TaskStepContext(deployed_envs=[record]))
    assert [request.url.path for request in requests if request.url.path == "/mcp"]
    assert _door_sent(requests) == {sent}


@pytest.mark.asyncio
@pytest.mark.parametrize(("door_token", "sent"), [(None, {}), ("door-tok", DOOR)])
async def test_the_in_process_proxy_registers_each_envs_mcp_server_with_its_door_header(monkeypatch, door_token, sent):
    from agent_env.task.episode import run as episode_run
    from agent_env.task.episode.steps import _EpisodeAgentStep

    registrations: list[dict] = []
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_MCP_CONFIG,
                                             "params": {"endpoint": "/mcp-config", "methods": {"add": {"method": "POST"}}}}]}}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp-config":
            registrations.append(json.loads(request.content))
            return httpx.Response(200, json={})
        return httpx.Response(200, json=card)

    monkeypatch.setattr(httpx, "AsyncClient", functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler)))

    class _Proxy:
        async def start(self) -> str:
            return "http://127.0.0.1:1"

    monkeypatch.setattr(episode_run, "InProcessProxy", _Proxy)
    record = _record(door_token)
    record.mcp_url = f"{GATEWAY}/mcp"
    episode = SimpleNamespace(_in_process=[], _attach=AsyncMock())
    episode._serve_in_process = functools.partial(episode_run.Episode._serve_in_process, episode)
    step = SimpleNamespace(agent_name="a", env_ids=["crm"], env_step_id=None, role=None)
    agent_step = _EpisodeAgentStep.__new__(_EpisodeAgentStep)
    agent_step.step, agent_step.sandboxed, agent_step.id = step, False, "play"
    context = TaskStepContext(deployed_envs=[record])
    with patch("agent_env.task.episode.steps._episode_of", return_value=episode), \
            patch("agent_env.task.episode.steps.bound_deployment", return_value=record):
        await agent_step.execute(context)
    assert registrations == [{"url": f"{GATEWAY}/mcp", **({"headers": sent} if sent else {})}]


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", [DEFAULT_STACK_LAYOUT, StackLayout.for_slot(2)])
async def test_a_slot_refuses_computers_on_sandboxes_of_their_own_and_the_default_stack_takes_them(layout):
    class _Stop(Exception):
        pass

    async def children(_url, _token):
        return []

    with patch.object(EnvironmentGatewayProvider, "in_vm", side_effect=_Stop), \
            pytest.raises(ValueError if layout.is_slot else _Stop, match="sandbox of its own" if layout.is_slot else None):
        await EnvironmentGatewayProvider()._deploy_gateway(
            SimpleNamespace(), state_instance=None, mcp_servers=[], mcp_server_images=[],
            existing_sandbox=SimpleNamespace(walls_verified=True), children=children, stack_layout=layout)
