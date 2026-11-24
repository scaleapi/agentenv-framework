"""Role auth on a gateway an env deploy runs on local compose. One raw client changing only AgentEnv-Role between /step
calls to act as another persona, which any caller could do before, is refused once that persona's role takes a token,
whatever the case it names the persona in: a member's agent token plays it, an open role on the same env still works,
and an MCP client with a member's token lists and calls tools while a plain header naming the persona is refused at
initialize. Requires a docker daemon; spins up a throwaway ``registry:2``. The tests share one deployed env, each with
roles of its own."""

import json
import uuid

import httpx
import pytest
import pytest_asyncio
from agentenv_protocol import client as protocol_v1
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.gateway.constants import AGENT_ENV_ROLE_HEADER, AGENT_TOKEN_HEADER, EXT_ROLES_URI
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs, local_stack_at

# Deploying a gateway on local compose, with the images it builds, takes minutes, as for every local stack test.
pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

WHOAMI = "items_whoami"
INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "probe", "version": "1"}}}


@pytest_asyncio.fixture(scope="module")
async def deployed(tmp_path_factory):
    """One items env for the module, on a stack of its own: deploying one is most of what a test here costs."""
    with pytest.MonkeyPatch.context() as monkeypatch, local_stack_at(monkeypatch, tmp_path_factory.mktemp("stack")):
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        items = MCPServerEnv.put(id=f"roles-items-{uid}", environment_name="items",
                                 docker_image_artifact=_put_items_image(f"roles-items-{uid}"))
        env = MultiEnv.put(id=f"roles-{uid}", mcp_server_envs=[items], name="office")
        deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
        try:
            yield deployed
        finally:
            await env.close()


def _persona(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:6]}@example.com"


async def _whoami(client: httpx.AsyncClient, headers: dict[str, str]) -> httpx.Response:
    return await client.post("/step", json={"action": "call_tool", "tool_name": WHOAMI, "arguments": {}},
                             headers=headers)


def _role(answer: httpx.Response) -> str:
    assert answer.status_code == 200, answer.text
    return json.loads(answer.json()["content"][0]["text"])["role"]


@pytest.mark.asyncio
async def test_switching_only_the_role_header_is_refused_once_the_persona_takes_a_token(deployed):
    brooke, denise, token = _persona("brooke"), _persona("denise"), f"agent-{uuid.uuid4().hex}"
    assert protocol_v1.find_extension_method(deployed.environment_card, EXT_ROLES_URI, "create") is not None
    async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
        before = [_role(await _whoami(client, {AGENT_ENV_ROLE_HEADER: who})) for who in (denise, brooke)]
        created = await client.post("/roles", json={"role": brooke, "auth": {"type": "token"}},
                                    headers=deployed.control_headers)
        switched = await _whoami(client, {AGENT_ENV_ROLE_HEADER: brooke})
        # the items service, like the others behind the gateway, may match a persona's address in any case
        respelled = await _whoami(client, {AGENT_ENV_ROLE_HEADER: brooke.upper()})
        still_open = _role(await _whoami(client, {AGENT_ENV_ROLE_HEADER: denise}))
        joined = await client.post("/roles/members", json={"role": brooke, "agent_token": token},
                                   headers=deployed.control_headers)
        member = _role(await _whoami(client, {AGENT_ENV_ROLE_HEADER: brooke, AGENT_TOKEN_HEADER: token}))
        harness = _role(await _whoami(client, {AGENT_ENV_ROLE_HEADER: brooke, **deployed.control_headers}))
        trajectory = (await client.get("/trajectory")).text

    assert before == [denise, brooke]
    assert created.status_code == 200 and joined.status_code == 200
    assert switched.status_code == 403 and brooke in switched.json()["error"]
    assert respelled.status_code == 403
    assert (still_open, member, harness) == (denise, brooke, brooke)
    calls = [event for event in map(json.loads, trajectory.splitlines())
             if event["event_type"] == "tool_call" and event.get("role") == brooke]
    assert [event["role_auth"] for event in calls] == ["header", "agent_token", "control_token"]
    assert token not in trajectory


@pytest.mark.asyncio
async def test_an_mcp_client_with_a_members_token_lists_and_calls_and_a_plain_header_is_refused_at_initialize(deployed):
    carol, token = _persona("carol"), f"agent-{uuid.uuid4().hex}"
    async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60, headers=deployed.control_headers) as harness:
        (await harness.post("/roles", json={"role": carol, "auth": {"type": "token"}})).raise_for_status()
        (await harness.post("/roles/members", json={"role": carol, "agent_token": token})).raise_for_status()

    async with (
        httpx.AsyncClient(headers={AGENT_ENV_ROLE_HEADER: carol, AGENT_TOKEN_HEADER: token},
                          timeout=httpx.Timeout(30, read=120)) as http,
        streamable_http_client(f"{deployed.gateway_url}/mcp", http_client=http) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        listed = [tool.name for tool in (await session.list_tools()).tools]
        called = await session.call_tool(WHOAMI, {})

    async with httpx.AsyncClient(timeout=60) as client:
        refused = await client.post(f"{deployed.gateway_url}/mcp", json=INITIALIZE,
                                    headers={AGENT_ENV_ROLE_HEADER: carol,
                                             "accept": "application/json, text/event-stream"})

    assert WHOAMI in listed
    assert not called.isError and json.loads(called.content[0].text)["role"] == carol
    assert refused.status_code == 403
