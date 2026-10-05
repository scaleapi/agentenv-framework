"""Base `Env` contract, and the card helpers on `DeployedEnv`.

`deploy` is part of the custom-env contract: the base raises `NotImplementedError`
(not `@abstractmethod`, which would block deserializing a read-only env) so the
`deploy_env` step fails clearly on an env that never implements it. Every built-in
env overrides `deploy`; this pins the base behavior a custom env inherits.

The card helpers read only the stored env card: capabilities and child envs come from it,
and invoke calls the endpoint and verb it advertises, joined onto the card's address.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json

import httpx
import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.env.env import (
    _GATEWAY_ONLY_FIELDS, DeployedEnv, DeployedGatewayEnv, DeployedSandboxEnv, Env, EnvCapabilityUnsupported, EnvNeedsGateway,
    EnvNeedsSandbox, gateway_url_of, require_gateway_url, require_sandbox,
)
from agent_env.env.gateway.constants import (
    EXT_CLOCK_URI, EXT_DISABLE_TOOL_URI, EXT_ENABLE_TOOL_URI, EXT_STATE_URI, EXT_STEP_URI, EXT_TRAJECTORY_URI, EXT_TRIGGERS_URI,
    WELL_KNOWN_PATH,
)
from agent_env.env.gateway.gateway import Gateway


class _DeploylessEnv(Env):
    type = "deployless_env_test"

    @classmethod
    def from_dict(cls, data: dict) -> "_DeploylessEnv":
        return cls(id=data["id"], version=data.get("version"), metadata=data.get("metadata"))


def test_base_deploy_raises_not_implemented():
    env = _DeploylessEnv(id="e1", version=1)
    with pytest.raises(NotImplementedError, match="must implement deploy"):
        asyncio.run(env.deploy())


def _ext(uri: str, endpoint: str, method: str) -> dict:
    return {"uri": uri, "params": {"endpoint": endpoint, "methods": {method: {"method": "POST"}}}}


# A backing server's own card, as it serves it; the gateway rewrites its /agentenv paths.
_SLACK = {"name": "slack", "url": "/agentenv", "capabilities": {"extensions": [
    _ext("urn:agentenv:set-errors/v1", "/agentenv/ext/set_errors", "set_errors"),
]}}
_LEAF = {"name": "notes", "url": "/agentenv", "capabilities": {"extensions": [_ext("urn:agentenv:set-errors/v1", "/agentenv/ext/set_errors", "set_errors")]}}
_BASES = ["https://gw.example", "https://gw.example/", "https://sandbox.example/sandbox/sb-01abc-18765"]


@pytest.fixture(scope="module")
def composed() -> dict:
    """The real gateway's composed card over mcp-slack, plus a backing server that serves no card."""
    gw = Gateway(host="127.0.0.1", port=0, server_name="env1234", internal_mcp_servers=[],
                 rest_proxy_urls={"mcp-slack": "http://slack:18765", "mcp-nocard": "http://nocard:18765"})

    async def fetch(client, key, base_url):
        return gw._rewrite_child_card(key, _SLACK) if key == "mcp-slack" else None

    gw._fetch_child_card = fetch
    return json.loads(asyncio.run(gw._serve_env_card()).body)


def _record(card: dict | None, base: str = _BASES[0]) -> DeployedEnv:
    return DeployedGatewayEnv(
        env_id="multi", env_version=1, gateway_url=base, mcp_url=f"{base.rstrip('/')}/mcp", db_web_url=None,
        sandbox_id="sb-1", instance_id="inst-1", environment_card_url=f"{base}{WELL_KNOWN_PATH}", environment_card=card,
    )


@pytest.fixture
def sent(monkeypatch) -> list[httpx.Request]:
    """Requests the protocol client sends, each answered with an empty 200."""
    requests, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    monkeypatch.setattr(protocol_v1.httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return requests


@pytest.mark.parametrize("base", _BASES, ids=["plain", "trailing-slash", "path-prefix"])
def test_environment_url_is_the_card_url_without_the_well_known_path(base):
    assert _record(None, base).environment_url == base.rstrip("/")


def test_supports_and_require_read_the_stored_card_by_method(composed):
    deployed = _record(composed)
    assert deployed.supports(EXT_STEP_URI, "step") and deployed.supports(EXT_CLOCK_URI, "set_time")
    assert not deployed.supports(EXT_STEP_URI, "undo") and not deployed.supports("urn:agentenv:unknown/v1", "get")
    deployed.require(EXT_STATE_URI, "get")
    with pytest.raises(EnvCapabilityUnsupported, match=r"^env 'multi' does not offer 'undo' on urn:agentenv:step/v1\.$"):
        deployed.require(EXT_STEP_URI, "undo")


@pytest.mark.asyncio
async def test_a_record_without_a_card_offers_nothing(sent):
    deployed = _record(None)
    assert not deployed.supports(EXT_STEP_URI, "step")
    assert deployed.get_child_env_card("slack") is None
    with pytest.raises(EnvCapabilityUnsupported):
        await deployed.invoke(EXT_STEP_URI, "step", {"action": "list_tools"})
    assert sent == []


def test_child_env_cards_come_from_the_card_by_name(composed):
    deployed = _record(composed)
    assert deployed.get_child_env_card("slack")["url"] == "/svc/mcp-slack/agentenv"
    assert deployed.get_child_env_card("nocard") is None  # the gateway omits a backing server that serves no card
    assert _record({**composed, "name": "slack", "children_environments": []}).get_child_env_card("slack") is None  # composed, not a leaf


def test_a_leaf_card_is_its_own_only_child_env():
    assert _record(_LEAF).get_child_env_card("notes") is _LEAF
    assert _record(_LEAF).get_child_env_card("slack") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("base", _BASES, ids=["plain", "trailing-slash", "path-prefix"])
@pytest.mark.parametrize("uri, method, params, verb, path", [
    (EXT_CLOCK_URI, "set_time", {"virtual_time": "2026-01-01T00:00:00Z"}, "PUT", "/clock/set-time"),
    (EXT_TRIGGERS_URI, "register", {"triggers": []}, "POST", "/triggers/register"),
    (EXT_STEP_URI, "step", {"action": "list_tools"}, "POST", "/step"),
    (EXT_STATE_URI, "get", None, "GET", "/state"),
    (EXT_DISABLE_TOOL_URI, "disable", {"role": "default", "tools": ["t"]}, "POST", "/tools/disable"),
    (EXT_ENABLE_TOOL_URI, "enable", {"role": "default", "tools": ["t"]}, "POST", "/tools/enable"),
    (EXT_CLOCK_URI, "state", None, "GET", "/clock/state"),
    (EXT_TRIGGERS_URI, "state", None, "GET", "/triggers/state"),
    (EXT_TRAJECTORY_URI, "get", None, "GET", "/trajectory"),
], ids=["clock-set_time", "triggers-register", "step", "state", "tools-disable", "tools-enable", "clock-state",
        "triggers-state", "trajectory-get"])
async def test_invoke_calls_the_endpoint_and_verb_the_card_advertises(sent, composed, base, uri, method, params, verb, path):
    await _record(composed, base).invoke(uri, method, params)
    [request] = sent
    assert (request.method, str(request.url).split("?")[0]) == (verb, base.rstrip("/") + path)
    assert (dict(request.url.params) if verb == "GET" else json.loads(request.content)) == (params or {})


@pytest.mark.asyncio
async def test_invoke_on_a_child_env_uses_the_child_card(sent, composed):
    await _record(composed).invoke("urn:agentenv:set-errors/v1", "set_errors", {"errors": []}, environment_name="slack")
    [request] = sent
    assert (request.method, str(request.url)) == ("POST", "https://gw.example/svc/mcp-slack/agentenv/ext/set_errors")


@pytest.mark.asyncio
async def test_invoke_raises_before_any_request_when_the_card_does_not_offer_it(sent, composed):
    deployed = _record(composed)
    with pytest.raises(EnvCapabilityUnsupported, match=r"^env 'multi' child env 'nocard' does not offer 'set_errors' on urn:agentenv:set-errors/v1\.$") as raised:
        await deployed.invoke("urn:agentenv:set-errors/v1", "set_errors", environment_name="nocard")
    assert (raised.value.environment_name, raised.value.instance_id) == ("nocard", "inst-1")
    with pytest.raises(EnvCapabilityUnsupported):
        await deployed.invoke(EXT_CLOCK_URI, "rewind")
    with pytest.raises(EnvCapabilityUnsupported):  # a card without its URL has no address to call
        await dataclasses.replace(deployed, environment_card_url=None).invoke(EXT_STEP_URI, "step")
    assert sent == []


def test_env_capability_unsupported_carries_its_fields():
    error = EnvCapabilityUnsupported("urn:agentenv:clock/v1", "sync_time", "multi", "inst-1", "slack")
    assert isinstance(error, RuntimeError)
    assert (error.capability, error.method, error.env_id, error.instance_id, error.environment_name) == (
        "urn:agentenv:clock/v1", "sync_time", "multi", "inst-1", "slack")


_CARD = {"name": "env1234", "additionalInterfaces": [{"url": "/custom-mcp", "transport": "mcp"}]}
_CARD_URL = f"https://gw.example/sb/x{WELL_KNOWN_PATH}"


def test_a_record_takes_its_mcp_url_and_server_name_from_its_card():
    record = DeployedGatewayEnv(env_id="e", env_version=1, gateway_url="https://gw.example/sb/x", sandbox_id="sb",
                                mcp_url="https://stale/mcp", mcp_server_name="stale", environment_card_url=_CARD_URL, environment_card=_CARD)
    assert (record.mcp_url, record.mcp_server_name) == ("https://gw.example/sb/x/custom-mcp", "env1234")
    assert dataclasses.replace(record, environment_card={"name": "env5678"}).mcp_url == "https://gw.example/sb/x/mcp"


@pytest.mark.parametrize("card, card_url", [(None, _CARD_URL), (_CARD, None)], ids=["no-card", "no-card-url"])
def test_a_record_without_its_card_keeps_its_stored_mcp_url_and_name(card, card_url):
    """Older records carry no card, so their stored values stand."""
    record = DeployedEnv.from_dict({"env_id": "e", "env_version": 1, "gateway_url": "https://old", "sandbox_id": "sb",
                                    "mcp_url": "https://old/mcp", "mcp_server_name": "old", "environment_card": card, "environment_card_url": card_url})
    assert (record.mcp_url, record.mcp_server_name) == ("https://old/mcp", "old")


def test_a_card_without_a_name_leaves_the_server_name_unset():
    record = DeployedEnv(env_id="e", env_version=1, environment_card_url=_CARD_URL, environment_card={"additionalInterfaces": []})
    assert (record.mcp_url, record.mcp_server_name) == ("https://gw.example/sb/x/mcp", None)


@pytest.mark.parametrize("doc, cls, provider_type", [
    ({"env_provider_type": "gateway", "gateway_url": "https://gw", "sandbox_id": "sb"}, DeployedGatewayEnv, "gateway"),
    ({"env_provider_type": "gateway", "sandbox_id": "sb"}, DeployedGatewayEnv, "gateway"),
    ({"gateway_url": "https://gw", "sandbox_id": "sb"}, DeployedGatewayEnv, "gateway"),
    ({"env_provider_type": None, "gateway_url": "https://gw", "sandbox_id": "sb"}, DeployedGatewayEnv, "gateway"),
    ({"env_provider_type": "newer", "gateway_url": "https://gw", "sandbox_id": "sb"}, DeployedGatewayEnv, "newer"),
    ({"env_provider_type": "newer", "sandbox_id": "sb"}, DeployedSandboxEnv, "newer"),
    ({"sandbox_id": "sb"}, DeployedSandboxEnv, None),
    ({"gateway_url": None, "db_web_url": "https://pg/", "sandbox_id": "sb"}, DeployedGatewayEnv, "gateway"),
    ({}, DeployedSandboxEnv, None),
    ({"env_provider_type": "newer"}, DeployedEnv, "newer"),
], ids=["gateway", "gateway-type-without-its-url", "before-the-type", "null-type", "newer-type-with-a-gateway", "newer-type-with-a-sandbox",
        "sandbox-shape", "gateway-data-without-its-url", "untyped-without-a-sandbox", "newer-type-without-a-sandbox"])
def test_a_stored_record_loads_as_its_type_or_else_its_shape_keeping_every_field(doc, cls, provider_type):
    record = DeployedEnv.from_dict({"env_id": "e", "env_version": 1, "mcp_url": "https://m/mcp", "unknown_key": 1, **doc})
    assert (type(record), record.env_provider_type, record.mcp_url) == (cls, provider_type, "https://m/mcp")
    assert {k: getattr(record, k) for k in ("gateway_url", "sandbox_id") if k in doc} == {k: doc[k] for k in ("gateway_url", "sandbox_id") if k in doc}


def test_a_gateway_record_is_filled_the_way_it_always_was():
    record = DeployedEnv.from_dict({"env_id": "e", "env_version": 1, "mcp_url": "https://m/mcp", "gateway_url": "https://gw", "sandbox_id": "sb",
                                    "sandbox_ids": None, "env_state_instance_ids": None})
    assert (record.gateway_mode, record.sandbox_ids, record.env_state_instance_ids, record.db_web_url) == ("performance", {}, [], None)


@pytest.mark.parametrize("record", [
    DeployedEnv(env_id="e", env_version=1, env_provider_type="newer", environment_card_url=_CARD_URL, environment_card=_CARD, instance_id="i1"),
    DeployedSandboxEnv(env_id="e", env_version=1, env_provider_type="server", sandbox_id="srv", sandbox_type="modal",
                       sandbox_ids={"mcp_server": {"slack": "srv"}}, environment_card_url=_CARD_URL, environment_card=_CARD),
    DeployedGatewayEnv(env_id="e", env_version=1, gateway_url="https://gw.example/sb/x", gateway_mode="consistent", sandbox_id="gw",
                       sandbox_type="modal_vm", sandbox_ids={"gateway_server": "gw"}, db_web_url="https://pg/", db_mcp_url="https://dm/mcp",
                       website_frontend_urls={"shop": "https://gw/shop/"}, vnc_url="https://vnc", env_state_instance_ids=["st-1"],
                       environment_card_url=_CARD_URL, environment_card=_CARD, environment_card_read_at_utc="2026-09-28T00:00:00+00:00",
                       metadata={"k": "v"}, instance_id="i1", created_at_utc="c", expires_at_utc="x"),
], ids=["base", "sandbox", "gateway"])
def test_a_record_round_trips_through_its_stored_form(record):
    assert DeployedEnv.from_dict(dataclasses.asdict(record)) == record


@pytest.mark.parametrize("cls", [DeployedEnv, DeployedSandboxEnv, DeployedGatewayEnv])
def test_each_record_class_loads_every_one_of_its_fields(cls):
    assert set(cls._fields_from({"env_id": "e", "env_version": 1})) == {f.name for f in dataclasses.fields(cls)}


def test_the_gateway_only_fields_are_the_gateway_classs_own():
    own = {f.name for f in dataclasses.fields(DeployedGatewayEnv)} - {f.name for f in dataclasses.fields(DeployedSandboxEnv)}
    assert set(_GATEWAY_ONLY_FIELDS) == own - {"gateway_mode", "env_provider_type"}


def test_only_a_gateway_record_has_a_gateway_url():
    gateway = DeployedGatewayEnv(env_id="e", env_version=1, gateway_url="https://gw", sandbox_id="gw")
    bare = DeployedSandboxEnv(env_id="e", env_version=1, sandbox_id="srv")
    assert (gateway_url_of(gateway), gateway_url_of(bare), gateway_url_of(None)) == ("https://gw", None, None)
    assert require_gateway_url(gateway, "step") == "https://gw"
    with pytest.raises(EnvNeedsGateway, match="^step needs a gateway; env 'e' was deployed without one$") as raised:
        require_gateway_url(bare, "step")
    assert (raised.value.what, raised.value.env_id) == ("step", "e")


def test_only_a_record_in_our_sandboxes_has_a_sandbox():
    contained = DeployedSandboxEnv(env_id="e", env_version=1, sandbox_id="srv")
    hosted = DeployedEnv(env_id="e", env_version=1, env_provider_type="hosted")
    assert require_sandbox(contained, "run_code") is contained
    with pytest.raises(EnvNeedsSandbox, match="^run_code needs a sandbox; env 'e' runs outside agent-env's sandboxes$") as raised:
        require_sandbox(hosted, "run_code")
    assert (raised.value.what, raised.value.env_id) == ("run_code", "e")
