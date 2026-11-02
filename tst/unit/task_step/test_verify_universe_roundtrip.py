"""VerifyUniverseLoadExportRoundtripStep exports each child env at the base the stored env card gives it, MCP server or
website, and a child env the card doesn't list through legacy /export-state. An environment of several files, not one
JSON state, is compared export to export only, as a non-JSON one is. Only the gateway HTTP boundary is mocked."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, call

import httpx
import pytest

from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact, FileArtifact, FileArtifactUniverse
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.task_step.context import TaskStepContext
from tst.unit.event_loop_probe import on_event_loop
from agent_env.task_step.task_steps.multienv_validator.verify_universe_roundtrip import VerifyUniverseLoadExportRoundtripStep

_GATEWAY = "https://gw"
# The card's address differs from gateway_url, so a URL built from the card is told apart from one built from /svc/... itself.
_CARD = "https://sandbox.example/sandbox/sb-1-18765"


@pytest.mark.asyncio
async def test_exports_listed_child_envs_at_their_card_base_and_an_unlisted_one_over_export_state(monkeypatch):
    sent = _mock_gateway(monkeypatch)
    record = _record(await _card(["mcp-items", "webitems"]))

    exported = await VerifyUniverseLoadExportRoundtripStep._export_all(record, ["items", "webitems", "email"])

    assert [(r.method, str(r.url)) for r in sent] == [
        ("POST", f"{_CARD}/svc/mcp-items/agentenv"),
        ("POST", f"{_CARD}/svc/webitems/agentenv"),
        ("GET", f"{_GATEWAY}/svc/mcp-email/export-state"),
    ]
    assert [json.loads(r.content)["method"] for r in sent[:2]] == ["data/get", "data/get"]
    assert exported == {"items": {"items": ["a"]}, "webitems": {"items": ["a"]}, "email": {"emails": []}}


@pytest.mark.asyncio
async def test_a_name_an_mcp_server_and_a_website_share_exports_the_mcp_server(monkeypatch):
    sent = _mock_gateway(monkeypatch)

    await VerifyUniverseLoadExportRoundtripStep._export_all(_record(await _card(["mcp-items", "items"])), ["items"])

    assert [(r.method, str(r.url)) for r in sent] == [("POST", f"{_CARD}/svc/mcp-items/agentenv")]


@pytest.mark.asyncio
async def test_a_record_without_a_stored_card_keeps_the_live_probe(monkeypatch):
    sent = _mock_gateway(monkeypatch)

    await VerifyUniverseLoadExportRoundtripStep._export_all(_record(None), ["items", "email"])

    assert [(r.method, str(r.url)) for r in sent] == [
        ("GET", f"{_GATEWAY}/svc/mcp-items{WELL_KNOWN_PATH}"),
        ("POST", f"{_GATEWAY}/svc/mcp-items/agentenv"),
        ("GET", f"{_GATEWAY}/svc/mcp-email{WELL_KNOWN_PATH}"),
        ("GET", f"{_GATEWAY}/svc/mcp-email/export-state"),
    ]


@pytest.mark.asyncio
async def test_both_export_rounds_read_the_deployed_record(monkeypatch):
    record = _record(await _card(["mcp-items"]))
    export_all = AsyncMock(return_value={"items": {"items": []}})
    monkeypatch.setattr(VerifyUniverseLoadExportRoundtripStep, "_export_all", export_all)
    monkeypatch.setattr(VerifyUniverseLoadExportRoundtripStep, "_create_universe_artifact", lambda *a: MagicMock(id="u-export1"))
    monkeypatch.setattr("agent_env.env.env.Env.get", lambda *a: _FakeEnv())
    universe = MagicMock(id="u", version=1)
    universe.get_environment_artifacts.return_value = [_environment("items", {"items.json": b'{"items": []}'})]
    monkeypatch.setattr("agent_env.artifact.EnvironmentUniverseArtifact.get", lambda *a: universe)

    step = VerifyUniverseLoadExportRoundtripStep(id="rt", version=None, env_id="env-x", universe_artifact_id="u", persist_result=False)
    ctx = await step.execute(TaskStepContext(deployed_envs=[record]))

    assert export_all.await_args_list == [call(record, ["items"]), call(record, ["items"])]
    assert ctx.metadata["verifications"]["universe_compatibility"]["compatible"] is True


@pytest.mark.asyncio
async def test_an_environment_of_several_files_is_compared_export_to_export_only(monkeypatch):
    record = _record(await _card(["mcp-desk"]))
    monkeypatch.setattr(VerifyUniverseLoadExportRoundtripStep, "_export_all", AsyncMock(return_value={"desk": {"files": ["a"]}}))
    monkeypatch.setattr(VerifyUniverseLoadExportRoundtripStep, "_create_universe_artifact", lambda *a: MagicMock(id="u-export1"))
    monkeypatch.setattr("agent_env.env.env.Env.get", lambda *a: _FakeEnv())
    universe = MagicMock(id="u", version=1)
    desk = _environment("desk", {"Desktop/a.txt": b"not json", "b.txt": b"{}"})
    universe.get_environment_artifacts.return_value = [desk]
    monkeypatch.setattr("agent_env.artifact.EnvironmentUniverseArtifact.get", lambda *a: universe)

    step = VerifyUniverseLoadExportRoundtripStep(id="rt", version=None, env_id="env-x", universe_artifact_id="u", persist_result=False)
    ctx = await step.execute(TaskStepContext(deployed_envs=[record]))

    verdict = ctx.metadata["verifications"]["universe_compatibility"]
    assert verdict["compatible"] is True and verdict["environments"]["desk"]["issues"] == []
    assert all(not file.load.called for file in desk.get_file_artifacts().values())


@pytest.mark.asyncio
async def test_the_universe_files_are_read_and_the_exports_uploaded_off_the_event_loop(monkeypatch):
    reads: list[bool] = []
    uploads: list[bool] = []

    def load():
        reads.append(on_event_loop())
        return b'{"items": []}'

    def put(**kwargs):
        uploads.append(on_event_loop())
        return MagicMock(id=kwargs["id"])

    record = _record(await _card(["mcp-items"]))
    monkeypatch.setattr(VerifyUniverseLoadExportRoundtripStep, "_export_all", AsyncMock(return_value={"items": {"items": []}}))
    monkeypatch.setattr("agent_env.env.env.Env.get", lambda *a: _FakeEnv())
    monkeypatch.setattr(FileArtifact, "put", staticmethod(put))
    for cls in (EnvironmentArtifact, EnvironmentUniverseArtifact, FileArtifactUniverse):
        monkeypatch.setattr(cls, "put", staticmethod(lambda **kwargs: MagicMock(id=kwargs["id"])))
    universe = MagicMock(id="u", version=1)
    environment = MagicMock(environment_name="items")
    environment.get_file_artifacts.return_value = {"items.json": MagicMock(load=load)}
    universe.get_environment_artifacts.return_value = [environment]
    monkeypatch.setattr(EnvironmentUniverseArtifact, "get", lambda *a: universe)

    step = VerifyUniverseLoadExportRoundtripStep(
        id="rt", version=None, env_id="env-x", universe_artifact_id="u", persist_result=False, emit_file_artifact_universe=True,
    )
    await step.execute(TaskStepContext(deployed_envs=[record]))

    assert reads == [False] and uploads == [False, False]


def _environment(name: str, contents: dict[str, bytes]) -> MagicMock:
    """An environment artifact of ``contents``' files by path."""
    environment = MagicMock(environment_name=name)
    environment.get_file_artifacts.return_value = {path: MagicMock(**{"load.return_value": data})
                                                   for path, data in contents.items()}
    return environment


async def _card(keys: list[str]) -> dict:
    """The real gateway's composed card, with a v1 backing server behind each key."""
    from agent_env.env.gateway.gateway import Gateway

    gw = Gateway(host="127.0.0.1", port=0, server_name="env1234", internal_mcp_servers=[],
                 rest_proxy_urls={key: f"http://{key}:18765" for key in keys})

    async def fetch(client, key, base_url):
        return gw._rewrite_child_card(key, {"name": key.removeprefix("mcp-"), "url": "/agentenv"})

    gw._fetch_child_card = fetch
    return json.loads((await gw._serve_env_card()).body)


def _record(card: dict | None) -> DeployedEnv:
    return DeployedGatewayEnv(env_id="env-x", env_version=1, gateway_url=_GATEWAY, mcp_url=f"{_GATEWAY}/mcp", db_web_url=None, sandbox_id="sb",
                       environment_card_url=f"{_CARD}{WELL_KNOWN_PATH}", environment_card=card)


def _mock_gateway(monkeypatch) -> list[httpx.Request]:
    """A gateway fronting v1 child envs and email (legacy, serves no card); return the requests."""
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        path = request.url.path
        if path.endswith(WELL_KNOWN_PATH):
            return httpx.Response(200, json={"name": "items"}) if "mcp-items" in path else httpx.Response(404)
        if path.endswith("/agentenv"):
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"parts": [{"kind": "data", "data": {"items": ["a"]}}]}})
        return httpx.Response(200, json={"emails": []})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent


class _FakeEnv:
    @classmethod
    async def from_deployed_env(cls, deployed):
        return cls()

    async def load_environment_universe_artifact(self, universe):
        return None
