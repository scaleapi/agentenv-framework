"""``_capture_universe_state`` publishes unsigned object URLs to
``context.metadata['snapshot_json_url']``, and reads each child env's route from the stored env card."""
from __future__ import annotations

import json

import pytest

import agent_env.artifact as artifact_mod
import agent_env.config as config_mod
from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedGatewayEnv
from agent_env.store import set_object_store
from agent_env.store.object_store import DEFAULT_CONTENT_TYPE
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import snapshot_agent_state as mod
from agentenv_protocol import client as protocol_v1
from tst.unit.event_loop_probe import on_event_loop
from tst.unit.store.fakes import ConfiguredObjectStore, FakeObjectStore

_PREFIX = "fake://artifact-bucket/agent_snapshots/oc_post_run_workspace_T/3-deadbeef/"


class _Store(FakeObjectStore):
    """Records each upload's key, and fails if asked to sign: the capture publishes plain object URLs."""

    def __init__(self):
        super().__init__(root="artifact-bucket")
        self.puts: list[str] = []

    def put(self, key, data, content_type=DEFAULT_CONTENT_TYPE, allow_overwrite=False):
        self.puts.append(key)
        return super().put(key, data, content_type, allow_overwrite)

    def signed_get_url(self, *a, **k):
        raise AssertionError(
            "snapshot_agent_state must not presign — publish the object URL and let the "
            "consumer sign it"
        )


class _StubServiceArtifact:
    def __init__(self, name):
        self.environment_name = name


class _StubUniverse:
    def get_environment_artifacts(self):
        return [_StubServiceArtifact("calendar"), _StubServiceArtifact("contacts")]


def _step():
    return mod.SnapshotAgentStateTaskStep(
        id="t-capture",
        version=None,
        artifact_id="oc_post_run_workspace_T",
        prompt_id="main",
        agent_name="openclaw-cli",
        env_id="env-1",
        universe_artifact_id="uni-1",
    )


def _ctx():
    ctx = TaskStepContext()
    ctx.deployed_envs = [
        DeployedGatewayEnv(
            env_id="env-1",
            env_version=1,
            gateway_url="https://gw",
            mcp_url="m",
            db_web_url=None,
            sandbox_id="sb-1",
        )
    ]
    return ctx


@pytest.mark.asyncio
async def test_capture_universe_state_publishes_unsigned_object_urls(monkeypatch):
    store = _Store()

    import agent_env.artifact as artifact_mod

    class _UniClass:
        @staticmethod
        def get(_id):
            return _StubUniverse()

    monkeypatch.setattr(artifact_mod, "EnvironmentUniverseArtifact", _UniClass)

    import agent_env.config as config_mod

    class _Cfg:
        def get_object_store_at(self, object_url):
            return store

    monkeypatch.setattr(config_mod, "get_config", lambda: _Cfg())

    async def _supports_v1(_base):
        return False

    monkeypatch.setattr(protocol_v1, "supports_v1", _supports_v1)

    async def _export_state(_gw, name):
        return {"service": name, "rows": 1}

    monkeypatch.setattr(legacy_protocol, "export_state", _export_state)

    ctx = _ctx()
    await _step()._capture_universe_state(ctx, _PREFIX)

    published = json.loads(ctx.metadata["snapshot_json_url"])
    assert published == {
        "calendar": f"{_PREFIX}services/calendar.json",
        "contacts": f"{_PREFIX}services/contacts.json",
    }
    assert {key.rsplit("/", 1)[-1] for key in store.puts} == {"calendar.json", "contacts.json"}


@pytest.mark.asyncio
async def test_each_service_state_uploads_off_the_event_loop(monkeypatch):
    on_loop: list[bool] = []

    class _LoopCheckingStore(_Store):
        def put(self, *a, **kw):
            on_loop.append(on_event_loop())
            return super().put(*a, **kw)

    store = _LoopCheckingStore()
    monkeypatch.setattr(artifact_mod, "EnvironmentUniverseArtifact", type("U", (), {"get": staticmethod(lambda _id: _StubUniverse())}))
    monkeypatch.setattr(config_mod, "get_config", lambda: type("Cfg", (), {"get_object_store_at": lambda self, object_url: store})())

    async def _no_v1(_base):
        return False

    async def _export_state(_gw, name):
        return {"service": name}

    monkeypatch.setattr(protocol_v1, "supports_v1", _no_v1)
    monkeypatch.setattr(legacy_protocol, "export_state", _export_state)

    await _step()._capture_universe_state(_ctx(), _PREFIX)

    assert on_loop == [False, False]


@pytest.mark.asyncio
async def test_capture_reads_child_envs_on_the_stored_card_and_takes_legacy_for_the_rest(monkeypatch):
    """calendar is on the card, so data/get goes to the card's address with no probe; contacts serves no card, so it goes legacy."""
    import agent_env.artifact as artifact_mod
    import agent_env.config as config_mod
    from agentenv_protocol import DataPart
    from agentenv_protocol.client import GetDataResponse

    store = _Store()
    monkeypatch.setattr(artifact_mod, "EnvironmentUniverseArtifact", type("U", (), {"get": staticmethod(lambda _id: _StubUniverse())}))
    monkeypatch.setattr(config_mod, "get_config", lambda: type("Cfg", (), {"get_object_store_at": lambda self, object_url: store})())

    async def _no_probe(_base):
        raise AssertionError("a stored card means no live probe")

    got, legacy = [], []

    async def _get_data(base, timeout=30):
        got.append(base)
        return GetDataResponse(parts=[DataPart(data={"rows": 1})])

    async def _export_state(gw, name):
        legacy.append((gw, name))
        return {"rows": 1}

    monkeypatch.setattr(protocol_v1, "supports_v1", _no_probe)
    monkeypatch.setattr(protocol_v1, "get_data", _get_data)
    monkeypatch.setattr(legacy_protocol, "export_state", _export_state)

    ctx = _carded_ctx(await _card("mcp-calendar"))

    await _step()._capture_universe_state(ctx, _PREFIX)

    assert got == ["https://sandbox.example/sb-1/svc/mcp-calendar"]
    assert legacy == [("https://gw", "contacts")]
    assert set(json.loads(ctx.metadata["snapshot_json_url"])) == {"calendar", "contacts"}


@pytest.mark.asyncio
async def test_a_service_that_exports_a_file_bundle_is_captured_from_its_export_state(monkeypatch):
    """calendar answers data/get with a bundle of its database, which is not JSON state: its state is read from
    /export-state at the card's address rather than skipped. contacts answers with its state."""
    import httpx
    from agentenv_protocol import DataPart, uploaded_file_part
    from agentenv_protocol.client import GetDataResponse

    store = _Store()
    monkeypatch.setattr(artifact_mod, "EnvironmentUniverseArtifact", type("U", (), {"get": staticmethod(lambda _id: _StubUniverse())}))
    monkeypatch.setattr(config_mod, "get_config", lambda: type("Cfg", (), {"get_object_store_at": lambda self, object_url: store})())

    async def _get_data(base, timeout=30):
        if base.endswith("/mcp-calendar"):
            return GetDataResponse(parts=[uploaded_file_part("calendar.zip", name="calendar.zip", mime_type="application/zip")])
        return GetDataResponse(parts=[DataPart(data={"contacts": 4})])

    real = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://sandbox.example/sb-1/svc/mcp-calendar/export-state":
            return httpx.Response(200, json={"events": 7})
        return httpx.Response(404)

    monkeypatch.setattr(protocol_v1, "get_data", _get_data)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))

    ctx = _carded_ctx(await _card("mcp-calendar", "mcp-contacts"))
    await _step()._capture_universe_state(ctx, _PREFIX)

    bodies = {url.rsplit("/", 1)[-1]: json.loads(data) for url, data in store.objects.items()}
    assert bodies == {"calendar.json": {"events": 7}, "contacts.json": {"contacts": 4}}
    assert set(json.loads(ctx.metadata["snapshot_json_url"])) == {"calendar", "contacts"}


async def _card(*keys: str) -> dict:
    """The real gateway's composed card over child envs at the given gateway keys (`mcp-{name}` or `{name}`)."""
    from agent_env.env.gateway.gateway import Gateway

    gw = Gateway(host="127.0.0.1", port=0, server_name="env1234", internal_mcp_servers=[],
                 rest_proxy_urls={key: f"http://{key}:18765" for key in keys})

    async def fetch(client, key, base_url):
        return gw._rewrite_child_card(key, {"name": key.removeprefix("mcp-"), "url": "/agentenv"})

    gw._fetch_child_card = fetch
    return json.loads((await gw._serve_env_card()).body)


def _carded_ctx(card: dict) -> TaskStepContext:
    """`_ctx()` with a stored card whose address differs from `gateway_url`."""
    from agent_env.env.gateway.constants import WELL_KNOWN_PATH

    ctx = _ctx()
    ctx.deployed_envs[0].environment_card_url = f"https://sandbox.example/sb-1{WELL_KNOWN_PATH}"
    ctx.deployed_envs[0].environment_card = card
    return ctx


@pytest.mark.asyncio
async def test_services_land_beside_a_capture_the_local_store_holds(monkeypatch, cli_routing):
    set_object_store(ConfiguredObjectStore())
    local = config_mod.get_config().get_object_store_for("@local/~/t")
    prefix = local.object_url("agent_snapshots/oc/3-deadbeef/")
    monkeypatch.setattr(artifact_mod, "EnvironmentUniverseArtifact", type("U", (), {"get": staticmethod(lambda _id: _StubUniverse())}))

    async def _state(_deployed, _gateway, name):
        return {"service": name}

    monkeypatch.setattr(legacy_protocol, "service_state", _state)
    ctx = _ctx()
    await _step()._capture_universe_state(ctx, prefix)

    published = json.loads(ctx.metadata["snapshot_json_url"])
    assert published == {
        name: local.object_url(f"agent_snapshots/oc/3-deadbeef/services/{name}.json") for name in ("calendar", "contacts")
    }
    assert json.loads(local.get(published["calendar"])) == {"service": "calendar"}
