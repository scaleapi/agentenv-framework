"""Several methods per extension: handlers given ``op`` merge into one card entry with an endpoint per method, with
the environment's extension_params beside them; declaration-only extensions; served_extensions choosing what is
served; the refusals at mount."""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

pytest.importorskip("mcp.server.fastmcp")

from agentenv_protocol import (  # noqa: E402
    AgentEnvEnvironment,
    AgentEnvFastMCPApplication,
    EnvironmentCapabilities,
    EnvironmentCard,
    EnvironmentExtension,
    client,
    environment_card,
    extension,
)
from agentenv_protocol.agent_env_environment import _extension_slug  # noqa: E402

_DISPLAY = "urn:agentenv:computer-display/v1"
_POINTER = "urn:agentenv:computer-pointer/v1"


@environment_card(name="display")
class _Display(AgentEnvEnvironment):
    def __init__(self) -> None:
        self.size = (1280, 800, 1.0)

    def extension_params(self, uri: str) -> dict:
        return {"modes": [[1280, 800], [1920, 1080]]} if uri == _DISPLAY else {}

    def declared_extensions(self) -> list[EnvironmentExtension]:
        return [EnvironmentExtension(uri=_POINTER, description="Pointer tools.", params={"buttons": ["left"]})]

    @extension(_DISPLAY, op="get", method="GET", description="Display resolution and scale.")
    async def display_get(self) -> dict:
        width, height, scale = self.size
        return {"width": width, "height": height, "scale": scale}

    @extension(_DISPLAY, op="set")
    async def display_set(self, width: int, height: int, scale: float = 1.0) -> dict:
        self.size = (width, height, scale)
        return await self.display_get()

    @extension("urn:agentenv:legacy/v1")
    async def legacy(self) -> dict:
        return {"legacy": True}


def _card(handler, card: EnvironmentCard | None = None) -> EnvironmentCard:
    return AgentEnvFastMCPApplication(card or EnvironmentCard(name="t"), handler).environment_card


def _extensions(card: EnvironmentCard) -> dict:
    return {e.uri: e for e in card.capabilities.extensions}


def _http(env: AgentEnvEnvironment) -> httpx.AsyncClient:
    return _client(env.create_app().streamable_http_app())


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


@pytest.mark.parametrize("uri, slug", [
    ("urn:agentenv:computer-display/v1", "computer-display"),
    ("urn:agentenv:computer-display/v12", "computer-display"),
    ("urn:trycua:windows/v1", "trycua/windows"),
    ("urn:agentenv:unversioned", "unversioned"),
    ("urn:agentenv:mail/vnext", "mail/vnext"),
])
def test_extension_slug_drops_the_agentenv_namespace_and_the_version(uri, slug):
    assert _extension_slug(uri) == slug


def test_methods_sharing_a_uri_merge_into_one_card_entry_with_their_own_endpoints():
    extensions = _extensions(_card(_Display()))
    display = extensions[_DISPLAY]
    assert display.description == "Display resolution and scale."
    assert display.params["modes"] == [[1280, 800], [1920, 1080]] and "endpoint" not in display.params
    methods = display.params["methods"]
    assert set(methods) == {"get", "set"}
    assert methods["get"]["method"] == "GET" and methods["get"]["endpoint"] == "/agentenv/ext/computer-display/get"
    assert methods["set"]["method"] == "POST" and methods["set"]["endpoint"] == "/agentenv/ext/computer-display/set"
    assert methods["set"]["request"]["required"] == ["width", "height"]
    assert extensions[_POINTER] == EnvironmentExtension(uri=_POINTER, description="Pointer tools.", params={"buttons": ["left"]})
    assert extensions["urn:agentenv:legacy/v1"].params["endpoint"] == "/agentenv/ext/legacy"


@pytest.mark.asyncio
async def test_each_method_is_served_at_its_endpoint_and_the_client_reaches_it_by_name():
    app = _Display().create_app().streamable_http_app()
    async with _client(app) as http:
        card = (await http.get("/.well-known/agent-env.json")).json()
        with patch("agentenv_protocol.client.httpx.AsyncClient", return_value=_client(app)):
            changed = await client.invoke_extension("http://t", card, _DISPLAY, {"width": 1920, "height": 1080}, method="set")
        got = await http.get("/agentenv/ext/computer-display/get")
        legacy = await http.post("/agentenv/ext/legacy", json={})
    assert changed == {"width": 1920, "height": 1080, "scale": 1.0}
    assert got.json() == {"width": 1920, "height": 1080, "scale": 1.0} and legacy.json() == {"legacy": True}


@pytest.mark.asyncio
async def test_an_explicit_path_is_the_methods_endpoint():
    class _Paths(AgentEnvEnvironment):
        @extension(_DISPLAY, op="get", method="GET", path="/display")
        async def get(self) -> dict:
            return {"ok": True}

    env = _Paths()
    assert _extensions(_card(env))[_DISPLAY].params["methods"]["get"]["endpoint"] == "/display"
    async with _http(env) as http:
        assert (await http.get("/display")).json() == {"ok": True}


def test_a_card_declared_extension_wins_over_the_merged_one():
    card = EnvironmentCard(name="d", capabilities=EnvironmentCapabilities(extensions=[EnvironmentExtension(uri=_DISPLAY)]))
    assert [e for e in _card(_Display(), card).capabilities.extensions if e.uri == _DISPLAY] == [EnvironmentExtension(uri=_DISPLAY)]


def test_a_plain_handler_without_the_hooks_serves_methods_with_no_extra_params():
    class _Plain:
        @extension(_DISPLAY, op="get")
        async def get(self) -> dict:
            return {}

    assert _extensions(_card(_Plain()))[_DISPLAY].params == {
        "methods": {"get": {"method": "POST", "endpoint": "/agentenv/ext/computer-display/get",
                            "request": {"type": "object", "properties": {}}}}}


def test_params_are_refused_with_op():
    with pytest.raises(TypeError, match="extension_params"):
        extension(_DISPLAY, op="get", params={"methods": {}})


def test_two_handlers_for_one_method_are_refused():
    class _Dup:
        @extension(_DISPLAY, op="get", path="/a")
        async def a(self) -> dict:
            return {}

        @extension(_DISPLAY, op="get", path="/b")
        async def b(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="Multiple handlers for extension urn:agentenv:computer-display/v1 method get"):
        _card(_Dup())


@pytest.mark.parametrize("first, second", [("get", None), (None, "get"), (None, None)])
def test_a_handler_without_op_shares_its_uri_with_no_other(first, second):
    class _Mixed:
        @extension(_DISPLAY, op=first, path="/a")
        async def a(self) -> dict:
            return {}

        @extension(_DISPLAY, op=second, path="/b")
        async def b(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="Multiple handlers for extension urn:agentenv:computer-display/v1$"):
        _card(_Mixed())


def test_methods_disagreeing_on_description_are_refused():
    class _Described:
        @extension(_DISPLAY, op="get", description="one")
        async def a(self) -> dict:
            return {}

        @extension(_DISPLAY, op="set", description="two")
        async def b(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="declare different description values"):
        _card(_Described())


def test_an_extension_both_declared_and_served_is_refused():
    class _Both(AgentEnvEnvironment):
        def declared_extensions(self) -> list[EnvironmentExtension]:
            return [EnvironmentExtension(uri=_DISPLAY)]

        @extension(_DISPLAY, op="get")
        async def a(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="both declared and served"):
        _card(_Both())


class _ReadOnly(_Display):
    def served_extensions(self, extensions: list[EnvironmentExtension]) -> list[EnvironmentExtension]:
        served = []
        for e in extensions:
            if e.uri == _DISPLAY:
                served.append(e.model_copy(update={"params": {**e.params, "methods": {"get": e.params["methods"]["get"]}}}))
            elif e.uri != "urn:agentenv:legacy/v1":
                served.append(e)
        return served


@pytest.mark.asyncio
async def test_served_extensions_chooses_the_extensions_and_methods_served():
    env = _ReadOnly()
    extensions = _extensions(_card(env))
    assert set(extensions[_DISPLAY].params["methods"]) == {"get"} and extensions[_DISPLAY].params["modes"]
    assert "urn:agentenv:legacy/v1" not in extensions and _POINTER in extensions
    async with _http(env) as http:
        got = await http.get("/agentenv/ext/computer-display/get")
        unset = await http.post("/agentenv/ext/computer-display/set", json={"width": 1920, "height": 1080})
        legacy = await http.post("/agentenv/ext/legacy", json={})
    assert got.json()["width"] == 1280 and unset.status_code == 404 and legacy.status_code == 404


@pytest.mark.parametrize("chosen, error", [
    (lambda exts: [*exts, EnvironmentExtension(uri="urn:agentenv:ghost/v1")],
     r"extensions with no @extension handler: \['urn:agentenv:ghost/v1'\]"),
    (lambda exts: [e.model_copy(update={"params": {**e.params, "methods": {**e.params["methods"], "reset": {}}}})
                   if e.uri == _DISPLAY else e for e in exts],
     r"methods of urn:agentenv:computer-display/v1 with no @extension handler: \['reset'\]"),
])
def test_served_extensions_naming_an_unhandled_extension_or_method_is_refused(chosen, error):
    class _Bad(_Display):
        def served_extensions(self, extensions):
            return chosen(extensions)

    with pytest.raises(ValueError, match=error):
        _card(_Bad())
