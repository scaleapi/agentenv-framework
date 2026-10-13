"""data/get through a write_namespace grant: the card advertisement, the handler binding, the client
call, and the answer that names where the export was uploaded."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Route

from agentenv_protocol import (
    DATA_OBJECTS_EXTENSION_URI,
    AgentEnvFastMCPApplication,
    DataPart,
    EnvironmentCapabilities,
    EnvironmentCard,
    EnvironmentExtension,
    FilePart,
    GetDataResponse,
    get_data,
    uploaded_file_part,
    uploaded_object_path,
)
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.transfers import HttpPostPolicyGrant, NamespaceUploader, WriteNamespaceGrant

_SIGNATURE = "policy-signature-secret"


def _grant(*, max_object_bytes: int = 1024) -> WriteNamespaceGrant:
    return WriteNamespaceGrant(
        root_path="snapshots/run-1",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        max_objects=1,
        max_object_bytes=max_object_bytes,
        max_total_bytes=max_object_bytes,
        write=HttpPostPolicyGrant(
            kind="http-post-policy",
            url="https://objects.example.test/upload",
            fields={"policy": _SIGNATURE},
            path_field="key",
            file_field="file",
        ),
    )


class _FakeMCP:
    def __init__(self) -> None:
        self.routes: list[Route] = []
        self.tools: dict[str, tuple] = {}
        self.settings = SimpleNamespace(streamable_http_path="/mcp")
        self._tool_manager = SimpleNamespace(get_tool=self.tools.get, call_tool=self._call_tool)

    async def _call_tool(self, name: str, arguments: dict, *_: object, **__: object):
        result = self.tools[name][1](**arguments)
        return await result if inspect.isawaitable(result) else result

    def custom_route(self, path: str, methods: list[str]):
        def deco(fn):
            self.routes.append(Route(path, fn, methods=methods))
            return fn
        return deco

    def tool(self, name: str | None = None, description: str | None = None):
        def deco(fn):
            self.tools[name or fn.__name__] = (description, fn)
            return fn
        return deco


class _InlineHandler:
    def __init__(self) -> None:
        self.calls = 0

    @get_data
    async def _state(self) -> list:
        self.calls += 1
        return [DataPart(data={"rows": 1})]


class _UploadingHandler:
    def __init__(self) -> None:
        self.grants: list[WriteNamespaceGrant | None] = []

    @get_data
    async def _state(self, write_namespace: WriteNamespaceGrant | None = None) -> list:
        self.grants.append(write_namespace)
        if write_namespace is None:
            return [DataPart(data={"rows": 1})]
        await NamespaceUploader(write_namespace).upload("slack.zip", b"PK\x03\x04bundle")
        return [uploaded_file_part("slack.zip", name="slack.zip", mime_type="application/zip")]


def _app(handler, card: EnvironmentCard | None = None) -> tuple[AgentEnvFastMCPApplication, Starlette]:
    mcp = _FakeMCP()
    application = AgentEnvFastMCPApplication(environment_card=card or EnvironmentCard(name="slack"), handler=handler)
    application.add_routes_to_app(mcp)
    return application, Starlette(routes=mcp.routes)


def _rpc(params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": "data/get", "params": params or {}}


async def _post(app: Starlette, body: dict) -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        return (await client.post("/agentenv", json=body)).json()


def _uris(card: EnvironmentCard) -> list[str]:
    return [e.uri for e in card.capabilities.extensions or []]


def test_a_handler_that_takes_a_write_namespace_is_advertised():
    application, _ = _app(_UploadingHandler())
    (extension,) = application.environment_card.capabilities.extensions
    assert extension.uri == DATA_OBJECTS_EXTENSION_URI
    assert extension.params is None


def test_a_handler_that_takes_none_is_not():
    application, _ = _app(_InlineHandler())
    assert _uris(application.environment_card) == []


def test_a_card_declared_data_objects_extension_wins():
    declared = EnvironmentExtension(uri=DATA_OBJECTS_EXTENSION_URI, description="declared")
    card = EnvironmentCard(name="slack", capabilities=EnvironmentCapabilities(extensions=[declared]))
    application, _ = _app(_UploadingHandler(), card)
    assert application.environment_card.capabilities.extensions == [declared]


@pytest.mark.asyncio
async def test_the_handler_gets_the_grant_sent_or_none():
    handler = _UploadingHandler()
    _, app = _app(handler)
    grant = _grant()

    await _post(app, _rpc())
    await _post(app, _rpc({"write_namespace": grant.model_dump(mode="json")}))

    assert handler.grants == [None, grant]


@pytest.mark.asyncio
async def test_a_handler_that_takes_no_grant_ignores_one_sent():
    handler = _InlineHandler()
    _, app = _app(handler)

    body = await _post(app, _rpc({"write_namespace": _grant().model_dump(mode="json")}))

    assert body["result"]["parts"] == [{"kind": "data", "data": {"rows": 1}}]
    assert handler.calls == 1


@pytest.mark.asyncio
async def test_an_invalid_grant_is_refused_without_echoing_it():
    handler = _UploadingHandler()
    _, app = _app(handler)
    sent = _grant().model_dump(mode="json")
    sent["root_path"] = "../elsewhere"

    body = await _post(app, _rpc({"write_namespace": sent}))

    assert body["error"]["code"] == -32602
    assert "root_path" in body["error"]["data"]["detail"]
    assert _SIGNATURE not in json.dumps(body)
    assert handler.grants == []


@pytest.mark.asyncio
async def test_params_given_by_position_are_refused():
    handler = _UploadingHandler()
    _, app = _app(handler)

    body = await _post(app, {"jsonrpc": "2.0", "id": 1, "method": "data/get", "params": [_grant().model_dump(mode="json")]})

    assert body["error"]["code"] == -32602
    assert _SIGNATURE not in json.dumps(body)
    assert handler.grants == []


@pytest.mark.asyncio
async def test_get_data_sends_no_params_without_a_grant_and_the_grant_with_one(monkeypatch):
    sent: list[dict] = []
    real = httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"parts": []}})

    monkeypatch.setattr(protocol_v1.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handle)))

    grant = _grant()
    await protocol_v1.get_data("http://env")
    await protocol_v1.get_data("http://env", write_namespace=grant)

    assert sent[0]["params"] == {}
    assert WriteNamespaceGrant.model_validate(sent[1]["params"]["write_namespace"]) == grant


@pytest.mark.asyncio
async def test_an_export_uploads_through_the_grant_and_the_answer_names_where(monkeypatch):
    """The real handler, uploader, client and reader together: the bundle is POSTed under the grant's root, and
    the answer names its path relative to it."""
    handler = _UploadingHandler()
    _, app = _app(handler)
    uploads: list[httpx.Request] = []

    def store(request: httpx.Request) -> httpx.Response:
        uploads.append(request)
        return httpx.Response(204)

    real_sync, real_async = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_sync(**{**kw, "transport": httpx.MockTransport(store)}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_async(transport=httpx.ASGITransport(app=app)))

    response = await protocol_v1.get_data("http://t", write_namespace=_grant())

    (part,) = response.parts
    assert uploaded_object_path(part) == "slack.zip"
    assert (part.file.name, part.file.mimeType) == ("slack.zip", "application/zip")
    (upload,) = uploads
    assert str(upload.url) == "https://objects.example.test/upload"
    assert b'name="key"\r\n\r\nsnapshots/run-1/slack.zip' in upload.content
    assert b"PK\x03\x04bundle" in upload.content


def test_the_uploaded_part_survives_the_wire():
    wire = GetDataResponse(parts=[uploaded_file_part("dir/gdrive.zip", name="gdrive.zip")]).model_dump(mode="json", exclude_none=True)
    (part,) = GetDataResponse.model_validate(json.loads(json.dumps(wire))).parts
    assert uploaded_object_path(part) == "dir/gdrive.zip"


@pytest.mark.parametrize("part", [
    FilePart(file={"uri": "export-snapshot", "name": "gdrive.zip"}),
    FilePart(file={"uri": "export-snapshot"}, metadata={"other": {"path": "x"}}),
    DataPart(data={"path": "x"}, metadata={DATA_OBJECTS_EXTENSION_URI: {"path": "x"}}),
], ids=["no-metadata", "other-metadata", "data-part"])
def test_a_part_that_names_no_upload_reads_as_none(part):
    assert uploaded_object_path(part) is None


@pytest.mark.parametrize("path", ["../escape.zip", "/abs.zip", "a//b.zip", "a/./b.zip", "a/", "", "a\\b.zip", None, 3])
def test_a_path_outside_the_root_is_refused(path):
    part = FilePart(file={"uri": "x"}, metadata={DATA_OBJECTS_EXTENSION_URI: {"path": path}})
    with pytest.raises(ValueError, match="not a normalized relative path"):
        uploaded_object_path(part)
    if isinstance(path, str):
        with pytest.raises(ValueError):
            uploaded_file_part(path)


def test_a_marker_that_is_not_an_object_is_refused():
    part = FilePart(file={"uri": "x"}, metadata={DATA_OBJECTS_EXTENSION_URI: "x.zip"})
    with pytest.raises(ValueError, match="not a normalized relative path"):
        uploaded_object_path(part)
