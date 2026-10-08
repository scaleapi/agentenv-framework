"""snapshot_env hands a service that takes the data-objects form a grant to upload its bundle with,
and registers the bundle where the service put it; every other service exports as before."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI, uploaded_file_part

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.config import set_object_store
from agent_env.store.base import GrantUnavailableError
from agent_env.store.object_store import MIN_GRANT_LIFETIME_SECONDS
from agent_env.task_step.task_steps import snapshot_env as snapshot_mod
from agent_env.task_step.task_steps.snapshot_env import ENV_SNAPSHOT_LIMITS, SnapshotEnvTaskStep, _snapshot_upload
from tst.unit.store.fakes import FakeObjectStore
from tst.unit.task_step.test_snapshot_env import _CARD, _FakeMultiEnv, _card, _carded
from tst.util.granting_object_store import GrantingObjectStore

_DATA_OBJECTS = {"uri": DATA_OBJECTS_EXTENSION_URI}
_S3_CREDENTIALS = {"uri": "urn:agentenv:add-s3-credentials/v1",
                   "params": {"endpoint": "/agentenv/ext/add_s3_credentials", "methods": {"add_s3_credentials": {"method": "POST"}}}}
_BUNDLE = b"PK\x03\x04bundle"


class _UngrantingStore(GrantingObjectStore):
    def issue_upload_policy(self, prefix_url, *, max_object_bytes, expires_in):
        raise GrantUnavailableError("temporary credentials sign this store's grants")


@pytest.fixture
def store(local_stores, tmp_path):
    granting = GrantingObjectStore(str(tmp_path / "objects"))
    set_object_store(granting)
    return granting


@pytest.fixture
def pushes(monkeypatch):
    pushed: list[str] = []

    async def push(base_url, card, timeout_seconds):
        pushed.append(base_url)

    monkeypatch.setattr(snapshot_mod, "_push_s3_credentials", push)
    return pushed


async def _record(extensions: list, *, sandbox_type: str | None = "local"):
    return dataclasses.replace(_carded(await _card({"mcp-gdrive": extensions})), sandbox_type=sandbox_type)


def _serve(monkeypatch, answer):
    """Answer data/get with ``answer(params)``, recording every request; any other GET streams the bundle."""
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/agentenv"):
            params = json.loads(request.content)["params"]
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": answer(params)})
        return httpx.Response(200, content=_BUNDLE)

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent


def _uploading(store):
    """A service that uploads its bundle under the grant it is handed and names where."""
    def answer(params):
        root = params["write_namespace"]["root_path"]
        store.put(f"{root}/gdrive.zip", _BUNDLE, content_type="application/zip")
        return {"parts": [uploaded_file_part("gdrive.zip", name="gdrive.zip", mime_type="application/zip").model_dump(mode="json", exclude_none=True)]}
    return answer


def _rpc_params(sent: list[httpx.Request]) -> list[dict]:
    return [json.loads(r.content)["params"] for r in sent if r.url.path.endswith("/agentenv")]


@pytest.mark.asyncio
async def test_a_service_taking_the_form_uploads_through_a_grant_and_is_registered_where_it_put_it(store, pushes, monkeypatch):
    record = await _record([_DATA_OBJECTS, _S3_CREDENTIALS])
    sent = _serve(monkeypatch, _uploading(store))

    result = await SnapshotEnvTaskStep.snapshot_env_state(
        env=_FakeMultiEnv(["gdrive"]), gateway_url="https://gw-1", snapshot_id="snap-g", deployed=record,
    )

    (params,) = _rpc_params(sent)
    grant = params["write_namespace"]
    assert grant["root_path"].startswith("agentenv-snapshots/")
    assert (grant["max_objects"], grant["max_object_bytes"]) == (1, ENV_SNAPSHOT_LIMITS.max_object_bytes)
    assert datetime.fromisoformat(grant["expires_at"]) >= datetime.now(UTC) + timedelta(seconds=MIN_GRANT_LIFETIME_SECONDS - 60)
    assert pushes == [_CARD]  # still pushed, for a bundle too large for the grant
    assert [str(r.url) for r in sent] == [f"{_CARD}/svc/mcp-gdrive/agentenv"]
    (bundle,) = EnvironmentUniverseArtifact.get(result.environment_universe_artifact_id).get_file_artifacts().values()
    assert bundle.object_url == store.object_url(f"{grant['root_path']}/gdrive.zip")
    assert (bundle.filename, bundle.content_type) == ("gdrive.zip", "application/zip")
    assert bundle.load() == _BUNDLE


@pytest.mark.asyncio
async def test_each_export_gets_a_namespace_of_its_own(store, pushes, monkeypatch):
    record = await _record([_DATA_OBJECTS])
    sent = _serve(monkeypatch, _uploading(store))

    for snapshot_id in ("snap-1", "snap-1"):
        await SnapshotEnvTaskStep.snapshot_env_state(
            env=_FakeMultiEnv(["gdrive"]), gateway_url="https://gw-1", snapshot_id=snapshot_id, deployed=record,
        )

    first, second = (params["write_namespace"]["root_path"] for params in _rpc_params(sent))
    assert first != second


@pytest.mark.asyncio
@pytest.mark.parametrize("extensions, store_cls, sandbox_type", [
    ([_S3_CREDENTIALS], GrantingObjectStore, "local"),
    ([_DATA_OBJECTS, _S3_CREDENTIALS], None, "local"),
    ([_DATA_OBJECTS, _S3_CREDENTIALS], GrantingObjectStore, "modal"),
    ([_DATA_OBJECTS, _S3_CREDENTIALS], GrantingObjectStore, None),
    ([_DATA_OBJECTS, _S3_CREDENTIALS], _UngrantingStore, "local"),
], ids=["form-not-advertised", "store-issues-no-grants", "grants-do-not-reach-the-sandbox", "sandbox-unknown", "grant-unavailable"])
async def test_otherwise_the_export_is_as_before(local_stores, tmp_path, pushes, monkeypatch, extensions, store_cls, sandbox_type):
    set_object_store(FakeObjectStore() if store_cls is None else store_cls(str(tmp_path / "objects")))
    record = await _record(extensions, sandbox_type=sandbox_type)
    sent = _serve(monkeypatch, lambda params: {"parts": [{"kind": "data", "data": {"rows": 1}}]})

    out = tmp_path / "export-tmp"
    suffix = await SnapshotEnvTaskStep._export_environment_to_file("https://gw-1", "gdrive", str(out), 30, record)

    assert suffix == ".json"
    assert json.loads(out.read_text()) == {"rows": 1}
    assert _rpc_params(sent) == [{}]
    assert pushes == [_CARD]


@pytest.mark.asyncio
async def test_a_service_that_does_not_use_the_grant_is_exported_as_before(store, pushes, monkeypatch, tmp_path):
    record = await _record([_DATA_OBJECTS])
    sent = _serve(monkeypatch, lambda params: {"parts": [{"kind": "file", "file": {"uri": "export-snapshot", "name": "gdrive.zip"}}]})

    out = tmp_path / "export-tmp"
    suffix = await SnapshotEnvTaskStep._export_environment_to_file("https://gw-1", "gdrive", str(out), 30, record)

    assert suffix == ".zip"
    assert out.read_bytes() == _BUNDLE
    assert "write_namespace" in _rpc_params(sent)[0]
    assert [str(r.url) for r in sent][-1] == f"{_CARD}/svc/mcp-gdrive/export-snapshot"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../elsewhere/gdrive.zip", "/gdrive.zip", "a/../../gdrive.zip"])
async def test_an_upload_named_outside_the_namespace_fails_the_service(store, pushes, monkeypatch, tmp_path, path):
    record = await _record([_DATA_OBJECTS])
    marked = {"kind": "file", "file": {"uri": path, "name": "gdrive.zip"}, "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": path}}}
    _serve(monkeypatch, lambda params: {"parts": [marked]})

    with pytest.raises(ValueError, match="not a normalized relative path"):
        await SnapshotEnvTaskStep._export_environment_to_file("https://gw-1", "gdrive", str(tmp_path / "out"), 30, record)


def test_the_grant_holds_no_more_than_one_upload_to_the_store_can(store, monkeypatch):
    monkeypatch.setattr(store, "max_single_upload_bytes", 5 * 1024**3, raising=False)

    upload = _snapshot_upload({"capabilities": {"extensions": [_DATA_OBJECTS]}}, 30, "local")

    assert (upload.grant.max_object_bytes, upload.grant.max_total_bytes) == (5 * 1024**3, 5 * 1024**3)
