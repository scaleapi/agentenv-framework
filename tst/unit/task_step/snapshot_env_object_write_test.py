"""snapshot_env hands a service that takes parts grants an object of its own to upload its bundle to, makes
the object from what it reports, and registers it there; every other reply aborts the write."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI, TRANSFERS_EXTENSION_URI, uploaded_object_part
from agentenv_protocol.transfers import Uploaded

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.config import set_object_store
from agent_env.store.base import GrantUnavailableError
from agent_env.task_step.task_steps import snapshot_env as snapshot_mod
from agent_env.task_step.task_steps.snapshot_env import ENV_SNAPSHOT_MAX_BYTES, SnapshotEnvTaskStep
from tst.unit.task_step.test_snapshot_env import _CARD, _FakeMultiEnv, _card, _carded
from tst.util.granting_object_store import GrantingObjectStore

_PARTS = {"uri": TRANSFERS_EXTENSION_URI, "params": {"write": ["http-post-policy", "http-put", "http-put-parts"]}}
_DATA_OBJECTS = {"uri": DATA_OBJECTS_EXTENSION_URI}
_S3_CREDENTIALS = {"uri": "urn:agentenv:add-s3-credentials/v1",
                   "params": {"endpoint": "/agentenv/ext/add_s3_credentials", "methods": {"add_s3_credentials": {"method": "POST"}}}}
_BUNDLE = b"PK\x03\x04bundle"


class _UngrantingStore(GrantingObjectStore):
    def begin_write(self, object_url, **kwargs):
        raise GrantUnavailableError("this store issues no object writes")


class _WatchedStore(GrantingObjectStore):
    """Records how each write it begins ends."""

    def __init__(self, root):
        super().__init__(root)
        self.ended: list[str] = []

    def begin_write(self, object_url, **kwargs):
        write = super().begin_write(object_url, **kwargs)
        complete, abort, ended = write.complete, write.abort, self.ended
        write.complete = lambda uploaded: (complete(uploaded), ended.append("completed"))[0]
        write.abort = lambda: (ended.append("aborted") if not write._settled else None, abort())[1]
        return write


class _GatedStore(_WatchedStore):
    """Holds the export at ``at`` (beginning the write, pushing creds, completing it) until released."""

    def __init__(self, root, at):
        super().__init__(root)
        self.at, self.reached, self.release = at, threading.Event(), threading.Event()

    def hold(self, at):
        if at == self.at:
            self.reached.set()
            self.release.wait(5)

    def begin_write(self, object_url, **kwargs):
        self.hold("begin")
        write = super().begin_write(object_url, **kwargs)
        complete = write.complete
        write.complete = lambda uploaded: (self.hold("complete"), complete(uploaded))[1]
        return write


@pytest.fixture
def store(local_stores, tmp_path):
    granting = _WatchedStore(str(tmp_path / "objects"))
    set_object_store(granting)
    return granting


@pytest.fixture
def pushes(monkeypatch):
    pushed: list[str] = []

    async def push(base_url, card, timeout_seconds):
        pushed.append(base_url)

    monkeypatch.setattr(snapshot_mod, "_push_s3_credentials", push)
    return pushed


async def _record(children: dict, *, sandbox_type: str | None = "local"):
    return dataclasses.replace(_carded(await _card(children)), sandbox_type=sandbox_type)


def _serve(monkeypatch, answers: dict):
    """Answer each service's data/get with ``answers[service](params)``, recording the params sent."""
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        service = next(seg for seg in request.url.path.split("/") if seg.startswith("mcp-")).removeprefix("mcp-")
        params = json.loads(request.content)["params"]
        sent.append((service, params))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": answers[service](params)})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent


def _key(params) -> str:
    return unquote(urlsplit(params["write_object"]["write"]["urls"][0]).path.lstrip("/"))


def _uploading(store, *, reported: int | None = None):
    """A service that PUTs its bundle to the object it is handed and says what it sent."""
    def answer(params):
        store.put(_key(params), _BUNDLE, content_type="application/zip")
        uploaded = Uploaded(size_bytes=len(_BUNDLE) if reported is None else reported)
        part = uploaded_object_part(uploaded, name="gdrive.zip", mime_type="application/zip")
        return {"parts": [part.model_dump(mode="json", exclude_none=True)]}
    return answer


def _data(params):
    return {"parts": [{"kind": "data", "data": {"rows": 1}}]}


@pytest.mark.asyncio
async def test_a_service_taking_parts_uploads_to_an_object_of_its_own_and_is_registered_there(
    store, pushes, monkeypatch, caplog
):
    record = await _record({"mcp-gdrive": [_PARTS, _DATA_OBJECTS, _S3_CREDENTIALS]})
    sent = _serve(monkeypatch, {"gdrive": _uploading(store)})

    with caplog.at_level(logging.INFO, logger=snapshot_mod.logger.name):
        result = await SnapshotEnvTaskStep.snapshot_env_state(
            env=_FakeMultiEnv(["gdrive"]), gateway_url="https://gw-1", snapshot_id="snap-o", deployed=record,
        )

    ((_, params),) = sent
    assert list(params) == ["write_object"]
    grant = params["write_object"]
    assert (grant["media_type"], grant["max_bytes"], grant["write"]["kind"]) == ("application/zip", ENV_SNAPSHOT_MAX_BYTES, "http-put-parts")
    key = _key(params)
    assert key.startswith("agentenv-snapshots/") and key.endswith("/gdrive.zip")
    assert pushes == [_CARD]
    assert store.ended == ["completed"]
    (bundle,) = EnvironmentUniverseArtifact.get(result.environment_universe_artifact_id).get_file_artifacts().values()
    assert bundle.object_url == store.object_url(key)
    assert (bundle.filename, bundle.content_type, bundle.load()) == ("gdrive.zip", "application/zip", _BUNDLE)
    assert f"gdrive exported through parts ({len(_BUNDLE)} bytes in 1 parts)" in caplog.text


@pytest.mark.asyncio
async def test_each_object_write_gets_an_object_of_its_own(store, pushes, monkeypatch):
    record = await _record({"mcp-gdrive": [_PARTS]})
    sent = _serve(monkeypatch, {"gdrive": _uploading(store)})

    for _ in range(2):
        await SnapshotEnvTaskStep.snapshot_env_state(
            env=_FakeMultiEnv(["gdrive"]), gateway_url="https://gw-1", snapshot_id="snap-1", deployed=record,
        )

    first, second = (_key(params) for _, params in sent)
    assert first != second


@pytest.mark.asyncio
async def test_a_reply_that_names_no_object_upload_aborts_the_write_and_exports_as_before(
    store, pushes, monkeypatch, tmp_path
):
    record = await _record({"mcp-gdrive": [_PARTS]})
    _serve(monkeypatch, {"gdrive": _data})

    out = tmp_path / "export-tmp"
    suffix = await SnapshotEnvTaskStep._export_environment_to_file("https://gw-1", "gdrive", str(out), 30, record)

    assert (suffix, json.loads(out.read_text())) == (".json", {"rows": 1})
    assert store.ended == ["aborted"]


@pytest.mark.asyncio
async def test_an_upload_the_store_cannot_make_into_the_object_fails_only_that_service(store, pushes, monkeypatch):
    record = await _record({"mcp-gdrive": [_PARTS], "mcp-slack": []})
    _serve(monkeypatch, {"gdrive": _uploading(store, reported=len(_BUNDLE) + 1), "slack": _data})

    with pytest.raises(RuntimeError, match=r"1/2 services were not exported.*gdrive"):
        await SnapshotEnvTaskStep.snapshot_env_state(
            env=_FakeMultiEnv(["gdrive", "slack"]), gateway_url="https://gw-1", snapshot_id="snap-f", deployed=record,
        )
    assert store.ended == ["aborted"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("extensions", "store_cls", "sandbox_type", "params"), [
    ([_DATA_OBJECTS], _WatchedStore, "local", "write_namespace"),
    ([_PARTS, _DATA_OBJECTS], _WatchedStore, "modal", None),
    ([_PARTS, _DATA_OBJECTS], _UngrantingStore, "local", "write_namespace"),
], ids=["parts-not-declared", "grants-do-not-reach-the-sandbox", "store-issues-no-object-write"])
async def test_otherwise_the_export_is_offered_as_before(
    local_stores, tmp_path, pushes, monkeypatch, caplog, extensions, store_cls, sandbox_type, params
):
    set_object_store(store_cls(str(tmp_path / "objects")))
    record = await _record({"mcp-gdrive": extensions}, sandbox_type=sandbox_type)
    sent = _serve(monkeypatch, {"gdrive": _data})

    with caplog.at_level(logging.INFO, logger=snapshot_mod.logger.name):
        suffix = await SnapshotEnvTaskStep._export_environment_to_file(
            "https://gw-1", "gdrive", str(tmp_path / "export-tmp"), 30, record
        )

    assert suffix == ".json"
    ((_, sent_params),) = sent
    assert list(sent_params) == ([] if params is None else [params])
    assert pushes == [_CARD]
    assert "gdrive exported through data in" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(("at", "ended"), [("begin", ["aborted"]), ("push", ["aborted"]), ("complete", ["completed"])])
async def test_an_export_cancelled_midway_leaves_no_write_open_or_aborted_mid_completion(
    local_stores, tmp_path, monkeypatch, at, ended
):
    store = _GatedStore(str(tmp_path / "objects"), at)
    set_object_store(store)
    monkeypatch.setattr(snapshot_mod, "_push_s3_credentials", lambda *_: asyncio.to_thread(store.hold, "push"))
    record = await _record({"mcp-gdrive": [_PARTS]})
    _serve(monkeypatch, {"gdrive": _uploading(store)})

    export = asyncio.ensure_future(
        SnapshotEnvTaskStep._export_environment_to_file("https://gw-1", "gdrive", str(tmp_path / "out"), 30, record)
    )
    assert await asyncio.to_thread(store.reached.wait, 5)
    export.cancel()
    await asyncio.sleep(0.05)
    store.release.set()
    with pytest.raises(asyncio.CancelledError):
        await export
    assert store.ended == ended
