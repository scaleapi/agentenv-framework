"""snapshot_env on an env deployed without a gateway, a computer env say: reached through the card its deployment
recorded, its environments the card's children or the env itself, and each environment's export kept as data/get
returned it, a file tree's files as parts by their paths, inline or uploaded under the grant without a download, and
what the env's own data/get returns named by computer_file_paths."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from agentenv_protocol import (DATA_OBJECTS_EXTENSION_URI, IntakeDeclaration, IntakeFormat, intake_extension,
                               uploaded_file_part)

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.config import set_object_store
from agent_env.env.env import DeployedSandboxEnv
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.task_step.task_steps.snapshot_env import SnapshotEnvTaskStep
from tst.unit.task_step.test_snapshot_env import _ctx
from tst.util.granting_object_store import GrantingObjectStore

COMPUTER = "http://127.0.0.1:18765"
FILE_TREE = intake_extension(IntakeDeclaration(get=[IntakeFormat(part="file", format="file-tree")])).model_dump(
    mode="json", exclude_none=True)
DATA_OBJECTS = {"uri": DATA_OBJECTS_EXTENSION_URI}


class _Desk:
    id, version, environment_name = "desk", 1, "desktop"


def _record(card: dict, network: dict | None = None) -> DeployedSandboxEnv:
    return DeployedSandboxEnv(env_id="desk", env_version=1, sandbox_id="sb-1", instance_id="desk-abc",
                              environment_card_url=f"{COMPUTER}{WELL_KNOWN_PATH}", environment_card=card,
                              sandbox_type="local", metadata={"computer": {"network": network}} if network else {})


def _computer(*extensions) -> dict:
    return {"name": "desktop", "capabilities": {"extensions": [FILE_TREE, *extensions]}}


def _inline(path: str, data: bytes) -> dict:
    return {"kind": "file", "file": {"bytes": base64.b64encode(data).decode(), "name": Path(path).name,
                                     "mimeType": "text/plain"},
            "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": path}}}


def _serve(monkeypatch, answers: dict) -> list[tuple[str, dict]]:
    """Answer each data/get with the parts ``answers`` gives for the path it's sent to, recording each path and its
    params; any other request fails the test."""
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        params = json.loads(request.content)["params"]
        sent.append((request.url.path, params))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1,
                                         "result": {"parts": answers[request.url.path](params)}})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent


async def _snapshot(card: dict, network: dict | None = None, **options) -> EnvironmentUniverseArtifact:
    result = await SnapshotEnvTaskStep.snapshot_env_state(env=_Desk(), gateway_url=None, snapshot_id="snap-d",
                                                          deployed=_record(card, network), **options)
    return EnvironmentUniverseArtifact.get(result.environment_universe_artifact_id)


@pytest.mark.asyncio
async def test_a_computer_keeps_each_file_its_data_get_returns_as_a_part_by_its_path(local_stores, monkeypatch):
    sent = _serve(monkeypatch, {"/agentenv": lambda params: [_inline("Documents/notes.txt", b"hello"),
                                                             _inline("Downloads/b/c.bin", b"\x00\x01")]})

    universe = await _snapshot(_computer())

    (desk,) = universe.get_environment_artifacts()
    assert (desk.environment_name, desk.file_artifact_ref) == ("desktop", None)
    assert [(part.kind, part.path, part.name, part.mime_type, file.filename, file.load())
            for part, file in desk.get_parts()] == [
        ("file", "Documents/notes.txt", "notes.txt", "text/plain", "notes.txt", b"hello"),
        ("file", "Downloads/b/c.bin", "c.bin", "text/plain", "c.bin", b"\x00\x01")]
    assert sorted(universe.get_file_artifacts()) == ["desktop/Documents/notes.txt", "desktop/Downloads/b/c.bin"]
    assert sent == [("/agentenv", {})]


@pytest.mark.asyncio
async def test_files_uploaded_under_the_grant_are_written_where_the_computer_put_them_without_a_download(
        local_stores, tmp_path, monkeypatch):
    store = GrantingObjectStore(str(tmp_path / "objects"))
    set_object_store(store)
    files = {"Documents/notes.txt": b"hello", "Downloads/c.bin": b"\x00"}

    def uploading(params):
        root = params["write_namespace"]["root_path"]
        for path, data in files.items():
            store.put(f"{root}/{path}", data, content_type="application/octet-stream")
        return [uploaded_file_part(path, name=Path(path).name, mime_type="text/plain").model_dump(
            mode="json", exclude_none=True) for path in files]

    sent = _serve(monkeypatch, {"/agentenv": uploading})

    (desk,) = (await _snapshot(_computer(DATA_OBJECTS))).get_environment_artifacts()

    ((path, params),) = sent
    root = params["write_namespace"]["root_path"]
    assert params["write_namespace"]["max_objects"] == 100_000
    assert [(part.path, file.object_url, file.load()) for part, file in desk.get_parts()] == [
        (path, store.object_url(f"{root}/{path}"), data) for path, data in files.items()]


@pytest.mark.asyncio
async def test_a_computer_whose_network_keeps_it_from_the_store_is_handed_no_grant_and_returns_its_files_inline(
        local_stores, tmp_path, monkeypatch):
    set_object_store(GrantingObjectStore(str(tmp_path / "objects")))
    sent = _serve(monkeypatch, {"/agentenv": lambda params: [_inline("Desktop/left.txt", b"x")]})

    (desk,) = (await _snapshot(_computer(DATA_OBJECTS), network={"mode": "allowlist"})).get_environment_artifacts()

    assert sent == [("/agentenv", {})]
    assert [(part.path, file.load()) for part, file in desk.get_parts()] == [("Desktop/left.txt", b"x")]


@pytest.mark.asyncio
async def test_its_own_data_get_takes_computer_file_paths_and_each_child_on_its_card_is_snapshotted(local_stores,
                                                                                                   monkeypatch):
    card = {"name": "desktop", "children_environments": [
        {"name": "desktop", "url": "/agentenv", "capabilities": {"extensions": [FILE_TREE]}},
        {"name": "mail", "url": "/svc/mail/agentenv", "capabilities": {"extensions": []}}]}
    sent = _serve(monkeypatch, {"/agentenv": lambda params: [_inline("Documents/notes.txt", b"hello")],
                                "/svc/mail/agentenv": lambda params: [{"kind": "data", "data": {"inbox": 2}}]})

    universe = await _snapshot(card, computer_file_paths=["Documents"])

    desk, mail = sorted(universe.get_environment_artifacts(), key=lambda environment: environment.environment_name)
    assert sorted(sent) == [("/agentenv", {"paths": ["Documents"]}), ("/svc/mail/agentenv", {})]
    assert [part.path for part, _ in desk.get_parts()] == ["Documents/notes.txt"]
    assert mail.parts is None and json.loads(mail.get_file_artifact().load()) == {"inbox": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("card, answer, kept", [
    (_computer(), [], []),
    ({"name": "desktop", "capabilities": {"extensions": []}},
     [{"kind": "data", "data": {"a": 1}}, {"kind": "data", "data": {"b": 2}}],
     [("data", "part-0.json", b'{"a": 1}'), ("data", "part-1.json", b'{"b": 2}')]),
], ids=["a-file-tree-of-no-files", "several-parts-of-no-file-tree"])
async def test_an_export_one_file_cant_keep_is_kept_as_its_parts(local_stores, monkeypatch, card, answer, kept):
    _serve(monkeypatch, {"/agentenv": lambda params: answer})

    (desk,) = (await _snapshot(card)).get_environment_artifacts()

    assert [(part.kind, file.filename, file.load()) for part, file in desk.get_parts()] == kept and desk.parts == [
        part for part, _ in desk.get_parts()]


@pytest.mark.asyncio
async def test_execute_snapshots_an_env_deployed_without_a_gateway_through_its_recorded_card(local_stores,
                                                                                            monkeypatch):
    sent = _serve(monkeypatch, {"/agentenv": lambda params: [_inline("Documents/notes.txt", b"hello")]})
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _Desk()))
    context = _ctx(_record(_computer()))

    await SnapshotEnvTaskStep(id="snap", version=None, env_id="desk", computer_file_paths=["Documents"]).execute(
        context)

    snapshotted = context.metadata["env_snapshotted_universes"]["snap"]
    universe = EnvironmentUniverseArtifact.get(snapshotted["id"], version=snapshotted["version"])
    assert sorted(universe.get_file_artifacts()) == ["desktop/Documents/notes.txt"]
    assert sent == [("/agentenv", {"paths": ["Documents"]})]
