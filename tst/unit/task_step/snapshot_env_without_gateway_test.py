"""snapshot_env on an env deployed without a gateway: reached through the card its deployment recorded, its
environments the card's children or the env itself, each kept as one file as through a gateway; an environment whose
data/get returns a file tree, a computer's, kept file by file by its paths under a grant sized for a file tree; and an
env that fails under the upload grant asked again without it."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from agentenv_protocol import DATA_OBJECTS_EXTENSION_URI, IntakeDeclaration, IntakeFormat, intake_extension

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
UNREACHABLE = {"error": {"code": -32000, "message": "The object store is temporarily unavailable.",
                         "data": "get_failed"}}


class _Desk:
    id, version, environment_name = "desk", 1, "desktop"


def _record(card: dict) -> DeployedSandboxEnv:
    return DeployedSandboxEnv(env_id="desk", env_version=1, sandbox_id="sb-1", instance_id="desk-abc",
                              environment_card_url=f"{COMPUTER}{WELL_KNOWN_PATH}", environment_card=card,
                              sandbox_type="local")


def _computer(*extensions) -> dict:
    return {"name": "desktop", "capabilities": {"extensions": [FILE_TREE, *extensions]}}


def _server(*extensions) -> dict:
    return {"name": "desktop", "capabilities": {"extensions": list(extensions)}}


def _inline(path: str, data: bytes) -> dict:
    return {"kind": "file", "file": {"bytes": base64.b64encode(data).decode(), "name": Path(path).name,
                                     "mimeType": "text/plain"},
            "metadata": {DATA_OBJECTS_EXTENSION_URI: {"path": path}}}


def _serve(monkeypatch, answers: dict) -> list[tuple[str, dict]]:
    """Answer each data/get with the parts ``answers`` gives for the path it's sent to, or the JSON-RPC error when it
    gives one, recording each path and its params; any other request fails the test."""
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        params = json.loads(request.content)["params"]
        sent.append((request.url.path, params))
        answer = answers[request.url.path](params)
        body = {"error": answer["error"]} if isinstance(answer, dict) else {"result": {"parts": answer}}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, **body})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent


async def _snapshot(card: dict) -> EnvironmentUniverseArtifact:
    result = await SnapshotEnvTaskStep.snapshot_env_state(env=_Desk(), gateway_url=None, snapshot_id="snap-d",
                                                          deployed=_record(card))
    return EnvironmentUniverseArtifact.get(result.environment_universe_artifact_id)


def _kept(universe: EnvironmentUniverseArtifact) -> dict[str, bytes]:
    """Each environment's one file's bytes, by the environment's name."""
    return {name.split("/", 1)[0]: file.load() for name, file in universe.get_file_artifacts().items()}


@pytest.mark.asyncio
async def test_each_child_on_the_recorded_card_is_reached_and_kept_as_one_file(local_stores, monkeypatch):
    card = {"name": "desk", "children_environments": [
        {"name": "mail", "url": "/svc/mail/agentenv", "capabilities": {"extensions": []}},
        {"name": "notes", "url": "/svc/notes/agentenv", "capabilities": {"extensions": []}}]}
    sent = _serve(monkeypatch, {"/svc/mail/agentenv": lambda params: [{"kind": "data", "data": {"inbox": 2}}],
                                "/svc/notes/agentenv": lambda params: [_inline("notes.zip", b"PK")]})

    universe = await _snapshot(card)

    assert _kept(universe) == {"mail": b'{"inbox": 2}', "notes": b"PK"}
    assert sorted(path for path, _ in sent) == ["/svc/mail/agentenv", "/svc/notes/agentenv"]


@pytest.mark.asyncio
async def test_the_env_itself_is_reached_when_its_card_has_no_children(local_stores, monkeypatch):
    sent = _serve(monkeypatch, {"/agentenv": lambda params: [{"kind": "data", "data": {"rows": 1}}]})

    universe = await _snapshot(_server())

    assert _kept(universe) == {"desktop": b'{"rows": 1}'}
    assert sent == [("/agentenv", {})]


@pytest.mark.asyncio
async def test_a_file_tree_is_kept_file_by_file_by_its_paths_under_a_grant_sized_for_it(local_stores, tmp_path,
                                                                                        monkeypatch):
    set_object_store(GrantingObjectStore(str(tmp_path / "objects")))
    sent = _serve(monkeypatch, {"/agentenv": lambda params: [_inline("Desktop/a.txt", b"a"), _inline("b.txt", b"b")]})

    universe = await _snapshot(_computer(DATA_OBJECTS))

    (desktop,) = universe.get_environment_artifacts()
    assert {path: file.load() for path, file in desktop.get_file_artifacts().items()} == {"Desktop/a.txt": b"a",
                                                                                          "b.txt": b"b"}
    ((_, params),) = sent
    assert params["write_namespace"]["max_objects"] == 100_000


@pytest.mark.asyncio
async def test_an_env_that_fails_under_the_grant_is_asked_again_without_one(local_stores, tmp_path, monkeypatch):
    set_object_store(GrantingObjectStore(str(tmp_path / "objects")))
    sent = _serve(monkeypatch, {"/agentenv": lambda params: UNREACHABLE if "write_namespace" in params
                                else [{"kind": "data", "data": {"rows": 1}}]})

    universe = await _snapshot(_server(DATA_OBJECTS))

    assert _kept(universe) == {"desktop": b'{"rows": 1}'}
    assert [(path, sorted(params)) for path, params in sent] == [("/agentenv", ["write_namespace"]), ("/agentenv", [])]


@pytest.mark.asyncio
async def test_an_env_that_fails_with_no_grant_is_asked_once(local_stores, monkeypatch):
    sent = _serve(monkeypatch, {"/agentenv": lambda params: UNREACHABLE})

    with pytest.raises(RuntimeError, match="data/get failed: The object store is temporarily unavailable"):
        await _snapshot(_server())
    assert sent == [("/agentenv", {})]


@pytest.mark.asyncio
async def test_execute_snapshots_an_env_deployed_without_a_gateway_through_its_recorded_card(local_stores,
                                                                                            monkeypatch):
    _serve(monkeypatch, {"/agentenv": lambda params: [{"kind": "data", "data": {"rows": 1}}]})
    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, id, version=None: _Desk()))
    context = _ctx(_record(_server()))

    await SnapshotEnvTaskStep(id="snap", version=None, env_id="desk").execute(context)

    snapshotted = context.metadata["env_snapshotted_universes"]["snap"]
    assert _kept(EnvironmentUniverseArtifact.get(snapshotted["id"], version=snapshotted["version"])) == {
        "desktop": b'{"rows": 1}'}
