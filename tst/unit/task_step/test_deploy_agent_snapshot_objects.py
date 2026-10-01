from __future__ import annotations

import threading

import httpx
import pytest

import agent_env.artifact.artifacts.file_artifact_universe as universe_mod
from agent_env.config import configure
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.snapshot_utils import agent_state_capture as capture
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from tst.util.granting_object_store import GrantingObjectStore

BUNDLE_KEY = "agent_snapshots/snapshot-1/3-deadbeef"


class _Universe:
    id = "snapshot-1"
    version = 3

    def __init__(self, store: GrantingObjectStore, files: tuple[str, ...]) -> None:
        for name in files:
            store.put(f"{BUNDLE_KEY}/{name}", b"snapshot bytes")
        self.bundle_object_url = store.object_url(BUNDLE_KEY) + "/"
        self.file_artifact_refs = None
        self.file_artifact_ids = {
            name: f"artifact-{index}" for index, name in enumerate(files)
        }


def _step() -> DeployAgentTaskStep:
    return DeployAgentTaskStep(
        id="deploy",
        version=None,
        agent_name="solver",
        agent_snapshot_files_artifact_id="snapshot-1",
        agent_snapshot_files_artifact_version=3,
        agent_snapshot_target_context_id="restored-context",
    )


def _card(*, objects: bool = True, legacy: bool = True) -> dict:
    variants = []
    if legacy:
        variants.append({"required": ["s3_prefix"]})
    if objects:
        variants.append({"required": ["objects"]})
    return {
        "capabilities": {
            "extensions": [
                {
                    "uri": "urn:agentenv:snapshot/v1",
                    "params": {
                        "methods": {
                            "load": {
                                "endpoint": "/custom/snapshot",
                                "request": {"oneOf": variants},
                            },
                        },
                    },
                }
            ]
        }
    }


@pytest.fixture
def store(tmp_path) -> GrantingObjectStore:
    store = GrantingObjectStore(str(tmp_path))
    configure(object_store=store)
    return store


def _install(
    monkeypatch, universe: _Universe, response: dict, *, status_code: int = 200
) -> list[dict]:
    monkeypatch.setattr(
        universe_mod.FileArtifactUniverse,
        "get",
        classmethod(lambda cls, artifact_id, version: universe),
    )
    requests: list[dict] = []

    async def fake_request(
        self, method, url, *, json=None, timeout=None, **kwargs
    ):  # noqa: A002
        requests.append(
            {"method": method, "url": url, "json": json, "timeout": timeout}
        )
        return httpx.Response(
            status_code, json=response, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    return requests


PORTABLE = (capture.SNAPSHOT_TRAJECTORY_OBJECT_NAME, capture.SNAPSHOT_WORKSPACE_OBJECT_NAME)
LEGACY = ("conversation.jsonl", "workspace.tar.gz")


@pytest.mark.asyncio
async def test_load_prefers_objects_for_a_portable_snapshot(monkeypatch, store):
    requests = _install(
        monkeypatch, _Universe(store, PORTABLE), {"context_id": "restored-context"}
    )
    context = TaskStepContext()

    await _step()._load_snapshot("https://agent", _card(), "agent-1", context, sandbox_type="local")

    sent = requests[0]
    assert sent["url"] == "https://agent/custom/snapshot"
    assert set(sent["json"]) == {"objects", "target_context_id"}
    assert set(sent["json"]["objects"]) == {"trajectory", "workspace"}
    assert all(
        descriptor["media_type"] == "application/octet-stream"
        and descriptor["size_bytes"] == len(b"snapshot bytes")
        for descriptor in sent["json"]["objects"].values()
    )
    assert {url.rsplit("/", 1)[-1] for url in store.granted} == set(PORTABLE)
    assert context.metadata["agent_loaded_snapshots"][0]["context_id"] == (
        "restored-context"
    )


@pytest.mark.asyncio
async def test_load_reads_object_metadata_off_the_event_loop(monkeypatch, store):
    _install(monkeypatch, _Universe(store, PORTABLE), {"context_id": "restored-context"})
    loop_threads = []
    metadata = store.get_object_metadata_at

    def recording(url):
        loop_threads.append(threading.current_thread() is threading.main_thread())
        return metadata(url)

    monkeypatch.setattr(store, "get_object_metadata_at", recording)

    await _step()._load_snapshot(
        "https://agent", _card(), "agent-1", TaskStepContext(), sandbox_type="local"
    )

    assert loop_threads == [False, False]


S3_BUNDLE_URL = f"s3://artifact-bucket/{BUNDLE_KEY}/"
EXPIRED_SIGNED_URL = (
    "https://objects.s3.region.example.test/agent_snapshots/snapshot-1/"
    "?X-Amz-Date=20000101T000000Z&X-Amz-Expires=3600&X-Amz-Signature=0"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("signed", [False, True], ids=["s3-url", "expired-signed-url"])
async def test_load_sends_a_legacy_snapshot_whole_as_its_prefix(monkeypatch, store, signed):
    universe = _Universe(store, LEGACY)
    universe.bundle_object_url = EXPIRED_SIGNED_URL if signed else S3_BUNDLE_URL
    requests = _install(monkeypatch, universe, {"ok": True, "context_id": "legacy-context"})
    context = TaskStepContext()

    await _step()._load_snapshot("https://agent", _card(), "agent-1", context, sandbox_type="local")

    assert requests[0]["json"] == {
        "s3_prefix": universe.bundle_object_url,
        "target_context_id": "restored-context",
    }
    assert not store.granted
    assert context.metadata["agent_loaded_snapshots"] == [
        {
            "agent_name": "solver",
            "context_id": "legacy-context",
            "source_artifact_id": "snapshot-1",
            "source_artifact_version": 3,
        }
    ]


@pytest.mark.asyncio
async def test_a_legacy_snapshot_on_the_local_store_is_refused(monkeypatch, store):
    requests = _install(monkeypatch, _Universe(store, LEGACY), {"ok": True})

    with pytest.raises(RuntimeError, match="cannot load a snapshot on a local object store"):
        await _step()._load_snapshot(
            "https://agent", _card(), "agent-1", TaskStepContext(), sandbox_type="local"
        )

    assert not store.granted
    assert not requests


@pytest.mark.asyncio
async def test_load_validates_the_object_response(monkeypatch, store):
    _install(monkeypatch, _Universe(store, PORTABLE[:1]), {"ok": True})

    with pytest.raises(RuntimeError, match="invalid object response"):
        await _step()._load_snapshot(
            "https://agent", _card(), "agent-1", TaskStepContext(), sandbox_type="local"
        )


@pytest.mark.asyncio
async def test_load_ignores_response_fields_it_does_not_know(monkeypatch, store):
    _install(
        monkeypatch,
        _Universe(store, PORTABLE[:1]),
        {"ok": True, "context_id": "restored-context"},
    )
    context = TaskStepContext()

    await _step()._load_snapshot("https://agent", _card(), "agent-1", context, sandbox_type="local")

    assert context.metadata["agent_loaded_snapshots"][0]["context_id"] == (
        "restored-context"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "card, grants",
    [(_card(objects=False), True), (_card(), False)],
    ids=["legacy-only-agent", "store-without-grants"],
)
async def test_a_portable_snapshot_loads_through_objects_or_not_at_all(
    monkeypatch, store, card, grants
):
    store.supports_transfer_grants = grants
    requests = _install(monkeypatch, _Universe(store, PORTABLE[:1]), {"context_id": "wrong"})

    with pytest.raises(RuntimeError, match="cannot load the configured portable"):
        await _step()._load_snapshot(
            "https://agent", card, "agent-1", TaskStepContext(), sandbox_type="local"
        )

    assert not store.granted
    assert not requests


@pytest.mark.asyncio
async def test_legacy_snapshot_is_not_sent_to_an_objects_only_agent(monkeypatch, store):
    requests = _install(
        monkeypatch, _Universe(store, ("workspace.tar.gz",)), {"context_id": "wrong"}
    )

    with pytest.raises(RuntimeError, match="cannot load the configured legacy"):
        await _step()._load_snapshot(
            "https://agent",
            _card(legacy=False),
            "agent-1",
            TaskStepContext(),
            sandbox_type="local",
        )

    assert not store.granted
    assert not requests


@pytest.mark.asyncio
async def test_a_legacy_load_error_keeps_the_agents_detail(monkeypatch, store):
    universe = _Universe(store, LEGACY)
    universe.bundle_object_url = S3_BUNDLE_URL
    _install(
        monkeypatch,
        universe,
        {"detail": "cannot load a workspace while an agent turn is running"},
        status_code=409,
    )

    with pytest.raises(httpx.HTTPStatusError, match="while an agent turn is running"):
        await _step()._load_snapshot(
            "https://agent", _card(), "agent-1", TaskStepContext(), sandbox_type="local"
        )
