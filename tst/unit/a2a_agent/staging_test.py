"""An agent the object store's grants cannot reach gets its objects through its own staging routes: pushed
before a call, pulled after it, and a changelog drained into the store while the agent works and before it
goes. The agent here is an SDK agent served in-process at an HTTPS URL, so every request agent-env and the
agent make, the agent's own calls to its staging included, runs without a network."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from agentenv_protocol.a2a_agent import (
    SKILL_CONFIG_V1,
    SNAPSHOT_V1,
    TRAJECTORY_V1,
    AgentEnvAgent,
    AgentIdentity,
    BundleSkillRequest,
    InlineSkillRequest,
    NamespaceChangelogEnableRequest,
    NamespaceUploader,
    ObjectChangelogApplyRequest,
    ObjectSnapshotLoadRequest,
    ObjectSnapshotSaveRequest,
    TaskObjectTrajectoryRequest,
    TaskRequest,
    TaskResult,
    TaskTrajectoryRequest,
    a2a_agent,
    download,
    extension,
    upload,
)
from starlette.testclient import TestClient

import agent_env.a2a_agent.staging as staging
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.a2a_agent.object_transfer import (
    changelog_apply_call,
    changelog_enable_call,
    fetch_trajectory,
    invoke_transfer,
    readable_parts,
    skill_add_call,
    snapshot_load_call,
    snapshot_save_call,
    TrajectoryUpload,
)
from agent_env.a2a_agent.staging import StagedObjectStore, StagingError, transfer_store
from agent_env.config import reset_config, set_object_store
from agent_env.store import LocalFilesystemObjectStore
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from tst.util.granting_object_store import GrantingObjectStore

URL = "https://agent.example.test"


@a2a_agent(identity=AgentIdentity(name="staging-test", description="test", version="1"))
class _Agent(AgentEnvAgent):
    """Moves every object through the SDK's transfer helpers, as any SDK agent does."""

    workdir: Path
    received: dict
    uploader: NamespaceUploader | None = None

    async def run(self, request: TaskRequest) -> TaskResult:
        return TaskResult.text("ok")

    @extension(SKILL_CONFIG_V1.add)
    async def add(self, request: InlineSkillRequest | BundleSkillRequest):
        for file in request.skill_bundle.files:
            await download(file.object, self.workdir / "skills" / request.name / file.path)
            self.received[f"skill/{file.path}"] = (self.workdir / "skills" / request.name / file.path).read_bytes()
        return {"name": request.name}

    @extension(TRAJECTORY_V1.get)
    async def trajectory(self, request: TaskTrajectoryRequest | TaskObjectTrajectoryRequest):
        uploaded = await upload(request.objects.trajectory, b'[{"type": "result"}]')
        return {"objects": {"trajectory": uploaded.model_dump(exclude_none=True)}}

    @extension(SNAPSHOT_V1.save)
    async def save(self, request: ObjectSnapshotSaveRequest):
        trajectory = await upload(request.objects.trajectory, b"conversation")
        workspace = await upload(request.objects.workspace, b"w" * 300_000)
        return {
            "context_id": request.context_id,
            "objects": {"trajectory": trajectory.model_dump(), "workspace": workspace.model_dump()},
        }

    @extension(SNAPSHOT_V1.load)
    async def load(self, request: ObjectSnapshotLoadRequest):
        for name in ("trajectory", "workspace"):
            await download(getattr(request.objects, name), self.workdir / "loaded" / name)
            self.received[f"snapshot/{name}"] = (self.workdir / "loaded" / name).read_bytes()
        return {"context_id": "restored"}

    @extension(SNAPSHOT_V1.changelog.enable)
    async def enable_changelog(self, request: NamespaceChangelogEnableRequest):
        type(self).uploader = NamespaceUploader(request.write_namespace)
        return {"roots": ["/app"]}

    @extension(SNAPSHOT_V1.changelog.apply)
    async def apply_changelog(self, request: ObjectChangelogApplyRequest):
        for increment in request.increments:
            await download(increment.object, self.workdir / "applied" / str(increment.sequence))
            self.received[f"applied/{increment.sequence}"] = (self.workdir / "applied" / str(increment.sequence)).read_bytes()
        return {"count": len(request.increments), "context_id": request.target_context_id}


@pytest.fixture
def agent(tmp_path, monkeypatch):
    """The agent's app, which every HTTP client in this process reaches at ``URL``."""
    monkeypatch.setenv("AGENTENV_STAGING_DIR", str(tmp_path / "agent-staging"))
    _Agent.workdir = tmp_path / "agent"
    _Agent.received = {}
    _Agent.uploader = None
    app = _Agent().create_app()
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_async_client(transport=httpx.ASGITransport(app=app), **kwargs)
    )
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: TestClient(app, base_url=URL, follow_redirects=False))
    return app


@pytest.fixture
def store(tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path / "store"))
    set_object_store(store)
    yield store
    reset_config()


def _card(app) -> dict:
    return TestClient(app).get("/.well-known/agent-card.json").json()


def _staged_paths(tmp_path) -> list[str]:
    objects = tmp_path / "agent-staging" / "objects"
    return sorted(p.relative_to(objects).as_posix() for p in objects.rglob("*") if p.is_file()) if objects.exists() else []


def _method(card, uri, name):
    return A2AAgent.operation(A2AAgent.find_extension(card, uri), name)


@pytest.mark.asyncio
async def test_a_remote_agent_gets_a_skill_bundle_through_its_staging(agent, store, tmp_path):
    store.put("skills/s/SKILL.md", b"# s")
    store.put("skills/s/ref/big.bin", b"b" * 3_000_000)
    card = _card(agent)
    granting = transfer_store(store, URL, card, sandbox_type="modal")
    method, path = _method(card, SKILL_CONFIG_V1.uri, "add")

    call = skill_add_call(
        method, granting, name="s", description="d", object_url=store.object_url("skills/s"), sandbox_type="modal"
    )
    await invoke_transfer(URL + path, call, verb="POST", operation="skill add", timeout=60, store=granting)

    assert isinstance(granting, StagedObjectStore) and call.mode == "objects"
    assert _Agent.received == {"skill/SKILL.md": b"# s", "skill/ref/big.bin": b"b" * 3_000_000}
    assert _staged_paths(tmp_path) == []  # the call's staging is cleared once it is answered


@pytest.mark.asyncio
async def test_a_file_part_reaches_a_remote_agent_through_its_staging_while_it_is_sent(agent, store, tmp_path):
    url = store.put("seeds/x.png", b"png bytes")
    parts = [{"kind": "file", "file": {"uri": url, "mimeType": "image/png", "name": "x.png"}}]

    async with readable_parts(parts, a2a_url=URL, card=_card(agent), sandbox_type="modal", expires_in=600) as sent:
        assert sent[0]["file"]["uri"].startswith(f"{URL}/")
        async with httpx.AsyncClient() as client:
            fetched = await client.get(sent[0]["file"]["uri"])
        assert fetched.content == b"png bytes"

    assert _staged_paths(tmp_path) == []


@pytest.mark.asyncio
async def test_a_file_part_whose_grant_needs_headers_is_staged_instead(agent, tmp_path):
    store = GrantingObjectStore(str(tmp_path / "store"), reaches=True, grant_headers={"x-token": "t"})
    set_object_store(store)
    url = store.put("seeds/x.png", b"png bytes")
    parts = [{"kind": "file", "file": {"uri": url, "mimeType": "image/png", "name": "x.png"}}]

    async with readable_parts(parts, a2a_url=URL, card=_card(agent), sandbox_type="modal", expires_in=600) as sent:
        async with httpx.AsyncClient() as client:
            fetched = await client.get(sent[0]["file"]["uri"])
        assert fetched.content == b"png bytes"


@pytest.mark.asyncio
async def test_what_the_agent_writes_is_in_the_store_when_the_call_returns(agent, store, tmp_path):
    card = _card(agent)
    granting = transfer_store(store, URL, card, sandbox_type="modal")
    save_method, path = _method(card, SNAPSHOT_V1.uri, "save")
    prefix = store.object_url("snapshots/one/")

    call = snapshot_save_call(
        save_method, granting, agent_name="a", context_id="ctx", capture_prefix=prefix, sandbox_type="modal"
    )
    await invoke_transfer(URL + path, call, verb="POST", operation="snapshot save", timeout=60, store=granting)

    assert store.get(store.object_url("snapshots/one/trajectory")) == b"conversation"
    assert store.get(store.object_url("snapshots/one/workspace")) == b"w" * 300_000
    assert _staged_paths(tmp_path) == []

    load_method, path = _method(card, SNAPSHOT_V1.uri, "load")
    granting = transfer_store(store, URL, card, sandbox_type="modal")
    call = snapshot_load_call(
        load_method, granting, agent_name="a", bundle_url=prefix, file_names={"trajectory", "workspace"},
        target_context_id=None, sandbox_type="modal",
    )
    await invoke_transfer(URL + path, call, verb="PUT", operation="snapshot load", timeout=60, store=granting)
    assert _Agent.received == {"snapshot/trajectory": b"conversation", "snapshot/workspace": b"w" * 300_000}


@pytest.mark.asyncio
async def test_a_trajectory_upload_lands_in_the_store(agent, store):
    card = _card(agent)
    granting = transfer_store(store, URL, card, sandbox_type="modal")
    _, path = _method(card, TRAJECTORY_V1.uri, "get")
    target = store.object_url("trajectories/t.json")

    fetched = await fetch_trajectory(
        URL + path, {"task_id": "t"}, upload=TrajectoryUpload.to(granting, target), store=granting
    )

    assert fetched.object_url == target
    assert store.get(target) == b'[{"type": "result"}]'


async def _enable_changelog(agent, store) -> tuple[str, staging.StagedNamespace]:
    card = _card(agent)
    granting = transfer_store(store, URL, card, sandbox_type="modal")
    method, path = _method(card, SNAPSHOT_V1.uri, "enable-changelog")
    namespace_url = store.object_url("agent_changelog/run/a")
    call = changelog_enable_call(
        method, granting, agent_name="a", namespace_url=namespace_url, expires_in=3600, sandbox_type="modal"
    )
    await invoke_transfer(URL + path, call, verb="POST", operation="changelog enable", timeout=60, store=granting)
    [namespace] = granting.namespaces
    return namespace_url, namespace


@pytest.mark.asyncio
async def test_a_staged_changelog_is_drained_into_the_store_and_applied_to_a_fresh_agent(agent, store, tmp_path):
    namespace_url, namespace = await _enable_changelog(agent, store)
    await _Agent.uploader.upload("000000.tar", b"first")
    await _Agent.uploader.upload("000003.tar", b"second")

    assert await staging.drain(namespace, store) == 2
    assert store.get(f"{namespace_url}/000000.tar") == b"first"
    assert store.get(f"{namespace_url}/000003.tar") == b"second"
    assert _staged_paths(tmp_path) == []
    assert await staging.drain(namespace, store) == 0

    card = _card(agent)
    granting = transfer_store(store, URL, card, sandbox_type="modal")
    method, path = _method(card, SNAPSHOT_V1.uri, "apply-changelog")
    call = changelog_apply_call(method, granting, agent_name="a", source_url=namespace_url, sandbox_type="modal")
    await invoke_transfer(URL + path, call, verb="PUT", operation="changelog apply", timeout=60, store=granting)
    assert _Agent.received == {"applied/0": b"first", "applied/3": b"second"}


@pytest.mark.asyncio
async def test_a_prompt_drains_as_the_agent_works_and_once_more_when_it_ends(agent, store, monkeypatch):
    monkeypatch.setattr(staging, "DRAIN_INTERVAL_SECONDS", 0.05)
    namespace_url, namespace = await _enable_changelog(agent, store)

    async with staging.draining([namespace]):
        await _Agent.uploader.upload("000000.tar", b"mid-prompt")
        for _ in range(100):
            if store.get_object_metadata_at(f"{namespace_url}/000000.tar") is not None:
                break
            await asyncio.sleep(0.02)
        assert store.get(f"{namespace_url}/000000.tar") == b"mid-prompt"  # before the prompt ends
        await _Agent.uploader.upload("000001.tar", b"last")
    assert store.get(f"{namespace_url}/000001.tar") == b"last"


@pytest.mark.asyncio
async def test_a_run_torn_down_drains_its_staged_changelogs_first(agent, store):
    namespace_url, namespace = await _enable_changelog(agent, store)
    await _Agent.uploader.upload("000000.tar", b"before teardown")
    context = TaskStepContext(metadata={"agent_changelog": [
        {"agent_name": "a", "object_url": namespace_url, "transfer_mode": "objects", "staging_url": namespace.staging_url},
    ]})

    await teardown_run(context)

    assert store.get(f"{namespace_url}/000000.tar") == b"before teardown"


def test_staging_is_only_for_agents_the_grants_cannot_reach(agent, store):
    card = _card(agent)
    assert transfer_store(store, URL, card, sandbox_type="local") is store  # the local grant server reaches it
    assert transfer_store(store, URL, {}, sandbox_type="modal") is store  # no staging: the inline forms, or a refusal
    assert transfer_store(store, "http://agent.example.test", card, sandbox_type="modal") is store  # no HTTPS grant
    assert isinstance(transfer_store(store, URL, card, sandbox_type="modal"), StagedObjectStore)


@pytest.mark.asyncio
async def test_a_full_staging_fails_the_call_without_naming_a_staged_path(tmp_path, monkeypatch, store):
    monkeypatch.setenv("AGENTENV_STAGING_MAX_BYTES", "1000")
    app = _Agent().create_app()
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_async_client(transport=httpx.ASGITransport(app=app), **kwargs)
    )
    _Agent.workdir, _Agent.received = tmp_path / "agent", {}
    store.put("skills/s/SKILL.md", b"x" * 5000)
    card = _card(app)
    granting = transfer_store(store, URL, card, sandbox_type="modal")
    method, path = _method(card, SKILL_CONFIG_V1.uri, "add")
    call = skill_add_call(
        method, granting, name="s", description="d", object_url=store.object_url("skills/s"), sandbox_type="modal"
    )

    with pytest.raises(StagingError, match="staging an object on the agent failed: its staging is full") as raised:
        await invoke_transfer(URL + path, call, verb="POST", operation="skill add", timeout=60, store=granting)
    assert granting.endpoint not in str(raised.value)
    assert granting.max_bytes == 1000  # refused from the card's limit, before anything was sent


class _CountingStore(LocalFilesystemObjectStore):
    lookups = 0

    def get_object_metadata_at(self, object_url):
        self.lookups += 1
        return super().get_object_metadata_at(object_url)


@pytest.mark.asyncio
async def test_a_push_sends_the_objects_as_the_call_described_them_without_asking_the_store_again(agent, tmp_path):
    store = _CountingStore(str(tmp_path / "counted"))
    set_object_store(store)
    try:
        files = {"SKILL.md": b"# s", "a.txt": b"a" * 10, "b.txt": b"b" * 20}
        for name, data in files.items():
            store.put(f"skills/s/{name}", data)
        card = _card(agent)
        granting = transfer_store(store, URL, card, sandbox_type="modal")
        method, path = _method(card, SKILL_CONFIG_V1.uri, "add")
        call = skill_add_call(
            method, granting, name="s", description="d", object_url=store.object_url("skills/s"), sandbox_type="modal"
        )
        described = store.lookups

        await invoke_transfer(URL + path, call, verb="POST", operation="skill add", timeout=60, store=granting)

        assert store.lookups == described
        assert _Agent.received == {f"skill/{name}": data for name, data in files.items()}
    finally:
        reset_config()


@pytest.mark.asyncio
async def test_a_pull_stops_at_the_size_its_grant_allows(agent, store):
    granting = transfer_store(store, URL, _card(agent), sandbox_type="modal")
    grant = granting.issue_write_grant(store.object_url("out/t.json"), media_type="application/json", max_bytes=10)
    async with httpx.AsyncClient() as client:  # an agent whose own client ignores the limit
        assert (await client.put(grant.url, content=b"x" * 100)).status_code == 201

    with pytest.raises(StagingError, match="larger than its grant allows"):
        await granting.pull()
    assert store.get_object_metadata_at(store.object_url("out/t.json")) is None


@pytest.mark.asyncio
async def test_an_increment_rewritten_after_it_was_drained_replaces_the_stored_copy(agent, store):
    namespace_url, namespace = await _enable_changelog(agent, store)
    await _Agent.uploader.upload("000000.tar", b"first")
    await staging.drain(namespace, store)
    await _Agent.uploader.upload("000000.tar", b"first")  # the same bytes again, as a retried upload sends
    await staging.drain(namespace, store)
    assert store.get(f"{namespace_url}/000000.tar") == b"first"

    await _Agent.uploader.upload("000000.tar", b"second")
    await staging.drain(namespace, store)
    assert store.get(f"{namespace_url}/000000.tar") == b"second"


@pytest.mark.asyncio
async def test_a_drain_that_keeps_failing_is_reported_once(monkeypatch, caplog, tmp_path):
    monkeypatch.setattr(staging, "DRAIN_INTERVAL_SECONDS", 0.01)
    local = LocalFilesystemObjectStore(str(tmp_path / "store"))
    unreachable = staging.StagedNamespace("https://gone.example.test/ext/staging/" + "n" * 32, local.object_url("ns"))

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_async_client(transport=httpx.MockTransport(refuse), **kwargs))
    monkeypatch.setattr(staging, "get_config", lambda: type("C", (), {"get_object_store_at": lambda self, url: local})())
    with caplog.at_level("WARNING", logger=staging.__name__):
        async with staging.draining([unreachable]):
            await asyncio.sleep(0.2)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2  # once while the prompt runs, once for the last drain
    assert "the agent could not be reached" in warnings[0].getMessage()
