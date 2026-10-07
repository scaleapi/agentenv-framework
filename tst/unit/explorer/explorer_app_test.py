"""The explorer over a real SQLite store: list/get/versions semantics and a real run."""

import asyncio
import dataclasses
import gzip
import json
import tempfile
import time
from importlib.metadata import version as installed_version
from urllib.parse import quote

import httpx
import pytest

pytest.importorskip("fastapi")  # the explorer is the optional [explorer] extra (fastapi, uvicorn)

from fastapi.testclient import TestClient

from agent_env.config import configure, get_config, reset_config, set_document_store, set_object_store, set_runner
from agent_env.config.errors import ConfigError
from agent_env.runner import store as run_store
from agent_env.runner.local_runner import LocalRunner
from agent_env.runner.runner import RunRecord, RunStatus
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore
from agent_env.store.routing import RoutingDocumentStore
from agent_env.task import Task
from agent_env.explorer.routers import objects as objects_router
from agent_env.explorer.routers import triggers as triggers_router
from tst.unit.store.fakes import FakeObjectStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Point at an empty config so the suite never reads the developer's own
    # .agentenv/config.toml. Without this a local `[explorer] static_dir` mounts a UI
    # and test_root_serves_health_when_no_ui_is_mounted fails on untouched code — the
    # tier is meant to need nothing, and that includes needing nothing of your machine.
    empty_config = tmp_path / "empty-config.toml"
    empty_config.write_text("")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(empty_config))
    configure()
    store = LocalSqliteDocumentStore(str(tmp_path / "documents.db"))
    set_document_store(store)
    set_runner(LocalRunner(workers=1))
    run_store.ensure_indexes()

    for doc in [
        {"id": "slack", "version": 1, "type": "docker_image", "created_at_utc": "2026-01-01T00:00:00Z"},
        {"id": "slack", "version": 2, "type": "docker_image", "created_at_utc": "2026-01-02T00:00:00Z"},
        {"id": "acme-universe", "version": 1, "type": "service_universe", "created_at_utc": "2026-01-03T00:00:00Z"},
    ]:
        store.insert("artifacts", doc)
    store.insert("envs", {"id": "slack-env", "version": 1, "type": "mcp_server", "created_at_utc": "2026-01-01T00:00:00Z"})
    store.insert("tasks", {"id": "t1", "version": 1, "created_at_utc": "2026-01-01T00:00:00Z"})
    store.insert("a2a_agents", {"id": "claude-code-cli", "version": 1, "created_at_utc": "2026-01-01T00:00:00Z"})
    store.insert("evals", {"id": "e1", "version": 1, "created_at_utc": "2026-01-01T00:00:00Z"})

    from agent_env.explorer.app import create_app
    # Loopback base_url so the Host-allow-list guard admits the request (a realistic
    # Host; TestClient's default "testserver" is intentionally refused).
    with TestClient(create_app(), base_url="http://localhost") as c:
        yield c
    reset_config()


def test_health_reports_the_resolved_backends(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["document_store"] == "LocalSqliteDocumentStore"
    assert body["runner"] == "local"


def test_guard_blocks_cross_site_and_foreign_hosts(client):
    # The actual attack: a page you visit while `up` runs makes a cross-site POST
    # that would dispatch a run. The browser tags it Sec-Fetch-Site: cross-site and
    # the guard refuses it before any side effect (CORS withholds the response, not
    # the request — so CORS alone would not stop this).
    assert client.post("/api/v1/tasks/t1/run", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert client.get("/api/v1/artifacts", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    # The explorer's own same-origin UI, and curl / the agent-env CLI (no Sec-Fetch-Site),
    # are unaffected.
    assert client.get("/api/v1/artifacts", headers={"sec-fetch-site": "same-origin"}).status_code == 200
    assert client.get("/api/v1/artifacts").status_code == 200
    # A non-loopback Host is refused (DNS-rebinding defense; the bind is loopback-only).
    assert client.get("/api/v1/artifacts", headers={"host": "evil.example"}).status_code == 421


def test_run_group_funnel_covers_every_run_status():
    # Every RunStatus must map to an explicit funnel bucket, so a newly-added status
    # can't silently fall through to "provisioning" in _group_status.
    from agent_env.explorer.routers.runs import _FUNNEL_BUCKET
    from agent_env.runner.runner import RunStatus

    assert set(_FUNNEL_BUCKET) == set(RunStatus)


def test_list_returns_latest_version_per_entity(client):
    body = client.get("/api/v1/artifacts").json()
    by_id = {i["id"]: i for i in body["items"]}
    assert set(by_id) == {"slack", "acme-universe"}
    assert by_id["slack"]["version"] == 2          # not v1
    # 3 rows, 2 entities: the pager total must count entities
    assert body["total"] == 2
    assert body["has_more"] is False


def test_universes_hub_is_artifacts_filtered_by_type(client):
    """Universes Hub has no routes of its own; it is a type filter over artifacts."""
    body = client.get("/api/v1/artifacts", params={"type": "service_universe"}).json()
    assert [i["id"] for i in body["items"]] == ["acme-universe"]
    assert body["total"] == 1


def test_type_filter_matches_documents_stored_under_an_aliased_spelling(tmp_path, monkeypatch):
    """Universes Hub is a type filter, so a renamed type must be reachable from either
    spelling — otherwise the rename silently hides every document written before it."""
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "[artifacts]\n"
        'impls = ["tst.unit.artifact.registry_test:_RenamedArtifact"]\n'
        'type_aliases = { legacy_renamed = "renamed_artifact" }\n'
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reset_config()
    configure()
    store = LocalSqliteDocumentStore(str(tmp_path / "documents.db"))
    set_document_store(store)
    set_runner(LocalRunner(workers=1))
    run_store.ensure_indexes()
    store.insert("artifacts", {"id": "before", "version": 1, "type": "legacy_renamed",
                               "created_at_utc": "2026-01-01T00:00:00Z"})
    store.insert("artifacts", {"id": "after", "version": 1, "type": "renamed_artifact",
                               "created_at_utc": "2026-01-02T00:00:00Z"})
    store.insert("artifacts", {"id": "other", "version": 1, "type": "file",
                               "created_at_utc": "2026-01-03T00:00:00Z"})

    from agent_env.explorer.app import create_app
    try:
        with TestClient(create_app(), base_url="http://localhost") as c:
            for requested in ("legacy_renamed", "renamed_artifact"):
                body = c.get("/api/v1/artifacts", params={"type": requested}).json()
                assert sorted(i["id"] for i in body["items"]) == ["after", "before"], requested
                assert body["total"] == 2
            unaliased = c.get("/api/v1/artifacts", params={"type": "file"}).json()
            assert [i["id"] for i in unaliased["items"]] == ["other"]
    finally:
        reset_config()
        reset_config()


@pytest.mark.parametrize("url_key", ["s3_url", "object_url"])
def test_detail_route_enriches_a_universe_stored_under_an_aliased_spelling(tmp_path, monkeypatch, url_key):
    """Enrichment picks its branch from the type, so a renamed universe must still
    resolve its refs — otherwise the detail response silently loses `files`. Its files
    keep their url whichever key their documents store it under."""
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "[artifacts]\n"
        'type_aliases = { old_universe = "file_artifact_universe" }\n'
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reset_config()
    configure()
    store = LocalSqliteDocumentStore(str(tmp_path / "documents.db"))
    set_document_store(store)
    set_runner(LocalRunner(workers=1))
    run_store.ensure_indexes()
    store.insert("artifacts", {"id": "f1", "version": 1, "type": "file", url_key: "s3://b/a.txt",
                               "content_type": "text/plain", "created_at_utc": "2026-01-01T00:00:00Z"})
    store.insert("artifacts", {"id": "u1", "version": 1, "type": "old_universe",
                               "file_artifact_refs": {"a.txt": {"id": "f1", "version": 1}},
                               "created_at_utc": "2026-01-02T00:00:00Z"})

    from agent_env.explorer.app import create_app
    try:
        with TestClient(create_app(), base_url="http://localhost") as c:
            body = c.get("/api/v1/artifacts/u1").json()
            assert body["files"] == [{
                "filename": "a.txt", "artifact_id": "f1", "version": 1,
                "content_type": "text/plain", "object_url": "s3://b/a.txt",
            }]
    finally:
        reset_config()
        reset_config()


def test_pagination_reports_has_more(client):
    body = client.get("/api/v1/artifacts", params={"limit": 1}).json()
    assert len(body["items"]) == 1
    assert body["total"] == 2
    assert body["has_more"] is True


def test_get_defaults_to_latest_and_honours_explicit_version(client):
    assert client.get("/api/v1/artifacts/slack").json()["version"] == 2
    assert client.get("/api/v1/artifacts/slack", params={"version": 1}).json()["version"] == 1
    assert client.get("/api/v1/artifacts/missing").status_code == 404


def test_versions_returns_a_bare_list_newest_first(client):
    """A bare list, not a PaginatedResponse."""
    body = client.get("/api/v1/artifacts/slack/versions").json()
    assert isinstance(body, list)
    assert [i["version"] for i in body] == [2, 1]


def test_versions_of_an_unknown_id_is_404_like_get(client):
    assert client.get("/api/v1/artifacts/missing/versions").status_code == 404
    assert client.get("/api/v1/artifacts/slack/versions", params={"offset": 5}).json() == []


@pytest.mark.parametrize("path", ["envs", "tasks", "agents", "evals"])
def test_every_surface_lists(client, path):
    body = client.get(f"/api/v1/{path}").json()
    assert body["total"] == 1 and len(body["items"]) == 1


def test_run_roundtrip_through_the_runner_seam(client, monkeypatch):
    """POST /run returns {workflow_id, instance_id} and the run reaches a terminal state."""
    from agent_env.task import Task

    class _Task:
        version = 1

        async def run(self, **kw):
            return kw.get("context")

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, tid, ver=None: _Task()))

    started = client.post("/api/v1/tasks/t1/run", json={"version": 1, "agent_model": "claude-x"})
    assert started.status_code == 200
    body = started.json()
    assert body["workflow_id"] and body["instance_id"] and body["status"] == "QUEUED"

    import time
    deadline = time.time() + 5
    run = None
    while time.time() < deadline:
        items = client.get("/api/v1/tasks/t1/runs").json()["items"]
        run = next((r for r in items if r["run_id"] == body["workflow_id"]), None)
        if run and run["status"] in {"COMPLETED", "FAILED", "CANCELED"}:
            break
        time.sleep(0.05)
    assert run and run["status"] == "COMPLETED", run
    assert run["instance_id"] == body["instance_id"]
    assert run["runner"] == "local"

    listed = client.get("/api/v1/tasks/t1/runs").json()
    assert listed["total"] == 1
    assert listed["items"][0]["run_id"] == body["workflow_id"]


def test_run_for_unknown_task_is_404(client):
    """Submitting a run for a task that does not exist must 404, not enqueue."""
    assert client.post("/api/v1/tasks/ghost/run", json={}).status_code == 404
    # and it must not have enqueued anything
    assert client.get("/api/v1/tasks/ghost/runs").json()["total"] == 0


NAMESPACED_IDS = [
    "@local/~/stuff/instances/foo",
    "@local/~/triage/runs",
    "@local/~/Dropbox (Personal)/a&b+c,d@e/tickets",
    "@local/~/Été/Straße/tâche",
    "team/triage/versions",
    "weird%2Fbare",
    "100%25done",
]
ID_ROUTE_COLLECTIONS = {"artifacts": "artifacts", "envs": "envs", "tasks": "tasks", "agents": "a2a_agents", "evals": "evals"}


@pytest.mark.parametrize("entity_id", NAMESPACED_IDS)
@pytest.mark.parametrize("path, collection", ID_ROUTE_COLLECTIONS.items())
def test_an_id_is_one_encoded_path_segment(client, path, collection, entity_id):
    """An id's own segments, even ones named like a route, never split it."""
    get_config().get_document_store().insert(
        collection, {"id": entity_id, "version": 1, "created_at_utc": "2026-01-04T00:00:00Z"})
    encoded = quote(entity_id, safe="")

    got = client.get(f"/api/v1/{path}/{encoded}")
    assert got.status_code == 200 and got.json()["id"] == entity_id
    assert [d["id"] for d in client.get(f"/api/v1/{path}/{encoded}/versions").json()] == [entity_id]
    assert client.get(f"/api/v1/{path}/{entity_id}").status_code == 404


@pytest.mark.parametrize("task_id", ["@local/~/triage/runs", "team%2F100%25/runs"])
def test_the_run_routes_address_a_task_whose_id_ends_in_a_route_name(client, monkeypatch, task_id):
    class _Task:
        version = 1

        async def run(self, **kw):
            return kw.get("context")

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, tid, ver=None: _Task()))
    get_config().get_document_store().insert("tasks", {"id": task_id, "version": 1, "created_at_utc": "2026-01-04T00:00:00Z"})
    base = f"/api/v1/tasks/{quote(task_id, safe='')}"

    single = client.post(f"{base}/run", json={"version": 1}).json()
    group = client.post(f"{base}/runs", json={"version": 1, "count": 1}).json()
    deadline = time.time() + 5
    while time.time() < deadline:
        runs = client.get(f"{base}/runs").json()["items"]
        if len(runs) == 2 and all(r["status"] == "COMPLETED" for r in runs):
            break
        time.sleep(0.05)

    assert {r["task_id"] for r in runs} == {task_id} and len(runs) == 2
    # The stub task records no instance, so seed the ones a real run would.
    for run in runs:
        get_config().get_document_store().insert("task_instances", {
            "instance_id": run["instance_id"], "task_id": task_id, "task_version": 1, "status": "completed",
            "total_steps": 1, "current_step": 1, "completed_steps": [{"step_id": "s", "status": "success"}],
        })
    assert single["instance_id"] in {i["instance_id"] for i in client.get(f"{base}/instances").json()["items"]}
    assert client.get(f"{base}/instances/{single['instance_id']}").json()["task_id"] == task_id
    assert client.get(f"{base}/instances/{single['instance_id']}/progress").json()["instance_id"] == single["instance_id"]
    assert group["run_group_id"] in {g["run_group_id"] for g in client.get(f"{base}/run-groups").json()["items"]}
    assert client.get(f"{base}/run-groups/{group['run_group_id']}").json()["task_id"] == task_id
    assert "event: complete" in client.get(f"{base}/run-groups/{group['run_group_id']}/stream").text
    assert client.post(f"{base}/cancel-run", params={"workflow_id": single["workflow_id"]}).status_code == 200


def test_a_bundle_runs_namespaced_instance_id_is_one_encoded_segment(client):
    task_id, instance_id = "@local/~/triage/runs", "@local/~/triage/runs-7f3a9c2e"
    store = get_config().get_document_store()
    store.insert("task_instances", {"instance_id": instance_id, "task_id": task_id, "task_version": 1,
                                    "status": "completed", "total_steps": 1, "current_step": 1, "completed_steps": []})
    store.insert("agent_env_a2a_conversations", {"task_instance_id": instance_id, "created_at_utc": "2026-01-04T00:00:00Z"})
    task, instance = quote(task_id, safe=""), quote(instance_id, safe="")

    assert client.get(f"/api/v1/tasks/{task}/instances/{instance}").json()["instance_id"] == instance_id
    assert client.get(f"/api/v1/tasks/{task}/instances/{instance}/progress").json()["instance_id"] == instance_id
    assert len(client.get(f"/api/v1/task-instances/{instance}/conversations").json()["conversations"]) == 1


def test_start_runs_caps_the_batch_size(client, monkeypatch):
    from agent_env.task import Task

    class _T:
        version = 1
        async def run(self, **kw):
            return kw.get("context")

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, tid, ver=None: _T()))
    over = client.post("/api/v1/tasks/t1/runs", json={"version": 1, "count": 101})
    assert over.status_code == 422 and "at most 100" in over.text
    assert client.post("/api/v1/tasks/t1/runs", json={"version": 1, "count": 1}).status_code == 200


def test_cors_default_is_loopback_only(client):
    foreign = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in foreign.headers}
    local = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert local.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_run_groups_are_not_truncated_at_500(client):
    gid = "rg-big"
    for i in range(600):
        run_store.insert_run(RunRecord(
            run_id=f"r{i}", runner="local", task_id="t1", task_version=1,
            instance_id=f"i{i}", status=RunStatus.COMPLETED,
            overrides={"metadata": {"run_group_id": gid}},
        ))
    group = client.get(f"/api/v1/tasks/t1/run-groups/{gid}").json()
    assert group["total"] == 600          # paged past the 500 fetch limit, not truncated


def test_run_group_list_uses_one_run_snapshot_and_only_page_instances(client, monkeypatch):
    store = get_config().get_document_store()
    query_calls = []
    original_query = store.find_many_by_id
    run_pages = []
    original_list_runs = run_store.list_runs

    def counted_query(collection, id_field, ids):
        if collection == "task_instances":
            query_calls.append(len(ids))
        return original_query(collection, id_field, ids)

    def counted_list_runs(*args, **kwargs):
        run_pages.append(kwargs.get("offset", 0))
        return original_list_runs(*args, **kwargs)

    monkeypatch.setattr(store, "find_many_by_id", counted_query)
    monkeypatch.setattr(run_store, "list_runs", counted_list_runs)
    for i in range(1000):
        run_store.insert_run(RunRecord(
            run_id=f"paged-{i}", runner="local", task_id="t1", task_version=1,
            instance_id=f"paged-i{i}", status=RunStatus.COMPLETED,
            created_at_utc=f"2026-01-{(i // 100) + 1:02d}T00:00:{i % 60:02d}Z",
            overrides={"metadata": {"run_group_id": f"rg-page-{i // 10}"}},
        ))
        store.insert("task_instances", {"instance_id": f"paged-i{i}", "completed_steps": []})

    response = client.get("/api/v1/tasks/t1/run-groups?limit=20")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 100
    assert len(body["items"]) == 20
    assert all(group["total"] == 10 for group in body["items"])
    assert query_calls == [200]
    assert run_pages == [0, 500, 1000]


def test_run_group_list_preserves_routed_instance_precedence(client, tmp_path):
    configured = get_config().get_document_store()
    local = LocalSqliteDocumentStore(str(tmp_path / "local.db"))
    router = RoutingDocumentStore(configured, local, tmp_path / "local.db")
    configured.insert("task_instances", {"instance_id": "shared", "current_step": 2, "total_steps": 4})
    local.insert("task_instances", {"instance_id": "shared", "current_step": 7, "total_steps": 9})
    run_store.insert_run(RunRecord(
        run_id="shared-run", runner="local", task_id="t1", task_version=1,
        instance_id="shared", status=RunStatus.RUNNING, created_at_utc="2026-01-01T00:00:00Z",
    ))
    set_document_store(router)

    group = client.get("/api/v1/tasks/t1/run-groups").json()["items"][0]
    assert (group["instances"][0]["current_step"], group["instances"][0]["total_steps"]) == (2, 4)


def test_run_group_list_batches_a_selected_group_larger_than_500(client, monkeypatch):
    gid = "rg-list-big"
    store = get_config().get_document_store()
    query_calls = []
    original_query = store.find_many_by_id

    def counted_query(collection, id_field, ids):
        if collection == "task_instances":
            query_calls.append(len(ids))
        return original_query(collection, id_field, ids)

    monkeypatch.setattr(store, "find_many_by_id", counted_query)
    for i in range(600):
        run_store.insert_run(RunRecord(
            run_id=f"list-big-{i}", runner="local", task_id="t1", task_version=1,
            instance_id=f"list-big-i{i}", status=RunStatus.COMPLETED,
            created_at_utc=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
            overrides={"metadata": {"run_group_id": gid}},
        ))
        store.insert("task_instances", {"instance_id": f"list-big-i{i}", "completed_steps": []})

    group = client.get("/api/v1/tasks/t1/run-groups?limit=1").json()["items"][0]
    assert group["run_group_id"] == gid
    assert group["total"] == 600
    assert len(group["instances"]) == 600
    assert query_calls == [500, 100]


def test_run_group_list_keeps_missing_instances_ties_and_cross_version_membership(client):
    first = "rg-tied-cross-version"
    for run_id, version, instance_id in (("cross-v1", 1, "cross-i1"), ("cross-v2", 2, "cross-i2")):
        run_store.insert_run(RunRecord(
            run_id=run_id, runner="local", task_id="t1", task_version=version,
            instance_id=instance_id, status=RunStatus.COMPLETED,
            created_at_utc="2026-01-04T00:00:00Z",
            overrides={"metadata": {"run_group_id": first}},
        ))
    get_config().get_document_store().insert("task_instances", {
        "instance_id": "cross-i1", "current_step": 2, "total_steps": 3,
    })
    run_store.insert_run(RunRecord(
        run_id="cross-tied-other", runner="local", task_id="t1", task_version=2,
        instance_id="missing-instance", status=RunStatus.COMPLETED,
        created_at_utc="2026-01-04T00:00:00Z",
        overrides={"metadata": {"run_group_id": "rg-tied-other"}},
    ))

    groups = client.get("/api/v1/tasks/t1/run-groups?task_version=2").json()["items"]
    assert [group["run_group_id"] for group in groups] == [first, "rg-tied-other"]
    selected = groups[0]
    assert selected["task_version"] == 2
    assert selected["total"] == 2
    assert {instance["instance_id"] for instance in selected["instances"]} == {"cross-i1", "cross-i2"}
    missing = next(instance for instance in selected["instances"] if instance["instance_id"] == "cross-i2")
    assert missing["current_step"] is None and missing["total_steps"] is None
    other = groups[1]
    assert other["total"] == 1
    assert other["instances"][0]["current_step"] is None


def test_run_group_stream_reloads_status_for_each_snapshot(client, monkeypatch):
    from agent_env.explorer.routers import runs as runs_router

    run_id = "stream-fresh"
    run_store.insert_run(RunRecord(
        run_id=run_id, runner="local", task_id="t1", task_version=1,
        instance_id="stream-instance", status=RunStatus.RUNNING,
        overrides={"metadata": {"run_group_id": "rg-stream"}},
    ))
    poll_count = 0

    async def finish_between_polls(_seconds):
        nonlocal poll_count
        poll_count += 1
        run_store.mark_terminal(run_id, RunStatus.COMPLETED)

    monkeypatch.setattr(runs_router.asyncio, "sleep", finish_between_polls)

    async def read_events():
        response = await runs_router.stream_run_group("t1", "rg-stream")
        return "".join([chunk async for chunk in response.body_iterator])

    events = asyncio.run(read_events())
    assert events.count("event: snapshot") == 2
    snapshots = [json.loads(line.removeprefix("data: ")) for line in events.splitlines() if line.startswith("data: ")]
    assert snapshots[0]["running"] == 1
    assert snapshots[1]["completed"] == 1
    assert poll_count == 1


def test_run_groups_tally_step_progress_for_the_batch_funnel(client):
    """The Batch progress strip reads `step_counts`; without it every step reads 0/N."""

    gid = "rg-funnel"
    outcomes = {
        "i-ok-1": [("deploy", "success"), ("prompt", "success")],
        "i-ok-2": [("deploy", "success"), ("prompt", "success")],
        "i-bad": [("deploy", "success"), ("prompt", "failure")],
        # A completion callback that raises records the failure next to the step's
        # earlier success; the run must count once, as failed, not in both columns.
        "i-callback": [("deploy", "success"), ("prompt", "success"), ("prompt", "failure")],
        "i-new": [],
    }
    for n, (instance_id, steps) in enumerate(outcomes.items()):
        run_store.insert_run(RunRecord(
            run_id=f"rf{n}", runner="local", task_id="t1", task_version=1,
            instance_id=instance_id, status=RunStatus.COMPLETED,
            overrides={"metadata": {"run_group_id": gid}},
        ))
        get_config().get_document_store().insert("task_instances", {
            "instance_id": instance_id, "task_id": "t1", "task_version": 1,
            "completed_steps": [{"step_id": s, "status": st} for s, st in steps],
        })

    expected = {"deploy": {"done": 4, "failed": 0}, "prompt": {"done": 2, "failed": 2}}
    assert client.get(f"/api/v1/tasks/t1/run-groups/{gid}").json()["step_counts"] == expected
    listed = {g["run_group_id"]: g for g in client.get("/api/v1/tasks/t1/run-groups").json()["items"]}
    assert listed[gid]["step_counts"] == expected


# --- Triggers ---------------------------------------------------------------


def _seed_trigger_run(status="completed", metadata=None, deployed_envs=None, task_id="t1", instance_id="i-trig"):
    get_config().get_document_store().insert("task_instances", {
        "instance_id": instance_id, "task_id": task_id, "task_version": 1, "status": status,
        "context": {"metadata": metadata or {}, "deployed_envs": deployed_envs or []},
    })


def _envelope(instance_id="i-trig"):
    return {
        "instance_id": instance_id, "env_id": "crm", "capture_source": "prompt_agent",
        "clock": {"rate": 60.0, "virtual_now": "2026-01-04T09:00:00+00:00"},
        "clock_read_at_utc": "2026-01-04T00:00:05+00:00",
        "state": {"triggers": [{"id": "follow-up", "status": "fired", "fire_count": 1}], "events": [{"kind": "fired"}]},
    }


def test_triggers_report_the_metadata_ledger(client):
    registrations = [{"env_id": "crm", "trigger_ids": ["follow-up"]}]
    _seed_trigger_run(metadata={"env_trigger_registrations": registrations, "agent_trigger_firings": [{"turn": 1}]})

    body = client.get("/api/v1/tasks/t1/instances/i-trig/triggers").json()

    assert set(triggers_router.TRIGGER_METADATA_KEYS) <= set(body)
    assert body["status"] == "completed" and body["source"] == "metadata"
    assert body["env_trigger_registrations"] == registrations
    assert body["agent_trigger_firings"] == [{"turn": 1}]
    assert body["env_trigger_state"] == {} and body["usersim_turn_outputs"] is None
    assert (body["state"], body["state_sources"], body["state_meta"]) == ({}, {}, {})


def test_triggers_read_a_finished_runs_saved_envelope(client, tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path / "obj"))
    set_object_store(store)
    url = store.put("env_trigger_state/instance_id=i-trig/crm-1.json", json.dumps(_envelope()).encode())
    _seed_trigger_run(metadata={"env_trigger_state": {"crm": {"object_url": url, "event_count": 1}}})

    body = client.get("/api/v1/tasks/t1/instances/i-trig/triggers").json()

    assert body["source"] == "artifact" and body["state_sources"] == {"crm": "artifact"}
    assert body["state"] == {"crm": _envelope()["state"]}
    assert body["state_meta"] == {"crm": {"clock": _envelope()["clock"], "clock_read_at_utc": _envelope()["clock_read_at_utc"]}}


def test_triggers_skip_an_envelope_outside_the_store_or_from_another_instance(client, tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path / "obj"))
    set_object_store(store)
    foreign = store.put("env_trigger_state/instance_id=other/db-1.json", json.dumps(_envelope("other")).encode())
    _seed_trigger_run(metadata={"env_trigger_state": {
        "crm": {"object_url": "s3://some-bucket/env_trigger_state/crm.json"},
        "db": {"object_url": foreign},
        "web": {"error": "capture budget of 60s exceeded"},
    }})

    body = client.get("/api/v1/tasks/t1/instances/i-trig/triggers").json()

    assert body["source"] == "metadata"
    assert (body["state"], body["state_sources"], body["state_meta"]) == ({}, {}, {})
    assert set(body["env_trigger_state"]) == {"crm", "db", "web"}


def test_triggers_read_the_live_gateway_while_running(client, monkeypatch):
    get_config().get_document_store().insert("env_instances", {"instance_id": "crm-abc", "env_id": "crm", "gateway_url": "http://gateway.test/"})
    _seed_trigger_run(status="running", deployed_envs=[
        {"env_id": "crm", "instance_id": "crm-abc", "gateway_url": "http://ignored.test"},
        {"env_id": "db", "instance_id": "db-gone"},
    ])
    live = {"http://gateway.test/triggers/state": {"triggers": [{"id": "follow-up", "status": "armed"}]},
            "http://gateway.test/clock/state": {"rate": 60.0}}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=live[str(request.url)]))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=transport, **kw))

    body = client.get("/api/v1/tasks/t1/instances/i-trig/triggers").json()

    assert body["source"] == "live" and body["state_sources"] == {"crm": "live", "db": "unavailable"}
    assert body["state"] == {"crm": live["http://gateway.test/triggers/state"]}
    assert body["state_meta"]["crm"]["clock"] == {"rate": 60.0}
    assert "db" not in body["state_meta"]


def test_triggers_for_an_unknown_instance_are_404(client):
    assert client.get("/api/v1/tasks/t1/instances/ghost/triggers").status_code == 404


def test_triggers_address_a_namespaced_task_and_instance(client):
    task_id, instance_id = "@local/~/triage/runs", "@local/~/triage/runs-7f3a9c2e"
    _seed_trigger_run(task_id=task_id, instance_id=instance_id, metadata={"env_trigger_registrations": []})

    got = client.get(f"/api/v1/tasks/{quote(task_id, safe='')}/instances/{quote(instance_id, safe='')}/triggers")

    assert got.status_code == 200 and got.json()["env_trigger_registrations"] == []


# --- Docs surface -----------------------------------------------------------


def test_docs_spec_carries_the_live_primitive_catalogue(client):
    """The Docs catalogue is built from the live registries this process dispatches on."""
    spec = client.get("/api/v1/docs/openapi").json()
    primitives = spec["x-agent-env-primitives"]

    assert primitives["version"] == 1
    for group in ("artifacts", "envs", "taskSteps"):
        assert primitives[group], f"{group} catalogue is empty"

    steps = {e["type"]: e for e in primitives["taskSteps"]}
    deploy = steps["deploy_env"]
    assert deploy["className"] == "DeployEnvTaskStep"
    assert deploy["module"] == "agent_env.task_step.task_steps.deploy_env"
    assert deploy["source"]["line"] > 0                 # jump-to-source, not just a name

    assert spec["x-agent-env-docs"]["source"] == "live"


def test_docs_spec_documents_constructor_arguments(client):
    """Each primitive references a schema of its constructor, with real JSON types."""
    spec = client.get("/api/v1/docs/openapi").json()
    steps = {e["type"]: e for e in spec["x-agent-env-primitives"]["taskSteps"]}
    ref = steps["deploy_env"]["component"]
    schema = spec["components"]["schemas"][ref.rsplit("/", 1)[1]]

    assert schema["required"][:3] == ["id", "version", "env_id"]
    assert schema["properties"]["env_id"] == {"type": "string"}
    assert schema["properties"]["ttl_seconds"]["type"] == "integer"
    assert schema["properties"]["ttl_seconds"]["default"] == 7200
    assert schema["properties"]["env_version"]["nullable"] is True


def test_docs_descriptions_are_not_inherited_from_the_base_class(client):
    """A step with no docstring reads blank, not TaskStep's."""
    spec = client.get("/api/v1/docs/openapi").json()
    steps = {e["type"]: e for e in spec["x-agent-env-primitives"]["taskSteps"]}
    assert "Base class for all task steps" not in steps["add_skills"]["description"]
    assert steps["reset_env"]["description"].startswith("Return a deployed env")


def test_every_operation_has_its_own_summary(client):
    """The Docs nav lists operations by summary alone, so no two may share one."""
    spec = client.get("/openapi.json").json()
    summaries = [op["summary"] for ops in spec["paths"].values() for op in ops.values()]

    assert len(summaries) == len(set(summaries))
    assert all(s == s.strip() for s in summaries)
    assert {"List Artifacts", "Get Environment", "List Task Versions", "Health"} <= set(summaries)


def test_openapi_version_is_the_installed_framework(client):
    assert client.get("/openapi.json").json()["info"]["version"] == installed_version("agentenv-framework")


def test_docs_metadata_reports_a_live_source(client):
    meta = client.get("/api/v1/docs/openapi/metadata").json()
    assert meta["source"] == "live"
    assert meta["openapi_version"].startswith("3.")
    # The keys the overview's version rows read (ui/src/lib/live-spec-versions.ts).
    assert meta["versions"]["agentenv-framework"]
    assert meta["versions"]["agentenv-protocol"]
    # No object-store provenance to report; the UI switches panels on `source`.
    assert "bucket" not in meta


def test_swagger_moves_but_the_openapi_document_stays_at_the_root(client):
    """Swagger moves under /api but /openapi.json stays at the root."""
    from agent_env.explorer.app import create_app

    app = create_app()
    assert app.docs_url == "/api/docs"
    assert app.openapi_url == "/openapi.json"
    assert client.get("/openapi.json").status_code == 200


def test_packaged_ui_is_absent_in_a_source_checkout():
    """`/` serves health JSON from a checkout, and the app when a wheel ships the UI.

    The wheel maps the built UI to `agent_env/explorer/static` (pyproject force-include), so an
    installed copy serves it with no Node toolchain. A source checkout has no `static/`
    until someone builds it, and must fall back to API-only rather than 404 at the root.
    """
    from agent_env.explorer.app import packaged_ui_dir

    assert packaged_ui_dir() is None


def test_root_serves_health_when_no_ui_is_mounted(client):
    body = client.get("/").json()
    assert body["status"] == "ok"


def test_instance_content_serves_object_store_bytes(client, tmp_path):
    """Content route proxies an object-store blob and refuses a url outside the store."""
    from agent_env.config import set_object_store
    from agent_env.store.object_store.local.store import LocalFilesystemObjectStore

    store = LocalFilesystemObjectStore(str(tmp_path / "obj"))
    set_object_store(store)
    url = store.put("prompt_agent_trajectories/t.json", b'{"hello":"world"}', content_type="application/json")

    ok = client.get("/api/v1/objects/content", params={"object_url": url})
    assert ok.status_code == 200
    assert ok.content == b'{"hello":"world"}'
    assert ok.headers["x-content-type-options"] == "nosniff"

    # url outside the configured store is refused
    bad = client.get(
        "/api/v1/objects/content", params={"object_url": "file:///etc/passwd"}
    )
    assert bad.status_code == 400


def test_object_metadata_reports_the_type_content_serves(client, tmp_path):
    store = LocalFilesystemObjectStore(str(tmp_path / "obj"))
    set_object_store(store)
    url = store.put("out/notes.json", b"{}")

    meta = client.get("/api/v1/objects/metadata", params={"object_url": url}).json()
    served = client.get("/api/v1/objects/content", params={"object_url": url}).headers["content-type"]
    assert meta["content_type"] == served == "application/json"


def test_object_routes_serve_only_the_stores_own_urls(client, tmp_path):
    """A bare path to an object in the local root is not one of the store's urls: both object
    routes refuse it, and serve the file:// url."""
    store = LocalFilesystemObjectStore(str(tmp_path / "obj"))
    set_object_store(store)
    url = store.put("a/b.txt", b"hi")
    bare = url.removeprefix("file://")

    for route in ("/api/v1/objects/content", "/api/v1/objects/metadata"):
        assert client.get(route, params={"object_url": url}).status_code == 200
        assert client.get(route, params={"object_url": bare}).status_code == 400


def test_content_names_any_file_for_the_browser_and_leaves_no_temp_file(client, tmp_path, monkeypatch):
    """Header values are Latin-1: a CJK name or a quote goes in ``filename*`` beside an ASCII
    fallback, rather than failing the response after the download to a temp file."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    store = LocalFilesystemObjectStore(str(tmp_path / "obj"))
    set_object_store(store)
    url = store.put('out/报告 "v2".txt', b"hi", content_type="text/plain")

    response = client.get("/api/v1/objects/content", params={"object_url": url})

    assert (response.status_code, response.content) == (200, b"hi")
    assert response.headers["content-disposition"] == (
        "inline; filename=\"__ _v2_.txt\"; filename*=UTF-8''%E6%8A%A5%E5%91%8A%20%22v2%22.txt"
    )
    assert list(scratch.iterdir()) == []


def test_content_that_fails_before_its_response_exists_removes_its_temp_file(client, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    store = LocalFilesystemObjectStore(str(tmp_path / "obj"))
    set_object_store(store)
    url = store.put("out/a.txt", b"hi")

    def fail(*args):
        raise RuntimeError("response could not be built")

    monkeypatch.setattr(objects_router, "_streamed", fail)
    with pytest.raises(RuntimeError, match="could not be built"):
        client.get("/api/v1/objects/content", params={"object_url": url})
    assert list(scratch.iterdir()) == []


def test_object_routes_refuse_a_url_that_names_no_object(client):
    set_object_store(FakeObjectStore())
    for route in ("/api/v1/objects/content", "/api/v1/objects/metadata"):
        response = client.get(route, params={"object_url": "fake://home/"})
        assert response.status_code == 400
        assert "names a bucket" in response.json()["detail"]


def test_content_forwards_the_stored_content_encoding(client):
    """Stores read the bytes as stored, so a gzip-encoded object reaches the browser as gzip
    and the browser decodes it."""
    store = FakeObjectStore()
    set_object_store(store)
    url = store.put("out/log.txt", gzip.compress(b"hello"), content_type="text/plain")
    store._metadata[url] = dataclasses.replace(store._metadata[url], content_encoding="gzip")

    response = client.get("/api/v1/objects/content", params={"object_url": url})

    assert response.headers["content-encoding"] == "gzip"
    assert response.content == b"hello"


def test_content_sandboxes_active_types(client, tmp_path):
    """HTML/SVG artifacts get CSP sandbox; inert types (png/json) do not.

    This is also the guard that the app-wide baseline headers in ``create_app`` do not
    layer over the byte proxy: blanketing them here would both weaken ``sandbox`` on an
    active type and add a misleading policy to a passive one.
    """
    from agent_env.config import set_object_store
    from agent_env.store.object_store.local.store import LocalFilesystemObjectStore

    store = LocalFilesystemObjectStore(str(tmp_path / "obj"))
    set_object_store(store)
    html_url = store.put("out/report.html", b"<h1>hi</h1>", content_type="text/html")
    png_url = store.put("out/plot.png", b"\x89PNG", content_type="image/png")

    html = client.get("/api/v1/objects/content", params={"object_url": html_url})
    assert html.headers.get("content-security-policy") == "sandbox"

    png = client.get("/api/v1/objects/content", params={"object_url": png_url})
    assert "content-security-policy" not in png.headers


def test_baseline_security_headers_on_app_responses(client):
    """Every non-object response carries the baseline hardening headers.

    The UI renders agent-produced docx/xlsx/pptx inside this unauthenticated origin.
    The previews each render into a scriptless sandboxed iframe; these headers are the
    second layer, so a regression that drops them should fail here rather than in a
    pen test.
    """
    resp = client.get("/health")
    csp = resp.headers.get("content-security-policy") or ""
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'self'" in csp
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    assert resp.headers.get("referrer-policy") == "no-referrer"


def test_single_run_group_reports_its_own_run(client, monkeypatch):
    """A run started via POST /run (no run_group_id) must appear inside its own group.

    `list_run_groups` keys a group-less run by its run id, so `_instances_for_group`
    has to apply the same fallback. When it did not, every "Start 1 Run" produced a
    Rollouts row reading "0 runs" with no instance — the group was listed but nothing
    could ever join to it.
    """
    from agent_env.task import Task

    class _Task:
        version = 1

        async def run(self, **kw):
            return kw.get("context")

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, tid, ver=None: _Task()))

    started = client.post("/api/v1/tasks/t1/run", json={}).json()

    groups = client.get("/api/v1/tasks/t1/run-groups").json()["items"]
    assert len(groups) == 1, groups
    group = groups[0]

    assert group["run_group_id"] == started["workflow_id"]
    assert group["total"] == 1, "the group must contain the run that created it"
    assert [i["instance_id"] for i in group["instances"]] == [started["instance_id"]]


def test_step_overrides_reach_the_step_under_the_key_it_reads(client):
    """The request field is `step_overrides`; the key steps read is `step_params`.

    `TaskStep.step_param_overrides` looks up `user_overrides["step_params"]`, and the
    hosted hub backend performs the same rename on its own run route. Passing the
    request name straight through silently dropped every per-step override.
    """
    from agent_env.explorer.routers.runs import RunRequest, _run_metadata

    overrides = {"deploy": {"cpu": 4}}
    meta = _run_metadata(RunRequest(step_overrides=overrides))

    assert meta["user_overrides"]["step_params"] == overrides
    assert "step_overrides" not in meta["user_overrides"]


def test_progress_reports_done_so_the_poller_can_stop(client):
    """The poller stops on `done`; emitting nothing left it running until unmount.

    Statuses are seeded lowercase because that is what the task-instance store writes
    (task/store.py) — seeding uppercase would keep passing if the route's `.upper()`
    were removed, which is the regression this exists to catch.

    `live_step_index` is deliberately absent: `current_step` is a count of completed
    steps, not an index into them, and steps can complete out of order under a
    `depends_on` DAG.
    """
    from agent_env.explorer.routers.runs import TASK_INSTANCES_COLLECTION
    from agent_env.config import get_config

    store = get_config().get_document_store()
    store.insert(TASK_INSTANCES_COLLECTION, {
        "instance_id": "i-running", "task_id": "t1", "task_version": 1,
        "status": "running", "current_step": 2, "total_steps": 4, "completed_steps": [],
    })
    store.insert(TASK_INSTANCES_COLLECTION, {
        "instance_id": "i-done", "task_id": "t1", "task_version": 1,
        "status": "completed", "current_step": 4, "total_steps": 4, "completed_steps": [],
    })

    running = client.get("/api/v1/tasks/t1/instances/i-running/progress").json()
    assert running["done"] is False, "a running instance must keep the poller alive"
    assert "live_step_index" not in running

    finished = client.get("/api/v1/tasks/t1/instances/i-done/progress").json()
    assert finished["done"] is True, "a terminal instance must stop the poller"


def test_api_docs_pages_are_exempt_from_the_app_csp(client):
    """Swagger UI / ReDoc load their bundles from a CDN; `default-src 'self'` blanks them.

    They render no agent-produced bytes (that all goes through the object proxy), so
    they are exempt rather than the policy being widened for every page.
    """
    for path in ("/api/docs", "/api/redoc", "/openapi.json"):
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert "content-security-policy" not in resp.headers, (
            f"{path} must not carry the app CSP — it loads CDN assets"
        )

    # ...while an ordinary route still does.
    assert "content-security-policy" in client.get("/health").headers


def test_bad_start_step_is_rejected_at_the_boundary(client, monkeypatch):
    """A non-integer start_step must 422, not 200-then-fail invisibly.

    `body.metadata` is copied through verbatim, so typing on RunRequest.start_step is
    not enough. Coercing it in the runner instead raises after submit has returned a
    run id: the run is marked FAILED without ever registering a task instance, so
    /progress 404s forever and the UI polls something it can never show.
    """
    from agent_env.task import Task

    class _Task:
        version = 1

        async def run(self, **kw):
            return kw.get("context")

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, tid, ver=None: _Task()))

    for bad in ("abc", {"a": 1}, [1], 3.9):
        resp = client.post("/api/v1/tasks/t1/run", json={"metadata": {"start_step": bad}})
        assert resp.status_code == 422, f"start_step={bad!r} should be refused"

    assert client.post("/api/v1/tasks/t1/run", json={"metadata": {"start_step": -1}}).status_code == 422
    # A valid one still runs.
    assert client.post("/api/v1/tasks/t1/run", json={"metadata": {"start_step": 0}}).status_code == 200


def test_run_metadata_merges_into_inherited_user_overrides(client):
    """Explicit request fields merge into metadata.user_overrides, they don't replace it.

    A re-run carries the prior instance's user_overrides in metadata; rebuilding the
    dict from body.overrides alone dropped them before Task.run saw them.
    """
    from agent_env.explorer.routers.runs import RunRequest, _run_metadata

    meta = _run_metadata(RunRequest(
        metadata={"user_overrides": {"agent_effort": "high", "step_params": {"deploy": {"cpu": 1}}}},
        step_overrides={"deploy": {"cpu": 4}},
    ))

    overrides = meta["user_overrides"]
    assert overrides["agent_effort"] == "high", "inherited keys survive"
    assert overrides["step_params"] == {"deploy": {"cpu": 4}}, "an explicit field wins over the inherited one"


def test_bad_inherited_user_overrides_is_rejected(client, monkeypatch):
    """A non-mapping metadata.user_overrides must 422, not 500 on the dict spread."""
    from agent_env.task import Task

    class _Task:
        version = 1

        async def run(self, **kw):
            return kw.get("context")

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, tid, ver=None: _Task()))

    for bad in ("oops", [1, 2], 7):
        resp = client.post(
            "/api/v1/tasks/t1/run",
            json={"metadata": {"user_overrides": bad}},
        )
        assert resp.status_code == 422, f"user_overrides={bad!r} should be refused"

    ok = client.post(
        "/api/v1/tasks/t1/run",
        json={"metadata": {"user_overrides": {"agent_effort": "high"}}},
    )
    assert ok.status_code == 200


def test_unsupported_run_fields_are_refused_by_either_route(client, monkeypatch):
    """Every runner-affecting key is checked on the merged metadata, not just the typed field.

    `_run_metadata` copies `body.metadata` verbatim, so validating only the RunRequest
    fields let a direct caller smuggle the same keys past the guard. Each case below
    reached the runner before `_validate_run_metadata` was applied to both.
    """
    from agent_env.task import Task

    class _Task:
        version = 1

        async def run(self, **kw):
            return kw.get("context")

    monkeypatch.setattr(Task, "get", classmethod(lambda cls, tid, ver=None: _Task()))

    def post(payload):
        return client.post("/api/v1/tasks/t1/run", json=payload).status_code

    # Resume is not implemented: refused as a typed field AND through metadata.
    assert post({"context_from_instance_id": "prior"}) == 501
    assert post({"context_json": {"metadata": {}}}) == 501
    assert post({"metadata": {"context_from_instance_id": "prior"}}) == 501
    assert post({"metadata": {"context_json": {}}}) == 501

    # Malformed values that would otherwise raise inside the worker, post-submit.
    assert post({"metadata": {"start_step": "abc"}}) == 422
    assert post({"metadata": {"start_step": -1}}) == 422
    assert post({"metadata": {"user_overrides": "oops"}}) == 422

    # The supported shape still runs.
    assert post({"start_step": 0, "metadata": {"user_overrides": {"agent_effort": "high"}}}) == 200


def test_cors_origins_narrows_the_csrf_guard_rather_than_disabling_it(tmp_path, monkeypatch):
    """Configuring cors_origins must not switch the Sec-Fetch guard off for all of /api.

    It used to (`enforce_csrf = not cors`), which on a no-auth plane traded a named
    allowlist for none at all: any site could drive a cross-site write, not just the
    ones opted in.
    """
    from fastapi.testclient import TestClient
    from agent_env.config import configure, reset_config, set_document_store, set_runner
    from agent_env.explorer.app import create_app
    from agent_env.runner.local_runner import LocalRunner
    from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore

    config = tmp_path / "config.toml"
    config.write_text('[explorer]\ncors_origins = ["http://example.com"]\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    configure()
    set_document_store(LocalSqliteDocumentStore(str(tmp_path / "d.db")))
    set_runner(LocalRunner(workers=1))

    try:
        with TestClient(create_app(), base_url="http://localhost") as c:
            def post(headers):
                return c.post("/api/v1/tasks/ghost/run", json={}, headers=headers).status_code

            assert post({"sec-fetch-site": "cross-site",
                         "origin": "http://evil.example"}) == 403
            # An allowed origin is admitted — 404 because the task does not exist,
            # which is the guard letting it through to the route.
            assert post({"sec-fetch-site": "cross-site",
                         "origin": "http://example.com"}) == 404
            assert post({"sec-fetch-site": "same-origin"}) == 404
    finally:
        reset_config()


def test_wildcard_cors_origin_is_refused_at_startup(tmp_path, monkeypatch):
    """`*` would admit every site to a no-auth plane, so it must fail loudly, not 403 silently."""
    from agent_env.explorer.app import create_app

    config = tmp_path / "config.toml"
    config.write_text('[explorer]\ncors_origins = ["*"]\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    configure()
    try:
        with pytest.raises(ConfigError, match="cors_origins"):
            create_app()
    finally:
        reset_config()
