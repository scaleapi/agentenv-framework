"""Namespace-addressed stores: ``@local`` documents, runs and objects go to the per-user local stores,
everything else to the configured ones, and an ``@local`` run can't write to a configured store."""

import inspect
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.store import get_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, get_config, reset_config, set_object_store
from agent_env.config.errors import ConfigError
from agent_env.config.paths import state_root
from agent_env.store import DuplicateKeyError, Filter, LocalSqliteDocumentStore, Sort, UpdateSpec, VersionedEntityStore
from agent_env.store.document_store import DocumentStore
from agent_env.store import routing
from agent_env.store.image_store import ImageStore, LocalRegistryImageStore, OciRegistryImageStore
from agent_env.store.object_store import LocalFilesystemObjectStore, ObjectStore
from agent_env.store.routing import (
    LocalRunImageStore,
    LocalRunObjectStore,
    LocalRunWriteError,
    OutwardReferenceError,
    RoutingDocumentStore,
    configured_store,
    run_scope,
)
from tst.store import conformance

LOCAL_TASK = "@local/~/bundle/tasks/t"
LOCAL_ENV = "@local/~/bundle/envs/e"
# Building a store from a config table isn't a call on a store, so a wrapper has nothing to route.
_NOT_ROUTED = {"from_config"}


@pytest.fixture
def stores(tmp_path, cli_routing):
    """A configured document store standing in for a shared one, routed beside the per-user store."""
    configured = LocalSqliteDocumentStore(str(tmp_path / "configured.db"))
    configure(document_store=configured)
    router = get_config().get_document_store()
    assert isinstance(router, RoutingDocumentStore)
    return router, configured, router.local


def _ids(store, collection, field="id"):
    return {doc[field] for doc in store.query(collection, Filter())}


# The suite's versioned-entity case writes bare entity ids, which an @local run refuses by design.
_CONFORMANCE = [
    pytest.param(case, mode, id=f"{mode}-{case.__name__}")
    for mode in ("configured", "merged", "local run")
    for case in conformance.CASES
    if not (mode == "local run" and case is conformance.versioned_entity_store_roundtrip)
]


@pytest.mark.parametrize("case, mode", _CONFORMANCE)
def test_the_router_passes_the_conformance_suite(case, mode, stores):
    router, _, local = stores
    if mode == "merged":
        local.insert("seed", {"id": "seed"})  # the local store exists, so every read merges both
    if mode == "local run":
        with run_scope(LOCAL_TASK):
            case(router, "coll")
    else:
        case(router, "coll")


def test_entity_documents_route_by_their_id(stores):
    router, configured, local = stores
    router.insert("envs", {"id": LOCAL_ENV, "version": 1})
    router.insert("envs", {"id": "rocket", "version": 1})

    assert _ids(local, "envs") == {LOCAL_ENV}
    assert _ids(configured, "envs") == {"rocket"}
    assert router.find_one("envs", Filter.of(id="rocket"))["id"] == "rocket"
    assert _ids(router, "envs") == {LOCAL_ENV, "rocket"}


def test_a_runs_records_follow_its_task_id(stores):
    router, configured, local = stores
    with run_scope(LOCAL_TASK):
        router.insert("env_instances", {"instance_id": "rocket-1a2b", "env_id": "rocket"})
    with run_scope("registry-task"):
        router.insert("env_instances", {"instance_id": "rocket-3c4d", "env_id": "rocket"})

    assert _ids(local, "env_instances", "instance_id") == {"rocket-1a2b"}
    assert _ids(configured, "env_instances", "instance_id") == {"rocket-3c4d"}


def test_outside_a_run_a_record_owned_by_an_local_entity_is_local(stores):
    router, configured, local = stores
    router.insert("runs", {"run_id": "local-1", "task_id": LOCAL_TASK})
    router.insert("runs", {"run_id": "local-2", "task_id": "registry-task"})

    assert _ids(local, "runs", "run_id") == {"local-1"}
    assert _ids(configured, "runs", "run_id") == {"local-2"}


def test_a_run_started_inside_an_local_run_stays_local(cli_routing):
    with run_scope(LOCAL_TASK):
        with run_scope("registry-task"):
            assert routing.in_local_run()
        assert routing.in_local_run()
    assert not routing.in_local_run()


def test_merged_reads_sort_page_and_count_across_both_stores(stores):
    router, configured, local = stores
    for rank in (1, 4, 5):
        local.insert("tasks", {"id": f"@local/~/b/tasks/t{rank}", "version": 1, "rank": rank, "project": "p"})
    for rank in (2, 3, 6):
        configured.insert("tasks", {"id": f"t{rank}", "version": 1, "rank": rank, "project": "p"})
    local.insert("tasks", {"id": "@local/~/b/tasks/t1", "version": 2, "rank": 7, "project": "p"})
    project = Filter.of(project="p")

    page = router.query("tasks", project, sort=Sort.by("rank", descending=False), limit=3, offset=2)
    assert [doc["rank"] for doc in page] == [3, 4, 5]
    assert router.find_one("tasks", project, sort=Sort.by("rank"))["rank"] == 7
    assert router.count("tasks", project) == 7
    latest = router.latest_per_id("tasks", project, sort=Sort.by("rank"), limit=2, offset=1)
    assert [doc["rank"] for doc in latest] == [6, 5]
    assert router.count_distinct("tasks", project) == 6


def test_reads_never_create_the_local_store(stores):
    router, _, _ = stores
    router.find_one("envs", Filter.of(id=LOCAL_ENV))
    router.query("tasks", Filter())
    router.count("runs", Filter())
    router.update("env_instances", Filter.of(instance_id="x"), UpdateSpec(set={"a": 1}))
    router.ensure_index("envs", ["id", "version"], unique=True)
    assert not state_root().exists()


def test_the_local_store_gets_its_indexes_before_its_first_write(stores):
    router, _, _ = stores
    router.ensure_index("envs", ["id", "version"], unique=True)
    router.insert("envs", {"id": LOCAL_ENV, "version": 1})
    with pytest.raises(DuplicateKeyError):
        router.insert("envs", {"id": LOCAL_ENV, "version": 1})


def test_an_update_goes_to_the_store_holding_its_match(stores):
    router, configured, local = stores
    local.insert("env_state_instances", {"instance_id": "esi-local", "n": 0})
    configured.insert("env_state_instances", {"instance_id": "esi-shared", "n": 0})
    bump = UpdateSpec(inc={"n": 1})

    assert router.update("env_state_instances", Filter.of(instance_id="esi-local"), bump) == 1
    assert router.update_one_and_get("env_state_instances", Filter.of(instance_id="esi-shared"), bump)["n"] == 1
    assert router.update("env_state_instances", Filter.of(instance_id="esi-missing"), bump) == 0
    assert local.find_one("env_state_instances", Filter.of(instance_id="esi-local"))["n"] == 1
    assert router.delete("env_state_instances", Filter.of(instance_id="esi-shared")) == 1
    assert configured.count("env_state_instances", Filter()) == 0


def test_an_upsert_creates_its_document_where_an_insert_would(stores):
    router, configured, local = stores
    router.update("env_artifacts", Filter.of(env_id=LOCAL_ENV, artifact_id="u"), UpdateSpec(set={"data": 1}), upsert=True)
    router.update("env_artifacts", Filter.of(env_id="rocket", artifact_id="u"), UpdateSpec(set={"data": 2}), upsert=True)
    with run_scope(LOCAL_TASK):
        router.replace("env_snapshots", Filter.of(env_id="rocket", instance_id="i"), {"env_id": "rocket", "instance_id": "i"}, upsert=True)

    assert _ids(local, "env_artifacts", "env_id") == {LOCAL_ENV}
    assert _ids(configured, "env_artifacts", "env_id") == {"rocket"}
    assert _ids(local, "env_snapshots", "instance_id") == {"i"}


def test_an_local_run_cant_write_a_bare_entity(stores):
    router, configured, _ = stores
    configured.insert("artifacts", {"id": "shared", "version": 1})
    with run_scope(LOCAL_TASK):
        with pytest.raises(LocalRunWriteError, match="'artifacts'"):
            router.insert("artifacts", {"id": "cli-rocket", "version": 1})
        with pytest.raises(LocalRunWriteError):
            router.update("artifacts", Filter.of(id="shared", version=1), UpdateSpec(set={"x": 1}))
        assert router.find_one("artifacts", Filter.of(id="shared"))["id"] == "shared"
    assert _ids(configured, "artifacts") == {"shared"}


def test_an_local_run_that_updates_a_configured_record_is_refused(stores):
    router, configured, _ = stores
    configured.insert("env_state_instances", {"instance_id": "esi-shared", "n": 0})
    with run_scope(LOCAL_TASK):
        with pytest.raises(LocalRunWriteError, match="configured store"):
            router.update("env_state_instances", Filter.of(instance_id="esi-shared"), UpdateSpec(inc={"n": 1}))
        with pytest.raises(LocalRunWriteError):
            router.delete("env_state_instances", Filter.of(instance_id="esi-shared"))
        assert router.update("env_state_instances", Filter.of(instance_id="esi-missing"), UpdateSpec(inc={"n": 1})) == 0
    assert configured.find_one("env_state_instances", Filter.of(instance_id="esi-shared"))["n"] == 0


def test_a_bare_entity_cant_reference_an_local_id(stores):
    router, configured, _ = stores
    with pytest.raises(OutwardReferenceError, match=r"steps\[1\]\.env_id"):
        router.insert("tasks", {"id": "t", "version": 1, "steps": [{"id": "a"}, {"id": "d", "env_id": LOCAL_ENV}]})
    configured.insert("tasks", {"id": "t", "version": 1})
    with pytest.raises(OutwardReferenceError, match=r"set\.agent_id"):
        router.update("tasks", Filter.of(id="t", version=1), UpdateSpec(set={"agent_id": "@local/~/bundle/agents/a"}))

    router.insert("tasks", {"id": LOCAL_TASK, "version": 1, "steps": [{"env_id": LOCAL_ENV}]})
    router.insert("tasks", {"id": "t2", "version": 1, "description": f"a copy of {LOCAL_TASK}"})
    assert _ids(configured, "tasks") == {"t", "t2"}


def test_routing_is_off_until_a_process_turns_it_on(tmp_path):
    configured = LocalSqliteDocumentStore(str(tmp_path / "configured.db"))
    configure(document_store=configured)
    assert get_config().get_document_store() is configured
    with run_scope(LOCAL_TASK):
        assert not routing.in_local_run()


@pytest.fixture
def probe_command():
    seen = []

    @click.command("routing-probe")
    def probe():
        seen.append(routing.namespace_routing_enabled())

    cli.add_command(probe)
    yield seen
    cli.commands.pop("routing-probe")


def test_the_cli_routes_for_the_length_of_a_command(probe_command):
    CliRunner().invoke(cli, ["routing-probe"], catch_exceptions=False)
    assert probe_command == [True]
    assert not routing.namespace_routing_enabled()

    routing.enable_namespace_routing()
    CliRunner().invoke(cli, ["routing-probe"], catch_exceptions=False)
    assert routing.namespace_routing_enabled()


def test_default_local_users_keep_the_local_namespace_in_its_own_file(cli_routing):
    config = get_config()
    router = config.get_document_store()
    assert isinstance(router, RoutingDocumentStore)
    assert router.configured.path == state_root() / "document_store" / "documents.db"
    assert router.local.path == state_root() / "document_store" / "local.db"
    router.insert("envs", {"id": LOCAL_ENV, "version": 1})
    router.insert("envs", {"id": "rocket", "version": 1})
    assert _ids(router.local, "envs") == {LOCAL_ENV}
    assert _ids(router.configured, "envs") == {"rocket"}
    page, total = router.latest_per_id_page("envs", Filter(), limit=0)
    assert {doc["id"] for doc in page} == {LOCAL_ENV, "rocket"}
    assert total == 2
    with run_scope(LOCAL_TASK):
        assert isinstance(config.get_object_store(), LocalFilesystemObjectStore)
        with pytest.raises(LocalRunWriteError):
            get_artifact_store().next_version("cli-derived")


@pytest.fixture
def object_stores(tmp_path, cli_routing):
    configured = LocalFilesystemObjectStore(str(tmp_path / "configured-objects"))
    configure(object_store=configured)
    return configured, get_config().get_object_store_for(LOCAL_ENV)


def test_objects_route_by_their_entity_and_by_url(object_stores):
    configured, local = object_stores
    config = get_config()
    assert config.get_object_store() is configured
    assert local is not configured and local.root == state_root() / "object_store"
    assert config.get_object_store_for("rocket") is configured

    assert config.get_object_store_at(local.object_url("k")) is local
    assert config.get_object_store_at(configured.object_url("k")) is configured
    assert config.get_object_store_at("s3://bucket/k") is configured
    assert config.get_object_store_at((state_root().parent / "elsewhere.txt").as_uri()) is configured
    with pytest.raises(ValueError, match="local object store"):
        config.get_object_store_to_write(configured.object_url("k"), LOCAL_ENV)


def test_an_local_run_writes_its_objects_locally_and_reads_by_url(object_stores, tmp_path):
    configured, local = object_stores
    shared = configured.put("shared/data.json", b"registry")
    source = tmp_path / "file.txt"
    source.write_text("x")
    with run_scope(LOCAL_TASK):
        store = get_config().get_object_store()
        written = store.put("prompt_agent_trajectories/p/x.json", b"trajectory")
        assert local.get(written) == b"trajectory"
        assert store.get(shared) == b"registry"
        with pytest.raises(LocalRunWriteError):
            store.put_file_at(configured.object_url("shared/other.json"), str(source))
        with pytest.raises(LocalRunWriteError):
            store.signed_put_url("s3://bucket/k")
        with pytest.raises(LocalRunWriteError):
            get_config().get_object_store_for("rocket")
    assert configured.list("") == ["shared/data.json"]


def test_images_route_by_namespace_and_by_host(cli_routing, monkeypatch):
    created = []
    monkeypatch.setattr(LocalRegistryImageStore, "ensure_repository", lambda self, repository: created.append(repository))
    configured = OciRegistryImageStore(registry_host="registry.example.com")
    configure(image_store=configured)
    config = get_config()
    assert config.get_image_store() is configured
    local = config.get_image_store_for(LOCAL_ENV)
    assert local.registry_host == "localhost:5000"
    assert config.get_image_store_at("localhost:5000/local/e-abc:v1") is local
    assert config.get_image_store_at("registry.example.com/rocket:v1") is configured

    with run_scope(LOCAL_TASK):
        view = config.get_image_store()
        assert view.image_ref("local/e-abc", "v1") == "localhost:5000/local/e-abc:v1"
        view.ensure_repository("local/e-abc")
        assert view.owns("localhost:5000/local/e-abc:v1") and view.owns("registry.example.com/rocket:v1")
        assert not view.owns("ghcr.io/example/rocket:v1")
        with pytest.raises(LocalRunWriteError):
            view.ensure_repository("rocket")
        with pytest.raises(LocalRunWriteError):
            config.get_image_store_for("rocket")
    assert created == ["local/e-abc"]


def test_an_local_run_refuses_a_bare_artifact_before_any_upload(object_stores):
    with run_scope(LOCAL_TASK):
        with pytest.raises(LocalRunWriteError, match="'cli-rocket'"):
            get_artifact_store().next_version("cli-rocket")


def test_what_default_local_use_left_behind_stays_out_of_routed_reads(stores):
    router, configured, _ = stores
    leftovers = LocalSqliteDocumentStore(str(state_root() / "document_store" / "documents.db"))
    for version in (1, 2, 3):
        leftovers.insert("tasks", {"id": "rocket", "version": version, "project": "p"})
    leftovers.insert("env_instances", {"instance_id": "rocket-env-old", "env_id": "rocket-env", "n": 0})
    configured.insert("tasks", {"id": "rocket", "version": 1, "project": "p"})
    router.insert("tasks", {"id": LOCAL_TASK, "version": 1, "project": "p"})

    for scope in (None, LOCAL_TASK):
        with run_scope(scope) if scope else nullcontext():
            latest = router.latest_per_id("tasks", Filter.of(project="p"))
            assert sorted((d["id"], d["version"]) for d in latest) == [(LOCAL_TASK, 1), ("rocket", 1)]
            assert router.count("tasks", Filter.of(project="p")) == 2
            assert router.query("env_instances", Filter.of(env_id="rocket-env")) == []
            assert router.update("env_instances", Filter.of(instance_id="rocket-env-old"), UpdateSpec(inc={"n": 1})) == 0
    assert leftovers.find_one("env_instances", Filter.of(instance_id="rocket-env-old"))["n"] == 0


def test_an_local_runs_records_stay_readable_and_updatable_after_the_run(stores):
    router, configured, _ = stores
    with run_scope(LOCAL_TASK):
        router.insert("env_instances", {"instance_id": "rocket-1a2b", "env_id": "rocket", "n": 0})
        router.insert("env_state_instances", {"instance_id": "esi-1a2b", "n": 0})
    assert router.find_one("env_instances", Filter.of(instance_id="rocket-1a2b"))["env_id"] == "rocket"
    assert router.update("env_state_instances", Filter.of(instance_id="esi-1a2b"), UpdateSpec(inc={"n": 1})) == 1
    assert router.find_one("env_state_instances", Filter.of(instance_id="esi-1a2b"))["n"] == 1
    assert configured.count("env_instances", Filter()) == 0


class _DatetimeStore(LocalSqliteDocumentStore):
    """Hands ``created_at_utc`` back as a datetime, the way Mongo does; SQLite keeps an ISO string."""

    def query(self, *args, **kwargs):
        docs = super().query(*args, **kwargs)
        return [{**d, "created_at_utc": datetime.fromisoformat(d["created_at_utc"])} if "created_at_utc" in d else d for d in docs]


def test_merged_sorts_compare_mongo_datetimes_with_sqlite_timestamps_and_put_absent_first(tmp_path, cli_routing):
    configured = _DatetimeStore(str(tmp_path / "configured.db"))
    configure(document_store=configured)
    router = get_config().get_document_store()
    start = datetime(2026, 9, 27, tzinfo=timezone.utc)
    for hour in (1, 3, 5):
        configured.insert("tasks", {"id": f"t{hour}", "version": 1, "created_at_utc": start + timedelta(hours=hour)})
    for hour in (2, 4):
        router.local.insert("tasks", {"id": f"@local/~/b/t{hour}", "version": 1, "created_at_utc": start + timedelta(hours=hour)})
    router.local.insert("tasks", {"id": "@local/~/b/undated", "version": 1})

    ascending = router.query("tasks", Filter(), sort=Sort.by("created_at_utc", descending=False))
    assert [d["id"].rsplit("/", 1)[-1] for d in ascending] == ["undated", "t1", "t2", "t3", "t4", "t5"]
    page = router.query("tasks", Filter(), sort=Sort.by("created_at_utc"), limit=2, offset=1)
    assert [d["id"].rsplit("/", 1)[-1] for d in page] == ["t4", "t3"]
    latest = router.latest_per_id("tasks", Filter(), sort=Sort.by("created_at_utc", descending=False))
    assert latest[-1]["id"] == "@local/~/b/undated"


def test_an_entitys_objects_stay_in_its_own_namespaces_store(object_stores, tmp_path):
    configured, local = object_stores
    source = tmp_path / "file.txt"
    source.write_text("x")
    with pytest.raises(ValueError, match="configured object store"):
        FileArtifact.put_at("bare-artifact", description="d", file_path=str(source), object_url=local.object_url("k/file.txt"))
    shared = configured.put("k/shared.txt", b"x")
    with pytest.raises(ValueError, match="configured object store"):
        FileArtifact.put_existing("bare-artifact", description="d", object_url=local.put("k/local.txt", b"x"))
    assert get_config().get_object_store_to_write(shared, "bare-artifact") is configured


def test_an_unreadable_local_store_is_a_config_error_naming_it(stores):
    router, _, local = stores
    local.path.parent.mkdir(parents=True)
    local.path.write_bytes(b"not a database")
    with pytest.raises(ConfigError, match="@local store at .*local.db"):
        router.query("tasks", Filter())


def test_reports_name_the_configured_store(stores, object_stores):
    router, configured, _ = stores
    assert configured_store(router) is configured
    with run_scope(LOCAL_TASK):
        assert configured_store(get_config().get_object_store()) is object_stores[0]


def test_overlapping_routing_scopes_keep_routing_on_until_the_last_ends():
    first, second = routing.namespace_routing(), routing.namespace_routing()
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    assert routing.namespace_routing_enabled()
    second.__exit__(None, None, None)
    assert not routing.namespace_routing_enabled()


def test_registering_a_bare_artifact_at_a_local_url_is_refused(object_stores):
    _, local = object_stores
    url = local.put("k/image.tar.gz", b"x")
    with pytest.raises(ValueError, match="configured object store"):
        DockerImageArtifact.put_tar("bare-image", description="d", image_name="img:v1", tar_gz_object_url=url)
    with pytest.raises(ValueError, match="configured object store"):
        FileArtifactUniverse.put("bare-universe", file_artifacts={"a": object()}, bundle_object_url=local.object_url("k/"))


def test_a_key_both_namespaces_recorded_reads_as_the_readers_own(stores):
    router, configured, _ = stores
    key = Filter.of(env_id="rocket", artifact_id="u")
    with run_scope(LOCAL_TASK):
        router.update("env_artifacts", key, UpdateSpec(set={"verdict": "local run"}), upsert=True)
    router.update("env_artifacts", key, UpdateSpec(set={"verdict": "registry run"}), upsert=True)

    assert router.find_one("env_artifacts", key)["verdict"] == "registry run"
    assert [d["verdict"] for d in router.query("env_artifacts", key)] == ["registry run", "local run"]
    assert router.update("env_artifacts", key, UpdateSpec(set={"seen": True})) == 1
    assert configured.find_one("env_artifacts", key)["seen"] is True
    with run_scope(LOCAL_TASK):
        assert router.find_one("env_artifacts", key)["verdict"] == "local run"


def test_an_id_recorded_in_both_stores_reduces_to_one_latest_row(stores):
    router, configured, local = stores
    with run_scope(LOCAL_TASK):
        router.insert("env_snapshots", {"id": "shared", "version": 2, "at": 1})
        router.insert("env_snapshots", {"id": "tied", "version": 1, "at": 2, "from": "local"})
        router.insert("env_snapshots", {"id": "local-only", "version": 1, "at": 3})
    router.insert("env_snapshots", {"id": "shared", "version": 1, "at": 4})
    router.insert("env_snapshots", {"id": "tied", "version": 1, "at": 5, "from": "configured"})

    by_time = Sort.by("at", descending=False)
    latest = router.latest_per_id("env_snapshots", Filter(), sort=by_time)
    assert [(d["id"], d["version"], d.get("from")) for d in latest] == [
        ("shared", 2, None), ("local-only", 1, None), ("tied", 1, "configured"),
    ]
    assert [d["id"] for d in router.latest_per_id("env_snapshots", Filter(), sort=by_time, limit=1, offset=1)] == ["local-only"]
    assert router.count_distinct("env_snapshots", Filter()) == 3
    page, total = router.latest_per_id_page(
        "env_snapshots", Filter(), sort=by_time, limit=1, offset=1,
    )
    assert [doc["id"] for doc in page] == ["local-only"]
    assert total == 3
    with run_scope(LOCAL_TASK):
        assert {d["id"]: d.get("from") for d in router.latest_per_id("env_snapshots", Filter())}["tied"] == "local"


def test_latest_version_lookup_uses_the_routed_namespace(stores):
    router, _, _ = stores
    versioned = VersionedEntityStore(router, "env_snapshots", dict, dict)
    router.insert("env_snapshots", {"id": "shared", "version": 1})
    local_id = "@local/~/bundle/env_snapshots/local-only"
    with run_scope(LOCAL_TASK):
        router.insert("env_snapshots", {"id": local_id, "version": 3})
    assert versioned.get("shared")["version"] == 1
    assert versioned.get(local_id)["version"] == 3


def test_the_local_namespace_file_cant_be_the_configured_store(tmp_path, cli_routing):
    configure(document_store=LocalSqliteDocumentStore(str(state_root() / "document_store" / "local.db")))
    with pytest.raises(ConfigError, match="kept for @local documents"):
        get_config().get_document_store()


def test_an_unreadable_local_store_is_a_config_error_on_writes_too(stores):
    router, _, local = stores
    local.path.parent.mkdir(parents=True)
    local.path.write_bytes(b"not a database")
    with pytest.raises(ConfigError, match="@local store at"):
        router.insert("envs", {"id": LOCAL_ENV, "version": 1})


@pytest.mark.parametrize("entity_id, refusal", [
    ("@local/a/../b", "'..' path segment"),
    ("x-@local/y", "contains an @local id but isn't one"),
    ("local/t-chaos-640285e449ba", "spelled like an encoded @local id"),
])
def test_a_raw_entity_write_is_checked_like_a_versioned_one(stores, entity_id, refusal):
    router, configured, local = stores
    with pytest.raises(ValueError, match=refusal):
        router.insert("envs", {"id": entity_id, "version": 1})
    with pytest.raises(ValueError, match=refusal):
        router.update("envs", Filter.of(id=entity_id, version=1), UpdateSpec(set={"x": 1}), upsert=True)
    assert configured.count("envs", Filter()) == 0
    assert not (state_root() / "document_store" / "local.db").exists()


def test_batch_instance_lookup_keeps_the_first_routed_copy(stores):
    router, configured, local = stores
    configured.ensure_index("task_instances", ["instance_id"], unique=True)
    local.ensure_index("task_instances", ["instance_id"], unique=True)
    configured.insert("task_instances", {"instance_id": "shared", "current_step": 2})
    local.insert("task_instances", {"instance_id": "shared", "current_step": 7})

    for scope in (nullcontext(), run_scope(LOCAL_TASK)):
        with scope:
            expected = router.find_one("task_instances", Filter.of(instance_id="shared"))
            assert router.find_many_by_id("task_instances", "instance_id", ["shared"]) == [expected]


def test_an_entity_store_defined_outside_core_is_routed_by_id(stores):
    router, configured, local = stores
    plugin_things = VersionedEntityStore(router, "plugin_things", serialize=dict, deserialize=dict)

    plugin_things.put({"id": LOCAL_ENV})
    plugin_things.put({"id": "registry-thing"})

    assert [d["id"] for d in local.query("plugin_things", Filter())] == [LOCAL_ENV]
    assert [d["id"] for d in configured.query("plugin_things", Filter())] == ["registry-thing"]
    with run_scope(LOCAL_TASK), pytest.raises(LocalRunWriteError):
        plugin_things.put({"id": "another-registry-thing"})


@pytest.mark.parametrize("base, wrapper", [
    (DocumentStore, RoutingDocumentStore),
    (ObjectStore, LocalRunObjectStore),
    (ImageStore, LocalRunImageStore),
])
def test_a_routing_wrapper_says_how_every_store_method_routes(base, wrapper):
    """A method its store gains later must be defined on the wrapper: inheriting the store's
    default would skip routing without anyone noticing."""
    public = {
        name for name, member in inspect.getmembers(base)
        if not name.startswith("_") and (callable(member) or isinstance(member, property))
    }
    assert public - set(vars(wrapper)) == _NOT_ROUTED


class _SharingStore(LocalFilesystemObjectStore):
    """A configured store that hands its credentials to the workloads agent-env deploys."""

    def shared_credentials_env(self) -> dict[str, str]:
        return {"AWS_ACCESS_KEY_ID": "configured-id", "AWS_SECRET_ACCESS_KEY": "configured-secret"}


def test_an_agent_deployed_in_an_at_local_run_gets_none_of_the_configured_stores_credentials(tmp_path):
    configured = _SharingStore(str(tmp_path / "configured"))
    routed = LocalRunObjectStore(configured, LocalFilesystemObjectStore(str(tmp_path / "local")))
    agent = A2AAgent(id="solver", version=1, docker_image_artifact=SimpleNamespace(image_name="img"))
    try:
        set_object_store(configured)
        assert agent._build_merged_env({}, 8000)["AWS_ACCESS_KEY_ID"] == "configured-id"
        set_object_store(routed)
        assert routed.shared_credentials_env() == {}
        assert "AWS_ACCESS_KEY_ID" not in agent._build_merged_env({}, 8000)
    finally:
        reset_config()
