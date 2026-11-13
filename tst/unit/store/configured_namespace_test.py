"""Configured namespaces: the ids the namespace resolver places in a namespace, ids derived from them, what a run of
their tasks records and every entity and object that run writes keep to the namespace's own stores; unplaced reads see
a namespace only inside its run or once a service opens it; a write asks the resolver fresh and fails when it can't."""

import inspect

import pytest

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.config import get_config, reset_config
from agent_env.config.errors import ConfigError
from agent_env.store import Filter, LocalSqliteDocumentStore, UpdateSpec
from agent_env.store.ids import id_bases, validate_namespace_name
from agent_env.store.namespace_resolver import NamespaceResolver, TableNamespaceResolver
from agent_env.store.object_store import LocalFilesystemObjectStore, ObjectStore
from agent_env.store.routing import (
    NamespaceDocumentStore,
    NamespaceRunObjectStore,
    NamespaceWriteError,
    PrefixedObjectStore,
    enable_namespace_routing,
    reading_namespaces,
    run_scope,
)
from tst.conftest import _reset_store_singletons
from tst.store import conformance

TEAM_TASK = "team-task"
TEAM_ENV = "team-env"
OTHER_ENV = "other-env"
_NOT_ROUTED = {"from_config"}
_RESOLVER = f"""
[namespace_resolver]
impl = "agent_env.store.namespace_resolver:TableNamespaceResolver"
config = {{ ids = {{ team-a = ["{TEAM_TASK}", "{TEAM_ENV}"], team-b = ["{OTHER_ENV}"] }} }}
"""


def _write_config(tmp_path, namespaces: str) -> None:
    (tmp_path / "config.toml").write_text(f"""
[stores.document]
impl = "agent_env.store.document_store:LocalSqliteDocumentStore"
config = {{ path = "{tmp_path / 'shared.db'}" }}

[stores.object]
impl = "agent_env.store.object_store:LocalFilesystemObjectStore"
config = {{ root = "{tmp_path / 'shared-objects'}" }}
{namespaces}
""")


def _namespace(tmp_path, name: str, objects: str = "") -> str:
    return f"""
[namespaces.{name}.document]
impl = "agent_env.store.document_store:LocalSqliteDocumentStore"
config = {{ path = "{tmp_path / f'{name}.db'}" }}

[namespaces.{name}.object]
impl = "agent_env.store.object_store:LocalFilesystemObjectStore"
config = {{ root = "{tmp_path / (objects or f'{name}-objects')}" }}
"""


@pytest.fixture
def configured(tmp_path, monkeypatch):
    """A shared store and two configured namespaces, ``team-a`` and ``team-b``, sharing one object root."""
    _write_config(tmp_path, _namespace(tmp_path, "team-a", "restricted") + _namespace(tmp_path, "team-b", "restricted")
                  + _RESOLVER)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()
    _reset_store_singletons()
    yield get_config()
    reset_config()
    _reset_store_singletons()


@pytest.fixture
def stores(configured):
    router = configured.get_document_store()
    assert isinstance(router, NamespaceDocumentStore)
    return router, router.configured, router.namespaces["team-a"]


class _Recording(NamespaceResolver):
    """Places ids from a table, recording each question, and fails while ``down``."""

    def __init__(self, table):
        self.table, self.asked, self.down = table, [], False

    def namespace_of(self, entity_id, *, fresh=False):
        self.asked.append((entity_id, fresh))
        if self.down:
            raise RuntimeError("restriction rows unreadable")
        return self.table.get(entity_id)


def _ids(store, collection, field="id"):
    return {doc[field] for doc in store.query(collection, Filter())}


# The suite's own collection is no entity's, so a run's writes all land in the run's namespace.
_CONFORMANCE = [
    pytest.param(case, mode, id=f"{mode}-{case.__name__}")
    for mode in ("shared", "opened", "run")
    for case in conformance.CASES
]


@pytest.mark.parametrize("case, mode", _CONFORMANCE)
def test_the_router_passes_the_conformance_suite(case, mode, stores):
    router, _, team = stores
    team.insert("seed", {"id": "seed"})
    if mode == "opened":
        with reading_namespaces("team-a"):
            case(router, "coll")
    elif mode == "run":
        with run_scope(TEAM_TASK):
            case(router, "coll")
    else:
        case(router, "coll")


def test_no_namespaces_leave_the_configured_store_unwrapped(tmp_path, monkeypatch):
    _write_config(tmp_path, "")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()

    assert isinstance(get_config().get_document_store(), LocalSqliteDocumentStore)
    assert isinstance(get_config().get_object_store(), LocalFilesystemObjectStore)
    assert get_config().configured_namespaces() == ()
    assert get_config().namespace_of(TEAM_TASK) is None


@pytest.mark.parametrize("table, message", [
    ('[namespaces.Team.document]\nimpl = "x:Y"\n[namespaces.Team.object]\nimpl = "x:Y"\n', "lowercase"),
    ('[namespaces.local.document]\nimpl = "x:Y"\n[namespaces.local.object]\nimpl = "x:Y"\n', "local namespace"),
    ('[namespaces.team-a.document]\nimpl = "x:Y"\n', "a document and an object table"),
])
def test_a_namespace_table_is_checked(tmp_path, monkeypatch, table, message):
    _write_config(tmp_path, table)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()

    with pytest.raises(ConfigError, match=message):
        get_config().get_document_store()


def test_namespaces_need_a_resolver(tmp_path, monkeypatch):
    _write_config(tmp_path, _namespace(tmp_path, "team-a"))
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()

    with pytest.raises(ConfigError, match=r"\[namespace_resolver\]"):
        get_config().get_document_store()


def test_a_resolver_naming_an_unconfigured_namespace_is_refused(stores):
    router, _, _ = stores
    router.resolver = TableNamespaceResolver({"team-c": ["stray"]})
    with pytest.raises(ConfigError, match="no \\[namespaces\\] table"):
        router.insert("envs", {"id": "stray", "version": 1})


def test_ids_keep_the_configured_stores_rules(stores):
    router, _, _ = stores
    router.check_id(TEAM_ENV)
    with pytest.raises(ValueError, match="reserved"):
        router.check_id("@team-a/envs/e")


def test_namespace_names_and_id_bases():
    validate_namespace_name("team-a")
    for bad in ("", "-a", "a-", "A", "a_b", "x" * 65, "local", "hf"):
        with pytest.raises(ValueError):
            validate_namespace_name(bad)
    assert id_bases("u__files__ab12") == ["u__files__ab12", "u__files", "u"]
    assert TableNamespaceResolver({"team-a": ["u"]}).namespace_of("u__files__ab12") == "team-a"
    assert TableNamespaceResolver({"team-a": ["u"]}).namespace_of("u2") is None


def test_entities_route_by_resolver_and_unplaced_reads_see_only_opened_namespaces(stores):
    router, shared, team = stores
    router.insert("envs", {"id": TEAM_ENV, "version": 1})
    router.insert("envs", {"id": f"{TEAM_ENV}__cli", "version": 1})
    router.insert("envs", {"id": "rocket", "version": 1})

    assert _ids(team, "envs") == {TEAM_ENV, f"{TEAM_ENV}__cli"}
    assert _ids(shared, "envs") == {"rocket"}
    assert router.find_one("envs", Filter.of(id=TEAM_ENV))["id"] == TEAM_ENV
    assert _ids(router, "envs") == {"rocket"}
    with reading_namespaces("team-a"):
        assert _ids(router, "envs") == {TEAM_ENV, f"{TEAM_ENV}__cli", "rocket"}
    with run_scope(TEAM_TASK):
        assert _ids(router, "envs") == {TEAM_ENV, f"{TEAM_ENV}__cli", "rocket"}
    with reading_namespaces("team-b"):
        assert _ids(router, "envs") == {"rocket"}


def test_writes_ask_the_resolver_fresh_and_fail_when_it_cant_answer(stores):
    router, shared, _ = stores
    router.resolver = resolver = _Recording({TEAM_ENV: "team-a"})
    router.insert("envs", {"id": TEAM_ENV, "version": 1})
    assert (TEAM_ENV, True) in resolver.asked

    resolver.down = True
    with pytest.raises(RuntimeError, match="unreadable"):
        router.insert("envs", {"id": "rocket", "version": 1})
    assert _ids(shared, "envs") == set()


def test_a_run_asks_once_where_its_task_is(stores):
    router, _, team = stores
    router.resolver = resolver = _Recording({TEAM_TASK: "team-a"})
    with run_scope(TEAM_TASK):
        router.insert("task_step_journal", {"instance_id": "ti_1", "step": "a"})
        router.insert("task_step_journal", {"instance_id": "ti_1", "step": "b"})
    assert resolver.asked.count((TEAM_TASK, True)) == 1
    assert len(team.query("task_step_journal", Filter())) == 2


def test_a_runs_records_and_the_records_of_a_namespaced_task_keep_to_the_namespace(stores):
    router, shared, team = stores
    with run_scope(TEAM_TASK):
        router.insert("task_instances", {"instance_id": "ti_1", "task_id": TEAM_TASK})
        router.insert("task_step_journal", {"instance_id": "ti_1", "step": "a"})
    router.insert("task_instances", {"instance_id": "ti_2", "task_id": TEAM_TASK})
    router.insert("task_instances", {"instance_id": "ti_3", "task_id": "bare-task"})

    assert _ids(team, "task_instances", "instance_id") == {"ti_1", "ti_2"}
    assert _ids(team, "task_step_journal", "instance_id") == {"ti_1"}
    assert _ids(shared, "task_instances", "instance_id") == {"ti_3"}
    assert router.find_one("task_instances", Filter.of(instance_id="ti_1")) is None
    assert router.find_one("task_instances", Filter.of(task_id=TEAM_TASK)) is not None
    with reading_namespaces("team-a"):
        assert router.find_one("task_instances", Filter.of(instance_id="ti_1"))["task_id"] == TEAM_TASK


def test_a_nested_run_keeps_the_outer_namespace_and_a_bare_outer_one_takes_the_inner_ones(stores):
    router, shared, team = stores
    with run_scope(TEAM_TASK), run_scope("bare-task"):
        router.insert("task_instances", {"instance_id": "ti_nested", "task_id": "bare-task"})
    with run_scope("bare-task"), run_scope(TEAM_TASK):
        router.insert("task_instances", {"instance_id": "ti_inner", "task_id": TEAM_TASK})
    assert _ids(team, "task_instances", "instance_id") == {"ti_nested", "ti_inner"}
    assert _ids(shared, "task_instances", "instance_id") == set()


def test_a_run_writes_new_bare_entities_to_its_namespace_but_no_new_version_of_a_shared_one(stores):
    router, shared, team = stores
    router.insert("artifacts", {"id": "shared-file", "version": 1})
    with run_scope(TEAM_TASK):
        router.insert("artifacts", {"id": "ti_1__recording", "version": 1})
        with pytest.raises(NamespaceWriteError, match="new version of 'shared-file'"):
            router.insert("artifacts", {"id": "shared-file", "version": 2})
        with pytest.raises(NamespaceWriteError, match="which is in 'team-b'"):
            router.insert("artifacts", {"id": OTHER_ENV, "version": 1})

    assert _ids(team, "artifacts") == {"ti_1__recording"}
    assert _ids(shared, "artifacts") == {"shared-file"}


def test_a_run_may_update_a_shared_entity(stores):
    router, shared, _ = stores
    router.insert("envs", {"id": "rocket", "version": 1, "metadata": {}})
    with run_scope(TEAM_TASK):
        router.update("envs", Filter.of(id="rocket", version=1), UpdateSpec(set={"metadata.checked": True}))
    assert shared.find_one("envs", Filter.of(id="rocket"))["metadata"] == {"checked": True}


def test_a_run_cannot_update_a_shared_run_record(stores):
    router, shared, _ = stores
    shared.insert("task_instances", {"instance_id": "ti_shared", "status": "running"})
    with run_scope(TEAM_TASK), pytest.raises(NamespaceWriteError, match="in the configured store"):
        router.update("task_instances", Filter.of(instance_id="ti_shared"), UpdateSpec(set={"status": "done"}))
    assert shared.find_one("task_instances", Filter.of(instance_id="ti_shared"))["status"] == "running"


def test_local_routing_runs_on_top_of_configured_namespaces(stores, configured):
    enable_namespace_routing()
    router = configured.get_document_store()
    router.insert("envs", {"id": "@local/~/b/envs/e", "version": 1})
    router.insert("envs", {"id": TEAM_ENV, "version": 1})

    assert _ids(router.local, "envs") == {"@local/~/b/envs/e"}
    assert _ids(stores[2], "envs") == {TEAM_ENV}


def test_a_runs_objects_go_to_its_namespace_under_its_prefix(configured, tmp_path):
    shared = configured.get_object_store()
    shared_url = shared.put("inputs/a.txt", b"shared")
    with run_scope(TEAM_TASK):
        store = configured.get_object_store()
        assert isinstance(store, NamespaceRunObjectStore)
        url = store.put("trajectories/ti_1.json", b"{}")
        assert store.get(shared_url) == b"shared"
        with pytest.raises(NamespaceWriteError, match="writes objects only to its own store"):
            store.signed_put_url(shared_url)
        other = configured.namespace_object_store("team-b").put("x", b"b")
        with pytest.raises(NamespaceWriteError, match="an object of 'team-b'"):
            store.get(other)

    assert (tmp_path / "restricted" / "team-a" / "trajectories" / "ti_1.json").read_bytes() == b"{}"
    assert not (tmp_path / "shared-objects" / "trajectories").exists()
    assert configured.get_object_store() is shared
    assert configured.get_object_store_at(url) is configured.namespace_object_store("team-a")


def test_entity_objects_follow_the_entity_and_the_run(configured):
    team = configured.namespace_object_store("team-a")
    assert configured.get_object_store_for(TEAM_ENV) is team
    assert configured.get_object_store_for(f"{TEAM_ENV}__files__ab12") is team
    assert configured.get_object_store_for("bare") is configured.get_object_store()
    with run_scope(TEAM_TASK):
        assert configured.get_object_store_for("bare") is team


def test_an_object_url_is_recorded_only_in_its_namespace(configured):
    team_url = configured.namespace_object_store("team-a").object_url("artifacts/x")
    shared_url = configured.get_object_store().object_url("artifacts/x")
    configured.check_object_url(TEAM_ENV, team_url)
    configured.check_object_url("bare", shared_url)
    with pytest.raises(ValueError, match="keeps its objects in the 'team-a'"):
        configured.check_object_url(TEAM_ENV, shared_url)
    with pytest.raises(ValueError, match="an object of 'team-a'"):
        configured.check_object_url("bare", team_url)
    with pytest.raises(ValueError, match="keeps its objects in the 'team-a'"):
        configured.check_object_url(f"{TEAM_ENV}__y", configured.namespace_object_store("team-b").object_url("y"))


def test_a_namespaced_artifact_round_trips_through_its_own_stores(stores, configured, tmp_path):
    _, shared, team = stores
    file_id = f"{TEAM_ENV}__readme"
    stored = FileArtifact.put_bytes(file_id, description="d", filename="f.txt", content=b"hi")

    assert stored.object_url.startswith(configured.namespace_object_store("team-a").object_url(""))
    assert _ids(team, "artifacts") == {file_id}
    assert _ids(shared, "artifacts") == set()
    assert get_artifact_store().get(file_id).object_url == stored.object_url
    assert configured.get_object_store_at(stored.object_url).get(stored.object_url) == b"hi"
    assert list((tmp_path / "shared-objects").rglob("f.txt")) == []


def test_the_prefixed_store_keeps_to_its_prefix(tmp_path):
    inner = LocalFilesystemObjectStore(str(tmp_path))
    store = PrefixedObjectStore(inner, "team-a/")
    url = store.put("k/v", b"1")
    outside = inner.put("k/v", b"2")

    assert store.owns(url) and not store.owns(outside)
    assert store.get_object_key(url) == "k/v"
    assert store.list("k/") == ["k/v"]
    assert store.read("k/v") == b"1"
    with pytest.raises(ValueError, match="prefix"):
        store.put_file_at(outside, str(tmp_path / "team-a" / "k" / "v"))
    with pytest.raises(ValueError, match="ends with '/'"):
        PrefixedObjectStore(inner, "team-a")


@pytest.mark.parametrize("wrapper", [PrefixedObjectStore, NamespaceRunObjectStore])
def test_the_object_wrappers_cover_every_store_method(wrapper):
    methods = {name for name, member in inspect.getmembers(ObjectStore) if callable(member) and not name.startswith("_")}
    covered = {name for name in methods
               if any(name in vars(cls) for cls in wrapper.__mro__ if cls is not ObjectStore and cls is not object)}
    assert methods - covered - _NOT_ROUTED == set()
