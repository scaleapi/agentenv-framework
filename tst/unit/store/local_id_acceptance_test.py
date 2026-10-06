"""``@local`` ids are accepted by the ``@local`` namespace's store alone, which the CLI routes them to;
every other store keeps refusing them, and the per-user store refuses bare ids spelled like one."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.a2a_agent.validator import A2AAgentValidator
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.cli import cli
from agent_env.config import configure, get_config, set_image_store, set_object_store
from agent_env.config.paths import state_root
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.snapshot_store import EnvSnapshot
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxProvider,
    reset_agent_sandbox_provider,
    set_agent_sandbox_provider,
)
from agent_env.store import Filter, LocalSqliteDocumentStore, VersionedEntityStore
from agent_env.store.base import NotFoundError
from agent_env.store.ids import fs_safe, key_segment
from agent_env.store.routing import LocalNamespaceDocumentStore
from agent_env.task import Task
from tst.unit.store.fakes import FakeDocumentStore, FakeImageStore, SigningObjectStore, reattach_for_snapshot

LOCAL_ID = "@local/t/Chaos"
LOCAL_ENV = "@local/~/bundle/envs/e"
UNIVERSE = "@local/~/bundle/artifacts/data"


def _documents() -> LocalSqliteDocumentStore:
    return LocalSqliteDocumentStore(str(state_root() / "document_store" / "documents.db"))


def _local() -> LocalSqliteDocumentStore:
    return LocalSqliteDocumentStore(str(state_root() / "document_store" / "local.db"))


def test_only_the_local_namespaces_store_takes_local_ids(tmp_path):
    namespace = LocalNamespaceDocumentStore(str(tmp_path / "local.db"))
    namespace.check_id(LOCAL_ID)
    for store in (LocalSqliteDocumentStore(str(tmp_path / "documents.db")), FakeDocumentStore()):
        with pytest.raises(ValueError, match="only the @local namespace's store holds"):
            store.check_id(LOCAL_ID)
    for refused, reason in (("registry-env", "isn't an @local id"), ("@other/x", "isn't an @local id"),
                            ("@local/a/../b", "'..' path segment")):
        with pytest.raises(ValueError, match=reason):
            namespace.check_id(refused)
    assert not (tmp_path / "local.db").exists()


@pytest.mark.parametrize("entity_id", [
    key_segment(LOCAL_ID), fs_safe(LOCAL_ID), f"{fs_safe(LOCAL_ID)}/data.json", "local/anything", "local-0123456789ab",
])
def test_a_local_store_refuses_a_bare_id_spelled_like_an_encoded_local_id(tmp_path, entity_id):
    with pytest.raises(ValueError, match="spelled like an encoded @local id"):
        LocalSqliteDocumentStore(str(tmp_path / "documents.db")).check_id(entity_id)
    FakeDocumentStore().check_id(entity_id)


@pytest.mark.parametrize("entity_id", [
    "local-dev-env", "local", "localhost/x", f"{fs_safe(LOCAL_ID)}-v2", "my-local-0123456789ab", "local-ABCDEF012345",
])
def test_bare_ids_that_merely_start_with_local_still_write(tmp_path, entity_id):
    LocalSqliteDocumentStore(str(tmp_path / "documents.db")).check_id(entity_id)


def test_the_router_checks_an_id_against_the_store_it_routes_to(tmp_path, cli_routing):
    configure(document_store=LocalSqliteDocumentStore(str(tmp_path / "configured.db")))
    things = VersionedEntityStore(get_config().get_document_store(), "envs", serialize=dict, deserialize=dict)

    assert things.put({"id": LOCAL_ID}) == 1 and things.put({"id": "registry-env"}) == 1
    for refused in ("@other/x", key_segment(LOCAL_ID)):
        with pytest.raises(ValueError):
            things.put({"id": refused})
    assert [d["id"] for d in _local().query("envs", Filter())] == [LOCAL_ID]


@pytest.mark.parametrize("entity", [{}, {"id": None}, {"id": 7}])
def test_an_entity_without_a_string_id_is_refused_before_any_store_is_asked(tmp_path, entity):
    things = VersionedEntityStore(FakeDocumentStore(), "envs", serialize=dict, deserialize=dict)
    with pytest.raises(ValueError, match="needs a string id"):
        things.put(entity)


def test_under_the_cli_an_local_artifact_is_written_to_and_read_from_the_local_namespaces_store(
    local_stores, tmp_path,
):
    source = tmp_path / "in"
    source.mkdir()
    (source / "a.txt").write_text("A")

    put = CliRunner().invoke(cli, ["artifact", "file-artifact-universe", "put-bundled", "--id", UNIVERSE, "--file-dir", str(source)])
    got = CliRunner().invoke(cli, ["artifact", "file-artifact-universe", "get", "--id", UNIVERSE, "--output-dir", str(tmp_path / "out")])

    assert put.exit_code == 0, put.output
    assert got.exit_code == 0, got.output
    assert (tmp_path / "out" / "a.txt").read_text() == "A"
    assert {d["id"] for d in _local().query("artifacts", Filter())} >= {UNIVERSE}
    assert all(not d["id"].startswith("@") for d in _documents().query("artifacts", Filter()))
    with pytest.raises(NotFoundError):
        FileArtifactUniverse.get(UNIVERSE)


def test_outside_the_cli_an_local_id_is_refused(local_stores, tmp_path):
    (tmp_path / "a.txt").write_text("A")
    with pytest.raises(ValueError, match="only the @local namespace's store holds"):
        FileArtifactUniverse.put_bundled(UNIVERSE, files={"a.txt": tmp_path / "a.txt"})
    assert not (state_root() / "document_store" / "local.db").exists()


GITHUB = "https://github.com/o/r/blob/main/Dockerfile"


@pytest.mark.parametrize("build", [
    lambda: DockerImageArtifact.put_from_github(id="@local/~/bundle/artifacts/image", dockerfile_github_url=GITHUB),
    lambda: MCPServerEnv.put_from_github(id="@local/~/bundle/envs/e", dockerfile_github_url=GITHUB),
    lambda: WebsiteEnv.put_from_github(
        id="@local/~/bundle/envs/site", backend_dockerfile_github_url=GITHUB, frontend_dockerfile_github_url=GITHUB,
    ),
], ids=["docker_image", "mcp_server", "website"])
def test_a_github_build_refuses_an_local_id_before_touching_a_registry_or_a_vm(local_stores, cli_routing, monkeypatch, build):
    images = FakeImageStore()
    set_image_store(images)
    monkeypatch.setattr("agent_env.providers.get_sandbox_provider", lambda *a, **k: pytest.fail("a build VM was requested"))

    with pytest.raises(ValueError, match="a GitHub build publishes to the configured registry"):
        asyncio.run(build())

    assert images.repositories == []


def _validation_runs_nothing(monkeypatch):
    universes = {u: SimpleNamespace(id=u, version=2, get_environment_artifacts=lambda: []) for u in ("registry-universe", UNIVERSE)}
    monkeypatch.setattr("agent_env.artifact.EnvironmentUniverseArtifact.get", lambda id, version=None: universes[id])
    finished = SimpleNamespace(deployed_envs=[], deployed_agents=[], instance_id="validation-run")
    monkeypatch.setattr(Task, "run", lambda self, **kwargs: asyncio.sleep(0, result=finished))


@pytest.mark.parametrize("validate, task_id", [
    (lambda: MCPServerEnv(id=LOCAL_ENV, version=1, docker_image_artifact=MagicMock(), environment_name="svc").validate(),
     f"{LOCAL_ENV}__validate-v1"),
    (lambda: WebsiteEnv(id=LOCAL_ENV, version=1, backend_docker_image_artifact=MagicMock(), frontend_docker_image_artifact=MagicMock(),
                        environment_name="svc").validate(),
     f"{LOCAL_ENV}__validate-v1"),
    (lambda: MultiEnv(id=LOCAL_ENV, version=1, mcp_server_envs=[]).validate(), f"{LOCAL_ENV}__validate-v1"),
    (lambda: MultiEnv(id=LOCAL_ENV, version=1, mcp_server_envs=[]).validate_universe_compatibility("registry-universe"),
     f"{LOCAL_ENV}__validate-universe-compat-v1-registry-universe-v2"),
    (lambda: MultiEnv(id="registry-env", version=1, mcp_server_envs=[]).validate_universe_compatibility(UNIVERSE),
     f"{UNIVERSE}__validate-universe-compat-v2-registry-env-v1"),
], ids=["mcp_server", "website", "multi", "compat-local-env", "compat-local-universe"])
def test_validating_an_local_entity_writes_its_task_under_it_to_the_local_store(local_stores, cli_routing, monkeypatch,
                                                                                validate, task_id):
    _validation_runs_nothing(monkeypatch)

    assert asyncio.run(validate()) == "validation-run"

    assert [d["id"] for d in _local().query("tasks", Filter())] == [task_id]
    assert not _documents().path.exists() or _documents().count("tasks", Filter()) == 0


class _RemoteProvider(SandboxProvider):
    async def create_sandbox(self, **kwargs):
        raise NotImplementedError


@pytest.fixture
def remote_agents():
    set_agent_sandbox_provider(_RemoteProvider())
    yield
    reset_agent_sandbox_provider()


class _Stopped(Exception):
    pass


def test_validating_an_local_agent_on_a_remote_provider_keeps_its_fixtures_and_task_in_the_local_stores(
    local_stores, cli_routing, remote_agents, tmp_path, monkeypatch,
):
    configured = SigningObjectStore(str(tmp_path / "configured-objects"))
    set_object_store(configured)
    agent = A2AAgent(id="@local/~/bundle/agents/a", version=1, docker_image_artifact=MagicMock())

    def stop(self, **kwargs):
        raise _Stopped

    monkeypatch.setattr(Task, "run", stop)

    with pytest.raises(_Stopped):
        asyncio.run(A2AAgentValidator.validate(agent))

    fixtures = f"a2a_validator/probe_fixtures/{key_segment(agent.id)}-v1"
    skills = f"a2a_validator/validator_skill/{key_segment(agent.id)}-v1"
    assert sorted(get_config().get_object_store_for(agent.id).list("a2a_validator/")) == [
        f"{fixtures}/clip.mp4", f"{fixtures}/red.png",
        *(f"{skills}/{name}/SKILL.md" for name in ("validator-probe-bundle", "validator-test-s3")),
    ]
    assert configured.list("") == []
    assert _local().count("tasks", Filter()) == 1
    assert not _documents().path.exists() or _documents().count("tasks", Filter()) == 0


def test_a_put_validates_an_local_entity_like_any_other(local_stores, monkeypatch):
    image_url = local_stores.get_object_store().put("artifacts/docker_image/srv-img/1/x.tar.gz", b"x")
    image = DockerImageArtifact.put_tar("srv-img", description="d", image_name="reg/srv:v1", tar_gz_s3_url=image_url)
    MCPServerEnv.put(id="srv", docker_image_artifact=image, environment_name="svc")
    _validation_runs_nothing(monkeypatch)

    result = CliRunner().invoke(cli, ["env", "multi", "put", "--id", "@local/~/bundle/envs/m", "--mcp-server", "srv", "--validate"])

    assert result.exit_code == 0, result.output
    assert [d["id"] for d in _local().query("envs", Filter())] == ["@local/~/bundle/envs/m"]
    assert [d["id"] for d in _local().query("tasks", Filter())] == ["@local/~/bundle/envs/m__validate-v1"]


def test_snapshotting_an_local_env_stops_at_the_local_store_it_cannot_presign_before_the_sandbox_runs_anything(
    local_stores, cli_routing, tmp_path, monkeypatch,
):
    configured = SigningObjectStore(str(tmp_path / "configured-objects"))
    set_object_store(configured)
    sandbox = reattach_for_snapshot(monkeypatch, LOCAL_ENV, 1, UNIVERSE, 1)

    with pytest.raises(RuntimeError, match="LocalFilesystemObjectStore can't presign uploads"):
        asyncio.run(EnvSnapshot.create("instance-1"))

    assert sandbox.scripts == []
    assert configured.list("") == [] and get_config().get_object_store_for(LOCAL_ENV).list("") == []


@pytest.mark.parametrize("env_id, universe_id", [(LOCAL_ENV, "registry-universe"), ("registry-env", UNIVERSE)],
                         ids=["local-env", "local-universe"])
def test_a_snapshot_of_an_env_and_a_universe_in_different_namespaces_is_refused_before_the_sandbox_is_reached(
    local_stores, cli_routing, monkeypatch, env_id, universe_id,
):
    reattach_for_snapshot(monkeypatch, env_id, 1, universe_id, 1)
    monkeypatch.setattr(MultiEnv, "from_deployed_env", classmethod(lambda cls, record: pytest.fail("the sandbox was reached")))

    with pytest.raises(ValueError, match="are in different namespaces"):
        asyncio.run(EnvSnapshot.create("instance-1"))


@pytest.mark.parametrize("derived", ["env-snapshot-@local/~/bundle/envs/e", "cua-vm-@local/~/bundle/envs/e", "x-@local/y"])
def test_under_the_cli_a_bare_id_derived_from_an_local_one_is_refused_before_any_upload(local_stores, cli_routing, derived):
    with pytest.raises(ValueError, match="contains an @local id but isn't one"):
        FileArtifact.put_bytes(id=derived, description="d", filename="f.txt", content=b"x")
    with pytest.raises(ValueError, match="contains an @local id but isn't one"):
        VersionedEntityStore(get_config().get_document_store(), "envs", serialize=dict, deserialize=dict).put({"id": derived})

    assert local_stores.get_object_store().list("") == []
    LocalSqliteDocumentStore(str(state_root() / "services.db")).check_id(derived)


class _Deployable:
    """An env or agent whose deploy records what a deployment does: a state instance naming no
    owner, and an instance of a bare infrastructure env."""

    def __init__(self, id):
        self.id, self.version, self.type = id, 1, "mcp_server"
        self.docker_image_artifact = SimpleNamespace(id="image")

    @classmethod
    def get(cls, id, version=None):
        return cls(id)

    async def deploy(self, **kwargs):
        store = get_config().get_document_store()
        store.insert("env_state_instances", {"instance_id": f"esi-{self.id}", "state_type": "none"})
        store.insert("env_instances", {"instance_id": f"gateway-for-{self.id}", "env_id": "gateway"})
        return SimpleNamespace(
            instance_id="i", mcp_url="m", gateway_url="g", db_web_url=None, db_mcp_url=None, vnc_url=None,
            website_frontend_urls=None, expires_at_utc=None, a2a_url="a", sandbox_id="s", agent_card={},
        )


@pytest.mark.parametrize("command, module, cls", [
    ("env", "agent_env.cli.env.deploy", "Env"),
    ("a2a-agent", "agent_env.cli.a2a_agent.deploy", "A2AAgent"),
])
@pytest.mark.parametrize("deployed, lands_locally", [("@local/~/bundle/envs/e", True), ("registry-env", False)])
def test_a_cli_deployment_records_follow_the_deployed_ids_namespace(
    local_stores, monkeypatch, command, module, cls, deployed, lands_locally,
):
    monkeypatch.setattr(importlib.import_module(module), cls, _Deployable)

    result = CliRunner().invoke(cli, [command, "deploy", "--id", deployed])

    assert result.exit_code == 0, result.output
    _assert_recorded(deployed, lands_locally, "env_instances", f"gateway-for-{deployed}")


@pytest.mark.parametrize("env_id, lands_locally", [("@local/~/bundle/envs/e", True), ("registry-env", False)])
def test_a_pre_initialized_env_state_follows_the_envs_namespace(local_stores, monkeypatch, env_id, lands_locally):
    async def acquire(**kwargs):
        get_config().get_document_store().insert("env_state_instances", {"instance_id": f"esi-{env_id}", "state_type": "db"})
        return SimpleNamespace(instance_id=f"esi-{env_id}", metadata={}, created_at_utc=None, expires_at_utc=None)

    async def prepare(names, instance):
        return None

    init = importlib.import_module("agent_env.cli.env.state.init")
    monkeypatch.setattr(init, "Env", _Deployable)
    monkeypatch.setattr(init, "_resolve_environment_names", lambda env: ["svc"])
    monkeypatch.setattr("agent_env.providers.env_state.acquire_state_for_deploy", acquire)
    monkeypatch.setattr("agent_env.providers.env_state.build_state_provider", lambda _type: SimpleNamespace(prepare=prepare))

    result = CliRunner().invoke(cli, ["env", "init-env-state", "--id", env_id, "--env-state-type", "db"])

    assert result.exit_code == 0, result.output
    _assert_recorded(env_id, lands_locally)


def _assert_recorded(owner, lands_locally, *also):
    holding, other = (_local(), _documents()) if lands_locally else (_documents(), _local())
    assert holding.find_one("env_state_instances", Filter.of(instance_id=f"esi-{owner}")) is not None
    if also:
        collection, instance_id = also
        assert holding.find_one(collection, Filter.of(instance_id=instance_id)) is not None
    assert not other.path.exists() or other.count("env_state_instances", Filter()) == 0
