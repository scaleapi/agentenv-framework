"""Tests for the EnvStateProvider abstraction + LocalPostgresStateProvider"""

import pytest

from agent_env.env.envs.service_db import DB_NAME, DB_PASSWORD, DB_USER, SERVICE_DB_PORT, ServiceDBConfig
from agent_env.providers.env_state import (
    EnvStateInstance,
    EnvStateInstanceStore,
    EnvStateProvider,
    LOCAL_POSTGRES_STATE_TYPE,
    LocalPostgresStateContext,
    LocalPostgresStateProvider,
    StateContext,
    build_state_provider,
    reset_env_state_instance_store,
    set_env_state_instance_store,
)
from agent_env.providers.env_state.store import ENV_STATE_INSTANCES_COLLECTION
from agent_env.store.document_store import Filter
from agent_env.config import get_config, set_document_store
from agent_env.providers.env_state.local_postgres import _STOCK_POSTGRES_IMAGE
from agent_env.store import EcrImageStore, OciRegistryImageStore
from tst.unit.store.fakes import FakeDocumentStore


@pytest.fixture(autouse=True)
def _fake_store():
    """Real EnvStateInstanceStore over an in-memory DocumentStore — exercises
    mint/timestamp/persist without touching a real backend."""
    store = EnvStateInstanceStore()
    set_document_store(FakeDocumentStore())
    set_env_state_instance_store(store)
    yield store
    reset_env_state_instance_store()


def test_build_state_provider_local_postgres():
    provider = build_state_provider(LocalPostgresStateProvider.type)
    assert isinstance(provider, LocalPostgresStateProvider)
    assert isinstance(provider, EnvStateProvider)


def test_local_postgres_provider_type_attr():
    assert LocalPostgresStateProvider.type == "local_postgres"


def test_build_state_provider_unknown_backend():
    with pytest.raises(ValueError, match="Unknown env state type"):
        build_state_provider("snowflake")


def test_build_state_provider_returns_fresh_instances():
    assert build_state_provider(LocalPostgresStateProvider.type) is not build_state_provider(
        LocalPostgresStateProvider.type
    )


@pytest.mark.asyncio
async def test_local_acquire_builds_and_persists_instance(_fake_store):
    provider = LocalPostgresStateProvider()
    ctx = LocalPostgresStateContext(environment_names=["slack", "email"], host="servicedb", ttl_seconds=60)
    instance = await provider.acquire(ctx)

    assert isinstance(instance, EnvStateInstance)
    assert instance.state_type == LOCAL_POSTGRES_STATE_TYPE
    assert instance.metadata == {"host": "servicedb"}
    # The store minted the id + stamped timestamps.
    assert instance.instance_id.startswith("esi-")
    assert instance.created_at_utc and instance.expires_at_utc
    # The live wiring URL is set on the returned object...
    expected = f"postgresql://{DB_USER}:{DB_PASSWORD}@servicedb:{SERVICE_DB_PORT}/{DB_NAME}"
    assert instance._db_url_base == expected
    # ...but is NEVER persisted: the stored doc omits it.
    assert "_db_url_base" not in instance.to_dict()
    persisted = _fake_store._doc_store.docs
    assert len(persisted) == 1
    assert persisted[0]["instance_id"] == instance.instance_id
    assert "_db_url_base" not in persisted[0]


@pytest.mark.asyncio
async def test_local_acquire_uses_context_host():
    """Container-mode host (i6pn) flows through the context into _db_url_base."""
    provider = LocalPostgresStateProvider()
    ctx = LocalPostgresStateContext(environment_names=["slack"], host="[fdaa::1]")
    instance = await provider.acquire(ctx)
    assert instance._db_url_base == f"postgresql://{DB_USER}:{DB_PASSWORD}@[fdaa::1]:{SERVICE_DB_PORT}/{DB_NAME}"


@pytest.mark.asyncio
async def test_local_acquire_requires_host():
    provider = LocalPostgresStateProvider()
    ctx = LocalPostgresStateContext(environment_names=["slack"], host="")
    with pytest.raises(ValueError, match="host"):
        await provider.acquire(ctx)


@pytest.mark.asyncio
async def test_local_acquire_rejects_base_state_context():
    # The base StateContext lacks environment_names/host; the provider requires its own
    # LocalPostgresStateContext so backend-specific inputs are explicit.
    provider = LocalPostgresStateProvider()
    with pytest.raises(TypeError, match="LocalPostgresStateContext"):
        await provider.acquire(StateContext())


@pytest.mark.asyncio
async def test_local_teardown_retires_record(_fake_store):
    provider = LocalPostgresStateProvider()
    instance = EnvStateInstance(state_type=LOCAL_POSTGRES_STATE_TYPE, instance_id="esi-abc")
    _fake_store._doc_store.insert(ENV_STATE_INSTANCES_COLLECTION, instance.to_dict())
    assert await provider.teardown(instance) is None
    # No container to drop for local; teardown only pulls expires_at_utc to now.
    stored = _fake_store._doc_store.find_one(
        ENV_STATE_INSTANCES_COLLECTION, Filter.of(instance_id="esi-abc")
    )
    assert stored["expires_at_utc"] is not None


def test_env_state_instance_to_dict_omits_transient_url():
    inst = EnvStateInstance(
        instance_id="esi-x", state_type=LOCAL_POSTGRES_STATE_TYPE,
        metadata={"host": "servicedb"}, _db_url_base="postgresql://secret@host/db",
    )
    d = inst.to_dict()
    assert d == {
        "instance_id": "esi-x",
        "state_type": LOCAL_POSTGRES_STATE_TYPE,
        "metadata": {"host": "servicedb"},
        "created_at_utc": None,
        "expires_at_utc": None,
    }
    # Round-trips (minus the transient url, which reattach re-mints).
    assert EnvStateInstance.from_dict(d)._db_url_base == ""


def test_local_store_spec():
    spec = LocalPostgresStateProvider().store_spec(["slack"])
    assert spec.env == {
        "POSTGRES_USER": DB_USER,
        "POSTGRES_PASSWORD": DB_PASSWORD,
        "POSTGRES_DB": DB_NAME,
    }
    assert spec.healthcheck == {
        "test": ["CMD-SHELL", f"pg_isready -U {DB_USER}"],
        "interval": "5s",
        "timeout": "5s",
        "retries": 5,
    }
    assert spec.port == SERVICE_DB_PORT
    assert 'CREATE SCHEMA IF NOT EXISTS "slack";' in spec.init_sql
    assert "public._changelog" in spec.init_sql


def test_get_init_script_schemas_and_changelog():
    """One quoted schema per service + the shared changelog objects."""
    script = LocalPostgresStateProvider.get_init_script(["slack", "website-browser"])
    assert 'CREATE SCHEMA IF NOT EXISTS "slack";' in script
    assert 'CREATE SCHEMA IF NOT EXISTS "website-browser";' in script
    assert "CREATE TABLE IF NOT EXISTS public._changelog" in script
    assert "CREATE OR REPLACE FUNCTION public._changelog_trigger_fn()" in script
    assert "CREATE OR REPLACE FUNCTION public._install_changelog_triggers(" in script


def test_get_init_script_empty_still_has_changelog():
    """Empty service list → no per-service schema, but the changelog SQL is still there.
    (Per-service schemas are double-quoted; the changelog SQL creates its own unquoted
    internal `agent_env_internal` schema, so assert on the quoted per-service form.)"""
    script = LocalPostgresStateProvider.get_init_script([])
    assert 'CREATE SCHEMA IF NOT EXISTS "' not in script
    assert "CREATE TABLE IF NOT EXISTS public._changelog" in script


def test_get_init_script_changelog_components():
    """The changelog SQL contains all required table columns, trigger ops, and install fn."""
    script = LocalPostgresStateProvider.get_init_script(["slack"])
    for col in ("schema_name TEXT NOT NULL", "table_name TEXT NOT NULL",
                "operation TEXT NOT NULL", "summary TEXT NOT NULL", "changed_fields JSONB"):
        assert col in script
    for op in ("TG_OP = 'INSERT'", "TG_OP = 'DELETE'", "TG_OP = 'UPDATE'"):
        assert op in script
    assert "_install_changelog_triggers(_schema_name TEXT)" in script
    assert "WHERE schemaname = _schema_name" in script


def test_get_init_script_row_id_uses_real_primary_key():
    """row_id derives from the table's actual PRIMARY KEY (pg_index), not a hardcoded 'id'."""
    script = LocalPostgresStateProvider.get_init_script(["slack"])
    assert "indisprimary" in script
    assert "_pk_cols" in script
    assert "row_to_json(OLD) ->> 'id'" not in script
    assert "row_to_json(NEW) ->> 'id'" not in script


def test_pg_url_builds_base_url():
    """_pg_url is the single base-URL builder from the store creds."""
    assert (
        LocalPostgresStateProvider._pg_url("servicedb")
        == f"postgresql://{DB_USER}:{DB_PASSWORD}@servicedb:{SERVICE_DB_PORT}/{DB_NAME}"
    )


def test_url_for_environment_scopes_search_path():
    """url_for_environment appends a %22-quoted search_path option scoping the base URL to the
    service's schema; here on the static export/snapshot instance (default_instance)."""
    provider = LocalPostgresStateProvider()
    inst = provider.default_instance()
    base = provider.base_url(inst)
    assert provider.url_for_environment("slack", instance=inst) == f"{base}?options=-c%20search_path%3D%22slack%22,public"
    # Hyphenated names must be %22-quoted (invalid in search_path otherwise).
    assert (
        provider.url_for_environment("website-browser", instance=inst)
        == f"{base}?options=-c%20search_path%3D%22website-browser%22,public"
    )


def test_url_for_environment_uses_ampersand_when_base_has_query():
    """Uses & (not ?) when the instance's base URL already carries query params."""
    provider = LocalPostgresStateProvider()
    inst = EnvStateInstance(
        state_type=provider.type, _db_url_base="postgresql://u:p@h:5432/d?sslmode=require"
    )
    assert provider.url_for_environment("slack", instance=inst) == (
        "postgresql://u:p@h:5432/d?sslmode=require&options=-c%20search_path%3D%22slack%22,public"
    )


_REGISTRY = "us-west1-docker.pkg.dev"
_REPOSITORIES = f"{_REGISTRY}/example-project/agentenv"


@pytest.fixture
def image_store():
    """A registry other than ECR's: Artifact Registry."""
    get_config().set_image_store(OciRegistryImageStore(_REGISTRY, "example-project/agentenv"))


@pytest.mark.parametrize(
    ("db_image", "used"),
    [
        (f"{_REPOSITORIES}/servicedb:v3", True),
        (f"{_REGISTRY}/other-project/agentenv/servicedb:v3", True),
        ("us-east1-docker.pkg.dev/example-project/agentenv/servicedb:v3", False),
        ("postgres:16-alpine", False),
    ],
    ids=["in-the-store", "same-registry-other-project", "other-registry", "docker-hub"],
)
def test_container_store_image_is_the_configured_one_only_when_the_image_store_owns_it(image_store, db_image, used, caplog):
    """Container mode pulls with the image store's credentials, so any other image falls back to
    stock postgres (postgres needs no customization), with a warning."""
    image = LocalPostgresStateProvider(ServiceDBConfig(db_image=db_image)).container_store_image()
    assert image == (db_image if used else _STOCK_POSTGRES_IMAGE)
    assert ("is not in the image store" in caplog.text) is not used


def test_an_ecr_image_store_keeps_its_images():
    ecr = "123456789012.dkr.ecr.us-west-2.amazonaws.com"
    get_config().set_image_store(EcrImageStore(registry_host=ecr))
    provider = LocalPostgresStateProvider(ServiceDBConfig(db_image=f"{ecr}/servicedb:tag"))
    assert provider.container_store_image() == f"{ecr}/servicedb:tag"


def test_sidecar_specs_skip_images_the_image_store_does_not_own(image_store, caplog):
    """The gating lives in the provider, not the gateway. The sidecar target host comes from the
    acquired instance's descriptor, not a separately-threaded arg."""
    instance = EnvStateInstance(
        state_type=LocalPostgresStateProvider.type, metadata={"host": "[fdaa::1]"}
    )

    # Both in the image store → both sidecars, pointed at the instance's host. The config is held
    # on the provider instance, not threaded through sidecar_specs.
    both = LocalPostgresStateProvider(
        service_db_config=ServiceDBConfig(db_web_image=f"{_REPOSITORIES}/pgweb:v1", db_mcp_image=f"{_REPOSITORIES}/db-mcp:v1")
    ).sidecar_specs(instance=instance)
    assert {s.name for s in both} == {"pgweb", "db-mcp"}
    assert all("[fdaa::1]" in next(iter(s.env.values())) for s in both)
    # pgweb runs locked to its single connection (Connect/Disconnect hidden) so a
    # contributor can't tear down the shared session the changelog export relies on.
    pgweb_spec = next(s for s in both if s.name == "pgweb")
    assert pgweb_spec.env["PGWEB_LOCK_SESSION"] == "true"

    # pgweb from Docker Hub + unset db-mcp → neither, and a warning for the configured one only.
    none = LocalPostgresStateProvider(
        service_db_config=ServiceDBConfig(db_web_image="sosedoff/pgweb", db_mcp_image=None)
    ).sidecar_specs(instance=instance)
    assert none == []
    assert "pgweb image sosedoff/pgweb is not in the image store" in caplog.text
    assert "db-mcp image" not in caplog.text


def test_container_images_name_what_the_container_path_starts(image_store):
    """The servicedb's image and each sidecar's the provider provisions, so a gateway can check them before it starts
    any."""
    provider = LocalPostgresStateProvider(ServiceDBConfig(db_image=f"{_REPOSITORIES}/servicedb:v3",
                                                          db_web_image=f"{_REPOSITORIES}/pgweb:v1",
                                                          db_mcp_image="crystaldba/postgres-mcp"))

    assert provider.container_images() == [f"{_REPOSITORIES}/servicedb:v3", f"{_REPOSITORIES}/pgweb:v1"]
