"""Tests for ServiceDBEnv and docker-compose generation with ServiceDB."""

from unittest.mock import MagicMock

from agent_env.env.envs.service_db import (
    DB_NAME,
    DB_PASSWORD,
    SERVICE_DB_PORT,
    DB_USER,
    ServiceDBConfig,
    ServiceDBEnv,
)
from agent_env.providers.gateway_provider import (
    DATABASE_SERVICE_NAME,
    DB_MCP_SERVICE_NAME,
    GatewayProvider,
    MCPServerConfig,
    PGWEB_SERVICE_NAME,
    WebsiteConfig,
)
from agent_env.providers.state import LocalPostgresStateProvider

# create_docker_compose requires an explicit state provider; these render tests are local-Postgres.
_LOCAL_STATE = LocalPostgresStateProvider()
_LOCAL_INSTANCE = _LOCAL_STATE.default_instance()  # static instance for render-only tests


def create_mock_artifact(image_name: str = "service-db:latest", artifact_id: str = "test-artifact") -> MagicMock:
    """Create a mock DockerImageArtifact."""
    artifact = MagicMock()
    artifact.id = artifact_id
    artifact.version = 1
    artifact.type = "docker_image"
    artifact.image_name = image_name
    return artifact


def create_service_db(**kwargs) -> ServiceDBEnv:
    """Create a ServiceDBEnv with default mock artifacts."""
    db_artifact = kwargs.pop("db_docker_image_artifact", create_mock_artifact())
    db_web_artifact = kwargs.pop("db_web_docker_image_artifact", create_mock_artifact(image_name="sosedoff/pgweb", artifact_id="test-db-web-artifact"))
    db_mcp_artifact = kwargs.pop("db_mcp_docker_image_artifact", create_mock_artifact(image_name="agent-env-db-mcp", artifact_id="test-db-mcp-artifact"))
    return ServiceDBEnv(db_docker_image_artifact=db_artifact, db_web_docker_image_artifact=db_web_artifact, db_mcp_docker_image_artifact=db_mcp_artifact, **kwargs)


class TestServiceDBEnv:
    """Tests for ServiceDBEnv class."""

    def test_to_config_carries_image_tags(self):
        """to_config surfaces the three docker image tags for compose/sidecar rendering."""
        service_db = create_service_db(id="test-db")
        cfg = service_db.to_config()
        assert cfg.db_image == service_db.db_docker_image_artifact.image_name
        assert cfg.db_web_image == service_db.db_web_docker_image_artifact.image_name
        assert cfg.db_mcp_image == service_db.db_mcp_docker_image_artifact.image_name

    def test_to_dict(self):
        """Test serialization to dict."""
        service_db = create_service_db(id="test-db", version=1)
        data = service_db.to_dict()
        assert data["id"] == "test-db"
        assert data["version"] == 1
        assert data["type"] == "service_db"
        assert data["db_docker_image_artifact"]["id"] == "test-artifact"
        assert data["db_docker_image_artifact"]["version"] == 1
        assert data["db_web_docker_image_artifact"]["id"] == "test-db-web-artifact"
        assert data["db_web_docker_image_artifact"]["version"] == 1
        assert data["db_mcp_docker_image_artifact"]["id"] == "test-db-mcp-artifact"
        assert data["db_mcp_docker_image_artifact"]["version"] == 1


class TestDockerComposeWithServiceDB:
    """Tests for docker-compose generation with ServiceDB."""

    def test_servicedb_service_included(self):
        """Test that servicedb service is included in docker-compose."""
        provider = GatewayProvider()
        artifact = create_mock_artifact("postgres:16-alpine")
        service_db = create_service_db(id="test-db", db_docker_image_artifact=artifact)
        servers = [MCPServerConfig(image="mcp-slack:latest", environment_name="slack")]

        compose = provider.create_docker_compose(
            mcp_servers=servers,            state_provider=LocalPostgresStateProvider(service_db_config=service_db.to_config()),
            state_instance=_LOCAL_INSTANCE,
        )

        assert f"{DATABASE_SERVICE_NAME}:" in compose
        assert "image: postgres:16-alpine" in compose

    def test_servicedb_environment_vars(self):
        """Test that servicedb has correct PostgreSQL environment variables."""
        provider = GatewayProvider()
        artifact = create_mock_artifact()
        service_db = create_service_db(id="test-db", db_docker_image_artifact=artifact)
        servers = [MCPServerConfig(image="mcp-slack:latest", environment_name="slack")]

        compose = provider.create_docker_compose(
            mcp_servers=servers,            state_provider=LocalPostgresStateProvider(service_db_config=service_db.to_config()),
            state_instance=_LOCAL_INSTANCE,
        )

        assert f"POSTGRES_USER={DB_USER}" in compose
        assert f"POSTGRES_PASSWORD={DB_PASSWORD}" in compose
        assert f"POSTGRES_DB={DB_NAME}" in compose

    def test_servicedb_healthcheck(self):
        """Test that servicedb has healthcheck configuration."""
        provider = GatewayProvider()
        artifact = create_mock_artifact()
        service_db = create_service_db(id="test-db", db_docker_image_artifact=artifact)
        servers = [MCPServerConfig(image="mcp-slack:latest", environment_name="slack")]

        compose = provider.create_docker_compose(
            mcp_servers=servers,            state_provider=LocalPostgresStateProvider(service_db_config=service_db.to_config()),
            state_instance=_LOCAL_INSTANCE,
        )

        assert "healthcheck:" in compose
        assert f"pg_isready -U {DB_USER}" in compose

    def test_mcp_servers_depend_on_servicedb(self):
        """Test that MCP servers depend on servicedb with service_healthy condition."""
        provider = GatewayProvider()
        artifact = create_mock_artifact()
        service_db = create_service_db(id="test-db", db_docker_image_artifact=artifact)
        servers = [
            MCPServerConfig(image="mcp-slack:latest", environment_name="slack"),
            MCPServerConfig(image="mcp-email:latest", environment_name="email"),
        ]

        compose = provider.create_docker_compose(
            mcp_servers=servers,            state_provider=LocalPostgresStateProvider(service_db_config=service_db.to_config()),
            state_instance=_LOCAL_INSTANCE,
        )

        assert "depends_on:" in compose
        assert f"{DATABASE_SERVICE_NAME}:" in compose
        assert "condition: service_healthy" in compose

    def test_mcp_servers_have_per_service_database_url(self):
        """Test that MCP servers have per-service DATABASE_URL with schema search_path."""
        provider = GatewayProvider()
        artifact = create_mock_artifact()
        service_db = create_service_db(id="test-db", db_docker_image_artifact=artifact)
        servers = [
            MCPServerConfig(image="mcp-slack:latest", environment_name="slack"),
            MCPServerConfig(image="mcp-email:latest", environment_name="email"),
        ]

        compose = provider.create_docker_compose(
            mcp_servers=servers,            state_provider=LocalPostgresStateProvider(service_db_config=service_db.to_config()),
            state_instance=_LOCAL_INSTANCE,
        )

        # Each service should have its own quoted schema in the search_path (%22 is URL-encoded ")
        slack_url = f"DATABASE_URL=postgresql://{DB_USER}:{DB_PASSWORD}@{DATABASE_SERVICE_NAME}:{SERVICE_DB_PORT}/{DB_NAME}?options=-c%20search_path%3D%22slack%22,public"
        email_url = f"DATABASE_URL=postgresql://{DB_USER}:{DB_PASSWORD}@{DATABASE_SERVICE_NAME}:{SERVICE_DB_PORT}/{DB_NAME}?options=-c%20search_path%3D%22email%22,public"
        assert slack_url in compose
        assert email_url in compose

    def test_website_backend_has_per_service_database_url(self):
        """Website backends get the same per-service search_path DATABASE_URL as MCP servers."""
        provider = GatewayProvider()
        service_db = create_service_db(id="test-db")
        compose = provider.create_docker_compose(
            mcp_servers=[],            website_configs=[WebsiteConfig(backend_image="shop-be", frontend_image="shop-fe", environment_name="shop")],
            state_provider=LocalPostgresStateProvider(service_db_config=service_db.to_config()),
            state_instance=_LOCAL_INSTANCE,
        )
        shop_url = f"DATABASE_URL=postgresql://{DB_USER}:{DB_PASSWORD}@{DATABASE_SERVICE_NAME}:{SERVICE_DB_PORT}/{DB_NAME}?options=-c%20search_path%3D%22shop%22,public"
        assert shop_url in compose

    def test_compose_static_instance_matches_live(self):
        """The export/snapshot render path feeds the static ``default_instance``; a live-acquired
        instance carrying the same base URL must render byte-identically (per-service schemas are
        derived at wire time, so only the base URL matters)."""
        from agent_env.providers.state import EnvStateInstance, LocalPostgresStateProvider

        provider = GatewayProvider()
        service_db = create_service_db(id="test-db")
        servers = [
            MCPServerConfig(image="mcp-slack:latest", environment_name="slack"),
            MCPServerConfig(image="mcp-email:latest", environment_name="email"),
        ]
        websites = [WebsiteConfig(backend_image="shop-be", frontend_image="shop-fe", environment_name="shop")]

        state_provider = LocalPostgresStateProvider(service_db_config=service_db.to_config())
        static = state_provider.default_instance()  # export/snapshot render path
        live = EnvStateInstance(  # live-acquired instance with the same base URL
            state_type=state_provider.type,
            metadata={"host": DATABASE_SERVICE_NAME},
            _db_url_base=static._db_url_base,
        )

        via_static = provider.create_docker_compose(
            mcp_servers=servers, website_configs=websites,
            state_provider=state_provider, state_instance=static,
        )
        via_live = provider.create_docker_compose(
            mcp_servers=servers, website_configs=websites,
            state_provider=state_provider, state_instance=live,
        )
        assert via_static == via_live

    def test_sidecar_urls_share_mcp_creds_source(self):
        """P1 regression: pgweb PGWEB_DATABASE_URL and db-mcp DATABASE_URI are built from the
        same DB_USER/DB_PASSWORD/DB_NAME as the MCP DATABASE_URL (one creds source in the
        provider), not hardcoded independently."""
        provider = GatewayProvider()
        service_db = create_service_db(id="test-db")  # defaults include pgweb + db-mcp images
        compose = provider.create_docker_compose(
            mcp_servers=[MCPServerConfig(image="mcp-slack:latest", environment_name="slack")],            state_provider=LocalPostgresStateProvider(service_db_config=service_db.to_config()),
            state_instance=_LOCAL_INSTANCE,
        )
        creds = f"{DB_USER}:{DB_PASSWORD}@{DATABASE_SERVICE_NAME}:{SERVICE_DB_PORT}/{DB_NAME}"
        assert f"PGWEB_DATABASE_URL=postgres://{creds}?sslmode=disable" in compose
        assert f"DATABASE_URI=postgresql://{creds}" in compose
        # pgweb is locked to the single connection so contributors can't
        # Disconnect (which would 500 the changelog export on Save & Next).
        assert "PGWEB_LOCK_SESSION=true" in compose

    def test_servicedb_has_init_script_volume(self):
        """Test that servicedb has volume mount for init script."""
        provider = GatewayProvider()
        artifact = create_mock_artifact()
        service_db = create_service_db(id="test-db", db_docker_image_artifact=artifact)
        servers = [MCPServerConfig(image="mcp-slack:latest", environment_name="slack")]

        compose = provider.create_docker_compose(
            mcp_servers=servers,            state_provider=LocalPostgresStateProvider(service_db_config=service_db.to_config()),
            state_instance=_LOCAL_INSTANCE,
        )

        assert "volumes:" in compose
        assert "init-schemas.sql:/docker-entrypoint-initdb.d/init-schemas.sql" in compose

    def test_gateway_depends_on_servicedb_and_mcp_servers(self):
        """Gateway depends on both the MCP servers and servicedb directly.

        The gateway connects to the service DB itself (SERVICE_DB_URL, used for
        query_state / changelog rollback), so it waits on servicedb being healthy
        in addition to the MCP servers."""
        provider = GatewayProvider()
        artifact = create_mock_artifact()
        service_db = create_service_db(id="test-db", db_docker_image_artifact=artifact)
        servers = [MCPServerConfig(image="mcp-slack:latest", environment_name="slack")]

        compose = provider.create_docker_compose(
            mcp_servers=servers,            state_provider=LocalPostgresStateProvider(service_db_config=service_db.to_config()),
            state_instance=_LOCAL_INSTANCE,
        )

        lines = compose.split("\n")
        in_gateway = False
        in_gateway_depends = False
        gateway_depends = []
        for line in lines:
            if line.strip() == "gateway:":
                in_gateway = True
            elif in_gateway and "depends_on:" in line:
                in_gateway_depends = True
            elif in_gateway and in_gateway_depends:
                stripped = line.strip()
                # Dict format: "slack:" at 6-space indent (depends_on children)
                if stripped.endswith(":") and line.startswith(" " * 6) and not line.startswith(" " * 8):
                    gateway_depends.append(stripped[:-1])
                # List format: "- slack"
                elif stripped.startswith("- "):
                    gateway_depends.append(stripped[2:])
                elif stripped and not line.startswith(" " * 6):
                    break

        assert "slack" in gateway_depends
        assert DATABASE_SERVICE_NAME in gateway_depends


def test_store_images_to_load_returns_servicedb_artifacts():
    """The local provider preloads the servicedb image set as artifacts (needs tar_gz_s3_url, which
    the image-name config can't give); it reuses the cached env (injected here to avoid Mongo)."""
    service_db = create_service_db(id="test-db")
    provider = LocalPostgresStateProvider()
    provider._service_db_env = service_db  # cached resolve, bypassing Env.get
    assert provider.store_images_to_load() == [
        service_db.db_docker_image_artifact,
        service_db.db_web_docker_image_artifact,
        service_db.db_mcp_docker_image_artifact,
    ]


def test_rendered_sidecar_service_names_matches_creation_gating():
    """The status-check gate (rendered_sidecar_service_names) must match render_sidecar_containers'
    per-image gating: pgweb iff db_web_image, db-mcp iff db_mcp_image. The config is held on the
    provider instance, so each variant is a separately-constructed provider."""
    both = LocalPostgresStateProvider(service_db_config=ServiceDBConfig(db_web_image="pgweb:1", db_mcp_image="dbmcp:1"))
    assert both.rendered_sidecar_service_names() == [PGWEB_SERVICE_NAME, DB_MCP_SERVICE_NAME]

    pgweb_only = LocalPostgresStateProvider(service_db_config=ServiceDBConfig(db_web_image="pgweb:1", db_mcp_image=None))
    assert pgweb_only.rendered_sidecar_service_names() == [PGWEB_SERVICE_NAME]

    none = LocalPostgresStateProvider(service_db_config=ServiceDBConfig(db_web_image=None, db_mcp_image=None))
    assert none.rendered_sidecar_service_names() == []

    # And the names it reports are exactly the services render_sidecar_containers emits.
    rendered = "\n".join(both.render_sidecar_containers(["svc"]))
    for name in both.rendered_sidecar_service_names():
        assert f"  {name}:" in rendered
