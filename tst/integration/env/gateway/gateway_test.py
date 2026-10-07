import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from typing import NamedTuple
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# Each test deploys a real GatewayEnv + MCP servers on sandbox VMs
# (~5-15 min per test). Mark the entire module as slow.
pytestmark = [pytest.mark.int_test_slow]

from agent_env.artifact import Artifact, CliArtifact, DockerImageArtifact, FileArtifact, FileArtifactUniverse, EnvironmentArtifact, EnvironmentUniverseArtifact
from agent_env.env import Env, GatewayEnv, MCPServerEnv, MultiEnv
from agentenv_protocol import DataPart, FilePart, client as protocol_v1
from agent_env.env.envs import WebsiteEnv
from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER, GatewayMode, TOOL_DISABLE_ACTION, TOOL_ENABLE_ACTION
from agent_env.env.gateway.get_time import (
    GET_TIME_TOOL_DESCRIPTION as GET_TIME_DESC,
    GET_TIME_TOOL_NAME as GET_TIME,
)
from agent_env.env.envs.service_db import ServiceDBEnv
from agent_env.env.envs.website_browser import (
    PLAYWRIGHT_MCP_VERSION,
    WEBSITE_BROWSER_IMAGE_TAG,
    WEBSITE_BROWSER_ENVIRONMENT_NAME,
)
from agent_env.config import get_config
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.util.capabilities import MCP_SERVER_SOURCES, missing_capability_reason, skip_without_remote_sandbox

logger = logging.getLogger(__name__)



REPO_ROOT = Path(__file__).resolve().parents[4]
TST_DATA_DIR = REPO_ROOT / "tst" / "data"
ENV_DIR = REPO_ROOT / "src" / "agent_env" / "env"
ENVS_DIR = ENV_DIR / "envs"


@dataclass
class DockerImageConfig:
    """Configuration for building a Docker image."""

    name: str  # Short name for filtering (e.g., "gateway", "slack", "email")
    dockerfile: Path
    tag: str  # Docker image tag (e.g., "env-gateway")
    context: Path
    service_name: str  # MCP service name for tool prefixes


GATEWAY_IMAGE = DockerImageConfig(
    name="gateway",
    dockerfile=ENV_DIR / "gateway" / "Dockerfile",
    tag="env-gateway",
    context=ENV_DIR,
    service_name="gateway",
)

SERVICE_DB_IMAGE = DockerImageConfig(
    name="service-db",
    dockerfile=ENVS_DIR / "service_db" / "Dockerfile",
    tag="agent-env-service-db",
    context=ENVS_DIR / "service_db",
    service_name="servicedb",
)

DB_WEB_IMAGE = DockerImageConfig(
    name="db-web",
    dockerfile=ENVS_DIR / "service_db" / "Dockerfile.db-web",
    tag="agent-env-db-web",
    context=ENVS_DIR / "service_db",
    service_name="pgweb",
)

DB_MCP_IMAGE = DockerImageConfig(
    name="db-mcp",
    dockerfile=ENVS_DIR / "service_db" / "Dockerfile.db-mcp",
    tag="agent-env-db-mcp",
    context=ENVS_DIR / "service_db",
    service_name="db-mcp",
)

ALL_MCP_SERVER_IMAGES = [
    DockerImageConfig(
        name="slack",
        dockerfile=TST_DATA_DIR / "slack_mcp" / "Dockerfile",
        tag="mcp-slack",
        context=TST_DATA_DIR,
        service_name="slack",
    ),
    DockerImageConfig(
        name="email",
        dockerfile=TST_DATA_DIR / "email_mcp" / "Dockerfile",
        tag="mcp-email",
        context=TST_DATA_DIR,
        service_name="email",
    ),
]

WEBSITE_BROWSER_IMAGE = DockerImageConfig(
    name="website-browser",
    dockerfile=ENVS_DIR / "website_browser" / "Dockerfile",
    tag=WEBSITE_BROWSER_IMAGE_TAG,
    context=ENVS_DIR / "website_browser",
    service_name=WEBSITE_BROWSER_ENVIRONMENT_NAME,
)

SLACK_WEBSITE_BACKEND_IMAGE = DockerImageConfig(
    name="slack-website-backend",
    dockerfile=TST_DATA_DIR / "slack_website" / "backend" / "Dockerfile",
    tag="slack-website-backend",
    context=TST_DATA_DIR,
    service_name="slack",
)

SLACK_WEBSITE_FRONTEND_IMAGE = DockerImageConfig(
    name="slack-website-frontend",
    dockerfile=TST_DATA_DIR / "slack_website" / "frontend" / "Dockerfile",
    tag="slack-website-frontend",
    context=TST_DATA_DIR,
    service_name="slack",
)


@pytest.fixture(scope="module")
def gateway_env() -> GatewayEnv:
    """Build gateway image, create artifact, and create GatewayEnv."""
    img = GATEWAY_IMAGE
    logger.info(f"Building {img.name} Docker image...")
    result = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64", "-f", str(img.dockerfile), "-t", img.tag, str(img.context)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{img.name} build failed: {result.stderr}")

    artifact = DockerImageArtifact.put(
        id=f"gateway-{img.name}",
        description=f"Gateway {img.name} server image",
        image_name=img.tag,
    )

    env = GatewayEnv.put(
        id="gateway",
        docker_image_artifact=artifact,
    )
    logger.info(f"Created GatewayEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def service_db_env() -> ServiceDBEnv:
    """Build service-db image, create artifact, and create ServiceDBEnv."""
    img = SERVICE_DB_IMAGE
    logger.info(f"Building {img.name} Docker image...")
    result = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64", "-f", str(img.dockerfile), "-t", img.tag, str(img.context)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{img.name} build failed: {result.stderr}")

    db_artifact = DockerImageArtifact.put(
        id=f"service-db-{img.name}",
        description=f"ServiceDB {img.name} image",
        image_name=img.tag,
    )

    db_web = DB_WEB_IMAGE
    logger.info(f"Building {db_web.name} Docker image...")
    result = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64", "-f", str(db_web.dockerfile), "-t", db_web.tag, str(db_web.context)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{db_web.name} build failed: {result.stderr}")

    db_web_artifact = DockerImageArtifact.put(
        id="db-web-service-db",
        description="db-web lightweight web UI for database inspection",
        image_name=db_web.tag,
    )

    # Build db-mcp image
    db_mcp = DB_MCP_IMAGE
    logger.info(f"Building {db_mcp.name} Docker image...")
    result = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64", "-f", str(db_mcp.dockerfile), "-t", db_mcp.tag, str(db_mcp.context)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{db_mcp.name} build failed: {result.stderr}")
    db_mcp_artifact = DockerImageArtifact.put(
        id="db-mcp-service-db",
        description="PostgreSQL MCP server for direct database access",
        image_name=db_mcp.tag,
    )

    env = ServiceDBEnv.put(
        id="service-db",
        db_docker_image_artifact=db_artifact,
        db_web_docker_image_artifact=db_web_artifact,
        db_mcp_docker_image_artifact=db_mcp_artifact,
    )
    logger.info(f"Created ServiceDBEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def website_browser_env() -> MCPServerEnv:
    """Build website-browser image, create artifact, and create MCPServerEnv."""
    img = WEBSITE_BROWSER_IMAGE
    logger.info(f"Building {img.name} Docker image...")
    result = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64",
         "--build-arg", f"PLAYWRIGHT_MCP_VERSION={PLAYWRIGHT_MCP_VERSION}",
         "-f", str(img.dockerfile), "-t", img.tag, str(img.context)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{img.name} build failed: {result.stderr}")

    artifact = DockerImageArtifact.put(
        id=f"website-browser-{img.name}",
        description="Website browser MCP server image",
        image_name=img.tag,
    )
    env = MCPServerEnv.put(
        id="website-browser",
        docker_image_artifact=artifact,
        environment_name=WEBSITE_BROWSER_ENVIRONMENT_NAME,
    )
    logger.info(f"Created website-browser MCPServerEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def slack_website_env() -> WebsiteEnv:
    """Build slack website frontend/backend images and create WebsiteEnv."""
    # Build backend
    be = SLACK_WEBSITE_BACKEND_IMAGE
    logger.info(f"Building {be.name} Docker image...")
    result = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64",
         "-f", str(be.dockerfile), "-t", be.tag, str(be.context)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{be.name} build failed: {result.stderr}")
    backend_artifact = DockerImageArtifact.put(
        id="integ-test-website-backend-slack",
        description="Slack website backend",
        image_name=be.tag,
    )

    # Build frontend
    fe = SLACK_WEBSITE_FRONTEND_IMAGE
    logger.info(f"Building {fe.name} Docker image...")
    result = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64",
         "-f", str(fe.dockerfile), "-t", fe.tag, str(fe.context)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{fe.name} build failed: {result.stderr}")
    frontend_artifact = DockerImageArtifact.put(
        id="integ-test-website-frontend-slack",
        description="Slack website frontend",
        image_name=fe.tag,
    )

    env = WebsiteEnv.put(
        id="integ-test-slack-website",
        backend_docker_image_artifact=backend_artifact,
        frontend_docker_image_artifact=frontend_artifact,
        environment_name="slack",
    )
    logger.info(f"Created WebsiteEnv: {env.id} version={env.version}")
    return env


# Function-scoped so the ids survive the per-test config reset; the env fixtures it
# depends on stay module-scoped, so the deploys still happen once.
@pytest.fixture(autouse=True)
def configure_default_envs(gateway_env, service_db_env, website_browser_env):
    """Set default env IDs for tests."""
    config = get_config()
    config.default_gateway_env_id = gateway_env.id
    config.default_service_db_env_id = service_db_env.id
    config.default_website_browser_env_id = website_browser_env.id
    yield


@pytest.fixture(scope="module")
def mcp_server_envs() -> dict[str, MCPServerEnv]:
    """Build MCP server images, create artifacts, and create MCPServerEnvs."""
    envs = {}
    for img in ALL_MCP_SERVER_IMAGES:
        logger.info(f"Building {img.name} Docker image...")
        result = subprocess.run(
            ["docker", "build", "--platform", "linux/amd64", "-f", str(img.dockerfile), "-t", img.tag, str(img.context)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{img.name} build failed: {result.stderr}")

        artifact = DockerImageArtifact.put(
            id=f"mcp-{img.name}",
            description=f"MCP {img.name} server image",
            image_name=img.tag,
        )

        env = MCPServerEnv.put(
            id=f"mcp-server-{img.name}",
            docker_image_artifact=artifact,
            environment_name=img.service_name,
        )
        envs[img.name] = env
        logger.info(f"Created MCPServerEnv: {env.id} version={env.version}")

    return envs


@pytest.fixture(scope="module")
def agentenv_items_env() -> MCPServerEnv:
    """Build the AgentEnvEnvironment-based in-memory items server and create its MCPServerEnv.

    Assembles a small build context (vendored agentenv_protocol package + server.py)
    so the image stays light (public PyPI deps only)."""
    src_pkg = REPO_ROOT / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol"
    server_dir = TST_DATA_DIR / "agentenv_mcp"
    with tempfile.TemporaryDirectory() as build_dir:
        bd = Path(build_dir)
        shutil.copytree(src_pkg, bd / "agentenv_protocol", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(server_dir / "server.py", bd / "server.py")
        shutil.copy(server_dir / "Dockerfile", bd / "Dockerfile")
        shutil.copy(server_dir / "seed.json", bd / "seed.json")
        logger.info("Building agentenv items Docker image...")
        result = subprocess.run(
            ["docker", "build", "--platform", "linux/amd64", "-t", "mcp-items", str(bd)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"items build failed: {result.stderr}")

    artifact = DockerImageArtifact.put(
        id="mcp-items",
        description="AgentEnvEnvironment in-memory items server image",
        image_name="mcp-items",
    )
    env = MCPServerEnv.put(
        id="mcp-server-items",
        docker_image_artifact=artifact,
        environment_name="items",
    )
    logger.info(f"Created MCPServerEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def agentenv_website_env() -> WebsiteEnv:
    """Build a Starlette AgentEnvEnvironment website backend (serves the v1 data-plane)
    and reuse the slack static frontend; create a WebsiteEnv.

    Mirrors agentenv_items_env but for a website: assembles a light build context
    (vendored agentenv_protocol + server.py) for the backend."""
    src_pkg = REPO_ROOT / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol"
    server_dir = TST_DATA_DIR / "agentenv_website"
    with tempfile.TemporaryDirectory() as build_dir:
        bd = Path(build_dir)
        shutil.copytree(src_pkg, bd / "agentenv_protocol", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(server_dir / "server.py", bd / "server.py")
        shutil.copy(server_dir / "Dockerfile", bd / "Dockerfile")
        logger.info("Building agentenv website backend Docker image...")
        result = subprocess.run(
            ["docker", "build", "--platform", "linux/amd64", "-t", "agentenv-website-backend", str(bd)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"website backend build failed: {result.stderr}")
    backend_artifact = DockerImageArtifact.put(
        id="integ-test-website-backend-webitems",
        description="AgentEnvEnvironment website backend (v1 data-plane)",
        image_name="agentenv-website-backend",
    )

    # Reuse the static slack frontend — its content is irrelevant to the backend v1 path.
    fe = SLACK_WEBSITE_FRONTEND_IMAGE
    result = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64", "-f", str(fe.dockerfile), "-t", fe.tag, str(fe.context)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{fe.name} build failed: {result.stderr}")
    frontend_artifact = DockerImageArtifact.put(
        id="integ-test-website-frontend-webitems",
        description="Static website frontend (reused)",
        image_name=fe.tag,
    )

    env = WebsiteEnv.put(
        id="integ-test-webitems-website",
        backend_docker_image_artifact=backend_artifact,
        frontend_docker_image_artifact=frontend_artifact,
        environment_name="webitems",
    )
    logger.info(f"Created v1 WebsiteEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(params=[
    pytest.param("scale", id="scale"),
    pytest.param(
        "modal",
        id="modal",
        marks=skip_without_remote_sandbox("modal"),
    ),
    pytest.param(
        # VM-mode Modal: one Modal VM sandbox runs the whole docker-compose stack
        # (same _deploy_via_vm path as scale). Requires Modal's experimental vm_runtime
        # to be enabled on the workspace.
        "modal_vm",
        id="modal_vm",
        marks=skip_without_remote_sandbox("modal_vm"),
    ),
])
def sandbox_provider(request):
    from agent_env.providers import (
        ModalSandboxProvider,
        ModalVmSandboxProvider,
        reset_env_sandbox_provider,
        reset_sandbox_provider,
        set_env_sandbox_provider,
        set_sandbox_provider,
    )

    if request.param == "modal":
        set_sandbox_provider(ModalSandboxProvider())
        set_env_sandbox_provider(ModalSandboxProvider())
    elif request.param == "modal_vm":
        set_sandbox_provider(ModalVmSandboxProvider())
        set_env_sandbox_provider(ModalVmSandboxProvider())
    try:
        yield request.param
    finally:
        reset_sandbox_provider()
        reset_env_sandbox_provider()


@pytest.fixture(scope="module")
def multi_env(mcp_server_envs) -> MultiEnv:
    """Create MultiEnv from MCP server envs."""
    env = MultiEnv.put(id="multi-slack-email", mcp_server_envs=[mcp_server_envs["slack"], mcp_server_envs["email"]], metadata={"category": "integration-test", "owner": "agent-env"})
    logger.info(f"Created MultiEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def multi_env_with_website(mcp_server_envs, slack_website_env) -> MultiEnv:
    """Create MultiEnv with MCP servers AND a website."""
    env = MultiEnv.put(
        id="multi-mcp-and-website",
        mcp_server_envs=[mcp_server_envs["slack"], mcp_server_envs["email"]],
        website_envs=[slack_website_env],
    )
    logger.info(f"Created MultiEnv with website: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def email_service_artifact() -> EnvironmentArtifact:
    file_artifact = FileArtifact.put(
        id="test-email-data",
        description="Generated email data with at least 100 emails",
        file_path=str(TST_DATA_DIR / "email_mcp" / "generated_data.json"),
    )
    return EnvironmentArtifact.put(
        id="test-email-service-data",
        environment_name="email",
        file_artifact=file_artifact,
    )


@pytest.fixture(scope="module")
def slack_service_artifact() -> EnvironmentArtifact:
    file_artifact = FileArtifact.put(
        id="test-slack-data",
        description="Generated slack data with users, channels, and messages",
        file_path=str(TST_DATA_DIR / "slack_mcp" / "generated_data.json"),
    )
    return EnvironmentArtifact.put(
        id="test-slack-service-data",
        environment_name="slack",
        file_artifact=file_artifact,
    )


@pytest.fixture(scope="module")
def metadata_file_artifact() -> FileArtifact:
    return FileArtifact.put(
        id="test-universe-metadata-config",
        description="Test metadata config file",
        file_path=str(TST_DATA_DIR / "email_artifact.json"),
    )


@pytest.fixture(scope="module")
def environment_universe_artifact(slack_service_artifact, email_service_artifact, metadata_file_artifact) -> EnvironmentUniverseArtifact:
    return EnvironmentUniverseArtifact.put(
        id="test-universe",
        environment_artifacts=[slack_service_artifact, email_service_artifact],
        metadata={"config": metadata_file_artifact},
    )


@pytest.fixture(scope="module")
def items_service_artifact() -> EnvironmentArtifact:
    return EnvironmentArtifact.put(
        id="test-items-snap-service-data",
        environment_name="items",
        file_artifact=FileArtifact.put_bytes(
            id="test-items-snap-data",
            description="items v1 snapshot seed",
            filename="items.json",
            content=json.dumps({"items": ["snap-x", "snap-y"]}).encode(),
            content_type="application/json",
        ),
    )


@pytest.fixture(scope="module")
def multi_env_items_email(agentenv_items_env, mcp_server_envs) -> MultiEnv:
    """MultiEnv mixing a v1 service (items) with a legacy service (email)."""
    env = MultiEnv.put(
        id="multi-items-email",
        mcp_server_envs=[agentenv_items_env, mcp_server_envs["email"]],
        metadata={"category": "integration-test", "owner": "agent-env"},
    )
    logger.info(f"Created mixed MultiEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def items_email_universe(items_service_artifact, email_service_artifact, metadata_file_artifact) -> EnvironmentUniverseArtifact:
    return EnvironmentUniverseArtifact.put(
        id="test-universe-items-email",
        environment_artifacts=[items_service_artifact, email_service_artifact],
        metadata={"config": metadata_file_artifact},
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_with_single_mcp_server(mcp_server_envs, email_service_artifact):
    """Test gateway with a single MCP server (email only) and file artifact loading.

    Legacy data-plane coverage: load_environment_artifact drives the /api/reset path via legacy_protocol.
    The AgentEnvEnvironment /v1/data:* path is covered by test_gateway_with_agentenv_environment."""

    async def assert_empty_database(session, tools_result, tool_names):
        assert "list_emails" in tool_names, f"Expected list_emails tool, got {tool_names}"
        assert "channels_list" not in tool_names, f"Should not have slack tools, got {tool_names}"

        result = await session.call_tool("list_emails", {"folder_name": "INBOX"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"BEFORE load_environment_artifact - list_emails result: {content_text[:300]}...")
        assert "total_emails\": 0" in content_text or '"total_emails": 0' in content_text, f"Expected empty database, got {content_text}"

    async def assert_generated_data(session, tools_result, tool_names):
        assert "list_emails" in tool_names, f"Expected list_emails tool, got {tool_names}"

        result = await session.call_tool("list_emails", {"folder_name": "INBOX"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"AFTER load_environment_artifact - list_emails result: {content_text[:300]}...")
        assert "alex.chen@techcorp.com" in content_text, f"Expected alex.chen@techcorp.com (generated_data), got {content_text}"

    email_env_ref = mcp_server_envs["email"]
    email_env = Env.get(email_env_ref.id, version=email_env_ref.version)
    assert email_env.id == email_env_ref.id
    assert email_env.version == email_env_ref.version

    try:
        result = await email_env.deploy()

        # Verify instance_id is set and from_instance_id roundtrips
        assert result.instance_id is not None, "Expected instance_id after deploy"
        assert result.created_at_utc is not None, "Expected created_at_utc after deploy"
        assert result.expires_at_utc is not None, "Expected expires_at_utc after deploy"
        assert result.environment_card_url == f"{result.gateway_url}/.well-known/agent-env.json"

        # Verify get returns matching DeployedEnv
        from agent_env.env.store import get_env_instance_store
        looked_up = get_env_instance_store().get(result.instance_id)
        assert looked_up.instance_id == result.instance_id
        assert looked_up.gateway_url == result.gateway_url
        assert looked_up.mcp_url == result.mcp_url
        assert looked_up.db_web_url == result.db_web_url
        assert looked_up.db_mcp_url == result.db_mcp_url
        assert looked_up.environment_card_url == result.environment_card_url
        assert looked_up.sandbox_id == result.sandbox_id
        assert looked_up.gateway_mode == GatewayMode.PERFORMANCE.value

        # Verify from_instance_id reconnects to correct env
        reconnected_env = await Env.from_instance_id(result.instance_id)
        assert isinstance(reconnected_env, MCPServerEnv)
        assert reconnected_env.id == email_env.id
        assert reconnected_env.version == email_env.version

        env_card = await protocol_v1.get_card(result.gateway_url)
        assert env_card["children_environments"] == []
        parent_uris = {e["uri"] for e in env_card["capabilities"]["extensions"]}
        assert {"urn:agentenv:disable-tool/v1", "urn:agentenv:enable-tool/v1"} <= parent_uris

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_empty_database)

        # Verify /step API parity: empty database
        async def assert_empty_database_via_step(client, gateway_url, tools, tool_names):
            assert "list_emails" in tool_names, f"Expected list_emails in /step tools, got {tool_names}"
            assert "channels_list" not in tool_names, f"Should not have slack tools via /step, got {tool_names}"

            resp = await _step_call_tool(client, gateway_url, "list_emails", {"folder_name": "INBOX"})
            content_text = _step_content_text(resp)
            assert "total_emails\": 0" in content_text or '"total_emails": 0' in content_text, f"Expected empty via /step, got {content_text}"

        await _verify_step_api(gateway_url=result.gateway_url, tool_verifier_callback=assert_empty_database_via_step)

        # Verify /step error handling
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{result.gateway_url}/step", json={"action": "call_tool", "tool_name": "nonexistent", "arguments": {}}, timeout=30)
            assert resp.status_code == 404, f"Expected 404 for unknown tool, got {resp.status_code}"
            assert "Unknown tool" in resp.json()["error"]

            resp = await client.post(f"{result.gateway_url}/step", json={"action": "bad_action"}, timeout=30)
            assert resp.status_code == 400, f"Expected 400 for unknown action, got {resp.status_code}"

            resp = await client.post(f"{result.gateway_url}/step", json={}, timeout=30)
            assert resp.status_code == 400, f"Expected 400 for missing action, got {resp.status_code}"

        # Verify /state endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{result.gateway_url}/state", timeout=30)
            response.raise_for_status()
            state = response.json()
            assert "mcp_servers" in state
            server_names = [s["name"] for s in state["mcp_servers"]]
            assert "email" in server_names
            assert "slack" not in server_names
            assert "gateway" not in server_names  # no websites configured
            email_server = next(s for s in state["mcp_servers"] if s["name"] == "email")
            email_tool_names = [t["name"] for t in email_server["tools"]]
            assert "list_emails" in email_tool_names
            list_emails_tool = next(t for t in email_server["tools"] if t["name"] == "list_emails")
            assert "description" in list_emails_tool
            assert "parameters" in list_emails_tool
            assert "properties" in list_emails_tool["parameters"]

        await email_env.load_environment_artifact(email_service_artifact)
        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_generated_data)

        # Verify /step API parity: loaded data
        async def assert_generated_data_via_step(client, gateway_url, tools, tool_names):
            assert "list_emails" in tool_names
            resp = await _step_call_tool(client, gateway_url, "list_emails", {"folder_name": "INBOX"})
            content_text = _step_content_text(resp)
            assert "alex.chen@techcorp.com" in content_text, f"Expected email data via /step, got {content_text}"

        await _verify_step_api(gateway_url=result.gateway_url, tool_verifier_callback=assert_generated_data_via_step)

        # Role-based tool filtering: disable list_emails for "default" role, verify it's hidden from
        # default callers but still accessible to a non-default role; then re-enable and verify recovery.
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{result.gateway_url}/tools/{TOOL_DISABLE_ACTION}",
                json={"role": "default", "tools": ["list_emails"]},
                timeout=30,
            )
            assert resp.status_code == 200, f"disable failed: {resp.status_code} {resp.text}"
            assert "list_emails" in resp.json()["disabled"]

            # /state surfaces the disabled mapping
            state_resp = await client.get(f"{result.gateway_url}/state", timeout=30)
            state_resp.raise_for_status()
            assert state_resp.json()["roles"] == {"default": {"disabled": ["list_emails"], "allowed": []}}

            # /step list_tools as default role: list_emails absent
            default_list = await client.post(
                f"{result.gateway_url}/step", json={"action": "list_tools"}, timeout=30,
            )
            default_list.raise_for_status()
            assert "list_emails" not in [t["name"] for t in default_list.json()["tools"]]

            # /step list_tools as cli role: list_emails present
            cli_list = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "list_tools"},
                headers={AGENT_ENV_ROLE_HEADER: "cli"},
                timeout=30,
            )
            cli_list.raise_for_status()
            assert "list_emails" in [t["name"] for t in cli_list.json()["tools"]]

            # /step call_tool as default: 403; as cli: 200
            default_call = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "call_tool", "tool_name": "list_emails", "arguments": {"folder_name": "INBOX"}},
                timeout=30,
            )
            assert default_call.status_code == 403, f"expected 403, got {default_call.status_code} {default_call.text}"

            cli_call = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "call_tool", "tool_name": "list_emails", "arguments": {"folder_name": "INBOX"}},
                headers={AGENT_ENV_ROLE_HEADER: "cli"},
                timeout=30,
            )
            assert cli_call.status_code == 200, f"cli call failed: {cli_call.status_code} {cli_call.text}"

        # MCP tools/list parity: default hides, cli shows
        async with streamable_http_client(
            result.mcp_url,
            http_client=httpx.AsyncClient(headers={AGENT_ENV_ROLE_HEADER: "default"}),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                assert "list_emails" not in [t.name for t in tools_result.tools]

        async with streamable_http_client(
            result.mcp_url,
            http_client=httpx.AsyncClient(headers={AGENT_ENV_ROLE_HEADER: "cli"}),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                assert "list_emails" in [t.name for t in tools_result.tools]

        # Re-enable: list_emails accessible to default role again
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{result.gateway_url}/tools/{TOOL_ENABLE_ACTION}",
                json={"role": "default", "tools": ["list_emails"]},
                timeout=30,
            )
            assert resp.status_code == 200, f"enable failed: {resp.status_code} {resp.text}"
            assert "list_emails" not in resp.json()["disabled"]

            recovered_list = await client.post(
                f"{result.gateway_url}/step", json={"action": "list_tools"}, timeout=30,
            )
            recovered_list.raise_for_status()
            assert "list_emails" in [t["name"] for t in recovered_list.json()["tools"]]

            recovered_call = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "call_tool", "tool_name": "list_emails", "arguments": {"folder_name": "INBOX"}},
                timeout=30,
            )
            assert recovered_call.status_code == 200, f"recovered call failed: {recovered_call.status_code} {recovered_call.text}"

        # Global disable (role="*"): tool should be hidden from every role, including never-seen ones,
        # and become visible to everyone again on global enable.
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{result.gateway_url}/tools/{TOOL_DISABLE_ACTION}",
                json={"role": "*", "tools": ["list_emails"]},
                timeout=30,
            )
            assert resp.status_code == 200, f"global disable failed: {resp.status_code} {resp.text}"
            assert resp.json() == {"role": "*", "disabled": ["list_emails"], "allowed": []}

            state_resp = await client.get(f"{result.gateway_url}/state", timeout=30)
            state_resp.raise_for_status()
            assert state_resp.json()["roles"]["*"] == {"disabled": ["list_emails"], "allowed": []}

            for role in (None, "cli", "marketing"):
                headers = {AGENT_ENV_ROLE_HEADER: role} if role else {}
                lst = await client.post(
                    f"{result.gateway_url}/step", json={"action": "list_tools"}, headers=headers, timeout=30,
                )
                lst.raise_for_status()
                assert "list_emails" not in [t["name"] for t in lst.json()["tools"]], f"global disable not honored for role={role!r}"

            blocked = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "call_tool", "tool_name": "list_emails", "arguments": {"folder_name": "INBOX"}},
                headers={AGENT_ENV_ROLE_HEADER: "marketing"},
                timeout=30,
            )
            assert blocked.status_code == 403, f"expected 403 for never-seen role under global disable, got {blocked.status_code} {blocked.text}"

        async with streamable_http_client(
            result.mcp_url,
            http_client=httpx.AsyncClient(headers={AGENT_ENV_ROLE_HEADER: "cli"}),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                assert "list_emails" not in [t.name for t in tools_result.tools]

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{result.gateway_url}/tools/{TOOL_ENABLE_ACTION}",
                json={"role": "*", "tools": ["list_emails"]},
                timeout=30,
            )
            assert resp.status_code == 200, f"global enable failed: {resp.status_code} {resp.text}"
            assert resp.json() == {"role": "*", "disabled": [], "allowed": ["list_emails"]}

            unblocked = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "call_tool", "tool_name": "list_emails", "arguments": {"folder_name": "INBOX"}},
                timeout=30,
            )
            assert unblocked.status_code == 200, f"call after global enable failed: {unblocked.status_code} {unblocked.text}"

        # Override scenario: globally disable everything via the wildcard tool sentinel ("*"), then
        # re-enable everything for one specific role. Verifies (a) per-role allow beats global deny,
        # and (b) wildcard rules apply to all tools without snapshotting (future-tool semantic).
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{result.gateway_url}/tools/{TOOL_DISABLE_ACTION}",
                json={"role": "*", "tools": "*"},
                timeout=30,
            )
            assert resp.status_code == 200, f"wildcard disable failed: {resp.status_code} {resp.text}"
            # Wildcard disable propagates: clears all prior rules and sets rules["*"]["*"] = True.
            assert resp.json() == {"role": "*", "disabled": ["*"], "allowed": []}

            # marketing has no rules — falls through to the global wildcard, blocked from everything.
            mkt_lst = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "list_tools"},
                headers={AGENT_ENV_ROLE_HEADER: "marketing"},
                timeout=30,
            )
            mkt_lst.raise_for_status()
            assert [t["name"] for t in mkt_lst.json()["tools"]] == []

            # Override: allow ALL tools for cli (per-role wildcard allow beats global wildcard deny).
            resp = await client.post(
                f"{result.gateway_url}/tools/{TOOL_ENABLE_ACTION}",
                json={"role": "cli", "tools": "*"},
                timeout=30,
            )
            assert resp.status_code == 200, f"override enable failed: {resp.status_code} {resp.text}"
            assert resp.json() == {"role": "cli", "disabled": [], "allowed": ["*"]}

            cli_lst = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "list_tools"},
                headers={AGENT_ENV_ROLE_HEADER: "cli"},
                timeout=30,
            )
            cli_lst.raise_for_status()
            cli_tools = [t["name"] for t in cli_lst.json()["tools"]]
            assert "list_emails" in cli_tools and len(cli_tools) > 1, f"cli wildcard allow should expose all tools, got {cli_tools}"

            # marketing still restricted — no override, still falls through to the global wildcard deny.
            mkt_lst = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "list_tools"},
                headers={AGENT_ENV_ROLE_HEADER: "marketing"},
                timeout=30,
            )
            mkt_lst.raise_for_status()
            assert [t["name"] for t in mkt_lst.json()["tools"]] == []

            # Repeating wildcard disable nukes prior overrides — cli loses its blanket allow.
            await client.post(
                f"{result.gateway_url}/tools/{TOOL_DISABLE_ACTION}",
                json={"role": "*", "tools": "*"},
                timeout=30,
            )
            cli_lst_after_reset = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "list_tools"},
                headers={AGENT_ENV_ROLE_HEADER: "cli"},
                timeout=30,
            )
            cli_lst_after_reset.raise_for_status()
            assert [t["name"] for t in cli_lst_after_reset.json()["tools"]] == [], "wildcard disable should reset prior cli override"

            # Cleanup: restore baseline so downstream test code (CLI artifact build) sees a normal registry.
            await client.post(
                f"{result.gateway_url}/tools/{TOOL_ENABLE_ACTION}",
                json={"role": "*", "tools": "*"},
                timeout=30,
            )

        # ModifyEnvToolAccessStep: exercise the task step end-to-end against the deployed gateway.
        from agent_env.task_step import ModifyEnvToolAccessStep
        from agent_env.task_step.context import TaskStepContext

        step_ctx = TaskStepContext(deployed_envs=[result])

        disable_step = ModifyEnvToolAccessStep(
            id="step-disable", version=None, env_id=email_env.id,
            action=TOOL_DISABLE_ACTION, role="step-role", tools=["list_emails"],
        )
        await disable_step.execute(step_ctx)

        async with httpx.AsyncClient() as client:
            state_after_disable = (await client.get(f"{result.gateway_url}/state", timeout=30)).json()
            assert state_after_disable["roles"]["step-role"] == {"disabled": ["list_emails"], "allowed": []}

            blocked = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "call_tool", "tool_name": "list_emails", "arguments": {"folder_name": "INBOX"}},
                headers={AGENT_ENV_ROLE_HEADER: "step-role"},
                timeout=30,
            )
            assert blocked.status_code == 403, f"expected 403 after step-driven disable, got {blocked.status_code} {blocked.text}"

        enable_step = ModifyEnvToolAccessStep(
            id="step-enable", version=None, env_id=email_env.id,
            action=TOOL_ENABLE_ACTION, role="step-role", tools=["list_emails"],
        )
        await enable_step.execute(step_ctx)

        async with httpx.AsyncClient() as client:
            state_after_enable = (await client.get(f"{result.gateway_url}/state", timeout=30)).json()
            assert state_after_enable["roles"]["step-role"] == {"disabled": [], "allowed": ["list_emails"]}

            unblocked = await client.post(
                f"{result.gateway_url}/step",
                json={"action": "call_tool", "tool_name": "list_emails", "arguments": {"folder_name": "INBOX"}},
                headers={AGENT_ENV_ROLE_HEADER: "step-role"},
                timeout=30,
            )
            assert unblocked.status_code == 200, f"expected 200 after step-driven enable, got {unblocked.status_code} {unblocked.text}"

        # Build CLI artifact against the live deployment (free coverage — reuses the existing deploy)
        from agent_env.task_step import BuildMcpCliTaskStep

        build_step = BuildMcpCliTaskStep(id=f"build-cli-{email_env.id}", version=None, env_id=email_env.id, command_name="email")
        ctx = TaskStepContext(deployed_envs=[result])
        await build_step.execute(ctx)

        cli_ref = ctx.metadata["cli_artifact"]
        cli_artifact = Artifact.get(cli_ref["id"], version=cli_ref["version"])
        assert isinstance(cli_artifact, CliArtifact)
        assert cli_artifact.command_name == "email"
        assert cli_artifact.entrypoint == "bin/email"
        assert cli_artifact.env_id == email_env.id
        assert cli_artifact.env_version == email_env.version
        assert cli_artifact.cli_object_url.startswith("s3://")

        universe = cli_artifact.get_cli_files()
        assert isinstance(universe, FileArtifactUniverse)
        cli_files = universe.get_file_artifacts()
        assert list(cli_files.keys()) == ["bin/email"]
        script = cli_files["bin/email"].load().decode("utf-8")
        assert script.startswith("#!/usr/bin/env python3\n")
        assert "AGENT_ENV_GATEWAY_URL" in script
        assert "AGENT_ENV_ROLE" in script
        assert AGENT_ENV_ROLE_HEADER in script
        assert 'DEFAULT_ROLE = "cli"' in script
        assert "'manifest_version': '0.1.0'" in script
        assert "class _ManifestGroup" in script
        assert "_list_tools" in script
        assert "@cli.command(" not in script

        # Dynamic-discovery end-to-end: run the generated CLI as a subprocess and verify --help
        # reflects gateway state changes immediately (no rebuild needed).
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(script)
            cli_script_path = tmp.name
        try:
            cli_env = {**os.environ, "AGENT_ENV_GATEWAY_URL": result.gateway_url}

            def _run_help() -> str:
                proc = subprocess.run(
                    [sys.executable, cli_script_path, "--help"],
                    env=cli_env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert proc.returncode == 0, f"--help failed: {proc.stderr}"
                return proc.stdout

            def _root_commands() -> set[str]:
                # Match the Commands: listing, not the whole help text — the CLI's own
                # description line contains the service name unconditionally.
                _, _, listing = _run_help().partition("Commands:")
                return {line.split()[0] for line in listing.splitlines() if line.strip()}

            # `Email` is its service's namesake, so the manifest's verbs render at the
            # root: `list`, not `email list`.
            assert "list" in _root_commands(), "expected root list command baseline"

            # Disable list_emails for role=cli and re-render.
            disable_for_cli = ModifyEnvToolAccessStep(
                id="cli-disable-list-emails", version=None, env_id=email_env.id,
                action=TOOL_DISABLE_ACTION, role="cli", tools=["list_emails"],
            )
            await disable_for_cli.execute(step_ctx)
            assert "list" not in _root_commands(), "root list should be hidden after cli disable"

            # Re-enable and verify it reappears.
            enable_for_cli = ModifyEnvToolAccessStep(
                id="cli-enable-list-emails", version=None, env_id=email_env.id,
                action=TOOL_ENABLE_ACTION, role="cli", tools=["list_emails"],
            )
            await enable_for_cli.execute(step_ctx)
            assert "list" in _root_commands(), "root list should reappear after cli re-enable"

            # Actually invoke the subcommand end-to-end to verify the dynamic Click command builds
            # and forwards arguments correctly.
            # `cli_script_path` is a temp CLI built earlier in this test and the args are static
            # literals, so the non-literal argv is not externally controllable.
            invoke = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit
                [sys.executable, cli_script_path, "list", "--folder-name", "INBOX"],
                env=cli_env, capture_output=True, text=True, timeout=30,
            )
            assert invoke.returncode == 0, f"email list invocation failed: stderr={invoke.stderr}"
            assert "alex.chen@techcorp.com" in invoke.stdout, f"unexpected email list output: {invoke.stdout[:500]}"
        finally:
            os.unlink(cli_script_path)

        # Cache-hit: seed env metadata with the just-built artifact ref, then create_cli()
        # should short-circuit and return it WITHOUT redeploying or rebuilding.
        email_env.update_metadata({**email_env.metadata, "cli_artifact": cli_ref})
        cached = await email_env.create_cli()
        assert cached.id == cli_artifact.id
        assert cached.version == cli_artifact.version
    finally:
        await email_env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_with_multi_mcp_server(sandbox_provider, multi_env, environment_universe_artifact):
    """Test gateway with multiple MCP servers (slack + email) and universe artifact loading."""

    async def assert_empty_databases(session, tools_result, tool_names):
        # Verify both tools are present
        assert "channels_list" in tool_names, f"Expected channels_list tool, got {tool_names}"
        assert "list_emails" in tool_names, f"Expected list_emails tool, got {tool_names}"

        # Verify tool schema is properly forwarded
        list_emails_tool = next((t for t in tools_result.tools if t.name == "list_emails"), None)
        assert list_emails_tool is not None, "list_emails tool not found"
        assert list_emails_tool.inputSchema is not None, "list_emails should have inputSchema"
        schema_str = str(list_emails_tool.inputSchema)
        assert "folder_name" in schema_str, f"list_emails schema should include 'folder_name' parameter, got {schema_str}"

        # Verify email database is empty
        result = await session.call_tool("list_emails", {"folder_name": "INBOX"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"BEFORE load - list_emails result: {content_text[:300]}...")
        assert "total_emails\": 0" in content_text or '"total_emails": 0' in content_text, f"Expected empty email database, got {content_text}"

        # Verify slack database is empty
        result = await session.call_tool("channels_list", {"channel_types": "public_channel"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"BEFORE load - channels_list result: {content_text[:300]}...")
        assert '"channels": []' in content_text, f"Expected empty slack channels, got {content_text}"

    async def assert_loaded_data(session, tools_result, tool_names):
        # Verify email data loaded
        result = await session.call_tool("list_emails", {"folder_name": "INBOX"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"AFTER load - list_emails result: {content_text[:300]}...")
        assert "alex.chen@techcorp.com" in content_text, f"Expected alex.chen@techcorp.com (email generated_data), got {content_text}"

        # Verify argument forwarding - call with folder_name="SENT" to confirm args are passed
        result = await session.call_tool("list_emails", {"folder_name": "SENT"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"AFTER load - list_emails(SENT) result: {content_text[:200]}...")
        assert "emails" in content_text.lower(), f"Expected emails in response, got {content_text}"

        # Verify slack data loaded
        result = await session.call_tool("channels_list", {"channel_types": "public_channel"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"AFTER load - channels_list result: {content_text[:300]}...")
        assert "general" in content_text, f"Expected general channel (slack generated_data), got {content_text}"

    # Verify Env.get roundtrip for MultiEnv
    multi_env_ref = multi_env
    multi_env = Env.get(multi_env_ref.id, version=multi_env_ref.version)
    assert multi_env.id == multi_env_ref.id
    assert multi_env.version == multi_env_ref.version
    assert len(multi_env.mcp_server_envs) == len(multi_env_ref.mcp_server_envs)
    assert multi_env.metadata == {"category": "integration-test", "owner": "agent-env"}

    try:
        result = await multi_env.deploy()
        assert multi_env._instance_id == result.instance_id, "Expected _instance_id set after deploy"
        assert result.gateway_mode == GatewayMode.PERFORMANCE.value

        if sandbox_provider == "modal":
            assert result.sandbox_type == "modal", f"expected sandbox_type=modal, got {result.sandbox_type!r}"
            ids = result.sandbox_ids
            assert "gateway_server" in ids and isinstance(ids["gateway_server"], str)
            assert set(ids.get("mcp_server", {}).keys()) == {"slack", "email"}, f"unexpected mcp_server keys: {ids.get('mcp_server')}"
            assert set(ids.get("service_db", {}).keys()) == {"servicedb", "pgweb", "db-mcp"}, f"unexpected service_db keys: {ids.get('service_db')}"
        elif sandbox_provider == "modal_vm":
            # VM mode runs the whole stack (all MCP servers + db) in ONE Modal VM via
            # docker-compose, so there is a single gateway_server sandbox id and NO
            # per-service mcp_server / service_db sandbox ids (those are container-mode only).
            assert result.sandbox_type == "modal_vm", f"expected sandbox_type=modal_vm, got {result.sandbox_type!r}"
            ids = result.sandbox_ids
            assert "gateway_server" in ids and isinstance(ids["gateway_server"], str)
            assert not ids.get("mcp_server"), f"VM mode should have no per-service mcp_server sandboxes, got {ids.get('mcp_server')}"
            assert not ids.get("service_db"), f"VM mode should have no separate service_db sandboxes, got {ids.get('service_db')}"

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_empty_databases)

        # Verify /step API parity: empty databases
        async def assert_empty_via_step(client, gateway_url, tools, tool_names):
            assert "channels_list" in tool_names, f"Expected channels_list via /step, got {tool_names}"
            assert "list_emails" in tool_names, f"Expected list_emails via /step, got {tool_names}"

            # Verify schema in /step list_tools response
            list_emails_tool = next(t for t in tools if t["name"] == "list_emails")
            assert "folder_name" in str(list_emails_tool["parameters"]), f"Expected folder_name in schema via /step"

            resp = await _step_call_tool(client, gateway_url, "list_emails", {"folder_name": "INBOX"})
            content_text = _step_content_text(resp)
            assert "total_emails\": 0" in content_text or '"total_emails": 0' in content_text, f"Expected empty email via /step, got {content_text}"

            resp = await _step_call_tool(client, gateway_url, "channels_list", {"channel_types": "public_channel"})
            content_text = _step_content_text(resp)
            assert '"channels": []' in content_text, f"Expected empty slack via /step, got {content_text}"

        await _verify_step_api(gateway_url=result.gateway_url, tool_verifier_callback=assert_empty_via_step)

        # Verify /state endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{result.gateway_url}/state", timeout=30)
            response.raise_for_status()
            state = response.json()
            server_names = [s["name"] for s in state["mcp_servers"]]
            assert "slack" in server_names
            assert "email" in server_names
            assert "gateway" not in server_names  # no websites configured
            slack_server = next(s for s in state["mcp_servers"] if s["name"] == "slack")
            assert any(t["name"] == "channels_list" for t in slack_server["tools"])
            assert "changelog_id" in state, f"Expected changelog_id in /state response"

        load_result = await multi_env.load_environment_universe_artifact(environment_universe_artifact)

        # Verify environment_universe was auto-tracked on the instance
        from agent_env.env.store import get_env_instance_store
        loaded_universe = get_env_instance_store().get_environment_universe(result.instance_id)
        assert loaded_universe is not None, "Expected service_universe after load"
        assert loaded_universe["id"] == environment_universe_artifact.id
        assert loaded_universe["version"] == environment_universe_artifact.version

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_loaded_data)

        # Verify /step API parity: loaded data
        async def assert_loaded_via_step(client, gateway_url, tools, tool_names):
            resp = await _step_call_tool(client, gateway_url, "list_emails", {"folder_name": "INBOX"})
            content_text = _step_content_text(resp)
            assert "alex.chen@techcorp.com" in content_text, f"Expected email data via /step, got {content_text}"

            resp = await _step_call_tool(client, gateway_url, "list_emails", {"folder_name": "SENT"})
            content_text = _step_content_text(resp)
            assert "emails" in content_text.lower(), f"Expected emails in SENT via /step, got {content_text}"

            resp = await _step_call_tool(client, gateway_url, "channels_list", {"channel_types": "public_channel"})
            content_text = _step_content_text(resp)
            assert "general" in content_text, f"Expected general channel via /step, got {content_text}"

        await _verify_step_api(gateway_url=result.gateway_url, tool_verifier_callback=assert_loaded_via_step)

        # Verify metadata was downloaded to VM
        assert "config" in load_result.metadata_filepaths
        config_path = load_result.metadata_filepaths["config"]
        assert config_path.startswith("/tmp/metadata-")
        assert config_path.endswith("/email_artifact.json")
        _, file_content, _ = await multi_env._sandbox.exec_with_output("cat", config_path)
        assert "user_email" in file_content

        # Verify changelog triggers via db-mcp's execute_sql tool
        async def assert_changelog_empty(session, tools_result, tool_names):
            assert "execute_sql" in tool_names, f"Expected execute_sql tool, got {tool_names}"
            result = await session.call_tool("execute_sql", {
                "sql": "SELECT count(*) as cnt FROM public._changelog",
            })
            content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
            logger.info(f"Changelog count after load: {content_text}")
            assert content_text == "[{'cnt': 0}]", f"Expected empty changelog after load, got {content_text}"

        await _verify_list_tools(mcp_url=result.db_mcp_url, tool_verifier_callback=assert_changelog_empty)

        # Make a modification via MCP tool
        async def send_email_for_changelog(session, tools_result, tool_names):
            assert "send_email" in tool_names, f"Expected send_email tool, got {tool_names}"
            await session.call_tool("send_email", {
                "to": ["test@example.com"],
                "subject": "Changelog test",
                "body": "Verifying audit triggers",
            })

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=send_email_for_changelog)

        # Verify the change appears in _changelog
        async def assert_changelog_has_entry(session, tools_result, tool_names):
            result = await session.call_tool("execute_sql", {
                "sql": "SELECT schema_name, table_name, operation, summary FROM public._changelog ORDER BY id",
            })
            content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
            logger.info(f"Changelog after send_email: {content_text}")
            assert "email" in content_text, f"Expected schema_name 'email' in changelog, got {content_text}"
            assert "INSERT" in content_text, f"Expected INSERT operation in changelog, got {content_text}"

        await _verify_list_tools(mcp_url=result.db_mcp_url, tool_verifier_callback=assert_changelog_has_entry)

        rehydrated = await MultiEnv.from_deployed_env(result)
        assert rehydrated._env_provider is not None
        assert rehydrated._sandbox is not None
        for child in rehydrated.mcp_server_envs:
            assert child._env_provider is rehydrated._env_provider
        if sandbox_provider == "modal":
            gp = rehydrated._env_provider
            assert gp._db_sandbox is not None
            assert gp._pgweb_sandbox is not None
            assert gp._db_mcp_sandbox is not None
            assert set(gp._environment_sandboxes.keys()) == {"slack", "email"}
            assert len(gp._container_sandboxes) == 6, f"expected 6 (gateway + 2 MCPs + 3 data-plane), got {len(gp._container_sandboxes)}"
        elif sandbox_provider == "modal_vm":
            # VM mode: db/pgweb/db-mcp are docker-compose services inside the one VM,
            # not separate sandboxes, so these container-mode handles stay None.
            gp = rehydrated._env_provider
            assert gp._db_sandbox is None
            assert gp._pgweb_sandbox is None
            assert gp._db_mcp_sandbox is None

        await rehydrated.load_environment_universe_artifact(environment_universe_artifact)

        rehydrate_probe_subject = "post-rehydrate-changelog-probe"

        async def send_probe_email(session, tools_result, tool_names):
            await session.call_tool("send_email", {
                "to": ["test@example.com"],
                "subject": rehydrate_probe_subject,
                "body": "Verifying changelog triggers re-install via rehydrated provider",
            })

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=send_probe_email)

        async def assert_probe_in_changelog(session, tools_result, tool_names):
            r = await session.call_tool("execute_sql", {
                "sql": f"SELECT summary FROM public._changelog WHERE summary LIKE '%{rehydrate_probe_subject}%'",
            })
            content_text = "".join(c.text for c in r.content if hasattr(c, "text"))
            logger.info(f"Post-rehydrate changelog query: {content_text}")
            assert rehydrate_probe_subject in content_text, f"Expected probe subject in changelog (triggers re-installed via rehydrated provider), got {content_text}"

        await _verify_list_tools(mcp_url=result.db_mcp_url, tool_verifier_callback=assert_probe_in_changelog)
    finally:
        await multi_env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_with_single_website_env(slack_website_env, slack_service_artifact):
    """Test gateway with a website env: list_website_urls, browser navigate/snapshot/close, and load_environment_artifact."""

    async def assert_website_tools(session, tools_result, tool_names):
        # 1. list_website_urls tool exists and returns the frontend URL
        assert "list_website_urls" in tool_names, f"Expected list_website_urls tool, got {tool_names}"
        result = await session.call_tool("list_website_urls", {})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        assert "http://slack-website-frontend:80" in content_text, f"Expected frontend URL, got {content_text}"

        # 2. browser_navigate to the website
        result = await session.call_tool("browser_navigate", {"url": "http://slack-website-frontend:80"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"browser_navigate result: {content_text[:500]}")
        assert "Slack Workspace" in content_text, f"Expected 'Slack Workspace' in navigate result, got {content_text}"

        # 3. browser_snapshot returns page content
        result = await session.call_tool("browser_snapshot", {})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"browser_snapshot result: {content_text[:500]}")
        assert "Slack Workspace" in content_text, f"Expected 'Slack Workspace' in snapshot, got {content_text}"

        # 4. browser_close then snapshot returns empty/blank page
        await session.call_tool("browser_close", {})
        result = await session.call_tool("browser_snapshot", {})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"browser_snapshot after close: {content_text[:500]}")
        assert "Slack Workspace" not in content_text, f"Expected empty page after close, got {content_text}"

    try:
        result = await slack_website_env.deploy()
        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_website_tools)

        # Verify /step API parity: website tools
        async def assert_website_tools_via_step(client, gateway_url, tools, tool_names):
            assert "list_website_urls" in tool_names, f"Expected list_website_urls via /step, got {tool_names}"

            resp = await _step_call_tool(client, gateway_url, "list_website_urls", {})
            content_text = _step_content_text(resp)
            assert "http://slack-website-frontend:80" in content_text, f"Expected frontend URL via /step, got {content_text}"

            resp = await _step_call_tool(client, gateway_url, "browser_navigate", {"url": "http://slack-website-frontend:80"})
            content_text = _step_content_text(resp)
            assert "Slack Workspace" in content_text, f"Expected 'Slack Workspace' via /step navigate, got {content_text}"

            resp = await _step_call_tool(client, gateway_url, "browser_snapshot", {})
            content_text = _step_content_text(resp)
            assert "Slack Workspace" in content_text, f"Expected 'Slack Workspace' via /step snapshot, got {content_text}"

            await _step_call_tool(client, gateway_url, "browser_close", {})
            resp = await _step_call_tool(client, gateway_url, "browser_snapshot", {})
            content_text = _step_content_text(resp)
            assert "Slack Workspace" not in content_text, f"Expected empty page after /step close, got {content_text}"

        await _verify_step_api(gateway_url=result.gateway_url, tool_verifier_callback=assert_website_tools_via_step)

        # Verify /state endpoint includes gateway tools for website
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{result.gateway_url}/state", timeout=30)
            response.raise_for_status()
            state = response.json()
            server_names = [s["name"] for s in state["mcp_servers"]]
            assert "gateway" in server_names
            gateway_entry = next(s for s in state["mcp_servers"] if s["name"] == "gateway")
            gateway_tool_names = [t["name"] for t in gateway_entry["tools"]]
            assert "list_website_urls" in gateway_tool_names

        # Verify REST proxy: empty channels before loading data
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{result.gateway_url}/svc/slack/api/channels?member_only=false", timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"BEFORE load_environment_artifact - channels: {data}")
            assert data["ok"] is True
            assert data["channels"] == []

        await slack_website_env.load_environment_artifact(slack_service_artifact)

        # Verify channels populated after loading
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{result.gateway_url}/svc/slack/api/channels?member_only=false", timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"AFTER load_environment_artifact - channels: {data}")
            channels = data["channels"]
            assert len(channels) > 0
            assert "general" in [c["name"] for c in channels]
    finally:
        await slack_website_env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_with_multi_env_including_website(multi_env_with_website, environment_universe_artifact):
    """Test MultiEnv with MCP servers + website: deploy, browse website, load universe artifact."""

    async def assert_all_tools_and_website(session, tools_result, tool_names):
        # MCP server tools
        assert "channels_list" in tool_names, f"Expected channels_list tool, got {tool_names}"
        assert "list_emails" in tool_names, f"Expected list_emails tool, got {tool_names}"
        # Website browser tools (auto-included when website_configs present)
        assert "list_website_urls" in tool_names, f"Expected list_website_urls tool, got {tool_names}"
        assert "browser_navigate" in tool_names, f"Expected browser_navigate tool, got {tool_names}"

        # Navigate to the website and verify it renders
        result = await session.call_tool("browser_navigate", {"url": "http://slack-website-frontend:80"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        logger.info(f"browser_navigate result: {content_text[:500]}")
        assert "Slack Workspace" in content_text, f"Expected 'Slack Workspace' in navigate result, got {content_text}"

        await session.call_tool("browser_close", {})

    async def assert_loaded_data(session, tools_result, tool_names):
        # Verify slack MCP data loaded
        result = await session.call_tool("channels_list", {"channel_types": "public_channel"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        assert "general" in content_text, f"Expected general channel in MCP, got {content_text}"

        # Verify email MCP data loaded
        result = await session.call_tool("list_emails", {"folder_name": "INBOX"})
        content_text = "".join(c.text for c in result.content if hasattr(c, "text"))
        assert "alex.chen@techcorp.com" in content_text, f"Expected email data, got {content_text}"

    # Verify Env.get roundtrip
    multi_env_ref = multi_env_with_website
    multi_env = Env.get(multi_env_ref.id, version=multi_env_ref.version)
    assert len(multi_env.mcp_server_envs) == 2
    assert len(multi_env.website_envs) == 1

    try:
        result = await multi_env.deploy()

        # Verify all tools present and website is navigable
        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_all_tools_and_website)

        # Verify /step API parity: all tools accessible
        async def assert_all_tools_via_step(client, gateway_url, tools, tool_names):
            assert "channels_list" in tool_names, f"Expected channels_list via /step, got {tool_names}"
            assert "list_emails" in tool_names, f"Expected list_emails via /step, got {tool_names}"
            assert "list_website_urls" in tool_names, f"Expected list_website_urls via /step, got {tool_names}"
            assert "browser_navigate" in tool_names, f"Expected browser_navigate via /step, got {tool_names}"

            resp = await _step_call_tool(client, gateway_url, "browser_navigate", {"url": "http://slack-website-frontend:80"})
            content_text = _step_content_text(resp)
            assert "Slack Workspace" in content_text, f"Expected 'Slack Workspace' via /step, got {content_text}"
            await _step_call_tool(client, gateway_url, "browser_close", {})

        await _verify_step_api(gateway_url=result.gateway_url, tool_verifier_callback=assert_all_tools_via_step)

        # Load universe artifact — slack data goes to MCP server (first match), email data goes to email MCP server
        load_result = await multi_env.load_environment_universe_artifact(environment_universe_artifact)

        # Verify environment_universe was auto-tracked on the instance
        from agent_env.env.store import get_env_instance_store
        loaded_universe = get_env_instance_store().get_environment_universe(result.instance_id)
        assert loaded_universe is not None, "Expected service_universe after load"
        assert loaded_universe["id"] == environment_universe_artifact.id
        assert loaded_universe["version"] == environment_universe_artifact.version

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_loaded_data)

        # Verify /step API parity: loaded data
        async def assert_loaded_data_via_step(client, gateway_url, tools, tool_names):
            resp = await _step_call_tool(client, gateway_url, "channels_list", {"channel_types": "public_channel"})
            content_text = _step_content_text(resp)
            assert "general" in content_text, f"Expected general channel via /step, got {content_text}"

            resp = await _step_call_tool(client, gateway_url, "list_emails", {"folder_name": "INBOX"})
            content_text = _step_content_text(resp)
            assert "alex.chen@techcorp.com" in content_text, f"Expected email data via /step, got {content_text}"

        await _verify_step_api(gateway_url=result.gateway_url, tool_verifier_callback=assert_loaded_data_via_step)

        # Verify metadata was downloaded to VM
        assert "config" in load_result.metadata_filepaths

        # Verify website backend also sees the data (shared DB schema)
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{result.gateway_url}/svc/slack/api/channels?member_only=false", timeout=30)
            response.raise_for_status()
            data = response.json()
            channels = data["channels"]
            assert len(channels) > 0, f"Expected channels in website backend, got {data}"
            assert "general" in [c["name"] for c in channels], f"Expected 'general' channel, got {channels}"
    finally:
        await multi_env.close()


async def _step_call_tool(client: httpx.AsyncClient, gateway_url: str, tool_name: str, arguments: dict) -> dict:
    """Call a tool via the /step REST API and return the parsed response."""
    response = await client.post(
        f"{gateway_url}/step",
        json={"action": "call_tool", "tool_name": tool_name, "arguments": arguments},
        timeout=630,
    )
    if response.status_code != 200:
        body = response.text
        logger.error(f"/step call_tool '{tool_name}' returned {response.status_code}: {body}")
        response.raise_for_status()
    return response.json()


def _step_content_text(step_response: dict) -> str:
    """Extract concatenated text from /step call_tool response content blocks."""
    return "".join(
        block["text"] for block in step_response.get("content", [])
        if block.get("type") == "text" and "text" in block
    )


async def _verify_step_api(gateway_url: str, tool_verifier_callback: callable):
    """Connect to gateway via /step REST API and verify tools.

    Args:
        gateway_url: The gateway base URL (e.g., "https://example.com").
        tool_verifier_callback: Async callable(client, gateway_url, tools, tool_names).
    """
    max_retries = 10
    for attempt in range(max_retries):
        logger.info(f"Testing /step API (attempt {attempt + 1}/{max_retries})...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{gateway_url}/step",
                    json={"action": "list_tools"},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                tools = data["tools"]
                tool_names = [t["name"] for t in tools]
                logger.info(f"Available tools via /step: {tool_names}")
                await tool_verifier_callback(client, gateway_url, tools, tool_names)
                break
        except AssertionError:
            raise
        except Exception as e:
            logger.info(f"  /step attempt {attempt + 1} failed: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(10)
            else:
                raise RuntimeError(f"/step API failed after {max_retries} attempts") from e

    logger.info("/step test passed!")


async def _verify_list_tools(mcp_url: str, tool_verifier_callback: callable):
    """Connect to gateway and verify tools are available.

    Args:
        mcp_url: The MCP endpoint URL (e.g., "https://example.com/mcp").
        tool_verifier_callback: Async callable to run assertions on the MCP session.
    """
    max_retries = 10
    for attempt in range(max_retries):
        logger.info(f"Testing gateway via tunnel (attempt {attempt + 1}/{max_retries})...")
        verified = False
        try:
            async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    tool_names = [t.name for t in tools_result.tools]
                    logger.info(f"Available tools: {tool_names}")
                    await tool_verifier_callback(session, tools_result, tool_names)
                    verified = True
        except BaseExceptionGroup as eg:
            assertion_errors, _ = eg.split(AssertionError)
            if assertion_errors:
                raise
            if verified:
                logger.info(f"  Verified (ignoring cleanup error)")
                break
            logger.info(f"  Attempt {attempt + 1} failed: {type(eg).__name__}")
            if attempt < max_retries - 1:
                await asyncio.sleep(10)
            else:
                raise RuntimeError(f"MCP client failed after {max_retries} attempts") from eg
        except Exception as e:
            logger.info(f"  Attempt {attempt + 1} failed: {type(e).__name__}")
            if attempt < max_retries - 1:
                await asyncio.sleep(10)
            else:
                raise RuntimeError(f"MCP client failed after {max_retries} attempts") from e
        else:
            break

    logger.info("Test passed!")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.dependency(name="test_snapshot_cli")
async def test_snapshot_cli(multi_env, environment_universe_artifact):
    """Test env snapshot: deploy MultiEnv, load universe, snapshot, verify."""
    from agent_env.artifact import Artifact
    from agent_env.cli.env.snapshot import run_snapshot
    from agent_env.env.snapshot_store import get_env_snapshot_store

    multi_env_ref = multi_env
    multi_env = Env.get(multi_env_ref.id, version=multi_env_ref.version)

    try:
        result = await multi_env.deploy()
        assert result.instance_id is not None

        await multi_env.load_environment_universe_artifact(environment_universe_artifact)

        # Run snapshot
        snapshot_result = await run_snapshot(result.instance_id)
        logger.info(f"Snapshot result: {snapshot_result}")

        # Verify env_snapshots record
        snapshot = get_env_snapshot_store().get(multi_env.id, environment_universe_artifact.id, result.instance_id)
        assert snapshot is not None, "Expected snapshot in env_snapshots"
        assert snapshot.env_id == multi_env.id
        assert snapshot.environment_universe_id == environment_universe_artifact.id
        assert snapshot.environment_universe_version == environment_universe_artifact.version
        assert snapshot.instance_id == result.instance_id
        assert snapshot.is_clean is True, "Expected clean snapshot (no agent modifications)"
        assert snapshot.db_image_artifact_id is not None
        assert snapshot.db_image_artifact_version > 0
        logger.info(f"Snapshot record: {snapshot}")

        # Verify get_clean returns the same snapshot
        clean_snapshot = get_env_snapshot_store().get_clean(multi_env.id, environment_universe_artifact.id)
        assert clean_snapshot is not None, "Expected clean snapshot via get_clean"
        assert clean_snapshot.instance_id == result.instance_id

        # Verify referenced DockerImageArtifact exists
        snapshot_artifact = Artifact.get(snapshot.db_image_artifact_id, snapshot.db_image_artifact_version)
        assert isinstance(snapshot_artifact, DockerImageArtifact)
        assert snapshot_artifact.tar_gz_object_url is not None
        logger.info(f"Snapshot artifact: {snapshot_artifact.id} v{snapshot_artifact.version}")
    finally:
        await multi_env.close()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.dependency(depends=["test_snapshot_cli"])
async def test_load_from_snapshot(multi_env, environment_universe_artifact):
    """Test loading a universe from a snapshot (fast path) instead of REST resets."""

    multi_env_ref = multi_env
    multi_env = Env.get(multi_env_ref.id, version=multi_env_ref.version)

    try:
        result = await multi_env.deploy()

        # Load universe — should use the snapshot created by test_snapshot_cli
        load_result = await multi_env.load_environment_universe_artifact(environment_universe_artifact)

        # Verify data was loaded correctly via MCP tools
        async def assert_loaded_data(session, tools_result, tool_names):
            assert "channels_list" in tool_names, f"Expected channels_list tool, got {tool_names}"
            assert "list_emails" in tool_names, f"Expected list_emails tool, got {tool_names}"

            r = await session.call_tool("channels_list", {"channel_types": "public_channel"})
            content = "".join(c.text for c in r.content if hasattr(c, "text"))
            assert "general" in content, f"Expected general channel, got {content}"

            r = await session.call_tool("list_emails", {"folder_name": "INBOX"})
            content = "".join(c.text for c in r.content if hasattr(c, "text"))
            assert "alex.chen@techcorp.com" in content, f"Expected email data, got {content}"

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_loaded_data)
        logger.info("Snapshot-based loading verified!")
    finally:
        await multi_env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_consistent_mode(multi_env, environment_universe_artifact):
    """Test gateway in CONSISTENT mode: deploy, load data, verify tool calls via MCP and /step."""

    multi_env_ref = multi_env
    multi_env = Env.get(multi_env_ref.id, version=multi_env_ref.version)

    try:
        result = await multi_env.deploy(gateway_mode=GatewayMode.CONSISTENT)
        assert result.gateway_mode == GatewayMode.CONSISTENT.value

        # Verify env instance persists the mode
        from agent_env.env.store import get_env_instance_store
        looked_up = get_env_instance_store().get(result.instance_id)
        assert looked_up.gateway_mode == GatewayMode.CONSISTENT.value

        await multi_env.load_environment_universe_artifact(environment_universe_artifact)

        # Verify tool calls work via MCP protocol
        async def assert_loaded_via_mcp(session, tools_result, tool_names):
            assert "channels_list" in tool_names
            assert "list_emails" in tool_names
            r = await session.call_tool("channels_list", {"channel_types": "public_channel"})
            content = "".join(c.text for c in r.content if hasattr(c, "text"))
            assert "general" in content, f"Expected general channel, got {content}"

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_loaded_via_mcp)

        # Verify tool calls work via /step API and changelog_id is returned
        async def assert_loaded_via_step(client, gateway_url, tools, tool_names):
            assert "channels_list" in tool_names
            assert "send_email" in tool_names

            # Read tool — should have a changelog_id (changelog is empty after trigger install, so None or 0 is fine)
            resp = await _step_call_tool(client, gateway_url, "channels_list", {"channel_types": "public_channel"})
            content_text = _step_content_text(resp)
            assert "general" in content_text, f"Expected general channel via /step, got {content_text}"
            assert "changelog_id" in resp, f"Expected changelog_id in /step response, got {resp.keys()}"

            # Write tool — should produce a positive changelog_id
            resp1 = await _step_call_tool(client, gateway_url, "send_email", {
                "to": ["test@example.com"], "subject": "changelog test 1", "body": "first",
            })
            changelog_id_1 = resp1.get("changelog_id")
            assert isinstance(changelog_id_1, int) and changelog_id_1 > 0, f"Expected positive changelog_id after write, got {changelog_id_1}"

            # Second write — changelog_id should strictly increase
            resp2 = await _step_call_tool(client, gateway_url, "send_email", {
                "to": ["test@example.com"], "subject": "changelog test 2", "body": "second",
            })
            changelog_id_2 = resp2.get("changelog_id")
            assert isinstance(changelog_id_2, int) and changelog_id_2 > changelog_id_1, (
                f"Expected changelog_id to increase: {changelog_id_1} -> {changelog_id_2}"
            )

        await _verify_step_api(gateway_url=result.gateway_url, tool_verifier_callback=assert_loaded_via_step)

        # Verify /state includes changelog_id matching the latest /step response
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{result.gateway_url}/state", timeout=30)
            response.raise_for_status()
            state = response.json()
            assert "changelog_id" in state, f"Expected changelog_id in /state response"
            assert isinstance(state["changelog_id"], int) and state["changelog_id"] > 0, (
                f"Expected positive changelog_id in /state after writes, got {state['changelog_id']}"
            )

    finally:
        await multi_env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_env_validator_task(mcp_server_envs):
    """Test MCPServerEnv.validate(): lazily creates task, runs validation, persists metadata."""
    from agent_env.task import Task

    email_env = mcp_server_envs["email"]
    instance_id = await email_env.validate()

    assert instance_id is not None
    logger.info(f"Validation instance_id: {instance_id}")

    instance = Task.get_instance(instance_id)
    assert instance.status == "completed", f"Expected completed, got {instance.status}"
    logger.info(f"Task instance: status={instance.status} steps={instance.total_steps}")

    env = Env.get(email_env.id, email_env.version)

    # Schema validation
    assert "mcp_tool_schema_validation" in env.metadata, f"Expected mcp_tool_schema_validation in env metadata, got keys: {list(env.metadata.keys())}"
    schema = env.metadata["mcp_tool_schema_validation"]
    assert schema["passed"] is True
    assert schema["total_tools"] > 0, "Expected at least one tool"
    logger.info(f"Schema validation: {schema}")

    # Tool correctness
    assert "mcp_tool_correctness_validation" in env.metadata, f"Expected mcp_tool_correctness_validation in env metadata, got keys: {list(env.metadata.keys())}"
    correctness = env.metadata["mcp_tool_correctness_validation"]
    assert correctness["total_tools"] > 0, "Expected at least one tool assessed"
    logger.info(f"Tool correctness: passed={correctness['passed']}, total={correctness['total_tools']}")
    for r in correctness.get("results", []):
        logger.info(f"  {r['tool_name']}: passed={r['passed']} justification={r.get('justification', '')[:100]}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_universe_compatibility(multi_env, environment_universe_artifact, monkeypatch):
    """MultiEnv.validate_universe_compatibility(): real env deploy + PV roundtrip + FileArtifactUniverse
    emission + judge-verdict merge, writing to env_artifacts.

    The agent-judge pipeline (deploy agent, prompt, LLM grading) is STUBBED — it runs out-of-process
    on Modal and is non-deterministic, so we stub it at the step boundary to test OUR wiring, not the
    model. A clean (all-pass) verdict is injected; gating correctness is covered by unit tests for
    apply_judge_verdict / judge_issues_from_verdict.
    """
    from agent_env.artifact import FileArtifactUniverse
    from agent_env.env.env_artifact_store import EnvArtifactType, get_env_artifact_store
    from agent_env.task_step import (
        DeployAgentTaskStep,
        LoadArtifactTaskStep,
        PromptAgentTaskStep,
        RubricsVerifierTaskStep,
        VerifyUniverseLoadExportRoundtripStep,
    )
    from agent_env.task_step.context import DeployedAgent, PromptResponse

    async def _stub_deploy_agent(self, context):
        context.deployed_agents.append(DeployedAgent(agent_name=self.agent_name, api_url="stub://judge", sandbox_id=None))
        return context

    async def _stub_load(self, context):
        return context  # skip staging the FAU onto a (non-existent) agent

    async def _stub_prompt(self, context):
        context.prompt_responses.append(PromptResponse(prompt_id=self.prompt_id, response="stub", agent_name=self.agent_name))
        return context

    async def _stub_rubric(self, context):
        # Clean verdict: every criterion passes (deterministic, no LLM).
        context.metadata.setdefault("verifications", {})[self.verifier_id] = {
            "results": [{"id": c["id"], "result": True, "score": 1.0, "justification": "stub"} for c in self.criteria],
            "score": 1.0,
        }
        return context

    monkeypatch.setattr(DeployAgentTaskStep, "execute", _stub_deploy_agent)
    monkeypatch.setattr(LoadArtifactTaskStep, "execute", _stub_load)
    monkeypatch.setattr(PromptAgentTaskStep, "execute", _stub_prompt)
    monkeypatch.setattr(RubricsVerifierTaskStep, "execute", _stub_rubric)

    instance_id = await multi_env.validate_universe_compatibility(universe_artifact_id=environment_universe_artifact.id)
    assert instance_id is not None
    logger.info(f"Validation instance_id: {instance_id}")

    store = get_env_artifact_store()
    doc = store.get(env_id=multi_env.id, env_version=multi_env.version, artifact_id=environment_universe_artifact.id, artifact_version=environment_universe_artifact.version, type=EnvArtifactType.UNIVERSE_COMPATIBILITY)
    assert doc is not None, "Expected record in env_artifacts collection"
    assert "compatible" in doc["data"]
    assert "services" in doc["data"]
    assert len(doc["data"]["services"]) > 0
    # PV clean + clean stubbed judge verdict => compatible. Judge verdict was merged in.
    assert doc["data"]["compatible"] is True, f"Expected compatible=True, got issues: {doc['data']['services']}"
    assert "agent_judge" in doc["data"], "Expected merged agent-judge verdict"
    logger.info(f"Compatible: {doc['data']['compatible']}, services: {list(doc['data']['services'].keys())}")

    # The round-trip step should have emitted a loadable FileArtifactUniverse with the 3 snapshots.
    fau_id = VerifyUniverseLoadExportRoundtripStep.file_artifact_universe_id(
        multi_env.id, multi_env.version, environment_universe_artifact.id, environment_universe_artifact.version
    )
    fau = FileArtifactUniverse.get(fau_id)
    assert fau is not None, f"Expected emitted FileArtifactUniverse '{fau_id}'"
    assert any(k.startswith("original/") for k in fau.file_artifact_ids)
    assert any(k.startswith("export_1/") for k in fau.file_artifact_ids)
    assert any(k.startswith("export_2/") for k in fau.file_artifact_ids)

    # Verify reverse lookup
    by_artifact = store.get_by_artifact(environment_universe_artifact.id, type=EnvArtifactType.UNIVERSE_COMPATIBILITY)
    assert any(r["env_id"] == multi_env.id for r in by_artifact)

    by_env = store.get_by_env(multi_env.id, type=EnvArtifactType.UNIVERSE_COMPATIBILITY)
    assert any(r["artifact_id"] == environment_universe_artifact.id for r in by_env)


@pytest.fixture(scope="module")
def cardless_probe_env() -> MCPServerEnv:
    """Build a CARDLESS AgentEnvEnvironment server and register it under a distinct env name.

    The image declares no @environment_card, so its identity comes only from the SDK chain.
    The SERVICE_NAME fallback was removed, leaving ENVIRONMENT_NAME (injected by the
    gateway compose) as the sole thing between this server and its class name."""
    src_pkg = REPO_ROOT / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol"
    server_dir = TST_DATA_DIR / "cardless_mcp"
    with tempfile.TemporaryDirectory() as build_dir:
        bd = Path(build_dir)
        shutil.copytree(src_pkg, bd / "agentenv_protocol", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(server_dir / "server.py", bd / "server.py")
        shutil.copy(server_dir / "Dockerfile", bd / "Dockerfile")
        logger.info("Building cardless probe Docker image...")
        result = subprocess.run(
            ["docker", "build", "--platform", "linux/amd64", "-t", "mcp-cardless-probe", str(bd)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cardless probe build failed: {result.stderr}")

    artifact = DockerImageArtifact.put(
        id="mcp-cardless-probe",
        description="Cardless AgentEnvEnvironment server (name resolution probe)",
        image_name="mcp-cardless-probe",
    )
    return MCPServerEnv.put(
        id="mcp-server-cardless-probe",
        docker_image_artifact=artifact,
        environment_name="probeitems",
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cardless_environment_resolves_name_from_environment_name(cardless_probe_env):
    """A cardless SDK server names itself from the injected ENVIRONMENT_NAME.

    This is the only path the SERVICE_NAME removal actually changed. If resolution regressed
    to the class name, the served card would be "UnnamedProbeEnv" and the templated tool
    "UnnamedProbeEnv_add_item" — so this fails loudly rather than silently mis-naming tools.
    """
    env = Env.get(cardless_probe_env.id, version=cardless_probe_env.version)
    try:
        result = await env.deploy()
        base = f"{result.gateway_url}/svc/mcp-probeitems"
        assert await protocol_v1.supports_v1(base) is True

        card = await protocol_v1.get_card(base)
        assert card["name"] == "probeitems", f"cardless server mis-resolved its name: {card['name']!r}"
        assert [t["name"] for t in card["capabilities"]["tools"]] == ["probeitems_add_item"]
        assert "UnnamedProbeEnv" not in json.dumps(card)
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_renamed_registration_serves_its_registered_name(agentenv_items_env, monkeypatch):
    """A carded server registered under another name serves that name, so a card-first load finds it.

    The image declares @environment_card(name="items"). If the declared name won, the composed card
    would list "items", the lookup for "renameditems" would miss, and the load would take legacy REST.
    """
    renamed = MCPServerEnv.put(
        id="mcp-server-renameditems",
        docker_image_artifact=agentenv_items_env.docker_image_artifact,
        environment_name="renameditems",
    )
    env = Env.get(renamed.id, version=renamed.version)
    try:
        result = await env.deploy()
        base = f"{result.gateway_url}/svc/mcp-renameditems"
        assert [c["name"] for c in result.environment_card["children_environments"]] == ["renameditems"]
        card = await protocol_v1.get_card(base)
        assert card["name"] == "renameditems"
        assert [t["name"] for t in card["capabilities"]["tools"]] == ["renameditems_add_item"]

        sent: list[tuple[str, str]] = []
        real_send = httpx.AsyncClient.send

        async def recording_send(self, request, *args, **kwargs):
            sent.append((request.method, request.url.path))
            return await real_send(self, request, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "send", recording_send)
        await env.load_environment_artifact(EnvironmentArtifact.put(
            id="test-renameditems-service-data",
            environment_name="renameditems",
            file_artifact=FileArtifact.put_bytes(
                id="test-renameditems-data",
                description="renameditems v1 seed",
                filename="items.json",
                content=json.dumps({"items": ["r"]}).encode(),
                content_type="application/json",
            ),
        ))
        monkeypatch.undo()
        routes = [r for r in sent if r[1].startswith("/svc/")]
        assert routes == [("POST", "/svc/mcp-renameditems/agentenv")] * 2, routes
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": ["r"]}
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_with_agentenv_environment(agentenv_items_env):
    """v1 data-plane (/v1/data:*) served by AgentEnvEnvironment + decorators.

    Counterpart to the legacy /api/reset coverage in test_gateway_with_single_mcp_server."""
    env = Env.get(agentenv_items_env.id, version=agentenv_items_env.version)
    try:
        result = await env.deploy()
        base = f"{result.gateway_url}/svc/mcp-items"

        # The deploy records the composed card its readiness probe read, and derives mcp_url from it.
        assert [c["name"] for c in result.environment_card["children_environments"]] == ["items"]
        assert result.environment_card_read_at_utc and result.mcp_url == f"{result.gateway_url}/mcp"

        assert await protocol_v1.supports_v1(base) is True
        card = await protocol_v1.get_card(base)
        assert card["protocolVersion"] == "1.0"
        assert card["url"] == "/agentenv" and card["preferredTransport"] == "JSONRPC"
        assert card["additionalInterfaces"] == [{"url": "/mcp", "transport": "mcp"}]
        assert {e["uri"] for e in card["capabilities"]["extensions"]} == {
            "urn:agentenv:disable-tool/v1", "urn:agentenv:set-errors/v1", "urn:agentenv:clock/v1",
            "urn:agentenv:export-as-file/v1",
        }
        # Decorator tools ride capabilities.tools ({environment_name} resolved); imperative tools stay handshake-only.
        assert [t["name"] for t in card["capabilities"]["tools"]] == ["items_add_item"]
        add_item_tool = protocol_v1.find_tool(card, "items_add_item")
        assert add_item_tool["description"] == "Add an item to the store."
        assert add_item_tool["inputSchema"]["required"] == ["item"]
        assert add_item_tool["inputSchema"]["properties"]["item"]["description"] == "The item to add."
        assert protocol_v1.find_tool(card, "list_items") is None
        # set-errors params are A2A-shaped; the request JSON-Schema is derived from the handler signature
        assert protocol_v1.extension_params(card, "urn:agentenv:set-errors/v1") == {
            "endpoint": "/agentenv/ext/set_errors",
            "methods": {
                "set_errors": {
                    "method": "POST",
                    "request": {
                        "type": "object",
                        "properties": {"tool_name": {"type": "string"}, "error_rate": {"type": "number"}},
                        "required": ["tool_name", "error_rate"],
                    },
                }
            },
        }
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": []}
        await protocol_v1.add_data(base, [DataPart(data={"items": ["a", "b"]})])
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": ["a", "b"]}

        await protocol_v1.reset_data(base)
        await protocol_v1.add_data(base, [FilePart(file={"uri": "file:///data/seed.json", "mimeType": "application/json"})])
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": ["seeded-a", "seeded-b"]}

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{base}/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/add", "params": {"parts": []}})
            assert r.status_code == 200 and r.json()["error"]["code"] == -32602
            r = await client.post(f"{base}/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/bogus"})
            assert r.json()["error"]["code"] == -32601

        await protocol_v1.reset_data(base)
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": []}
        await protocol_v1.add_data(base, [DataPart(data={"items": ["x"]})])

        async def assert_tool_sees_items(session, tools_result, tool_names):
            assert "list_items" in tool_names, f"Expected list_items tool, got {tool_names}"
            assert "items_add_item" in tool_names, f"Expected items_add_item tool, got {tool_names}"
            tool_result = await session.call_tool("list_items", {})
            content = "".join(c.text for c in tool_result.content if hasattr(c, "text"))
            assert "x" in content, f"Expected item 'x' via MCP tool, got {content}"
            # _verify_list_tools retries on transient errors — no exact counts after mutations.
            add_result = await session.call_tool("items_add_item", {"item": "w", "times": 2})
            add_content = "".join(c.text for c in add_result.content if hasattr(c, "text"))
            assert '"count"' in add_content, f"Expected count payload from items_add_item, got {add_content}"
            tool_result = await session.call_tool("list_items", {})
            content = "".join(c.text for c in tool_result.content if hasattr(c, "text"))
            assert "w" in content, f"Expected @tool-added item 'w' via list_items, got {content}"

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_tool_sees_items)

        # Exercise load_environment_artifact's v1 branch end-to-end (supports_v1 -> reset + add inline).
        items_artifact = EnvironmentArtifact.put(
            id="test-items-service-data",
            environment_name="items",
            file_artifact=FileArtifact.put_bytes(
                id="test-items-data",
                description="items v1 seed",
                filename="items.json",
                content=json.dumps({"items": ["y", "z"]}).encode(),
                content_type="application/json",
            ),
        )
        await env.load_environment_artifact(items_artifact)
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": ["y", "z"]}

        # Non-JSON EnvironmentArtifact loads via the file:// FilePart branch (no content_type special-case).
        csv_artifact = EnvironmentArtifact.put(
            id="test-items-csv-service-data",
            environment_name="items",
            file_artifact=FileArtifact.put_bytes(
                id="test-items-csv-data",
                description="items non-json seed",
                filename="items.csv",
                content=b"a,b\n1,2",
                content_type="text/csv",
            ),
        )
        await env.load_environment_artifact(csv_artifact)
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": ["file:text/csv:a,b\n1,2"]}

        # Environment Card extension end-to-end (urn:agentenv:set-errors/v1): a server-local
        # error-injection capability declared with @extension. Invoking it makes the named tool
        # start raising at the given rate; calls succeed before and produce errors after.
        async def _count_list_items_errors(n: int) -> int:
            errors = 0
            async with streamable_http_client(result.mcp_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    for _ in range(n):
                        try:
                            res = await session.call_tool("list_items", {})
                        except Exception:
                            errors += 1
                            continue
                        text = "".join(c.text for c in res.content if hasattr(c, "text"))
                        try:
                            ok = not getattr(res, "isError", False) and "items" in json.loads(text)
                        except Exception:
                            ok = False
                        if not ok:
                            errors += 1
            return errors

        assert await _count_list_items_errors(20) == 0  # no injection yet
        ack = await protocol_v1.invoke_extension(
            base, card, "urn:agentenv:set-errors/v1",
            {"tool_name": "list_items", "error_rate": 0.75},
        )
        assert ack == {"tool_name": "list_items", "error_rate": 0.75}
        assert await _count_list_items_errors(20) >= 1  # P(0 errors) = 0.25**20 ~ 1e-12

        # Environment Card extension end-to-end (urn:agentenv:disable-tool/v1): the card
        # advertises a tool-disable capability; reading it via the client accessor and invoking
        # the advertised gateway endpoint removes a named tool from the aggregated tools/list.
        card = await protocol_v1.get_card(base)
        disable_params = protocol_v1.extension_params(card, "urn:agentenv:disable-tool/v1")
        assert disable_params, f"disable-tool extension not advertised: {card.get('capabilities')}"

        # An extension the card does NOT advertise (e.g. enable-tool) is tolerated: the accessors
        # return nothing rather than raising, so probing unknown / forward-compat URIs is safe and
        # yields no invocable endpoint.
        assert protocol_v1.find_extension(card, "urn:agentenv:enable-tool/v1") is None
        assert protocol_v1.extension_params(card, "urn:agentenv:enable-tool/v1") == {}

        async def assert_list_items_present(session, tools_result, tool_names):
            assert "list_items" in tool_names, f"expected list_items before disable, got {tool_names}"

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_list_items_present)

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{result.gateway_url}{disable_params['endpoint']}",
                json={"role": "default", "tools": ["list_items"]},
            )
            assert r.status_code == 200, r.text
            assert "list_items" in r.json()["disabled"]

        async def assert_list_items_disabled(session, tools_result, tool_names):
            assert "list_items" not in tool_names, f"list_items should be disabled, got {tool_names}"

        await _verify_list_tools(mcp_url=result.mcp_url, tool_verifier_callback=assert_list_items_disabled)

        assert result.environment_card_url == f"{result.gateway_url}/.well-known/agent-env.json"
        env_card = await protocol_v1.get_card(result.gateway_url)
        assert env_card["url"] == "/agentenv"
        parent_uris = {e["uri"] for e in env_card["capabilities"]["extensions"]}
        assert {"urn:agentenv:disable-tool/v1", "urn:agentenv:enable-tool/v1"} <= parent_uris

        items_child = protocol_v1.find_child(env_card, "items")
        assert items_child is not None, f"items child missing: {env_card.get('children_environments')}"
        child_endpoints = {e["uri"]: (e.get("params") or {}).get("endpoint") for e in items_child["capabilities"]["extensions"]}
        assert child_endpoints["urn:agentenv:set-errors/v1"] == "/svc/mcp-items/agentenv/ext/set_errors"
        assert child_endpoints["urn:agentenv:disable-tool/v1"] == "/tools/disable"

        ack = await protocol_v1.invoke_extension(
            result.gateway_url, items_child, "urn:agentenv:set-errors/v1",
            {"tool_name": "list_items", "error_rate": 0.5},
        )
        assert ack == {"tool_name": "list_items", "error_rate": 0.5}

        await protocol_v1.reset_data(result.gateway_url)
        await protocol_v1.add_data(result.gateway_url, [DataPart(data={"items": ["env-added"]})])
        assert (await protocol_v1.get_data(result.gateway_url)).parts[0].data == {"items": ["env-added"]}
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_snapshot_agent_state_v1_and_fallback(multi_env_items_email, items_email_universe):
    """snapshot_agent_state._capture_universe_state over a mixed env: items captured via
    v1 get_data, email via the legacy export_state fallback."""
    from agent_env.env import legacy_protocol
    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.task_steps.snapshot_agent_state import SnapshotAgentStateTaskStep

    env = Env.get(multi_env_items_email.id, version=multi_env_items_email.version)
    try:
        result = await env.deploy()
        await env.load_environment_universe_artifact(items_email_universe)

        items_base = legacy_protocol.environment_base_url(result.gateway_url, "items", mcp=True)
        email_base = legacy_protocol.environment_base_url(result.gateway_url, "email", mcp=True)
        assert await protocol_v1.supports_v1(items_base) is True
        assert await protocol_v1.supports_v1(email_base) is False

        assert result.environment_card_url == f"{result.gateway_url}/.well-known/agent-env.json"
        env_card = await protocol_v1.get_card(result.gateway_url)
        assert protocol_v1.find_child(env_card, "items") is not None
        assert protocol_v1.find_child(env_card, "email") is None
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{result.gateway_url}/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/reset"})
            assert "exactly one" in r.json()["error"]["message"]

        step = SnapshotAgentStateTaskStep(
            id="snap-v1-fallback", version=None,
            artifact_id="snap-v1-fallback-artifact", prompt_id="unused",
            env_id=env.id, universe_artifact_id=items_email_universe.id,
        )
        context = TaskStepContext(deployed_envs=[result])
        prefix = get_config().get_object_store().object_url("agent_snapshots/snap-v1-fallback-test/")
        await step._capture_universe_state(context, prefix)

        from agent_env.artifact.store import get_artifact_store

        urls = json.loads(context.metadata["snapshot_json_url"])
        assert set(urls) == {"items", "email"}
        # snapshot_json_url holds unsigned object refs (the consumer re-signs); read
        # them via the store rather than httpx, which can't fetch the store's scheme.
        assert all(u.startswith(prefix) for u in urls.values())
        store = get_artifact_store()
        items_state = json.loads(store.get_object(urls["items"]))
        email_state = json.loads(store.get_object(urls["email"]))
        assert items_state == {"items": ["snap-x", "snap-y"]}
        assert isinstance(email_state, dict) and email_state
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_verify_universe_roundtrip_export_all_v1_and_fallback(multi_env_items_email, items_email_universe):
    """verify_universe_roundtrip._export_all over a mixed env: items, which the env card lists, exported via
    v1 get_data; email, which serves no card, via the legacy export_state fallback."""
    from agent_env.env import legacy_protocol
    from agent_env.task_step.task_steps.multienv_validator.verify_universe_roundtrip import (
        VerifyUniverseLoadExportRoundtripStep,
    )

    env = Env.get(multi_env_items_email.id, version=multi_env_items_email.version)
    try:
        result = await env.deploy()
        await env.load_environment_universe_artifact(items_email_universe)

        items_base = legacy_protocol.environment_base_url(result.gateway_url, "items", mcp=True)
        email_base = legacy_protocol.environment_base_url(result.gateway_url, "email", mcp=True)
        assert await protocol_v1.supports_v1(items_base) is True
        assert await protocol_v1.supports_v1(email_base) is False

        exported = await VerifyUniverseLoadExportRoundtripStep._export_all(result, ["items", "email"])
        assert exported["items"] == {"items": ["snap-x", "snap-y"]}
        assert isinstance(exported["email"], dict) and exported["email"]
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_with_website_env_v1(agentenv_website_env):
    """v1 data-plane branch of WebsiteEnv.load_environment_artifact: supports_v1 -> reset + add inline,
    served by a Starlette AgentEnvStarletteApplication backend behind the gateway."""
    env = Env.get(agentenv_website_env.id, version=agentenv_website_env.version)
    try:
        result = await env.deploy()
        base = f"{result.gateway_url}/svc/webitems"

        assert await protocol_v1.supports_v1(base) is True
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": []}

        assert result.environment_card_url == f"{result.gateway_url}/.well-known/agent-env.json"
        env_card = await protocol_v1.get_card(result.gateway_url)
        assert any(c.get("url") == "/svc/webitems/agentenv" for c in (env_card.get("children_environments") or [])), \
            f"webitems backend not composed as a child: {env_card.get('children_environments')}"

        web_artifact = EnvironmentArtifact.put(
            id="test-webitems-service-data",
            environment_name="webitems",
            file_artifact=FileArtifact.put_bytes(
                id="test-webitems-data",
                description="webitems v1 seed",
                filename="webitems.json",
                content=json.dumps({"items": ["w1", "w2"]}).encode(),
                content_type="application/json",
            ),
        )
        await env.load_environment_artifact(web_artifact)
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": ["w1", "w2"]}

        # Non-JSON EnvironmentArtifact loads via the file:// FilePart branch.
        csv_artifact = EnvironmentArtifact.put(
            id="test-webitems-csv-service-data",
            environment_name="webitems",
            file_artifact=FileArtifact.put_bytes(
                id="test-webitems-csv-data",
                description="webitems non-json seed",
                filename="webitems.csv",
                content=b"a,b\n1,2",
                content_type="text/csv",
            ),
        )
        await env.load_environment_artifact(csv_artifact)
        assert (await protocol_v1.get_data(base)).parts[0].data == {"items": ["file:text/csv:a,b\n1,2"]}
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_triggers_extension(sandbox_provider, mcp_server_envs, email_service_artifact):
    """Exercise urn:agentenv:triggers/v1 against a deployed gateway + email MCP: action + permission
    (real tool-visibility change), a state check pipeline, the /step path + watch_roles gate, recursion
    safety (a search_emails decoy stays armed through the check's internal reads), tool + verify, and
    the add/fire/remove/re-add lifecycle. The nl action is covered by test_gateway_triggers_nl_executor."""
    email_ref = mcp_server_envs["email"]
    email_env = Env.get(email_ref.id, version=email_ref.version)

    async def _state(client, gw):
        resp = await client.get(f"{gw}/triggers/state", timeout=30)
        resp.raise_for_status()
        return resp.json()

    async def _poll(client, gw, want_fired, timeout_s=120):
        import time as _time
        deadline = _time.monotonic() + timeout_s
        last = {}
        while _time.monotonic() < deadline:
            st = await _state(client, gw)
            last = {t["id"]: t["status"] for t in st["triggers"]}
            if all(last.get(i) == "fired" for i in want_fired):
                return st
            await asyncio.sleep(3)
        raise AssertionError(f"triggers {want_fired} not all fired; last statuses={last}")

    try:
        result = await email_env.deploy()
        gw = result.gateway_url
        await email_env.load_environment_artifact(email_service_artifact)  # so search_emails returns data

        # Types: action+permission (grant), state check-pipeline (state), action+sensor (decoy),
        # tool+verify (toolact); the lifecycle leg below adds a fourth. nl -> test_gateway_triggers_nl_executor.
        triggers = [
            {"id": "grant", "when": {"type": "action", "tool": "list_emails"},
             "actions": [{"type": "permission", "action": "enable", "role": "default",
                          "tools": ["send_email"]}]},
            {"id": "state", "when": {"type": "state", "check": {"steps": [
                {"tool": "search_emails", "args": {"query": "alex.chen", "folder_name": "INBOX"},
                 "predicate": {"regex": r"alex\.chen@techcorp\.com"}}]}},
             "actions": []},
            {"id": "decoy", "when": {"type": "action", "tool": "search_emails"},
             "actions": []},
            # tool fire-action + typed verify; both call search_emails internally, so must not fire the decoy.
            {"id": "toolact", "when": {"type": "action", "tool": "list_emails"},
             "actions": [{"type": "tool", "tool": "search_emails",
                          "args": {"query": "alex.chen", "folder_name": "INBOX"},
                          "verify": {"tool": "search_emails",
                                     "args": {"query": "alex.chen", "folder_name": "INBOX"},
                                     "predicate": {"regex": r"alex\.chen@techcorp\.com"}}}]},
        ]

        async with httpx.AsyncClient() as client:
            # Baseline: pre-disable send_email for role default so the grant has a visible effect.
            r = await client.post(f"{gw}/tools/{TOOL_DISABLE_ACTION}",
                                  json={"role": "default", "tools": ["send_email"]}, timeout=30)
            r.raise_for_status()
            lt = await client.post(f"{gw}/step", json={"action": "list_tools"},
                                   headers={AGENT_ENV_ROLE_HEADER: "default"}, timeout=30)
            assert "send_email" not in [t["name"] for t in lt.json()["tools"]], "send_email should start disabled"

            # Register the trigger set over the deployed gateway's extension endpoint.
            reg = await client.post(f"{gw}/triggers/register",
                                    json={"watch_roles": ["default"], "triggers": triggers}, timeout=30)
            assert reg.status_code == 200, f"register failed: {reg.status_code} {reg.text}"
            assert set(reg.json()["added"]) == {"grant", "state", "decoy", "toolact"}

            # MCP path: an agent (role default) calls list_emails -> grant fires; the State check
            # runs (calling search_emails INTERNALLY) and fires.
            async with streamable_http_client(
                result.mcp_url, http_client=httpx.AsyncClient(headers={AGENT_ENV_ROLE_HEADER: "default"}),
            ) as (rs, ws, _):
                async with ClientSession(rs, ws) as session:
                    await session.initialize()
                    call = await session.call_tool("list_emails", {"folder_name": "INBOX"})
                    assert not call.isError, f"list_emails errored: {call.content}"

            st = await _poll(client, gw, want_fired={"grant", "state", "toolact"})
            statuses = {t["id"]: t["status"] for t in st["triggers"]}

            # `tool` fire-action + `verify` both ran (typed acceptance passed).
            assert any(e["kind"] == "verify_ok" and e.get("trigger_id") == "toolact" for e in st["events"]), (
                "toolact tool-action verify did not pass")

            # RECURSION SAFETY: the State check AND toolact's tool-action/verify all issued internal
            # search_emails reads; the decoy (armed on search_emails) must NOT have fired from them.
            assert statuses["decoy"] == "armed", (
                f"decoy fired from engine-internal search_emails reads — recursion-safety violated: {statuses}")

            # Permission fire-action really took effect: send_email now visible to role default.
            lt = await client.post(f"{gw}/step", json={"action": "list_tools"},
                                   headers={AGENT_ENV_ROLE_HEADER: "default"}, timeout=30)
            assert "send_email" in [t["name"] for t in lt.json()["tools"]], "grant did not enable send_email"

            # /step path + role gate: a harness-role search_emails must NOT fire the decoy...
            hr = await client.post(f"{gw}/step",
                                   json={"action": "call_tool", "tool_name": "search_emails",
                                         "arguments": {"query": "alex.chen", "folder_name": "INBOX"}},
                                   headers={AGENT_ENV_ROLE_HEADER: "harness"}, timeout=60)
            assert hr.status_code == 200, f"harness /step failed: {hr.text}"
            await asyncio.sleep(4)
            assert {t["id"]: t["status"] for t in (await _state(client, gw))["triggers"]}["decoy"] == "armed", (
                "harness-role /step call fired a watched trigger")

            # ...but a default-role /step search_emails DOES fire it (external call over /step).
            dr = await client.post(f"{gw}/step",
                                   json={"action": "call_tool", "tool_name": "search_emails",
                                         "arguments": {"query": "alex.chen", "folder_name": "INBOX"}},
                                   timeout=60)
            assert dr.status_code == 200, f"default /step failed: {dr.text}"
            st = await _poll(client, gw, want_fired={"decoy"})

            # Full add / fire / remove / no-fire / re-add / fire lifecycle: remove stops firing (checked
            # at event level, not list membership); re-add works (removed ids are re-addable).
            life = {"id": "life", "when": {"type": "action", "tool": "list_emails"},
                    "actions": []}

            async def _fire_list_emails():
                resp = await client.post(f"{gw}/step",
                                         json={"action": "call_tool", "tool_name": "list_emails",
                                               "arguments": {"folder_name": "INBOX"}}, timeout=60)
                assert resp.status_code == 200, f"/step list_emails failed: {resp.text}"

            # (1) ADD — additive: the original three survive alongside the new one.
            add = await client.post(f"{gw}/triggers/register", json={"triggers": [life]}, timeout=30)
            assert add.status_code == 200 and set(add.json()["all"]) == {"grant", "state", "decoy", "toolact", "life"}, add.text

            # (2) FIRES on a default-role list_emails.
            await _fire_list_emails()
            await _poll(client, gw, want_fired={"life"})

            # (3) REMOVE.
            rm = await client.post(f"{gw}/triggers/remove", json={"ids": ["life"]}, timeout=30)
            assert rm.status_code == 200 and rm.json()["removed"] == ["life"], f"remove failed: {rm.text}"
            assert "life" not in {t["id"] for t in (await _state(client, gw))["triggers"]}, "remove left 'life' armed"

            # (4) DOES NOT FIRE after removal — the same provoking call emits no new 'life' event.
            seq_before = (await _state(client, gw))["events"][-1]["seq"]
            await _fire_list_emails()
            await asyncio.sleep(4)
            new_life = [e for e in (await _state(client, gw))["events"]
                        if e["seq"] > seq_before and e.get("trigger_id") == "life"]
            assert not new_life, f"removed trigger still emitted events: {new_life}"

            # (5) RE-ADD the same id (no retirement) ...
            readd = await client.post(f"{gw}/triggers/register", json={"triggers": [life]}, timeout=30)
            assert readd.status_code == 200 and "life" in readd.json()["all"], f"re-add failed: {readd.text}"

            # (6) ... and it FIRES again (a fresh armed instance).
            await _fire_list_emails()
            st = await _poll(client, gw, want_fired={"life"})

            # No firing failed anywhere.
            bad = [e for e in st["events"] if e["kind"] in ("failed", "eval_error")]
            assert not bad, f"unexpected trigger failures: {bad}"
    finally:
        await email_env.close()


async def _triggers_state(client, gw: str) -> dict:
    resp = await client.get(f"{gw}/triggers/state", timeout=30)
    resp.raise_for_status()
    return resp.json()


async def _poll_triggers(client, gw: str, want_fired: set[str], timeout_s: int = 120) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        st = await _triggers_state(client, gw)
        last = {t["id"]: t["status"] for t in st["triggers"]}
        if all(last.get(i) == "fired" for i in want_fired):
            return st
        await asyncio.sleep(3)
    raise AssertionError(f"triggers {want_fired} not all fired; last statuses={last}")


async def _trajectory_events(client, gw: str) -> list[dict]:
    resp = await client.get(f"{gw}/trajectory", timeout=30)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


async def _send_email(client, gw: str, subject: str) -> str:
    """Send over /step as role default (a watched call) and return the new email_id."""
    resp = await client.post(f"{gw}/step",
                             json={"action": "call_tool", "tool_name": "send_email",
                                   "arguments": {"recipients": ["alex.chen@techcorp.com"],
                                                 "subject": subject, "content": "body"}}, timeout=60)
    assert resp.status_code == 200, f"send_email failed: {resp.text}"
    return json.loads(_step_content_text(resp.json()))["email_id"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_barrier_orders_a_templated_mirror(sandbox_provider, mcp_server_envs):
    """A barrier at `provoking_call` makes a mirror observable before the provoking call answers.

    The gsuite shape in miniature: `send_email` returns `{"email_id": ...}`, a repeat trigger
    templates that id into a `move_email` mirror, and the barrier means the moved mail is readable
    with no sleep and no polling. Leg A asserts the guarantee; on its own it could pass without a
    barrier, because a ~20ms mirror usually beats the next read anyway. Leg D is what makes it
    meaningful: the same wait point with a 1ms bound must give up, answer the call regardless, and
    say so in the trajectory — so only the bound differs and the observable behaviour flips. Leg B
    covers the other half of the fold: a call is held once, under the widest of its barriers, so a
    tight bound must not shrink a sibling's.
    """
    email_ref = mcp_server_envs["email"]
    email_env = Env.get(email_ref.id, version=email_ref.version)

    mirror = {"id": "mirror", "when": {"type": "action", "tool": "send_email", "repeat": True},
              "barrier": {"at": "provoking_call"},
              "actions": [{"type": "tool", "tool": "move_email",
                           "args": {"email_id": "${result.email_id}",
                                    "from_folder": "SENT", "to_folder": "DRAFT"}}]}
    try:
        result = await email_env.deploy()
        gw = result.gateway_url

        async with httpx.AsyncClient() as client:
            reg = await client.post(f"{gw}/triggers/register",
                                    json={"watch_roles": ["default"], "triggers": [mirror]}, timeout=30)
            assert reg.status_code == 200, f"register failed: {reg.status_code} {reg.text}"

            # --- Leg A: the guarantee. No sleep and no poll between the send and the reads.
            email_id = await _send_email(client, gw, "barrier-leg-a")

            moved = await client.post(f"{gw}/step",
                                      json={"action": "call_tool", "tool_name": "get_email_by_id",
                                            "arguments": {"email_id": email_id, "folder_name": "DRAFT"}},
                                      timeout=60)
            assert moved.status_code == 200, moved.text
            body = json.loads(_step_content_text(moved.json()))
            assert body.get("email_id") == email_id, (
                f"the mirror had not landed when the provoking call answered — barrier did not hold: {body}")

            row = {t["id"]: t for t in (await _triggers_state(client, gw))["triggers"]}["mirror"]
            assert row["fire_count"] == 1 and row["status"] == "armed", row
            assert row["failure_count"] == 0 and row["barrier"] == {"at": "provoking_call"}, row

            # `${result.email_id}` resolved against the real ack, not sent literally.
            mirrors = [e for e in await _trajectory_events(client, gw)
                       if e.get("event_type") == "internal_tool_call" and e.get("trigger_id") == "mirror"]
            assert mirrors and mirrors[-1]["ok"] is True, mirrors
            assert mirrors[-1]["arguments"]["email_id"] == email_id, (
                f"the mirror was sent an unresolved or wrong id: {mirrors[-1]['arguments']}")

            # --- Leg C: repeat. A second watched call fires the same registration again.
            await _send_email(client, gw, "barrier-leg-c")
            row = {t["id"]: t for t in (await _triggers_state(client, gw))["triggers"]}["mirror"]
            assert row["fire_count"] == 2 and row["status"] == "armed", row

            # --- Leg B: registration is additive, so this 1ms barrier joins `mirror` on the same
            # call. An omitted timeout means the gateway default, so the widest bound is 30s and
            # both fires still land; a tight trigger must not starve its siblings.
            sibling = {"id": "tight_sibling",
                       "when": {"type": "action", "tool": "send_email", "repeat": True},
                       "barrier": {"at": "provoking_call", "timeout_seconds": 0.001},
                       "actions": [{"type": "tool", "tool": "list_emails",
                                    "args": {"folder_name": "SENT"}}]}
            reg = await client.post(f"{gw}/triggers/register", json={"triggers": [sibling]}, timeout=30)
            assert reg.status_code == 200, reg.text

            held_id = await _send_email(client, gw, "barrier-leg-b")
            assert not [e for e in await _trajectory_events(client, gw)
                        if e.get("event_type") == "trigger_barrier_timeout"], (
                "a 1ms sibling shrank the bound — an omitted timeout must resolve to the default "
                "before widening, not inherit whatever a sibling asked for")
            rows = {t["id"]: t for t in (await _triggers_state(client, gw))["triggers"]}
            assert rows["mirror"]["fire_count"] == 3 and rows["tight_sibling"]["fire_count"] == 1, rows
            assert not any(rows[i]["failure_count"] for i in ("mirror", "tight_sibling")), rows

            # --- Leg D: the same 1ms bound, now the only barrier on its provoking call, must give
            # up and answer anyway.
            solo = {"id": "solo",
                    "when": {"type": "action", "tool": "get_email_by_id", "repeat": True},
                    "barrier": {"at": "provoking_call", "timeout_seconds": 0.001},
                    "actions": [{"type": "tool", "tool": "list_emails",
                                 "args": {"folder_name": "INBOX"}}]}
            reg = await client.post(f"{gw}/triggers/register", json={"triggers": [solo]}, timeout=30)
            assert reg.status_code == 200, reg.text

            read = await client.post(f"{gw}/step",
                                     json={"action": "call_tool", "tool_name": "get_email_by_id",
                                           "arguments": {"email_id": held_id, "folder_name": "DRAFT"}},
                                     timeout=60)
            assert read.status_code == 200, read.text  # 200 == fail-open, not an agent-visible error

            timeouts = [e for e in await _trajectory_events(client, gw)
                        if e.get("event_type") == "trigger_barrier_timeout"]
            assert timeouts, "a 1ms barrier did not record a timeout — the bound was not honoured"
            assert timeouts[-1]["at"] == "provoking_call" and timeouts[-1]["timeout_s"] == 0.001, timeouts[-1]
            assert timeouts[-1]["trigger_ids"] == ["solo"], timeouts[-1]

            # Fail-open abandons the wait, it does not cancel the fire: give it a moment and the
            # action has still run, with no failure recorded.
            for _ in range(15):
                rows = {t["id"]: t for t in (await _triggers_state(client, gw))["triggers"]}
                if rows["solo"]["fire_count"]:
                    break
                await asyncio.sleep(2)
            assert rows["solo"]["fire_count"] == 1, f"the abandoned fire never completed: {rows['solo']}"
            assert rows["solo"]["failure_count"] == 0, rows["solo"]
    finally:
        await email_env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_trigger_rearms_after_a_failed_mirror(sandbox_provider, mcp_server_envs):
    """A failed action re-arms the trigger and increments a sticky tally, on a real deployment.

    v1 parked a trigger at `failed` on its first failure, permanently. Now nothing parks, so the
    per-trigger counters are the only lasting record — and a regression here is silent, because a
    parked trigger and an idle one look identical in `status`.

    The failure has to be a real MCP error: the fixture servers under tst/data return error JSON
    without setting `isError` (unlike servers built on the full BaseService, which flags handled errors), so
    an error *payload* reads as success to the engine. A type-invalid argument is rejected by the tool
    manager before the body runs, which is a genuine failure on any server.
    """
    email_ref = mcp_server_envs["email"]
    email_env = Env.get(email_ref.id, version=email_ref.version)

    breaks = {"id": "breaks", "when": {"type": "action", "tool": "send_email", "repeat": True},
              "actions": [{"type": "tool", "tool": "get_email_by_index",
                           "args": {"idx": "${result.email_id}"}}]}  # an id where an int is required
    try:
        result = await email_env.deploy()
        gw = result.gateway_url

        async with httpx.AsyncClient() as client:
            reg = await client.post(f"{gw}/triggers/register",
                                    json={"watch_roles": ["default"], "triggers": [breaks]}, timeout=30)
            assert reg.status_code == 200, f"register failed: {reg.status_code} {reg.text}"

            await _send_email(client, gw, "rearm-1")
            for _ in range(20):
                st = await _triggers_state(client, gw)
                row = {t["id"]: t for t in st["triggers"]}["breaks"]
                if row["failure_count"]:
                    break
                await asyncio.sleep(2)

            kinds = [e["kind"] for e in st["events"] if e.get("trigger_id") == "breaks"]
            assert "action_failed" in kinds and "failed" in kinds, kinds
            assert row["status"] == "armed", f"a failed action parked the trigger: {row}"
            assert row["failure_count"] == 1 and row["last_failure_at"], row
            assert row["fire_count"] == 0, f"a failed fire counted as a fire: {row}"

            # Still serving: the next watched call fails again rather than being ignored.
            await _send_email(client, gw, "rearm-2")
            for _ in range(20):
                row = {t["id"]: t for t in (await _triggers_state(client, gw))["triggers"]}["breaks"]
                if row["failure_count"] == 2:
                    break
                await asyncio.sleep(2)
            assert row["failure_count"] == 2 and row["status"] == "armed", row
            assert row["fire_count"] == 0, row
    finally:
        await email_env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_trigger_registration_is_fail_loud(sandbox_provider, mcp_server_envs):
    """Every registration 400 the engine promises, against a gateway that has discovered its tools.

    The unknown-tool check is the reason this belongs here and not in the unit tier: it is skipped
    until tool discovery has completed, which a deployed gateway has done and a task's step chain has
    not — so this is the only place it runs at all.
    """
    email_ref = mcp_server_envs["email"]
    email_env = Env.get(email_ref.id, version=email_ref.version)
    action_when = {"type": "action", "tool": "send_email"}
    state_when = {"type": "state", "check": {"tool": "list_emails", "predicate": {"exists": True}}}
    cases = [
        ("unknown action tool", {"id": "b", "when": action_when,
                                 "actions": [{"type": "tool", "tool": "no_such_tool", "args": {}}]},
         "unknown tool"),
        ("unknown when.tool", {"id": "b", "when": {"type": "action", "tool": "no_such_tool"},
                               "actions": []}, "unknown tool"),
        ("barrier not an object", {"id": "b", "when": action_when, "barrier": True, "actions": []},
         "barrier must be an object"),
        ("unknown barrier.at", {"id": "b", "when": action_when,
                                "barrier": {"at": "next_call"}, "actions": []}, "barrier.at must be one of"),
        ("unknown barrier key", {"id": "b", "when": action_when,
                                 "barrier": {"at": "provoking_call", "timout_seconds": 5}, "actions": []},
         "unknown key"),
        ("non-positive timeout", {"id": "b", "when": action_when,
                                  "barrier": {"at": "provoking_call", "timeout_seconds": 0}, "actions": []},
         "positive, finite"),
        ("repeat on a state trigger", {"id": "b", "when": dict(state_when, repeat=True), "actions": []},
         "only valid on action triggers"),
        ("ctx placeholder on a state trigger", {"id": "b", "when": state_when,
                                                "actions": [{"type": "tool", "tool": "move_email",
                                                             "args": {"email_id": "${result.email_id}"}}]},
         "only action triggers carry"),
    ]
    try:
        result = await email_env.deploy()
        gw = result.gateway_url

        async with httpx.AsyncClient() as client:
            # Force discovery, so the unknown-tool checks are live rather than skipped.
            lt = await client.post(f"{gw}/step", json={"action": "list_tools"}, timeout=30)
            assert "send_email" in [t["name"] for t in lt.json()["tools"]], lt.text

            for label, spec, fragment in cases:
                r = await client.post(f"{gw}/triggers/register", json={"triggers": [spec]}, timeout=30)
                assert r.status_code == 400, f"{label}: expected 400, got {r.status_code} {r.text}"
                assert fragment in r.text, f"{label}: 400 did not name the problem: {r.text}"

            # None of them registered anything.
            assert (await _triggers_state(client, gw))["triggers"] == [], "a rejected spec was still added"

            # An unknown TOP-LEVEL key is not a rejection — v1 ignores those, and still does.
            ok = await client.post(f"{gw}/triggers/register",
                                   json={"triggers": [{"id": "tolerated", "when": action_when,
                                                       "actions": [], "sync": True}]}, timeout=30)
            assert ok.status_code == 200, f"an unknown top-level key was rejected: {ok.text}"
    finally:
        await email_env.close()


# One internal MCP round trip each, measured at ~8ms. A `queued` trigger exists only while a fire
# is still running, so the chain is what holds that window open long enough to sample from outside
# the deployment -- there is no slow tool to lean on, and a fire cannot be paused from the client.
# Sized for ~1s, an order of magnitude over the tunnel round trip; the actions themselves are
# free next to deploy and teardown, so the margin costs nothing.
_QUEUED_FIRE_ACTIONS = 120


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_queued_trigger_is_not_reported_settled(sandbox_provider, mcp_server_envs):
    """A `queued` trigger has work outstanding, so the capture path must not read it as settled.

    `queued` means a fire is running *and* at least one more provoking call is waiting -- strictly
    more outstanding than `firing`. The prompt-agent capture path classifies on `firing` alone, so a
    queued trigger reads as settled: the settle poll stops on its first sample, and
    `_capture_is_final` asserts the record can no longer change while no mirror has run yet.

    Both halves are checked against the real `/triggers/state` payload, and the claim is then shown
    to have been premature: at the sample `fire_count` is 0, and two mirrors land afterwards.
    """
    email_ref = mcp_server_envs["email"]
    email_env = Env.get(email_ref.id, version=email_ref.version)

    padding = [{"type": "tool", "tool": "list_emails", "args": {"folder_name": "INBOX"}}]
    slow = {"id": "slow", "when": {"type": "action", "tool": "send_email", "repeat": True},
            "actions": padding * _QUEUED_FIRE_ACTIONS}
    try:
        result = await email_env.deploy()
        gw = result.gateway_url

        async with httpx.AsyncClient() as client:
            reg = await client.post(f"{gw}/triggers/register",
                                    json={"watch_roles": ["default"], "triggers": [slow]}, timeout=30)
            assert reg.status_code == 200, f"register failed: {reg.status_code} {reg.text}"

            # No barrier on purpose: the provoking call returns while its own fire runs, so the
            # second send is detected mid-fire and queues instead of starting a fire of its own.
            await _send_email(client, gw, "queued-a")
            await _send_email(client, gw, "queued-b")

            captured, row = None, None
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                state = await _triggers_state(client, gw)
                row = {t["id"]: t for t in state["triggers"]}["slow"]
                if row["status"] == "queued":
                    captured = state
                    break
                if row["status"] == "armed":
                    break
                await asyncio.sleep(0.05)
            assert captured is not None, (
                f"never sampled `queued` -- the fire outran the poll, so raise "
                f"_QUEUED_FIRE_ACTIONS (last status {row['status']!r}, fire_count {row['fire_count']})")

            row = {t["id"]: t for t in captured["triggers"]}["slow"]
            assert row["pending"] >= 1, row
            # Two sends, so `queued` means the first fire has not yet finished its chain.
            assert row["fire_count"] == 0, row

            assert PromptAgentTaskStep._is_settling(captured) is True, (
                f"the settle poll would stop here with {row['pending']} call(s) queued: {row}")
            assert PromptAgentTaskStep._capture_is_final(captured) is False, (
                f"the capture claimed the record was final with {row['pending']} call(s) queued "
                f"and no mirror yet run: {row}")

            # The claim really was premature: both mirrors land after that sample.
            for _ in range(30):
                rows = {t["id"]: t for t in (await _triggers_state(client, gw))["triggers"]}
                if rows["slow"]["status"] == "armed" and not rows["slow"]["pending"]:
                    break
                await asyncio.sleep(2)
            assert rows["slow"]["status"] == "armed" and rows["slow"]["fire_count"] == 2, rows["slow"]
    finally:
        await email_env.close()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.int_test_slow
async def test_gateway_triggers_nl_executor(sandbox_provider, mcp_server_envs, email_service_artifact):
    """Deployed e2e for the nl fire-action: a real claude-code A2A executor agent realizes an NL event
    over A2A, gated by the engine's typed verify. The one trigger type the hermetic test can't cover
    (needs a reachable model-backed executor). Reliability is engineered — the executor role is locked
    to one tool, the payload is literal, and verify+retry gates firing. int_test_slow, modal_vm only."""
    import time
    import uuid

    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
    from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep

    if sandbox_provider != "modal_vm":
        pytest.skip("nl-executor e2e runs on modal_vm only (VM-mode agent deploy)")

    email_ref = mcp_server_envs["email"]
    email_env = Env.get(email_ref.id, version=email_ref.version)

    marker = f"TRIGPROBE-{uuid.uuid4().hex[:8]}"
    executor_system_prompt = (
        "# Environment Event Executor\n\n"
        "Each message describes exactly one change to make in the workspace. Perform exactly that "
        "change with your available tools — treat quoted text as a literal payload to reproduce "
        "character-for-character; never rephrase, add, or omit anything. Then reply DONE with a "
        "one-line summary, or FAILED and the reason. Never do anything beyond the single change."
    )
    nl_trigger = {
        "id": "nlact",
        "when": {"type": "action", "tool": "list_emails"},
        "actions": [{
            "type": "nl",
            "instruction": (f"Send an email to alex.chen@techcorp.com with the subject line exactly "
                            f"'{marker}' and a short one-sentence body. Reproduce the subject "
                            f"character-for-character."),
            "verify": {"tool": "search_emails", "args": {"query": marker, "folder_name": "SENT"},
                       "predicate": {"regex": marker}},
        }],
    }

    ctx = None
    result = await email_env.deploy()
    try:
        gw = result.gateway_url
        await email_env.load_environment_artifact(email_service_artifact)
        ctx = TaskStepContext(deployed_envs=[result], metadata={})

        async with httpx.AsyncClient() as client:
            # Lock the executor role to exactly one tool (deny-all, then allow send_email). Wrong
            # realizations become structurally impossible, and the executor's own call stays outside
            # watch_roles (no cascade).
            r = await client.post(f"{gw}/tools/{TOOL_DISABLE_ACTION}",
                                  json={"role": "executor", "tools": ["*"]}, timeout=30)
            r.raise_for_status()
            r = await client.post(f"{gw}/tools/{TOOL_ENABLE_ACTION}",
                                  json={"role": "executor", "tools": ["send_email"]}, timeout=30)
            r.raise_for_status()

            # Deploy the real claude-code executor (sonnet), pointed at the gateway MCP, role=executor.
            deploy_exec = DeployAgentTaskStep(
                id="nl-deploy-executor", version=None, env_ids=[email_env.id],
                a2a_agent_id="claude-code-cli", agent_name="executor",
                agent_description="nl trigger executor (constrained role)",
                system_prompt=executor_system_prompt, role="executor",
                sandbox_type="modal_vm", ttl_seconds=1800)
            ctx = await deploy_exec.execute(ctx)
            executor = next(a for a in ctx.deployed_agents if a.agent_name == "executor")
            assert executor.role == "executor", f"executor role not set: {executor.role!r}"

            # The one-tool constraint really held (determinism lever, verified live).
            lt = await client.post(f"{gw}/step", json={"action": "list_tools"},
                                   headers={AGENT_ENV_ROLE_HEADER: "executor"}, timeout=30)
            exec_tools = [t["name"] for t in lt.json()["tools"]]
            assert exec_tools == ["send_email"], f"executor tool surface not locked to one tool: {exec_tools}"

            # Register the nl trigger; the step resolves executor_agent_name -> a2a_url + role.
            reg = RegisterEnvTriggersStep(
                id="nl-register", version=None, env_id=email_env.id, triggers=[nl_trigger],
                watch_roles=["default"], executor_agent_name="executor", executor_timeout_seconds=180)
            ctx = await reg.execute(ctx)

            # Provoke: a default-role list_emails fires the nl action.
            dr = await client.post(f"{gw}/step",
                                   json={"action": "call_tool", "tool_name": "list_emails",
                                         "arguments": {"folder_name": "INBOX"}}, timeout=60)
            assert dr.status_code == 200, f"provoke failed: {dr.text}"

            # Poll for the executor to realize the event and the typed verify to pass.
            async def _state():
                resp = await client.get(f"{gw}/triggers/state", timeout=30)
                resp.raise_for_status()
                return resp.json()

            deadline = time.monotonic() + 300
            st = await _state()
            while time.monotonic() < deadline:
                st = await _state()
                if {t["id"]: t["status"] for t in st["triggers"]}.get("nlact") in ("fired", "failed"):
                    break
                await asyncio.sleep(5)

            status = {t["id"]: t["status"] for t in st["triggers"]}
            events = st["events"]
            assert status.get("nlact") == "fired", (
                f"nl trigger did not fire; status={status}, events={[e['kind'] for e in events]}")
            assert any(e["kind"] == "verify_ok" and e.get("trigger_id") == "nlact" for e in events), (
                f"no verify_ok for nlact; events={events}")

            # Independent of the firing log: the executor really sent the marker email (lands in SENT).
            se = await client.post(f"{gw}/step",
                                   json={"action": "call_tool", "tool_name": "search_emails",
                                         "arguments": {"query": marker, "folder_name": "SENT"}},
                                   headers={AGENT_ENV_ROLE_HEADER: "harness"}, timeout=30)
            assert marker in se.text, f"marker email not found in SENT: {se.text[:400]}"

            # Cascade-safety: nlact was detected exactly once (from the provoke), NOT from the
            # executor's own send_email (executor role is outside watch_roles).
            detected = [e for e in events if e["kind"] == "detected" and e.get("trigger_id") == "nlact"]
            assert len(detected) == 1, f"nlact detected {len(detected)} times (cascade?): {events}"
    finally:
        await email_env.close()


# ─── The agent-facing `get_time` clock read ─────────────────────────────────────────────────────
# The tool manager and the `_server_tools` mirror are different structures read by different
# routes, so checking MCP `tools/list` alone would miss a broken mirror — hence all three surfaces.
class _Surfaces(NamedTuple):
    mcp: dict          # tools/list over MCP, name -> Tool
    step: set          # /step list_tools names
    servers: set       # /state mcp_servers names


async def _get_time_surfaces(client, gw: str, mcp_url: str) -> _Surfaces:
    async with streamable_http_client(mcp_url, http_client=httpx.AsyncClient()) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            mcp = {t.name: t for t in (await session.list_tools()).tools}
    step = (await client.post(f"{gw}/step", json={"action": "list_tools"}, timeout=30)).json()["tools"]
    state = (await client.get(f"{gw}/state", timeout=30)).json()
    return _Surfaces(mcp=mcp, step={t["name"] for t in step},
                     servers={s["name"] for s in state.get("mcp_servers", [])})


async def _get_time_values(client, gw: str, mcp_url: str) -> tuple[str, str]:
    """Read get_time over BOTH call paths. /step routes through `_step_call_gateway_tool`, a wholly
    separate branch reached only because get_time has no `_tool_server_urls` entry."""
    payload = json.loads(await _clock_call_tool(mcp_url, GET_TIME, {}))
    assert set(payload) == {"current_time"}, payload
    r = await client.post(f"{gw}/step",
                          json={"action": "call_tool", "tool_name": GET_TIME, "arguments": {}},
                          timeout=30)
    assert r.status_code == 200, f"/step get_time failed: {r.text}"
    return payload["current_time"], json.loads(_step_content_text(r.json()))["current_time"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_clock_extension(sandbox_provider, mcp_server_envs):
    """clock/v1 against a deployed gateway: advertised + off-by-default (404 unarmed), arm at a rate,
    verify it advances on its own with real time, re-arm at a new rate/t0, then clear (next read 404).

    Also carries the `get_time` assertions: the tool is gated on exactly the arm/re-arm/clear
    transitions this test already walks, so it rides along rather than paying its own deployment."""
    import time
    from datetime import datetime, timezone

    email_ref = mcp_server_envs["email"]
    env = Env.get(email_ref.id, version=email_ref.version)
    t0a, t0b = "2026-06-01T00:00:00Z", "2027-01-01T00:00:00Z"

    def _dt(iso):
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)

    async def _read(client, gw):
        return await client.get(f"{gw}/clock/time", timeout=30)

    async def _rate_check(client, gw, rate, sleep_s=5.0):
        # Two reads bracketing a real sleep; virtual_delta / real_delta ~= rate. Generous band absorbs jitter.
        w1 = time.monotonic()
        v1 = _dt((await _read(client, gw)).json()["virtual_time"])
        await asyncio.sleep(sleep_s)
        v2 = _dt((await _read(client, gw)).json()["virtual_time"])
        w2 = time.monotonic()
        real = w2 - w1
        virtual = (v2 - v1).total_seconds()
        assert virtual > 0, f"rate={rate}: clock did not advance"
        ratio = virtual / real
        assert 0.5 * rate <= ratio <= 1.5 * rate, (
            f"rate={rate}: observed {ratio:.1f}x (virtual {virtual:.0f}s over real {real:.2f}s)")

    try:
        result = await env.deploy()
        gw = result.gateway_url

        async with httpx.AsyncClient() as client:
            # Off-by-default + unconditional advertisement.
            assert (await _read(client, gw)).status_code == 404, "unarmed clock should 404"
            card = (await client.get(f"{gw}/.well-known/agent-env.json", timeout=30)).json()
            uris = [e["uri"] for e in card["capabilities"]["extensions"]]
            assert "urn:agentenv:clock/v1" in uris, f"clock/v1 not advertised: {uris}"

            # Baseline: absent on every surface while unarmed. Not named `base` — the host narrative
            # below rebinds that to a datetime.
            unarmed = await _get_time_surfaces(client, gw, result.mcp_url)
            # Non-vacuity: a gateway whose lifespan discovery failed answers /step with 503, so this
            # asserts the backing tools are really there and the "not in" checks below mean something.
            assert "list_emails" in unarmed.step, f"discovery did not populate _server_tools: {unarmed.step}"
            assert GET_TIME not in unarmed.mcp, sorted(unarmed.mcp)
            assert GET_TIME not in unarmed.step, sorted(unarmed.step)
            assert "gateway" not in unarmed.servers, unarmed.servers

            # A REJECTED arm must register nothing: the ClockError 400 returns before the tool is
            # added. Inverting that ordering exposes a tool whose handler silently answers WALL time.
            assert (await client.put(f"{gw}/clock/set-time",
                                     json={"virtual_time": t0a, "virtual_seconds_per_real_second": 1e9},
                                     timeout=30)).status_code == 400
            assert GET_TIME not in (await _get_time_surfaces(client, gw, result.mcp_url)).mcp
            assert (await _read(client, gw)).status_code == 404, "rejected arm must not arm the clock"

            # Arm at t0a, rate=3600 (1 real s -> 1 virtual h); baseline near t0a.
            armed_at = time.monotonic()
            arm = await client.put(f"{gw}/clock/set-time", json={"virtual_time": t0a, "virtual_seconds_per_real_second": 3600}, timeout=30)
            assert arm.status_code == 200 and arm.json()["armed"] is True and arm.json()["virtual_seconds_per_real_second"] == 3600, arm.text
            base = _dt((await _read(client, gw)).json()["virtual_time"])
            assert 0 <= (base - _dt(t0a)).total_seconds() <= 3600 * 6, "baseline should be near t0a"
            st = (await client.get(f"{gw}/clock/state", timeout=30)).json()
            assert st["armed"] is True and st["virtual_seconds_per_real_second"] == 3600 and st["t0"] == t0a, st
            # gateway self-determined its own server-reachable clock URL (modal i6pn)
            assert st.get("env_get_time_url", "").startswith("http://") and st["env_get_time_url"].endswith("/clock/time"), st

            # Arming registered get_time on all three surfaces, purely additively.
            armed = await _get_time_surfaces(client, gw, result.mcp_url)
            assert GET_TIME in armed.mcp and GET_TIME in armed.step, sorted(armed.step)
            assert set(armed.mcp) - {GET_TIME} == set(unarmed.mcp), "MCP list not purely additive"
            assert armed.step - {GET_TIME} == unarmed.step, "/step list not purely additive"
            assert armed.servers - {"gateway"} == unarmed.servers, armed.servers
            tool = armed.mcp[GET_TIME]
            assert tool.description == GET_TIME_DESC
            assert tool.inputSchema == {"type": "object", "properties": {}, "additionalProperties": False}
            assert tool.annotations.readOnlyHint is True, tool.annotations
            # Bound to the LIVE armed clock, not a stale t0 captured at registration. Both call
            # paths, generous window because the clock is running at 3600x.
            for v in await _get_time_values(client, gw, result.mcp_url):
                drift = (_dt(v) - _dt(t0a)).total_seconds()
                assert 0 <= drift <= 3600 * (time.monotonic() - armed_at + 2), (v, t0a, drift)

            # rate cap: 1 real s = 1 virtual day is the ceiling.
            assert (await client.put(f"{gw}/clock/set-time",
                                     json={"virtual_time": t0a, "virtual_seconds_per_real_second": 1e9}, timeout=30)).status_code == 400
            await _rate_check(client, gw, rate=3600)

            # Re-arm with a DIFFERENT rate + new t0: re-anchors and applies the new rate.
            rearmed_at = time.monotonic()
            arm2 = await client.put(f"{gw}/clock/set-time", json={"virtual_time": t0b, "virtual_seconds_per_real_second": 60}, timeout=30)
            assert arm2.status_code == 200 and arm2.json()["virtual_seconds_per_real_second"] == 60, arm2.text
            base2 = _dt((await _read(client, gw)).json()["virtual_time"])
            assert 0 <= (base2 - _dt(t0b)).total_seconds() <= 60 * 6, "re-arm should reset to near t0b"
            await _rate_check(client, gw, rate=60)

            # A re-arm re-anchors the handler and must NOT double-register: the mirror appends,
            # guarded only by an early return, so a duplicate would show up here.
            rearmed = await _get_time_surfaces(client, gw, result.mcp_url)
            assert set(rearmed.mcp) == set(armed.mcp), "re-arm changed the tool list"
            gw_tools = [s for s in (await client.get(f"{gw}/state", timeout=30)).json()["mcp_servers"]
                        if s["name"] == "gateway"]
            assert len(gw_tools) == 1 and [t["name"] for t in gw_tools[0]["tools"]] == [GET_TIME], gw_tools
            for v in await _get_time_values(client, gw, result.mcp_url):
                drift = (_dt(v) - _dt(t0b)).total_seconds()
                assert 0 <= drift <= 60 * (time.monotonic() - rearmed_at + 2), (v, t0b, drift)

            cl = await client.post(f"{gw}/clock/clear", timeout=30)
            assert cl.status_code == 200 and cl.json()["armed"] is False, cl.text
            assert (await _read(client, gw)).status_code == 404, "cleared clock should 404"
            assert (await client.get(f"{gw}/clock/state", timeout=30)).json() == {"armed": False}, "state after clear"

            # Clear round-trips the tool surface byte-for-byte back to the unarmed baseline.
            cleared = await _get_time_surfaces(client, gw, result.mcp_url)
            assert set(cleared.mcp) == set(unarmed.mcp), "MCP list not restored after clear"
            assert cleared.step == unarmed.step, "/step list not restored after clear"
            assert cleared.servers == unarmed.servers, "the 'gateway' entry was not popped"
            # Unreachable by name, not merely unlisted: a stale registration would answer 2026.
            gone = await client.post(f"{gw}/step",
                                     json={"action": "call_tool", "tool_name": GET_TIME, "arguments": {}},
                                     timeout=30)
            assert gone.status_code == 404 and "Unknown tool" in gone.json()["error"], gone.text

            # Clear -> re-arm round trip, frozen so the value assertions are EXACT. One
            # equality pins second precision, the +00:00 -> Z rewrite, and microsecond stripping.
            refroze = await client.put(f"{gw}/clock/set-time",
                                       json={"virtual_time": CLOCK_DEMO_T, "virtual_seconds_per_real_second": 0},
                                       timeout=30)
            assert refroze.status_code == 200, refroze.text
            frozen = await _get_time_surfaces(client, gw, result.mcp_url)
            assert GET_TIME in frozen.mcp and GET_TIME in frozen.step, "re-registration failed"
            over_mcp, over_step = await _get_time_values(client, gw, result.mcp_url)
            n_mcp, n_step = 1, 1   # counted, not hardcoded: adding a phase must not break the tally
            assert over_mcp == CLOCK_DEMO_T == over_step, (over_mcp, over_step)
            # Producer agreement: the agent-facing tool and the server-facing pull endpoint are the
            # same clock. Exact only because the clock is frozen.
            assert (await _read(client, gw)).json()["virtual_time"] == over_mcp
            # An agent that hallucinates an argument must still get an answer, not a 500.
            stray = await client.post(f"{gw}/step",
                                      json={"action": "call_tool", "tool_name": GET_TIME,
                                            "arguments": {"timezone": "UTC"}}, timeout=30)
            assert stray.status_code == 200, stray.text
            assert json.loads(_step_content_text(stray.json()))["current_time"] == CLOCK_DEMO_T
            n_step += 1

            # One complete, virtual_time-stamped call/result pair per call on
            # BOTH paths — they log through different code, so only here are they proved equivalent.
            # Filtered to the frozen clock, which only this phase's calls carry.
            traj = (await client.get(f"{gw}/trajectory", timeout=30)).text
            events = [json.loads(line) for line in traj.splitlines() if line.strip()]
            calls = [e for e in events
                     if e.get("event_type") == "tool_call"
                     and e.get("tool_call", {}).get("function_name") == GET_TIME
                     and e.get("virtual_time") == CLOCK_DEMO_T]
            over_mcp_ev = [e for e in calls if e.get("source") is None]
            over_step_ev = [e for e in calls if e.get("source") == "step_api"]
            assert over_mcp_ev, f"the MCP call left no trajectory record: {calls}"
            assert over_step_ev, f"the /step call left no trajectory record: {calls}"
            # the zero-arg contract is recorded as called, not just as declared
            assert over_mcp_ev[0]["tool_call"]["arguments"] == {}, over_mcp_ev[0]
            # every call is paired with a result carrying the value the agent actually received
            results = {e.get("tool_call_event_id"): e for e in events
                       if e.get("event_type") == "tool_call_result"}
            for call in calls:
                res = results.get(call["event_id"])
                assert res is not None, f"unpaired tool_call {call['event_id']}: {call}"
                assert res["virtual_time"] == CLOCK_DEMO_T, res
                assert CLOCK_DEMO_T in json.dumps(res["tool_call_result"]), res
            # ...and nothing logged it twice: one record per call actually made, on each path
            assert len(over_mcp_ev) == n_mcp, f"expected {n_mcp} MCP records, got {len(over_mcp_ev)}"
            assert len(over_step_ev) == n_step, f"expected {n_step} /step records, got {len(over_step_ev)}"
            assert len(calls) == n_mcp + n_step, [e.get("source") for e in calls]

            # Role-gated like any other tool, so a deny-all role loses env time unless re-enabled —
            # pinned as deliberate. Last phase: it mutates the gateway-wide role rules.
            await client.post(f"{gw}/tools/{TOOL_DISABLE_ACTION}",
                              json={"role": "locked", "tools": ["*"]}, timeout=30)
            locked = await client.post(f"{gw}/step", json={"action": "list_tools"},
                                       headers={AGENT_ENV_ROLE_HEADER: "locked"}, timeout=30)
            assert GET_TIME not in [t["name"] for t in locked.json()["tools"]], "wildcard deny did not hide get_time"
            denied = await client.post(f"{gw}/step",
                                       json={"action": "call_tool", "tool_name": GET_TIME, "arguments": {}},
                                       headers={AGENT_ENV_ROLE_HEADER: "locked"}, timeout=30)
            assert denied.status_code == 403, denied.text
            assert GET_TIME in (await _get_time_surfaces(client, gw, result.mcp_url)).step, "default role lost it"
            await client.post(f"{gw}/tools/{TOOL_ENABLE_ACTION}",
                              json={"role": "locked", "tools": [GET_TIME]}, timeout=30)
            relocked = await client.post(f"{gw}/step", json={"action": "list_tools"},
                                         headers={AGENT_ENV_ROLE_HEADER: "locked"}, timeout=30)
            assert GET_TIME in [t["name"] for t in relocked.json()["tools"]], "re-enable did not restore it"
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_clock_rate_accuracy(sandbox_provider, mcp_server_envs):
    """Across a matrix of rates on a live gateway: arm, bracket a real sleep, and verify virtual time
    advanced ~rate * real_elapsed (frozen at rate=0)."""
    import time
    from datetime import datetime, timezone

    email_ref = mcp_server_envs["email"]
    env = Env.get(email_ref.id, version=email_ref.version)
    t0 = "2026-06-01T00:00:00Z"
    rates = [0, 1, 60, 3600, 86400]  # frozen, real-time, minute/hour/day compression (86400 = cap)
    sleep_s = 5.0

    def _dt(iso):
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)

    async def _vtime(client, gw):
        return _dt((await client.get(f"{gw}/clock/time", timeout=30)).json()["virtual_time"])

    try:
        result = await env.deploy()
        gw = result.gateway_url
        async with httpx.AsyncClient() as client:
            for rate in rates:
                arm = await client.put(f"{gw}/clock/set-time", json={"virtual_time": t0, "virtual_seconds_per_real_second": rate}, timeout=30)
                assert arm.status_code == 200 and arm.json()["virtual_seconds_per_real_second"] == rate, f"rate={rate}: {arm.text}"
                w1 = time.monotonic()
                v1 = await _vtime(client, gw)
                await asyncio.sleep(sleep_s)
                v2 = await _vtime(client, gw)
                real = time.monotonic() - w1
                virtual = (v2 - v1).total_seconds()
                if rate == 0:
                    assert virtual == 0, f"rate=0 must freeze; advanced {virtual}s over {real:.1f}s real"
                else:
                    assert virtual > 0, f"rate={rate}: clock did not advance"
                    ratio = virtual / real
                    assert 0.5 * rate <= ratio <= 1.5 * rate, (
                        f"rate={rate}: observed {ratio:.1f}x (virtual {virtual:.0f}s / real {real:.2f}s)")

            # Re-arm re-anchors to a new t0.
            t0b = "2027-01-01T00:00:00Z"
            assert (await client.put(f"{gw}/clock/set-time", json={"virtual_time": t0b, "virtual_seconds_per_real_second": 1}, timeout=30)).status_code == 200
            assert 0 <= (await _vtime(client, gw) - _dt(t0b)).total_seconds() <= 6, "re-arm should reset to near t0b"
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_time_trigger_extension(sandbox_provider, mcp_server_envs):
    """when.type=='time' clock triggers against a deployed gateway + email MCP.

    Proves the event-anchored flagship (+24h after submit -> a correction lands, seen as a real
    permission mutation) and a fixed recurrence that catches up and terminates via `count`."""
    email_ref = mcp_server_envs["email"]
    env = Env.get(email_ref.id, version=email_ref.version)
    t0 = "2026-06-01T00:00:00Z"

    async def _state(client, gw):
        r = await client.get(f"{gw}/triggers/state", timeout=30)
        r.raise_for_status()
        return {t["id"]: t for t in r.json()["triggers"]}, r.json()["events"]

    async def _drive(client, gw):
        # A watched (role default) call: fires the `submit` anchor once, and ticks time-trigger eval.
        r = await client.post(f"{gw}/step",
                              json={"action": "call_tool", "tool_name": "list_emails",
                                    "arguments": {"folder_name": "INBOX"}}, timeout=60)
        assert r.status_code == 200, f"/step list_emails failed: {r.text}"

    async def _drive_until(client, gw, done, timeout_s=90):
        import time as _time
        deadline = _time.monotonic() + timeout_s
        triggers, events = {}, []
        while _time.monotonic() < deadline:
            await _drive(client, gw)
            triggers, events = await _state(client, gw)
            if done(triggers):
                return triggers, events
            await asyncio.sleep(2)
        raise AssertionError(f"condition not met in {timeout_s}s; last={triggers}")

    triggers = [
        {"id": "submit", "when": {"type": "action", "tool": "list_emails"}, "actions": []},
        # FLAGSHIP: +24 virtual-hours after the agent 'submits', enable send_email (a discoverable env change).
        {"id": "corr", "when": {"type": "time", "after": "submit", "offset": "PT24H"},
         "actions": [{"type": "permission", "action": "enable", "role": "default", "tools": ["send_email"]}]},
        # Fixed recurring, terminates after 3 arrivals (proves catch-up + count).
        {"id": "pulse", "when": {"type": "time", "every": "PT12H", "count": 3}, "actions": []},
    ]

    try:
        result = await env.deploy()
        gw = result.gateway_url
        async with httpx.AsyncClient() as client:
            # Pre-disable send_email so the flagship's permission mutation is observable.
            await client.post(f"{gw}/tools/{TOOL_DISABLE_ACTION}",
                              json={"role": "default", "tools": ["send_email"]}, timeout=30)
            # Arm the clock fast (rate=86400: 1 real s = 1 virtual day, so +24h lands in ~1 real s).
            arm = await client.put(f"{gw}/clock/set-time",
                                   json={"virtual_time": t0, "virtual_seconds_per_real_second": 86400}, timeout=30)
            assert arm.status_code == 200 and arm.json()["armed"] is True, arm.text
            reg = await client.post(f"{gw}/triggers/register",
                                    json={"watch_roles": ["default"], "triggers": triggers}, timeout=30)
            assert reg.status_code == 200, f"register failed: {reg.text}"

            # First watched call fires `submit`, which stamps corr's concrete mark (an 'anchored' event).
            await _drive(client, gw)
            tstate, events = await _drive_until(client, gw, lambda t: t["submit"]["status"] == "fired")
            anchored = [e for e in events if e["kind"] == "anchored" and e.get("trigger_id") == "corr"]
            assert anchored and anchored[0]["anchor"] == "submit", f"corr was not anchored: {events[-6:]}"

            # Keep driving watched calls; virtual time crosses submit+24h -> corr fires; pulse catches up to 3.
            tstate, events = await _drive_until(
                client, gw, lambda t: t["corr"]["status"] == "fired" and t["pulse"]["status"] == "fired")
            assert tstate["corr"]["fire_count"] == 1, tstate["corr"]
            assert tstate["pulse"]["fire_count"] == 3, tstate["pulse"]  # count-terminated at 3 arrivals

            # The flagship's env mutation actually took effect: send_email is now visible to role default.
            lt = await client.post(f"{gw}/step", json={"action": "list_tools"},
                                   headers={AGENT_ENV_ROLE_HEADER: "default"}, timeout=30)
            assert "send_email" in [t["name"] for t in lt.json()["tools"]], "corr did not enable send_email"

            # Observability: the mark is the anchor instant + PT24H, and the arrival fires at it.
            from datetime import datetime, timedelta, timezone

            def _dt(iso):
                return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)

            corr_anchored = [e for e in events if e["kind"] == "anchored" and e.get("trigger_id") == "corr"]
            corr_fired = [e for e in events if e["kind"] == "fired" and e.get("trigger_id") == "corr"]
            assert corr_anchored and corr_fired, (corr_anchored, corr_fired)
            # Loose lower bound: _emit re-reads the clock just after the mark, and at rate 86400 even
            # 100ms of real skew is ~2.4 virtual hours. Still kills an offset-dropped regression (~0h).
            offset = _dt(corr_anchored[0]["mark"]) - _dt(corr_anchored[0]["virtual_time"])
            assert timedelta(hours=12) < offset <= timedelta(hours=24), (offset, corr_anchored)
            assert corr_fired[0]["mark"] == corr_anchored[0]["mark"], (corr_fired, corr_anchored)
            # a count-terminated recurrence records each arrival's mark: t0 + n*PT12H, exactly
            assert [e["mark"] for e in events if e["kind"] == "fired" and e.get("trigger_id") == "pulse"] == [
                "2026-06-01T12:00:00Z", "2026-06-02T00:00:00Z", "2026-06-02T12:00:00Z"]

            bad = [e for e in events if e["kind"] in ("failed", "eval_error")]
            assert not bad, f"unexpected trigger failures: {bad}"
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gateway_autonomous_time_trigger(sandbox_provider, mcp_server_envs):
    """The autonomous driver: time-triggers fire with ZERO agent tool calls.

    This makes no /step calls at all — only GETs of /triggers/state, which are not watched tool
    calls — so any firing comes solely from the gateway's background poller."""
    email_ref = mcp_server_envs["email"]
    env = Env.get(email_ref.id, version=email_ref.version)
    t0 = "2026-06-01T00:00:00Z"

    async def _state(client, gw):
        r = await client.get(f"{gw}/triggers/state", timeout=30)
        r.raise_for_status()
        body = r.json()
        return {t["id"]: t for t in body["triggers"]}, body

    async def _wait_until(client, gw, done, timeout_s=40):
        import time as _time
        deadline = _time.monotonic() + timeout_s
        triggers, body = {}, {}
        while _time.monotonic() < deadline:
            await asyncio.sleep(1)  # NO /step calls — only the background driver advances anything
            triggers, body = await _state(client, gw)
            if done(triggers):
                return triggers, body
        raise AssertionError(f"autonomous firing did not occur in {timeout_s}s; last={triggers}")

    from datetime import datetime, timezone

    def _dt(iso):
        """Parse: '…00.5Z' string-sorts BEFORE '…00Z' because '.' < 'Z'."""
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)

    # Both are sensors (actions: []): the `fired` status is the signal, no executor / env mutation needed.
    triggers = [
        {"id": "solo", "when": {"type": "time", "at": "PT2H"}, "actions": []},   # one-shot ~2 real s in
        {"id": "beat", "when": {"type": "time", "every": "PT1H"}, "actions": []},  # ~1 arrival / real s
    ]

    try:
        result = await env.deploy()
        gw = result.gateway_url
        async with httpx.AsyncClient() as client:
            # rate 3600: 1 real s = 1 virtual h, so PT2H lands in ~2 real s and PT1H arrives ~1/real s.
            arm = await client.put(f"{gw}/clock/set-time",
                                   json={"virtual_time": t0, "virtual_seconds_per_real_second": 3600}, timeout=30)
            assert arm.status_code == 200 and arm.json()["armed"] is True, arm.text
            reg = await client.post(f"{gw}/triggers/register",
                                    json={"watch_roles": ["default"], "triggers": triggers}, timeout=30)
            assert reg.status_code == 200, f"register failed: {reg.text}"

            # Wait WITHOUT ever driving a tool call — the driver alone must fire these.
            tstate, body = await _wait_until(
                client, gw, lambda t: t["solo"]["status"] == "fired" and t["beat"]["fire_count"] >= 3)
            events = body["events"]
            assert tstate["solo"]["fire_count"] == 1, tstate["solo"]
            # Not `== "armed"`: the snapshot can land mid-fire. What matters is it never terminates.
            assert tstate["beat"]["status"] in ("armed", "firing"), tstate["beat"]
            assert tstate["beat"]["fire_count"] >= 3, tstate["beat"]
            bad = [e for e in events if e["kind"] in ("failed", "eval_error")]
            assert not bad, f"unexpected trigger failures: {bad}"

            # Observability. Only rate/poll-phase-independent facts: a `mark` is exact
            # (t0 + schedule), `virtual_time` depends on which tick observed it so is only bounded.
            assert body["events_dropped"] == 0, body["events_dropped"]
            assert tstate["solo"]["type"] == "time" and tstate["beat"]["type"] == "time", tstate
            assert tstate["solo"]["when"] == {"type": "time", "at": "PT2H"}, tstate["solo"]["when"]
            assert tstate["beat"]["when"] == {"type": "time", "every": "PT1H"}, tstate["beat"]["when"]

            fired = [e for e in events if e["kind"] == "fired"]
            assert len(fired) >= 4, fired  # else the all(...) below are vacuous
            assert all("mark" in e and "virtual_time" in e for e in fired), fired
            assert all(_dt(e["virtual_time"]) >= _dt(e["mark"]) for e in fired), \
                f"an arrival cannot be emitted before the mark it was due at: {fired}"
            # on the virtual axis, not real time
            assert all(_dt(t0) <= _dt(e["virtual_time"]) < _dt("2026-06-04T00:00:00Z") for e in fired), fired

            assert [e["mark"] for e in fired if e["trigger_id"] == "solo"] == ["2026-06-01T02:00:00Z"]
            beat_marks = [e["mark"] for e in fired if e["trigger_id"] == "beat"]
            assert beat_marks[:3] == ["2026-06-01T01:00:00Z", "2026-06-01T02:00:00Z",
                                      "2026-06-01T03:00:00Z"], beat_marks  # t0 + n*PT1H, in order

            detected = [e for e in events if e["kind"] == "detected"]
            assert detected and all(e["provoking"] == {"source": "clock"} for e in detected), detected
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sync_env_clock_step(sandbox_provider, multi_env_items_email):
    """SyncEnvClockTaskStep against a MultiEnv mixing a clock-AWARE server (items advertises clock/v1 +
    sync_time) and a LEGACY one (email, no v1 card): the step arms the gateway clock, then fans out
    sync_time — items is invoked for real and reads the gateway's virtual time back through the handed
    env_get_time_url (proving the self-determined URL is server->gateway reachable), while email is tolerated
    (skipped). Strict mode fails loud when a server lacks the extension."""
    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep

    env = Env.get(multi_env_items_email.id, version=multi_env_items_email.version)
    t0 = "2026-06-01T00:00:00Z"
    try:
        result = await env.deploy()
        gw = result.gateway_url

        ctx = await SyncEnvClockTaskStep(
            id="sync", version=None, env_id=env.id, virtual_time=t0, virtual_seconds_per_real_second=1,
        ).execute(TaskStepContext(deployed_envs=[result], metadata={}))
        cfg = ctx.metadata["clock_configurations"][-1]

        # clock-aware server (items): real sync_time invoked; it read the gateway's virtual time back
        # through env_get_time_url -> proves the self-determined URL is server->gateway reachable.
        synced = {s["service"]: s for s in cfg["synced"]}
        assert "items" in synced, f"items should have synced: {cfg}"
        assert "2026-06-01" in json.dumps(synced["items"]["result"]), synced["items"]

        # legacy server (email): tolerated skip (no v1 card).
        skipped = {s["service"]: s["reason"] for s in cfg["skipped"]}
        assert skipped.get("email") == "no_env_card", f"email should skip: {cfg}"
        assert cfg["env_get_time_url"].endswith("/clock/time"), cfg

        # gateway clock is armed by the step.
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{gw}/clock/time", timeout=30)
            assert r.status_code == 200 and r.json()["virtual_time"].startswith("2026-06-01"), r.text

        # strict mode fails loud when any server lacks clock/v1 (email, no v1 card).
        with pytest.raises((RuntimeError, httpx.HTTPStatusError)):
            await SyncEnvClockTaskStep(
                id="strict", version=None, env_id=env.id, virtual_time=t0, virtual_seconds_per_real_second=1,
                tolerate_missing_sync_time=False,
            ).execute(TaskStepContext(deployed_envs=[result], metadata={}))
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Cross-repo checkpoint: a REAL MCP server built from sources outside this repository
# (reminder), carrying the clock/v1 ClockConsumerMixin, consumes the gateway's virtual clock. Proof
# is behavioral: reminder_get_due_reminders ("due <= now") reflects the ARMED virtual time.
# ─────────────────────────────────────────────────────────────────────────────
import secrets as _secrets


def _mcp_server_build_ctx(server: str) -> Path:
    """The Docker build context for ``server``: ``AGENT_ENV_TEST_MCP_SERVERS_DIR``, a directory holding
    one ``<server>/Dockerfile`` per server. Skips the test when it is unset or lacks ``server``."""
    root = os.environ.get("AGENT_ENV_TEST_MCP_SERVERS_DIR")
    build_ctx = Path(root).expanduser() if root else None
    if build_ctx is None or not (build_ctx / server / "Dockerfile").exists():
        pytest.skip(missing_capability_reason(MCP_SERVER_SOURCES))
    return build_ctx


CLOCK_DEMO_T = "2019-06-15T12:00:00Z"  # canonical virtual time — deliberately far from real "now"
# Reminders straddling T: three due on/before noon UTC, two after. At real time (2026) ALL are due.
_REMINDER_CLOCK_DATA = {"reminders": [
    {"reminder_id": "c10cc0de0000000000000001", "title": "Renew passport", "due_datetime": "2019-06-10T09:00:00Z", "description": "before T", "repetition_unit": None, "repetition_value": None, "time_notified": None},
    {"reminder_id": "c10cc0de0000000000000002", "title": "Dentist visit", "due_datetime": "2019-06-14T15:00:00Z", "description": "before T", "repetition_unit": None, "repetition_value": None, "time_notified": None},
    {"reminder_id": "c10cc0de0000000000000003", "title": "Morning standup", "due_datetime": "2019-06-15T08:00:00Z", "description": "before T (same day AM)", "repetition_unit": None, "repetition_value": None, "time_notified": None},
    {"reminder_id": "c10cc0de0000000000000004", "title": "Afternoon sync", "due_datetime": "2019-06-15T15:00:00Z", "description": "after T (same day PM)", "repetition_unit": None, "repetition_value": None, "time_notified": None},
    {"reminder_id": "c10cc0de0000000000000005", "title": "Quarterly review", "due_datetime": "2019-06-20T10:00:00Z", "description": "after T", "repetition_unit": None, "repetition_value": None, "time_notified": None},
]}
_DUE_AT_T = {"Renew passport", "Dentist visit", "Morning standup"}       # due_datetime <= T
_NOT_DUE_AT_T = {"Afternoon sync", "Quarterly review"}                    # due_datetime > T


@pytest.fixture(scope="module")
def reminder_env() -> MCPServerEnv:
    """Build the real `reminder` server, which carries the clock/v1 ClockConsumerMixin."""
    build_ctx = _mcp_server_build_ctx("reminder")
    logger.info("Building reminder Docker image...")
    build = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64",
         "-f", str(build_ctx / "reminder" / "Dockerfile"),
         "-t", "mcp-reminder-clock", str(build_ctx)],
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(f"reminder build failed: {build.stderr[-2000:]}")
    rid = _secrets.token_hex(3)
    artifact = DockerImageArtifact.put(
        id=f"mcp-reminder-clock-{rid}", description="reminder server with clock/v1 consumer", image_name="mcp-reminder-clock",
    )
    env = MCPServerEnv.put(
        id=f"mcp-server-reminder-clock-{rid}", docker_image_artifact=artifact, environment_name="reminder",
    )
    logger.info(f"Created reminder MCPServerEnv: {env.id} version={env.version}")
    return env


@pytest.fixture(scope="module")
def reminder_clock_multi_env(reminder_env) -> MultiEnv:
    """Wrap reminder in a MultiEnv (the proven deploy + universe-load path)."""
    return MultiEnv.put(
        id=f"multi-reminder-clock-{_secrets.token_hex(3)}", mcp_server_envs=[reminder_env],
        metadata={"category": "integration-test", "owner": "agent-env"},
    )


@pytest.fixture(scope="module")
def reminder_clock_universe() -> EnvironmentUniverseArtifact:
    sa = EnvironmentArtifact.put(
        id=f"reminder-clock-service-data-{_secrets.token_hex(3)}", environment_name="reminder",
        file_artifact=FileArtifact.put_bytes(
            id=f"reminder-clock-data-{_secrets.token_hex(3)}", description="reminders straddling the canonical virtual time",
            filename="reminders.json", content=json.dumps(_REMINDER_CLOCK_DATA).encode(), content_type="application/json",
        ),
    )
    return EnvironmentUniverseArtifact.put(id=f"reminder-clock-universe-{_secrets.token_hex(3)}", environment_artifacts=[sa])


@pytest.mark.asyncio
@pytest.mark.integration
async def test_reminder_clock_consumer(sandbox_provider, reminder_clock_multi_env, reminder_clock_universe):
    """Checkpoint: the real reminder server consumes the gateway virtual clock.

    Deploy reminder behind the clock-producer gateway, load reminders straddling a canonical
    virtual time T (far from real now), then arm the gateway clock to T + fan out sync_time.
    reminder_get_due_reminders ("due <= now") must follow the VIRTUAL clock: on the real wall
    clock (~2026) every 2019 reminder is due; after arming to 2019-06-15T12:00:00Z only the ones
    due <= T are — that before/after gap is the proof the server is on the gateway's virtual clock.
    """
    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep

    env = Env.get(reminder_clock_multi_env.id, version=reminder_clock_multi_env.version)
    try:
        result = await env.deploy()
        await env.load_environment_universe_artifact(reminder_clock_universe)

        async def _due_titles() -> set:
            async with streamable_http_client(result.mcp_url, http_client=httpx.AsyncClient()) as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    res = await session.call_tool("reminder_get_due_reminders", {"limit": 250})
                    text = "".join(c.text for c in res.content if hasattr(c, "text"))
                    return {rem["title"] for rem in json.loads(text)["reminders"]}

        # reminder advertises the clock/v1 consumer extension on its card.
        card = await protocol_v1.get_card(f"{result.gateway_url}/svc/mcp-reminder")
        assert "urn:agentenv:clock/v1" in {e["uri"] for e in card["capabilities"]["extensions"]}, card

        # BEFORE arming: real wall clock (~2026) -> every 2019 reminder is due.
        before = await _due_titles()
        assert before == _DUE_AT_T | _NOT_DUE_AT_T, f"expected all reminders due on wall clock, got {sorted(before)}"

        # Arm the gateway clock to the canonical T and fan sync_time out to reminder.
        ctx = await SyncEnvClockTaskStep(
            id="sync", version=None, env_id=env.id, virtual_time=CLOCK_DEMO_T, virtual_seconds_per_real_second=1,
        ).execute(TaskStepContext(deployed_envs=[result], metadata={}))
        cfg = ctx.metadata["clock_configurations"][-1]
        synced = {s["service"]: s for s in cfg["synced"]}
        assert "reminder" in synced, f"reminder should have synced: {cfg}"
        # The server read the armed virtual time back through env_get_time_url (proves the reroute is live).
        assert "2019-06-15" in json.dumps(synced["reminder"]["result"]), synced["reminder"]

        # AFTER arming: virtual clock (2019-06-15T12:00Z) -> only reminders due <= T.
        after = await _due_titles()
        assert after == _DUE_AT_T, f"expected only due<=T reminders on the virtual clock, got {sorted(after)}"
        # Armed through the real SyncEnvClockTaskStep rather than the raw route, and on the
        # same virtual clock the reminder server just demonstrated. Prefix, not equality: rate=1.
        gt = json.loads(await _clock_call_tool(result.mcp_url, GET_TIME, {}))
        assert set(gt) == {"current_time"}, gt
        assert gt["current_time"].startswith("2019-06-15T12:"), gt

        logger.info(f"CLOCK DEMO OK: wall-clock due={sorted(before)}  ;  virtual({CLOCK_DEMO_T}) due={sorted(after)}")
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet batch 1: gmail, calendar_mcp, slack, google_calendar, strava each
# built with the clock/v1 ClockConsumerMixin consume the gateway's virtual clock.
# Same proof shape as reminder above: a temporal tool's output shifts wall -> T.
# ─────────────────────────────────────────────────────────────────────────────
import io as _io
import zipfile as _zipfile


@dataclass
class _ClockServerSpec:
    server: str             # <server>/Dockerfile subdir of the servers' build context
    env_name: str           # environment_name / card name; gateway route is /svc/mcp-<env_name>
    universe_filename: str   # single EnvironmentArtifact filename the deployed reset path loads
    universe_bytes: bytes


def _build_clock_mcp_env(spec: "_ClockServerSpec") -> MCPServerEnv:
    build_ctx = _mcp_server_build_ctx(spec.server)
    tag = f"mcp-{spec.env_name}-clock"
    logger.info(f"Building {spec.server} Docker image...")
    build = subprocess.run(
        ["docker", "build", "--platform", "linux/amd64",
         "-f", str(build_ctx / spec.server / "Dockerfile"),
         "-t", tag, str(build_ctx)],
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(f"{spec.server} build failed: {build.stderr[-2000:]}")
    rid = _secrets.token_hex(3)
    artifact = DockerImageArtifact.put(
        id=f"mcp-{spec.env_name}-clock-{rid}", description=f"{spec.server} server with clock/v1 consumer", image_name=tag,
    )
    return MCPServerEnv.put(
        id=f"mcp-server-{spec.env_name}-clock-{rid}", docker_image_artifact=artifact,
        environment_name=spec.env_name,
    )


def _clock_universe(spec: "_ClockServerSpec") -> EnvironmentUniverseArtifact:
    ctype = "application/zip" if spec.universe_filename.endswith(".zip") else "application/json"
    sa = EnvironmentArtifact.put(
        id=f"{spec.env_name}-clock-service-data-{_secrets.token_hex(3)}", environment_name=spec.env_name,
        file_artifact=FileArtifact.put_bytes(
            id=f"{spec.env_name}-clock-data-{_secrets.token_hex(3)}", description=f"{spec.env_name} universe straddling the canonical virtual time",
            filename=spec.universe_filename, content=spec.universe_bytes, content_type=ctype,
        ),
    )
    return EnvironmentUniverseArtifact.put(id=f"{spec.env_name}-clock-universe-{_secrets.token_hex(3)}", environment_artifacts=[sa])


async def _clock_call_tool(mcp_url: str, name: str, args: dict) -> str:
    async with streamable_http_client(mcp_url, http_client=httpx.AsyncClient()) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            res = await session.call_tool(name, args)
            return "".join(c.text for c in res.content if hasattr(c, "text"))


async def _arm_gateway_clock(env, result, virtual_time: str = CLOCK_DEMO_T, rate: float = 1) -> dict:
    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
    ctx = await SyncEnvClockTaskStep(
        id="sync", version=None, env_id=env.id, virtual_time=virtual_time, virtual_seconds_per_real_second=rate,
    ).execute(TaskStepContext(deployed_envs=[result], metadata={}))
    return ctx.metadata["clock_configurations"][-1]


async def _deploy_clock_env(spec: "_ClockServerSpec"):
    """Build + register + deploy <spec.server> behind the clock-producer gateway; load its universe (TTL <= 20 min)."""
    mcp_env = _build_clock_mcp_env(spec)
    multi = MultiEnv.put(
        id=f"multi-{spec.env_name}-clock-{_secrets.token_hex(3)}", mcp_server_envs=[mcp_env],
        metadata={"category": "integration-test", "owner": "agent-env"},
    )
    env = Env.get(multi.id, version=multi.version)
    result = await env.deploy(ttl_seconds=1200)
    await env.load_environment_universe_artifact(_clock_universe(spec))
    return env, result


async def _assert_advertises_clock(result, svc_slug: str) -> None:
    card = await protocol_v1.get_card(f"{result.gateway_url}/svc/{svc_slug}")
    assert "urn:agentenv:clock/v1" in {e["uri"] for e in card["capabilities"]["extensions"]}, card


# ── gmail: gmail_search_messages "newer_than:7d" — empty on wall (2026), returns the 2019 msg at T ──
_GMAIL_CLOCK_SEED = {
    "users": [{"emailAddress": "alice@example.com"}],
    "labels": [],
    "threads": [
        {"id": "alice@example.com/T_inside", "user_email": "alice@example.com", "messages": [
            {"id": "alice@example.com/<inside@ex>", "thread_id": "alice@example.com/T_inside", "user_email": "alice@example.com",
             "labelIds": ["INBOX", "UNREAD"], "from": "carol@example.com <Carol>", "to": "alice@example.com",
             "subject": "Inside 7d window", "date": "Mon, 10 Jun 2019 09:00:00 +0000", "body": "Inside the 7d-before-T window."}]},
        {"id": "alice@example.com/T_outside", "user_email": "alice@example.com", "messages": [
            {"id": "alice@example.com/<outside@ex>", "thread_id": "alice@example.com/T_outside", "user_email": "alice@example.com",
             "labelIds": ["INBOX"], "from": "dan@example.com <Dan>", "to": "alice@example.com",
             "subject": "Outside 7d window", "date": "Wed, 01 May 2019 09:00:00 +0000", "body": "Outside the 7d-before-T window."}]},
    ],
}


def _gmail_universe_zip() -> bytes:
    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as z:
        z.writestr("users.json", json.dumps(_GMAIL_CLOCK_SEED["users"]))
        z.writestr("labels.json", json.dumps(_GMAIL_CLOCK_SEED["labels"]))
        for i, thread in enumerate(_GMAIL_CLOCK_SEED["threads"]):
            z.writestr(f"threads/{i:08d}.json", json.dumps(thread))
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gmail_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("gmail", "gmail", "gmail.zip", _gmail_universe_zip())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-gmail")
        args = {"query": "newer_than:7d", "user_email": "alice@example.com"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "gmail_search_messages", args))
        assert before["resultSizeEstimate"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        synced = {s["service"]: s for s in cfg["synced"]}
        assert "gmail" in synced, cfg
        assert "2019-06-15" in json.dumps(synced["gmail"]["result"]), synced["gmail"]
        after = json.loads(await _clock_call_tool(result.mcp_url, "gmail_search_messages", args))
        assert after["resultSizeEstimate"] == 1, after
        logger.info(f"GMAIL CLOCK OK: wall={before['resultSizeEstimate']} armed={after['resultSizeEstimate']}")
    finally:
        await env.close()


# ── calendar_mcp: read_today — empty on wall (2026), returns the 2019-06-15 event at T (not the 06-20 decoy) ──
_CALENDAR_CLOCK_DATA = {"events": [
    {"event_id": "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1", "title": "Virtual-date standup", "start_datetime": "2019-06-15T09:00:00Z", "end_datetime": "2019-06-15T10:00:00Z", "organizer": "", "tag": "work", "description": "on the armed virtual date", "location": "Room A", "attendees": []},
    {"event_id": "c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1c1", "title": "Later-week sync", "start_datetime": "2019-06-20T09:00:00Z", "end_datetime": "2019-06-20T10:00:00Z", "organizer": "", "tag": "work", "description": "2019 but not the T day", "location": "Room C", "attendees": []},
]}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_calendar_mcp_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("calendar_mcp", "calendar", "data.json", json.dumps(_CALENDAR_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-calendar")

        def _titles(text: str) -> set:
            return {e["title"] for e in json.loads(text)["events"]}

        before = _titles(await _clock_call_tool(result.mcp_url, "calendar_read_today_calendar_events", {}))
        assert "Virtual-date standup" not in before, before
        cfg = await _arm_gateway_clock(env, result)
        synced = {s["service"]: s for s in cfg["synced"]}
        assert "calendar" in synced, cfg
        after = _titles(await _clock_call_tool(result.mcp_url, "calendar_read_today_calendar_events", {}))
        assert after == {"Virtual-date standup"}, after
    finally:
        await env.close()


# ── slack: conversations_history "7d" — empty on wall (2026), returns the two June-2019 msgs at T ──
_SLACK_CLOCK_DATA = {
    "users": [{"id": "U_ALICE", "name": "alice", "real_name": "Alice Straddle", "display_name": "alice", "email": "alice@example.com"}],
    "channels": [{"id": "C_STRADDLE", "name": "straddle", "is_channel": True, "is_private": False, "is_general": False, "creator": "U_ALICE", "members": ["U_ALICE"]}],
    "messages": [
        {"channel": "C_STRADDLE", "ts": "1556668800.000100", "user": "U_ALICE", "text": "May 1 message (outside 7d-before-T)", "type": "message"},
        {"channel": "C_STRADDLE", "ts": "1560300000.000100", "user": "U_ALICE", "text": "June 12 message (inside 7d-before-T)", "type": "message"},
        {"channel": "C_STRADDLE", "ts": "1560556800.000100", "user": "U_ALICE", "text": "June 15 message (inside 7d-before-T)", "type": "message"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_slack_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("slack", "slack", "data.json", json.dumps(_SLACK_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-slack")
        args = {"channel_id": "C_STRADDLE", "limit": "7d"}

        def _ts(text: str) -> set:
            return {m["ts"] for m in json.loads(text)["messages"]}

        before = _ts(await _clock_call_tool(result.mcp_url, "slack_conversations_history", args))
        assert before == set(), before
        cfg = await _arm_gateway_clock(env, result)
        synced = {s["service"]: s for s in cfg["synced"]}
        assert "slack" in synced, cfg
        after = _ts(await _clock_call_tool(result.mcp_url, "slack_conversations_history", args))
        assert after == {"1560556800.000100", "1560300000.000100"}, after
    finally:
        await env.close()


# ── google_calendar: create_event stamps `created` — wall (2026) vs exactly T when armed ──
_GCAL_CLOCK_DATA = {
    "calendars": [{"calendar_id": "alice@example.com", "summary": "Alice — primary", "owner_email": "alice@example.com", "shared_with_emails": [], "visibility": "private", "time_zone": "UTC", "primary": True, "access_role": "owner"}],
    "events": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_google_calendar_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("gcal", "gcal", "data.json", json.dumps(_GCAL_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-gcal")
        args = {"calendarId": "alice@example.com", "event": {"summary": "Clock probe", "start": {"dateTime": "2019-06-15T13:00:00Z"}, "end": {"dateTime": "2019-06-15T14:00:00Z"}}}
        before = json.loads(await _clock_call_tool(result.mcp_url, "gcal_create_event", args))
        assert not before["created"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        synced = {s["service"]: s for s in cfg["synced"]}
        assert "gcal" in synced, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "gcal_create_event", args))
        # rate=1 clock ticks a beat past T between arm and this call, so match the virtual minute, not the exact second.
        assert after["created"].startswith("2019-06-15T12:00:"), after
    finally:
        await env.close()


# ── strava: LOW observability — a newly starred segment sorts ABOVE the 2019-08-01 seed on wall,
#    BELOW it when armed at T (starred_date value is never surfaced; only list order reflects the clock). ──
_STRAVA_CLOCK_DATA = {
    "athletes": [{"user_id": "strava_persona_001", "athlete_id": 9001, "firstname": "Ada", "lastname": "Persona", "is_user": True, "email": "ada@persona.test"}],
    "oauth_tokens": [{"token": "strava_tok_persona_001", "user_id": "strava_persona_001"}],
    "segments": [
        {"segment_id": 2001, "name": "Old Faithful Loop", "activity_type": "Run"},
        {"segment_id": 2002, "name": "Riverside Sprint", "activity_type": "Run"},
        {"segment_id": 2003, "name": "Hilltop Climb", "activity_type": "Run"},
    ],
    "starred_segments": [{"user_id": "strava_persona_001", "segment_id": 2001, "starred_date": "2019-08-01T00:00:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_strava_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("strava", "strava", "data.json", json.dumps(_STRAVA_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-strava")

        def _order(text: str) -> list:
            return [int(s["id"]) for s in json.loads(text)["segments"]]

        # BEFORE arming: starring stamps wall time (~2026) -> new star (2002) sorts ABOVE the 2019-08-01 seed (2001).
        await _clock_call_tool(result.mcp_url, "strava_star_segment", {"segment_id": 2002})
        before = _order(await _clock_call_tool(result.mcp_url, "strava_get_starred_segments", {}))
        assert before.index(2002) < before.index(2001), before
        cfg = await _arm_gateway_clock(env, result)
        synced = {s["service"]: s for s in cfg["synced"]}
        assert "strava" in synced, cfg
        # AFTER arming: starring stamps virtual T (2019-06-15) -> new star (2003) sorts BELOW the 2019-08-01 seed (2001).
        await _clock_call_tool(result.mcp_url, "strava_star_segment", {"segment_id": 2003})
        after = _order(await _clock_call_tool(result.mcp_url, "strava_get_starred_segments", {}))
        assert after.index(2003) > after.index(2001), after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet batch 2: google_drive, fitbit, apple_health, myfitnesspal, garmin_health.
# Same proof shape — a temporal tool's output shifts wall -> T when the clock is armed.
# ─────────────────────────────────────────────────────────────────────────────

# ── google_drive: create_folder mints createdTime — wall (~2026) vs T when armed. Needs a zip universe. ──
_GDRIVE_CLOCK_DATA = {
    "drive_users": [{"id": "usr_alice", "email": "alice@example.com", "display_name": "Alice Owner"}],
    "drive_files": [
        {"id": "d_reports", "name": "Reports", "mime_type": "application/vnd.google-apps.folder", "parents": [], "owner_email": "alice@example.com", "owners": ["alice@example.com"], "visibility": "private", "created_time": "2019-05-01T00:00:00Z", "modified_time": "2019-05-20T00:00:00Z"},
        {"id": "f_q2notes", "name": "Q2 Notes", "mime_type": "application/vnd.google-apps.document", "parents": ["d_reports"], "owner_email": "alice@example.com", "owners": ["alice@example.com"], "visibility": "private", "created_time": "2019-06-01T09:00:00Z", "modified_time": "2019-06-10T09:00:00Z"},
        {"id": "f_roadmap", "name": "Roadmap", "mime_type": "application/vnd.google-apps.document", "parents": ["d_reports"], "owner_email": "alice@example.com", "owners": ["alice@example.com"], "visibility": "private", "created_time": "2019-07-01T09:00:00Z", "modified_time": "2019-08-01T09:00:00Z"},
    ],
    "drive_sheets": [],
}


def _gdrive_universe_zip() -> bytes:
    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.json", json.dumps(_GDRIVE_CLOCK_DATA))
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_google_drive_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("gdrive", "gdrive", "gdrive.zip", _gdrive_universe_zip())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-gdrive")
        args = {"name": "clock-probe", "parentId": "root"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "gdrive_create_folder", args))
        assert not before["createdTime"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "gdrive" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "gdrive_create_folder", args))
        assert after["createdTime"].startswith("2019-06-15T12:00:"), after
    finally:
        await env.close()


# ── fitbit: get_steps (default date = today) — empty on wall (2026), returns the 2019-06-15 row at T ──
_FITBIT_CLOCK_DATA = {
    "user_profiles": [{"user_id": "persona_001", "encoded_id": "FB001", "display_name": "Jordan", "full_name": "Jordan Rivera", "email": "jordan.rivera@example.com", "is_user": True, "timezone": "UTC", "member_since": "2018-01-10"}],
    "oauth_tokens": [{"token": "fitbit-token-persona-001", "user_id": "persona_001"}],
    "daily_stats": [
        {"user_id": "persona_001", "date": "2019-06-14", "steps": 7011, "distance": 5.1, "distance_unit": "km", "calories": 2320, "floors": 8},
        {"user_id": "persona_001", "date": "2019-06-15", "steps": 8432, "distance": 6.2, "distance_unit": "km", "calories": 2475, "floors": 11},
        {"user_id": "persona_001", "date": "2019-06-16", "steps": 9120, "distance": 6.8, "distance_unit": "km", "calories": 2510, "floors": 13},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fitbit_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("fitbit", "fitbit", "fitbit.json", json.dumps(_FITBIT_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-fitbit")
        before = json.loads(await _clock_call_tool(result.mcp_url, "fitbit_get_steps", {}))
        assert before["total_steps"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        assert "fitbit" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "fitbit_get_steps", {}))
        assert after["total_steps"] == 8432, after
        assert {s["date"] for s in after["steps"]} == {"2019-06-15"}, after
    finally:
        await env.close()


# ── apple_health: get_steps(days=7) — empty on wall (2026), returns the trailing-7d 2019 set at T ──
_APPLE_HEALTH_CLOCK_DATA = {
    "user_profiles": [{"user_id": "persona_001", "is_user": True, "height_cm": 178.0, "weight_kg": 78.3, "date_of_birth": "1992-06-15", "sex": "M", "email": "john.doe@email.com"}],
    "oauth_tokens": [{"token": "apple_health_tok_persona_001", "user_id": "persona_001"}],
    "step_records": [
        {"record_id": 1, "user_id": "persona_001", "date": "2019-06-08", "total_steps": 9100},
        {"record_id": 2, "user_id": "persona_001", "date": "2019-06-10", "total_steps": 10250},
        {"record_id": 3, "user_id": "persona_001", "date": "2019-06-14", "total_steps": 8760},
        {"record_id": 4, "user_id": "persona_001", "date": "2019-06-15", "total_steps": 12030},
        {"record_id": 5, "user_id": "persona_001", "date": "2019-06-20", "total_steps": 7400},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_apple_health_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("apple_health", "apple_health", "apple_health.json", json.dumps(_APPLE_HEALTH_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-apple_health")
        before = json.loads(await _clock_call_tool(result.mcp_url, "apple_health_get_steps", {"days": 7}))
        assert before["count"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        assert "apple_health" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "apple_health_get_steps", {"days": 7}))
        assert {r["date"] for r in after["records"]} == {"2019-06-08", "2019-06-10", "2019-06-14", "2019-06-15"}, after
    finally:
        await env.close()


# ── myfitnesspal: get_daily_summary (default date = today) — empty on wall (2026), the 2019-06-15 day at T ──
_MYFITNESSPAL_CLOCK_DATA = {
    "user_profiles": [{"persona_id": "persona_001", "email": "john.doe@email.com", "password": "test-password", "first_name": "John", "last_name": "Doe", "is_user": True, "height_cm": 180, "weight_kg": 80, "age": 30, "gender": "male", "activity_level": "moderate", "goal": "maintain", "calorie_goal": 2200, "protein_goal_g": 140, "carbs_goal_g": 250, "fat_goal_g": 70, "water_goal_ml": 2500, "created_at": "2019-06-01T08:00:00Z"}],
    "foods": [
        {"food_id": "FOOD001", "name": "Scrambled Eggs", "brand": None, "serving_size": "2", "serving_unit": "eggs", "calories": 180, "protein_g": 12, "carbs_g": 2, "fat_g": 14, "fiber_g": 0, "sugar_g": 1, "sodium_mg": 160, "verified": True},
        {"food_id": "FOOD002", "name": "Oatmeal", "brand": "Quaker", "serving_size": "1", "serving_unit": "cup", "calories": 150, "protein_g": 5, "carbs_g": 27, "fat_g": 3, "fiber_g": 4, "sugar_g": 1, "sodium_mg": 0, "verified": True},
    ],
    "food_logs": [
        {"persona_id": "persona_001", "log_date": "2019-06-14", "meal_type": "Dinner", "food_id": "FOOD001", "servings": 1, "logged_at": "2019-06-14T19:00:00Z"},
        {"persona_id": "persona_001", "log_date": "2019-06-15", "meal_type": "Breakfast", "food_id": "FOOD001", "servings": 2, "logged_at": "2019-06-15T08:00:00Z"},
        {"persona_id": "persona_001", "log_date": "2019-06-15", "meal_type": "Breakfast", "food_id": "FOOD002", "servings": 1, "logged_at": "2019-06-15T08:05:00Z"},
        {"persona_id": "persona_001", "log_date": "2019-06-16", "meal_type": "Lunch", "food_id": "FOOD002", "servings": 1, "logged_at": "2019-06-16T12:00:00Z"},
    ],
    "exercises": [{"exercise_id": "EX001", "name": "Running (6 mph)", "category": "cardio", "calories_per_minute": 10, "met_value": 9.8, "description": "Jog"}],
    "exercise_logs": [{"persona_id": "persona_001", "log_date": "2019-06-15", "exercise_id": "EX001", "duration_minutes": 30, "calories_burned": 300, "logged_at": "2019-06-15T18:00:00Z"}],
    "water_logs": [
        {"persona_id": "persona_001", "log_date": "2019-06-15", "amount_ml": 500, "logged_at": "2019-06-15T09:00:00Z"},
        {"persona_id": "persona_001", "log_date": "2019-06-15", "amount_ml": 750, "logged_at": "2019-06-15T13:00:00Z"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_myfitnesspal_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("myfitnesspal", "myfitnesspal", "myfitnesspal.json", json.dumps(_MYFITNESSPAL_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-myfitnesspal")
        before = json.loads(await _clock_call_tool(result.mcp_url, "myfitnesspal_get_daily_summary", {}))
        assert before["summary"]["calories"]["food"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        assert "myfitnesspal" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "myfitnesspal_get_daily_summary", {}))
        assert after["summary"]["date"] == "2019-06-15", after
        assert after["summary"]["calories"]["food"] == 510, after
        assert after["summary"]["entries"]["food"] == 2, after
    finally:
        await env.close()


# ── garmin_health: get_daily_stats (default date = today) — null on wall (2026), the 2019-06-15 row at T ──
_GARMIN_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "Jane Runner", "email": "jane.runner@example.com", "phone": "+1-415-555-0142", "address": "742 Marina Blvd, San Francisco, CA 94123", "is_user": True}],
    "daily_stats": [
        {"user_id": "persona_001", "date": "2019-06-14", "steps": 9800, "distance_meters": 7800, "calories_total": 2300, "calories_active": 600, "floors_climbed": 8, "intensity_minutes": 45, "moderate_intensity_minutes": 30, "vigorous_intensity_minutes": 15},
        {"user_id": "persona_001", "date": "2019-06-15", "steps": 11500, "distance_meters": 9200, "calories_total": 2510, "calories_active": 690, "floors_climbed": 12, "intensity_minutes": 60, "moderate_intensity_minutes": 40, "vigorous_intensity_minutes": 20},
        {"user_id": "persona_001", "date": "2019-06-16", "steps": 10200, "distance_meters": 8100, "calories_total": 2380, "calories_active": 640, "floors_climbed": 9, "intensity_minutes": 50, "moderate_intensity_minutes": 33, "vigorous_intensity_minutes": 17},
    ],
    "activities": [{"user_id": "persona_001", "activity_id": "ACT-U001-001", "activity_type": "running", "activity_name": "Morning Run", "start_time": "2019-06-15T09:00:00", "duration_seconds": 3010, "distance_meters": 8575, "calories": 714, "avg_hr": 148, "max_hr": 174}],
    "oauth_tokens": [{"token": "garmin_health_tok_persona_001", "user_id": "persona_001"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_garmin_health_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("garmin_health", "garmin_health", "garmin_health.json", json.dumps(_GARMIN_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-garmin_health")
        before = json.loads(await _clock_call_tool(result.mcp_url, "garmin_health_get_daily_stats", {}))
        assert before["stats"] is None, before
        cfg = await _arm_gateway_clock(env, result)
        assert "garmin_health" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "garmin_health_get_daily_stats", {}))
        assert after["date"] == "2019-06-15", after
        assert after["stats"]["steps"] == 11500, after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet batch 3: google_docs, google_sheets, google_slides, renpho, amazon_fresh.
# ─────────────────────────────────────────────────────────────────────────────

async def _gsuite_created_stamp(mcp_url: str, create_tool: str, create_args: dict, search_tool: str, list_key: str) -> str:
    """Create a doc/sheet/deck (which mints created_time at virtual-now), then read the stamp back via search."""
    await _clock_call_tool(mcp_url, create_tool, create_args)
    text = await _clock_call_tool(mcp_url, search_tool, {"query": create_args["title"]})
    return json.loads(text)[list_key][0]["createdTime"]


# ── google_docs: create_document mints createdTime — wall (~2026) vs T when armed (read back via search). ──
_GDOCS_CLOCK_DATA = {"docs_documents": [
    {"document_id": "doc-pre-t-001", "title": "Planning Notes (pre-T)", "owner_email": "alice@example.com", "shared_with_emails": [], "visibility": "private", "body_text": "Q1 planning notes.\n", "style_runs": [], "revision_id": "r1", "drive_file_id": "f_doc_pre_t_001", "created_time": "2019-01-10T09:00:00Z", "modified_time": "2019-03-01T09:00:00Z"},
    {"document_id": "doc-post-t-002", "title": "Roadmap (post-T)", "owner_email": "alice@example.com", "shared_with_emails": [], "visibility": "private", "body_text": "H2 roadmap.\n", "style_runs": [], "revision_id": "r1", "drive_file_id": "f_doc_post_t_002", "created_time": "2019-12-20T15:00:00Z", "modified_time": "2019-12-20T15:30:00Z"},
]}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_google_docs_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("gdocs", "gdocs", "gdocs.json", json.dumps(_GDOCS_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-gdocs")
        before = await _gsuite_created_stamp(result.mcp_url, "gdocs_create_document", {"title": "gdocs-clock-before", "bodyText": "probe"}, "gdocs_search_documents", "documents")
        assert not before.startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "gdocs" in {s["service"] for s in cfg["synced"]}, cfg
        after = await _gsuite_created_stamp(result.mcp_url, "gdocs_create_document", {"title": "gdocs-clock-after", "bodyText": "probe"}, "gdocs_search_documents", "documents")
        assert after.startswith("2019-06-15T12:00:"), after
    finally:
        await env.close()


# ── google_sheets: create_spreadsheet mints createdTime — wall vs T (read back via search). ──
_GSHEETS_CLOCK_DATA = {"sheets_spreadsheets": [
    {"spreadsheet_id": "ss-seed-001", "title": "Seed Budget", "owner_email": "alice@example.com", "shared_with_emails": [], "visibility": "private", "drive_file_id": "f_seed_001", "created_time": "2019-06-14T09:00:00Z", "modified_time": "2019-06-16T09:00:00Z", "sheets": [{"sheet_id": 0, "title": "Sheet1", "index": 0, "row_count": 1000, "column_count": 26, "grid": [["Name", "Value"], ["Alpha", 1]]}]},
]}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_google_sheets_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("gsheets", "gsheets", "gsheets.json", json.dumps(_GSHEETS_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-gsheets")
        before = await _gsuite_created_stamp(result.mcp_url, "gsheets_create_spreadsheet", {"title": "gsheets-clock-before"}, "gsheets_search_spreadsheets", "spreadsheets")
        assert not before.startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "gsheets" in {s["service"] for s in cfg["synced"]}, cfg
        after = await _gsuite_created_stamp(result.mcp_url, "gsheets_create_spreadsheet", {"title": "gsheets-clock-after"}, "gsheets_search_spreadsheets", "spreadsheets")
        assert after.startswith("2019-06-15T12:00:"), after
    finally:
        await env.close()


# ── google_slides: create_presentation mints createdTime — wall vs T (read back via search). ──
_GSLIDES_CLOCK_DATA = {"slides_presentations": [
    {"presentation_id": "pres_seed_straddle", "title": "Seed Deck", "owner_email": "alice@example.com", "shared_with_emails": [], "visibility": "private", "drive_file_id": "f_seed", "revision_id": "r1", "created_time": "2019-05-01T00:00:00Z", "modified_time": "2019-05-10T00:00:00Z", "pages": [{"object_id": "slide_1", "index": 0, "layout": "TITLE", "elements": [{"object_id": "el_1", "kind": "SHAPE", "placeholder_type": "TITLE", "text": "Seed"}]}]},
]}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_google_slides_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("gslides", "gslides", "gslides.json", json.dumps(_GSLIDES_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-gslides")
        before = await _gsuite_created_stamp(result.mcp_url, "gslides_create_presentation", {"title": "gslides-clock-before"}, "gslides_search_presentations", "presentations")
        assert not before.startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "gslides" in {s["service"] for s in cfg["synced"]}, cfg
        after = await _gsuite_created_stamp(result.mcp_url, "gslides_create_presentation", {"title": "gslides-clock-after"}, "gslides_search_presentations", "presentations")
        assert after.startswith("2019-06-15T12:00:"), after
    finally:
        await env.close()


# ── renpho: get_weight_trend(days=7) — empty on wall (2026), the 2019 trailing-week trend at T. ──
_RENPHO_CLOCK_DATA = {
    "user_profiles": [{"persona_id": "persona_001", "email": "john.doe@email.com", "is_user": True, "nickname": "John", "birthday": "1990-05-15", "height": 175.0, "gender": "male", "goal_weight": 75.0, "unit_system": "metric", "created_at": "2019-05-01T08:00:00Z", "password": "test-password"}],
    "measurements": [
        {"persona_id": "persona_001", "timestamp": "2019-05-20T08:00:00Z", "weight": 82.0, "bmi": 26.8, "body_fat": 22.0, "muscle_mass": 60.0, "visceral_fat": 9, "metabolic_age": 40, "bmr": 1700, "created_at": "2019-05-20T08:00:00Z"},
        {"persona_id": "persona_001", "timestamp": "2019-06-10T08:00:00Z", "weight": 80.0, "bmi": 26.1, "body_fat": 21.0, "muscle_mass": 60.5, "visceral_fat": 9, "metabolic_age": 40, "bmr": 1710, "created_at": "2019-06-10T08:00:00Z"},
        {"persona_id": "persona_001", "timestamp": "2019-06-14T08:00:00Z", "weight": 78.5, "bmi": 25.6, "body_fat": 20.5, "muscle_mass": 61.0, "visceral_fat": 8, "metabolic_age": 39, "bmr": 1720, "created_at": "2019-06-14T08:00:00Z"},
    ],
    "scales": [{"persona_id": "persona_001", "device_id": "SCALE001", "device_name": "Bathroom Scale", "model": "Renpho Elis 1", "mac_address": "AA:BB:CC:DD:EE:FF", "last_sync": "2019-06-14T08:00:00Z", "status": "active"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_renpho_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("renpho", "renpho", "renpho.json", json.dumps(_RENPHO_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-renpho")
        before = json.loads(await _clock_call_tool(result.mcp_url, "renpho_get_weight_trend", {"days": 7}))["weight_trend"]
        assert before["measurements_count"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        assert "renpho" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "renpho_get_weight_trend", {"days": 7}))["weight_trend"]
        assert after["measurements_count"] == 2, after
        assert after["trend"] == "losing", after
        assert after["start_weight"] == 80.0 and after["end_weight"] == 78.5, after
    finally:
        await env.close()


# ── amazon_fresh: get_purchase_history — auto-ship orders accrue up to now; wall (~2026) has none in 2019,
#    armed at T only the 5 cycles elapsed by 2019-06-15 exist. ──
_AMAZON_FRESH_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "John Doe", "email": "john.doe@email.com", "phone": "+1-415-555-0123", "address": "123 Main Street, San Francisco, CA 94102", "is_user": True, "password": "test-password"}],
    "products": [{"id": "af_milk", "name": "Organic Whole Milk", "description": "1 gallon organic whole milk", "price": 5.49, "currency": "USD", "category": "Dairy", "unit": "gallon", "inventory_count": 50, "created_at": "2018-01-01T00:00:00Z"}],
    "product_variants": [{"id": "afv_milk_1gal", "product_id": "af_milk", "name": "1 Gallon", "price": 5.49, "sku": "MILK-1GAL", "inventory_count": 50, "attributes": "{}"}],
    "policies": [], "faqs": [], "carts": [], "cart_lines": [], "orders": [], "order_items": [], "payment_methods": [],
    "subscriptions": [
        {"id": "sub_milk_past", "user_id": "persona_001", "product_id": "af_milk", "quantity": 1, "frequency": "monthly", "status": "active", "created_at": "2019-01-01T00:00:00Z"},
        {"id": "sub_milk_future", "user_id": "persona_001", "product_id": "af_milk", "quantity": 1, "frequency": "monthly", "status": "active", "created_at": "2019-11-01T00:00:00Z"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_amazon_fresh_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("amazon_fresh", "amazon_fresh", "amazon_fresh.json", json.dumps(_AMAZON_FRESH_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-amazon_fresh")
        before = json.loads(await _clock_call_tool(result.mcp_url, "amazon_fresh_get_purchase_history", {"limit": 100}))
        assert "2019" not in {o["created_at"][:4] for o in before["orders"]}, before
        cfg = await _arm_gateway_clock(env, result)
        assert "amazon_fresh" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "amazon_fresh_get_purchase_history", {"limit": 100}))
        assert len(after["orders"]) == 5, after
        assert {o["created_at"][:4] for o in after["orders"]} == {"2019"}, after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet batch 4: walmart, target, hotel_management, stayncook, fintrack.
# ─────────────────────────────────────────────────────────────────────────────

# ── walmart: checkout mints an order_id date prefix (WMT-YYYYMMDD-...) at virtual-now. ──
_WALMART_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "John Doe", "email": "john.doe@email.com", "is_user": True, "walmart_plus_member": True, "shipping_address": "123 Main Street, San Francisco, CA 94102", "password": "test-password"}],
    "products": [{"product_id": "WMT00000001", "name": "Bananas", "price": 0.55, "currency": "USD", "category": "Grocery", "department": "Grocery & Essentials", "brand": "Fresh Produce", "in_stock": True, "stock_quantity": 100}],
    "cart_items": [{"user_id": "persona_001", "product_id": "WMT00000001", "quantity": 3, "added_at": "2019-06-10T09:00:00Z"}],
    "orders": [{"order_id": "WMT-20190601-AAAAAAAA", "user_id": "persona_001", "status": "delivered", "total": 12.50, "currency": "USD", "fulfillment_type": "delivery", "created_at": "2019-06-01T10:00:00Z", "shipping_address": "123 Main Street, San Francisco, CA 94102"}],
    "order_items": [{"order_id": "WMT-20190601-AAAAAAAA", "product_id": "WMT00000001", "quantity": 2, "price_at_purchase": 0.55}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_walmart_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("walmart", "walmart", "walmart.json", json.dumps(_WALMART_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-walmart")
        before = json.loads(await _clock_call_tool(result.mcp_url, "walmart_checkout", {"fulfillment_type": "delivery"}))
        assert not before["order_id"].startswith("WMT-2019"), before
        await _clock_call_tool(result.mcp_url, "walmart_add_to_cart", {"product_id": "WMT00000001", "quantity": 1})  # refill (checkout emptied the cart)
        cfg = await _arm_gateway_clock(env, result)
        assert "walmart" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "walmart_checkout", {"fulfillment_type": "delivery"}))
        assert after["order_id"].startswith("WMT-20190615-"), after
    finally:
        await env.close()


# ── target: checkout mints an order_id date prefix (TGT-YYYYMMDD-...) at virtual-now. ──
_TARGET_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "Alex Thompson", "email": "alex@example.com", "password": "hunter2", "is_user": True, "circle_member": True, "shipping_address": "1 Main St, Chicago, IL 60601", "preferred_store": {"store_id": "T-1771", "name": "Chicago State St", "address": "1 State St, Chicago, IL"}}],
    "products": [{"product_id": "TGT00000001", "name": "Wireless Headphones", "price": 29.99, "category": "Electronics", "department": "Electronics", "brand": "Acme", "in_stock": True, "stock_quantity": 12}],
    "cart_items": [{"cart_item_id": "CART001", "user_id": "persona_001", "product_id": "TGT00000001", "quantity": 1, "added_at": "2019-06-14T09:00:00Z"}],
    "orders": [{"order_id": "TGT-20190610-AAAA0001", "user_id": "persona_001", "status": "delivered", "total": 15.0, "fulfillment_type": "delivery", "created_at": "2019-06-10T08:00:00Z"}],
    "order_items": [{"order_item_id": "toi-0001", "order_id": "TGT-20190610-AAAA0001", "product_id": "TGT00000001", "quantity": 1, "price_at_purchase": 15.0}],
    "payment_methods": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_target_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("target", "target", "target.json", json.dumps(_TARGET_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-target")
        before = json.loads(await _clock_call_tool(result.mcp_url, "target_checkout", {"fulfillment_type": "delivery"}))
        assert not before["order_id"].startswith("TGT-2019"), before
        await _clock_call_tool(result.mcp_url, "target_add_to_cart", {"product_id": "TGT00000001", "quantity": 1})  # refill (checkout emptied the cart)
        cfg = await _arm_gateway_clock(env, result)
        assert "target" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "target_checkout", {"fulfillment_type": "delivery"}))
        assert after["order_id"].startswith("TGT-20190615-"), after
    finally:
        await env.close()


# ── hotel_management: get_todays_checkins filters check_in_date == today — 0 on wall, the 2019-06-15 booking at T. ──
_HOTEL_CLOCK_DATA = {
    "rooms": [{"room_number": "101", "room_type": "Standard King", "floor": 1, "beds": "1 King", "view": "Garden", "base_rate": 189.0}, {"room_number": "205", "room_type": "Ocean View Double", "floor": 2, "beds": "2 Queen", "view": "Ocean", "base_rate": 349.0}],
    "guests": [{"guest_id": "G001", "first_name": "John", "last_name": "Smith", "email": "john.smith@example.com", "phone": "555-0101", "address": "123 Main St", "loyalty_tier": "Gold", "total_stays": 5, "lifetime_spend": 4250.0, "preferences": "High floor"}, {"guest_id": "G002", "first_name": "Jane", "last_name": "Doe", "email": "jane.doe@example.com", "phone": "555-0102", "address": "456 Oak Ave", "loyalty_tier": "Silver", "total_stays": 2, "lifetime_spend": 1200.0, "preferences": "Quiet room"}],
    "reservations": [
        {"confirmation_code": "APH-TODAY", "guest_name": "John Smith", "guest_id": "G001", "room_number": "101", "room_type": "Standard King", "check_in_date": "2019-06-15", "check_out_date": "2019-06-18", "status": "Confirmed", "rate_per_night": 189.0, "booking_source": "Direct", "special_requests": "Late check-in", "total_amount": 567.0},
        {"confirmation_code": "APH-DEPART", "guest_name": "Jane Doe", "guest_id": "G002", "room_number": "205", "room_type": "Ocean View Double", "check_in_date": "2019-06-12", "check_out_date": "2019-06-15", "status": "Checked-In", "rate_per_night": 349.0, "booking_source": "Booking.com", "special_requests": None, "total_amount": 1047.0},
        {"confirmation_code": "APH-FUTURE", "guest_name": "John Smith", "guest_id": "G001", "room_number": "101", "room_type": "Standard King", "check_in_date": "2019-06-20", "check_out_date": "2019-06-22", "status": "Confirmed", "rate_per_night": 189.0, "booking_source": "Direct", "special_requests": None, "total_amount": 378.0},
    ],
    "emails": [{"email_id": "E001", "from_name": "John Smith", "from_email": "john.smith@example.com", "date": "2019-06-10", "time": "14:30", "subject": "Reservation Confirmation APH-TODAY", "body": "Please arrange a late check-in for June 15.", "confirmation_code": "APH-TODAY"}],
    "occupancy_history": [{"date": "2019-06-14", "occupancy_percent": 78.5, "rooms_sold": 157, "adr": 275.0, "revpar": 215.88, "revenue": 43175.0}, {"date": "2019-06-15", "occupancy_percent": 82.0, "rooms_sold": 164, "adr": 285.0, "revpar": 233.7, "revenue": 46740.0}],
    "ota_pricing": [{"room_type": "Standard King", "date": "2019-06-15", "direct_price": 189.0, "booking_com_price": 199.0, "booking_com_commission": 15.0, "expedia_price": 195.0, "expedia_commission": 18.0, "hotels_com_price": 197.0, "hotels_com_commission": 16.0}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_hotel_management_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("hotel_management", "hotel", "hotel.json", json.dumps(_HOTEL_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-hotel")
        before = json.loads(await _clock_call_tool(result.mcp_url, "hotel_get_todays_checkins", {}))
        assert before["total_checkins"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        assert "hotel" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "hotel_get_todays_checkins", {}))
        assert after["total_checkins"] == 1, after
        assert "APH-TODAY" in {c["confirmation_code"] for c in after["checkins"]}, after
    finally:
        await env.close()


# ── stayncook: listing_details returns a 60-day availability window from today — 2026 window on wall, 2019 at T. ──
_STAYNCOOK_CLOCK_DATA = {"listings": [
    {"listing_id": "stayncooktest01", "title": "Sunny Loft Near the Bay", "city": "San Francisco", "country_code": "US", "state_or_region": "California", "price_per_night": 180.0, "max_guests": 4, "bedrooms": 2, "beds": 2, "bathrooms": 1.5, "property_type": "Loft", "amenities": ["Wifi", "Kitchen", "Air conditioning"], "description": "A bright loft a short walk from the waterfront.", "host_name": "Dana Kim", "rating": 4.85, "num_reviews": 120, "images": ["https://example.com/img1.jpg"], "allows_pets": False, "instant_book": True},
]}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_stayncook_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("stayncook", "stayncook", "stayncook.json", json.dumps(_STAYNCOOK_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-stayncook")
        before = json.loads(await _clock_call_tool(result.mcp_url, "stayncook_listing_details", {"id": "stayncooktest01"}))
        assert "2019" not in {d[:4] for d in before["available_dates"]}, before
        cfg = await _arm_gateway_clock(env, result)
        assert "stayncook" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "stayncook_listing_details", {"id": "stayncooktest01"}))
        assert {d[:4] for d in after["available_dates"]} == {"2019"}, after
    finally:
        await env.close()


# ── fintrack: get_upcoming_billings filters next_billing_date in [today, today+N] — 0 on wall, 2 in the 2019 window at T. ──
_FINTRACK_CLOCK_DATA = {
    "users": [{"user_id": "U001", "name": "Alice Johnson", "email": "alice.johnson@example.com", "status": "Active"}],
    "accounts": [], "transactions": [],
    "subscriptions": [
        {"subscription_id": "S001", "user_id": "U001", "service_name": "CloudStream Video", "amount": 15.99, "billing_frequency": "monthly", "next_billing_date": "2019-07-01", "status": "Active"},
        {"subscription_id": "S002", "user_id": "U001", "service_name": "GymPass", "amount": 45.00, "billing_frequency": "monthly", "next_billing_date": "2019-06-20", "status": "Active"},
        {"subscription_id": "S003", "user_id": "U001", "service_name": "News Daily", "amount": 9.99, "billing_frequency": "monthly", "next_billing_date": "2019-06-10", "status": "Active"},
        {"subscription_id": "S004", "user_id": "U001", "service_name": "Annual Cloud Backup", "amount": 99.00, "billing_frequency": "yearly", "next_billing_date": "2019-08-20", "status": "Active"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fintrack_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("fintrack", "fintrack", "fintrack.json", json.dumps(_FINTRACK_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-fintrack")
        before = json.loads(await _clock_call_tool(result.mcp_url, "fintrack_get_upcoming_billings", {"within_days": 30}))
        assert before["upcoming_count"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        assert "fintrack" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "fintrack_get_upcoming_billings", {"within_days": 30}))
        assert after["upcoming_count"] == 2, after
        assert {s["service_name"] for s in after["subscriptions"]} == {"CloudStream Video", "GymPass"}, after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet batch 5: buildium, oracle_gl, sap_subledger, obsidian, airtable.
# ─────────────────────────────────────────────────────────────────────────────

# ── buildium: get_lease_renewals filters lease_end <= now+90d — both leases on wall, only the near one at T. ──
_BUILDIUM_CLOCK_DATA = {
    "properties": [{"id": 1, "name": "Straddle Court", "address": "1 T St", "city": "Austin", "state": "TX", "zip": "78701", "year_built": 2000, "total_units": 2, "parking_spots": 2, "notes": None}],
    "units": [{"id": 1, "property_id": 1, "unit_number": "A1", "bedrooms": 1, "bathrooms": 1.0, "sqft": 500.0, "rent_amount": 1500.0, "status": "occupied", "notes": None}, {"id": 2, "property_id": 1, "unit_number": "A2", "bedrooms": 2, "bathrooms": 1.0, "sqft": 700.0, "rent_amount": 2000.0, "status": "occupied", "notes": None}],
    "tenants": [{"id": 1, "first_name": "Ada", "last_name": "Near", "email": "ada@ex.com", "phone": "+1-512-000-0001", "unit_id": 1, "lease_start": "2018-07-15", "lease_end": "2019-07-15", "rent_amount": 1500.0, "security_deposit": 1500.0, "status": "active"}, {"id": 2, "first_name": "Bo", "last_name": "Far", "email": "bo@ex.com", "phone": "+1-512-000-0002", "unit_id": 2, "lease_start": "2019-01-15", "lease_end": "2020-01-15", "rent_amount": 2000.0, "security_deposit": 2000.0, "status": "active"}],
    "leases": [{"id": 1, "tenant_id": 1, "unit_id": 1, "lease_start": "2018-07-15", "lease_end": "2019-07-15", "rent_amount": 1500.0, "security_deposit": 1500.0, "lease_status": "current", "notes": None}, {"id": 2, "tenant_id": 2, "unit_id": 2, "lease_start": "2019-01-15", "lease_end": "2020-01-15", "rent_amount": 2000.0, "security_deposit": 2000.0, "lease_status": "current", "notes": None}],
    "work_orders": [], "vendors": [], "applicants": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_buildium_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("buildium", "buildium", "buildium.json", json.dumps(_BUILDIUM_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-buildium")
        before = json.loads(await _clock_call_tool(result.mcp_url, "buildium_get_lease_renewals", {"within_days": 90}))
        assert before["expiring_count"] == 2, before
        cfg = await _arm_gateway_clock(env, result)
        assert "buildium" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "buildium_get_lease_renewals", {"within_days": 90}))
        assert after["expiring_count"] == 1, after
        assert {l["id"] for l in after["leases"]} == {1}, after
    finally:
        await env.close()


# ── oracle_gl: create_journal_entry stamps created_at at virtual-now (idempotent per call). ──
_ORACLE_GL_CLOCK_DATA = {
    "fiscal_periods": [{"id": "FP-2019-05", "period_label": "2019-05 (May 2019)", "fiscal_year": 2019, "fiscal_quarter": 2, "start": "2019-05-01", "end": "2019-05-31", "status": "closed", "bd3_lock_at": "2019-06-05T22:00:00Z", "bd5_close_at": "2019-06-07T22:00:00Z", "regulatory_deadlines": {}}, {"id": "FP-2019-06", "period_label": "2019-06 (June 2019)", "fiscal_year": 2019, "fiscal_quarter": 2, "start": "2019-06-01", "end": "2019-06-30", "status": "open", "bd3_lock_at": "2019-07-03T22:00:00Z", "bd5_close_at": "2019-07-08T22:00:00Z", "regulatory_deadlines": {}}],
    "accounts": [{"number": "11000", "name": "Cash - Operating", "type": "asset", "normal_balance": "debit", "status": "active", "entity": "Bank N.A.", "cost_center": "CC-100", "restricted": False, "current_balance": 0.0}, {"number": "40000", "name": "Service Revenue", "type": "revenue", "normal_balance": "credit", "status": "active", "entity": "Bank N.A.", "cost_center": "CC-100", "restricted": False, "current_balance": 0.0}],
    "journal_entries": [], "transactions": [], "subledger_feeds": [], "subledger_feed_runs": [],
}
_ORACLE_GL_JE_ARGS = {"period_id": "FP-2019-06", "posting_date": "2019-06-15", "description": "Clock probe entry", "lines": [{"account_number": "11000", "debit": 100.0, "credit": 0.0}, {"account_number": "40000", "debit": 0.0, "credit": 100.0}]}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_oracle_gl_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("oracle_gl", "oracle_gl", "oracle_gl.json", json.dumps(_ORACLE_GL_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-oracle_gl")
        before = json.loads(await _clock_call_tool(result.mcp_url, "oracle_gl_create_journal_entry", _ORACLE_GL_JE_ARGS))
        assert not before["journal_entry"]["created_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "oracle_gl" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "oracle_gl_create_journal_entry", _ORACLE_GL_JE_ARGS))
        assert after["journal_entry"]["created_at"].startswith("2019-06-15T12:00"), after
    finally:
        await env.close()


# ── sap_subledger: pay_ap_invoice defaults payment_date to now (one-way transition -> pay a different invoice per phase). ──
_SAP_SUBLEDGER_CLOCK_DATA = {
    "ap_invoices": [
        {"id": "AP-INV-1001", "sap_document_number": "5100001234", "vendor_id": "V-ACME", "vendor_name": "Acme Supply Co", "invoice_number": "INV-2019-0442", "invoice_date": "2019-05-28", "posting_date": "2019-06-01", "gl_account_number": "21000", "amount": 8500.00, "currency": "USD", "status": "approved", "approver": "admin.controller", "due_date": "2019-06-30", "document_id": "DOC-INV-1001"},
        {"id": "AP-INV-1002", "sap_document_number": "5100001299", "vendor_id": "V-GLOBEX", "vendor_name": "Globex Corp", "invoice_number": "INV-2019-0777", "invoice_date": "2019-06-20", "posting_date": "2019-06-20", "gl_account_number": "21000", "amount": 4200.00, "currency": "USD", "status": "approved", "approver": "admin.controller", "due_date": "2019-07-15", "document_id": "DOC-INV-1002"},
    ],
    "fixed_assets": [], "prepaid_amortization_schedules": [], "subledger_transactions": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sap_subledger_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("sap_subledger", "sap_subledger", "sap_subledger.json", json.dumps(_SAP_SUBLEDGER_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-sap_subledger")
        before = json.loads(await _clock_call_tool(result.mcp_url, "sap_subledger_pay_ap_invoice", {"invoice_id": "AP-INV-1001", "actor_role_tags": ["admin.controller"]}))
        assert not before["payment_date"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "sap_subledger" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "sap_subledger_pay_ap_invoice", {"invoice_id": "AP-INV-1002", "actor_role_tags": ["admin.controller"]}))
        assert after["payment_date"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── obsidian: recent(days=7) filters modified_at >= now-7d — empty on wall, the two June-2019 notes at T. ──
_OBSIDIAN_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "username": "johndoe", "vault_name": "My Vault", "is_user": True, "email": "john.doe@email.com", "password": "test-password", "created_at": "2019-01-01T00:00:00Z"}],
    "notes": [
        {"note_id": 1, "user_id": "persona_001", "path": "Recent Note", "title": "Recent Note", "content": "# Recent Note\nUpdated recently.", "folder": "", "created_at": "2019-06-14T09:00:00Z", "modified_at": "2019-06-14T09:00:00Z", "size_bytes": 33},
        {"note_id": 2, "user_id": "persona_001", "path": "Also Recent", "title": "Also Recent", "content": "# Also Recent\nAlso updated.", "folder": "", "created_at": "2019-06-10T12:00:00Z", "modified_at": "2019-06-10T12:00:00Z", "size_bytes": 32},
        {"note_id": 3, "user_id": "persona_001", "path": "Old Note", "title": "Old Note", "content": "# Old Note\nStale.", "folder": "", "created_at": "2019-01-15T12:00:00Z", "modified_at": "2019-01-15T12:00:00Z", "size_bytes": 20},
    ],
    "tags": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_obsidian_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("obsidian", "obsidian", "obsidian.json", json.dumps(_OBSIDIAN_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-obsidian")
        before = json.loads(await _clock_call_tool(result.mcp_url, "obsidian_recent", {"days": 7}))
        assert before["count"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        assert "obsidian" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "obsidian_recent", {"days": 7}))
        assert after["count"] == 2, after
        assert {n["path"] for n in after["notes"]} == {"Recent Note", "Also Recent"}, after
    finally:
        await env.close()


# ── airtable: create_record stamps created_time at virtual-now (idempotent per call). ──
_AIRTABLE_CLOCK_DATA = {
    "bases": [{"id": "appClockProbe001", "name": "Clock Probe"}],
    "tables": [{"id": "tblTasks00000001", "name": "Tasks", "base_id": "appClockProbe001", "description": "Clock probe tasks", "fields": [{"id": "fldName00000001", "name": "Name", "type": "singleLineText"}, {"id": "fldStatus000001", "name": "Status", "type": "singleSelect", "options": {"choices": [{"name": "Open"}, {"name": "Done"}]}}, {"id": "fldDueDate0001", "name": "Due Date", "type": "date"}], "views": [{"id": "viwAll00000001", "name": "All", "type": "grid"}]}],
    "records": [
        {"id": "recBeforeT00001", "table_id": "tblTasks00000001", "fields": {"Name": "Filed before T", "Status": "Done", "Due Date": "2019-05-01"}, "created_time": "2019-05-01T09:00:00Z", "modified_time": "2019-05-10T09:00:00Z"},
        {"id": "recAfterT000001", "table_id": "tblTasks00000001", "fields": {"Name": "Due after T", "Status": "Open", "Due Date": "2019-07-01"}, "created_time": "2019-06-20T09:00:00Z", "modified_time": "2019-06-20T09:00:00Z"},
    ],
}
_AIRTABLE_CREATE_ARGS = {"base_id": "appClockProbe001", "table_name": "Tasks", "fields": {"Name": "Clock probe record", "Status": "Open"}}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_airtable_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("airtable", "airtable", "airtable.json", json.dumps(_AIRTABLE_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-airtable")
        before = json.loads(await _clock_call_tool(result.mcp_url, "airtable_create_record", _AIRTABLE_CREATE_ARGS))
        assert not before["created_time"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "airtable" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "airtable_create_record", _AIRTABLE_CREATE_ARGS))
        assert after["created_time"].startswith("2019-06-15T12:"), after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet batch 6 (final Phase A): records_vault, system, zillow.
# ─────────────────────────────────────────────────────────────────────────────

# ── records_vault: list_documents(retention_expired=True) filters retention_expires_at <= now — present on wall, gone at T. ──
_RECORDS_VAULT_CLOCK_DATA = {
    "retention_policies": [{"code": "SOX_7Y", "description": "SOX 7-year retention", "retention_years": 7, "regulatory_basis": "SOX", "permits_expiry": True}],
    "classifications": [{"code": "internal", "description": "Internal", "requires_elevated_role": False}],
    "documents": [{"id": "doc_straddle_001", "kind": "audit_evidence", "classification": "internal", "retention_policy_code": "SOX_7Y", "retention_expires_at": "2022-06-15T00:00:00Z", "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "size_bytes": 0, "status": "active", "uploaded_at": "2015-06-15T00:00:00Z", "uploaded_by": "seed"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_records_vault_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("records_vault", "records_vault", "records_vault.json", json.dumps(_RECORDS_VAULT_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-records_vault")
        before = json.loads(await _clock_call_tool(result.mcp_url, "records_vault_list_documents", {"retention_expired": True}))
        assert any(d["id"] == "doc_straddle_001" for d in before["documents"]), before
        cfg = await _arm_gateway_clock(env, result)
        assert "records_vault" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "records_vault_list_documents", {"retention_expired": True}))
        assert not any(d["id"] == "doc_straddle_001" for d in after["documents"]), after
    finally:
        await env.close()


# ── system: get_current_time returns "now" directly — the cleanest observable (wall ~2026 vs 2019-06-15 at T). ──
_SYSTEM_CLOCK_DATA = {"settings": {"timezone": "UTC", "date_format": "YYYY-MM-DD HH:MM:SS"}}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_system_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("system", "system", "system.json", json.dumps(_SYSTEM_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-system")
        before = json.loads(await _clock_call_tool(result.mcp_url, "system_get_current_time", {}))
        assert not before["current_datetime"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "system" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "system_get_current_time", {}))
        assert after["current_datetime"].startswith("2019-06-15"), after
        assert after["current_weekday"] == "Saturday", after
    finally:
        await env.close()


# ── zillow: search auto-records a searched_at stamp; read it back via get_search_history (DESC sort -> assert set membership). ──
_ZILLOW_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "Alex Thompson", "email": "alex@email.com", "is_user": True, "preferences": "{}"}],
    "properties": [{"id": 1, "zpid": "Z1001", "address": "123 Ocean View Drive", "city": "San Francisco", "state": "CA", "zip_code": "94122", "latitude": 37.7599, "longitude": -122.4941, "price": 1850000, "zestimate": 1875000, "rent_zestimate": 6500, "bedrooms": 4, "bathrooms": 3.5, "sqft": 2400, "lot_size": 4500, "year_built": 1965, "home_type": "Single Family", "property_status": "For Sale", "days_on_zillow": 12, "views": 100, "saves": 5, "price_per_sqft": 770.8, "hoa_fee": 0, "property_tax": 20000, "description": "Ocean view home.", "features": "[]", "images_count": 10, "listing_agent": "Jane Doe", "listing_broker": "ACME Realty", "last_sold_date": "2015-06-01", "last_sold_price": 1200000}],
    "market_data": [], "saved_properties": [], "search_history": [], "scheduled_tours": [], "oauth_tokens": [{"token": "zillow_tok_persona_001", "user_id": "persona_001"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_zillow_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("zillow", "zillow", "zillow.json", json.dumps(_ZILLOW_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-zillow")
        await _clock_call_tool(result.mcp_url, "zillow_search_properties", {"city": "San Francisco"})
        before = json.loads(await _clock_call_tool(result.mcp_url, "zillow_get_search_history", {"limit": 10}))
        assert not any(r["searched_at"].startswith("2019") for r in before["search_history"]), before
        cfg = await _arm_gateway_clock(env, result)
        assert "zillow" in {s["service"] for s in cfg["synced"]}, cfg
        await _clock_call_tool(result.mcp_url, "zillow_search_properties", {"city": "San Francisco"})
        after = json.loads(await _clock_call_tool(result.mcp_url, "zillow_get_search_history", {"limit": 10}))
        assert any(r["searched_at"].startswith("2019-06-15T12:0") for r in after["search_history"]), after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet PHASE B batch 1: mortgage_los, polymarket, ring, smart_lock, blackline (all HIGH).
# ─────────────────────────────────────────────────────────────────────────────

# ── mortgage_los: pipeline summary — a loan closing 2019-06-20 counts as "closing this week" only at T. ──
_MORTGAGE_LOS_CLOCK_DATA = {
    "staff": [{"id": "stf_001", "name": "Alice Officer", "email": "alice@example.com", "role": "loan_officer", "hire_date": "2017-01-05", "is_active": True}],
    "lenders": [{"id": "lnd_001", "name": "Prime Lender", "contact_email": "lender@example.com", "phone": "555-1111"}],
    "borrowers": [{"id": "brw_001", "first_name": "Jane", "last_name": "Doe", "email": "jane@example.com", "status": "active", "assigned_loan_officer": "stf_001"}],
    "loans": [{"id": "loan_001", "borrower_id": "brw_001", "loan_number": "LN-2019-00001", "loan_type": "conventional", "loan_purpose": "purchase", "property_address": "123 Main St", "purchase_price": 400000.0, "loan_amount": 320000.0, "rate": 4.5, "status": "processing", "lender_id": "lnd_001", "assigned_lo": "stf_001", "assigned_processor": "stf_001", "closing_date": "2019-06-20", "rate_lock_expiration": "2019-06-19", "created_at": "2019-06-10T09:00:00Z"}],
    "conditions": [{"id": "cond_001", "loan_id": "loan_001", "condition_type": "prior_to_closing", "description": "Provide updated bank statements", "status": "outstanding", "issued_date": "2019-06-11"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_mortgage_los_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("mortgage_los", "mortgage_los", "mortgage_los.json", json.dumps(_MORTGAGE_LOS_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-mortgage_los")
        before = json.loads(await _clock_call_tool(result.mcp_url, "mortgage_los_get_pipeline_summary", {}))
        assert before["loans_closing_this_week"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        assert "mortgage_los" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "mortgage_los_get_pipeline_summary", {}))
        assert after["loans_closing_this_week"] == 1, after
    finally:
        await env.close()


# ── polymarket: closing-soon markets — a market ending 2019-06-15T18:00Z is within 24h only at T. ──
_POLYMARKET_CLOCK_DATA = {
    "events": [{"event_id": "evt_test_001", "title": "2019 Test Event", "slug": "test-2019", "category": "politics", "end_date": 1560621600.0, "volume_total": 1000.0, "closed": 0, "created_at": 1559347200.0}],
    "markets": [{"market_id": "mkt_test_001", "event_id": "evt_test_001", "condition_id": "cond_test_1", "question": "Will the 2019 test market resolve YES?", "slug": "will-2019-test-resolve-yes", "outcomes": ["Yes", "No"], "outcome_prices": None, "start_date": 1559347200.0, "end_date": 1560621600.0, "volume_total": 5000.0, "volume_24h": 1200.0, "volume_1w": 3000.0, "volume_1m": 4500.0, "liquidity": 800.0, "last_trade_price": 0.55, "category": "politics", "closed": 0, "active": 1, "resolution_status": "pending"}],
    "market_tokens": [{"token_id": "tok_test_yes", "market_id": "mkt_test_001", "outcome_name": "Yes", "outcome_index": 0, "current_price": 0.55}, {"token_id": "tok_test_no", "market_id": "mkt_test_001", "outcome_name": "No", "outcome_index": 1, "current_price": 0.45}],
    "price_history": [], "orderbook_levels": [], "market_holders": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_polymarket_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("polymarket", "polymarket", "polymarket.json", json.dumps(_POLYMARKET_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-polymarket")
        before = json.loads(await _clock_call_tool(result.mcp_url, "polymarket_get_closing_soon_markets", {"hours": 24, "limit": 20}))
        assert not any(m["market_id"] == "mkt_test_001" for m in before["items"]), before
        cfg = await _arm_gateway_clock(env, result)
        assert "polymarket" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "polymarket_get_closing_soon_markets", {"hours": 24, "limit": 20}))
        assert any(m["market_id"] == "mkt_test_001" for m in after["items"]), after
    finally:
        await env.close()


# ── ring: list_alarm_codes — the guest code (expires 2019-07-01) reads active only at T, else expired. ──
_RING_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "Alex Thompson", "is_user": True, "email": "alex.thompson@email.com"}],
    "oauth_tokens": [{"token": "ring_tok_persona_001", "user_id": "persona_001"}],
    "alarm_systems": [{"alarm_id": "alarm_home", "user_id": "persona_001", "mode": "disarmed", "last_changed_at": "2019-06-14T22:00:00Z"}],
    "alarm_codes": [
        {"code_id": "alc_primary", "user_id": "persona_001", "name": "Primary", "code": "1357", "expires_at": None, "status": "active", "created_at": "2019-05-01T09:00:00Z"},
        {"code_id": "alc_guest", "user_id": "persona_001", "name": "Guest", "code": "9090", "expires_at": "2019-07-01T00:00:00Z", "status": "active", "created_at": "2019-06-10T10:00:00Z"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ring_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("ring", "ring", "ring.json", json.dumps(_RING_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-ring")
        before = {c["name"]: c["state"] for c in json.loads(await _clock_call_tool(result.mcp_url, "ring_list_alarm_codes", {}))["alarm_codes"]}
        assert before.get("Guest") == "expired", before
        cfg = await _arm_gateway_clock(env, result)
        assert "ring" in {s["service"] for s in cfg["synced"]}, cfg
        after = {c["name"]: c["state"] for c in json.loads(await _clock_call_tool(result.mcp_url, "ring_list_alarm_codes", {}))["alarm_codes"]}
        assert after.get("Guest") == "active", after
    finally:
        await env.close()


# ── smart_lock: list_access_codes — window states flip (Cleaner expired→active, Contractor expired→scheduled) at T. ──
_SMART_LOCK_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "Alex Thompson", "is_user": True, "email": "alex.thompson@email.com"}],
    "oauth_tokens": [{"token": "smartlock_tok_persona_001", "user_id": "persona_001"}],
    "locks": [{"lock_id": "lock_front", "user_id": "persona_001", "name": "Front Door", "location": "Front Entry", "brand": "August", "model": "Wi-Fi Smart Lock", "state": "locked", "battery_percent": 88, "is_jammed": False, "last_changed_at": "2019-06-15T07:01:00Z"}],
    "access_codes": [
        {"code_id": "ac_family", "user_id": "persona_001", "name": "Family", "code": "2468", "lock_ids": ["lock_front"], "starts_at": None, "ends_at": None, "status": "active", "created_at": "2019-01-04T18:00:00Z"},
        {"code_id": "ac_cleaner", "user_id": "persona_001", "name": "House Cleaner", "code": "4821", "lock_ids": ["lock_front"], "starts_at": "2019-06-15T09:00:00Z", "ends_at": "2019-06-15T17:00:00Z", "status": "active", "created_at": "2019-06-13T14:20:00Z"},
        {"code_id": "ac_contractor", "user_id": "persona_001", "name": "Contractor", "code": "7777", "lock_ids": ["lock_front"], "starts_at": "2019-06-20T08:00:00Z", "ends_at": "2019-06-20T17:00:00Z", "status": "active", "created_at": "2019-06-14T10:00:00Z"},
    ],
    "lock_events": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_smart_lock_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("smart_lock", "smart_lock", "smart_lock.json", json.dumps(_SMART_LOCK_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-smart_lock")
        before = {c["name"]: c["state"] for c in json.loads(await _clock_call_tool(result.mcp_url, "smart_lock_list_access_codes", {}))["access_codes"]}
        assert before.get("House Cleaner") == "expired", before
        cfg = await _arm_gateway_clock(env, result)
        assert "smart_lock" in {s["service"] for s in cfg["synced"]}, cfg
        after = {c["name"]: c["state"] for c in json.loads(await _clock_call_tool(result.mcp_url, "smart_lock_list_access_codes", {}))["access_codes"]}
        assert after.get("House Cleaner") == "active" and after.get("Contractor") == "scheduled", after
    finally:
        await env.close()


# ── blackline: update_exception SLA gate — an open exception past its 2019-06-15T20:00Z SLA is rejected on wall, accepted at T. ──
_BLACKLINE_CLOCK_DATA = {
    "exceptions": [{"id": "EX-0001", "type": "posting_error", "urgency": "high", "state": "investigating", "identified_at": "2019-06-14T09:00:00Z", "identified_by": "j.reyes", "related_period_id": "FP-2019-06", "description": "Suspense account 19900 carries an unexplained $42,000 posting-error variance.", "sla_due_at": "2019-06-15T20:00:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_blackline_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("blackline", "blackline", "blackline.json", json.dumps(_BLACKLINE_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-blackline")
        args = {"exception_id": "EX-0001", "root_cause": "Feed-timing mismatch on suspense clearing; correcting entry drafted.", "actor": "j.reyes"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "blackline_update_exception", args))
        assert not before["success"] and before.get("code") == "EX.SLA_OVERDUE", before
        cfg = await _arm_gateway_clock(env, result)
        assert "blackline" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "blackline_update_exception", args))
        assert after["success"], after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet PHASE B batch 2: amazon, fresh_direct, instacart, gofundme, jira (all HIGH).
# amazon/fresh_direct/instacart mirror the shipped amazon_fresh subscription-synthesis pattern.
# ─────────────────────────────────────────────────────────────────────────────

_AMAZON_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "John Doe", "email": "john.doe@email.com", "phone": "+1-415-555-0123", "is_user": True, "prime_member": True, "default_address": "123 Main Street, San Francisco, CA 94102", "password": "test-password"}],
    "products": [{"asin": "B08MILK001", "title": "Organic Whole Milk", "description": "1 gallon organic whole milk", "price": 5.49, "currency": "USD", "category": "Grocery", "in_stock": True}],
    "cart_items": [], "orders": [], "order_items": [], "payment_methods": [],
    "subscriptions": [
        {"id": "sub_milk_past", "user_id": "persona_001", "product_id": "B08MILK001", "quantity": 1, "frequency": "monthly", "status": "active", "created_at": "2019-01-01T00:00:00Z"},
        {"id": "sub_milk_future", "user_id": "persona_001", "product_id": "B08MILK001", "quantity": 1, "frequency": "monthly", "status": "active", "created_at": "2019-11-01T00:00:00Z"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_amazon_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("amazon", "amazon", "amazon.json", json.dumps(_AMAZON_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-amazon")
        before = json.loads(await _clock_call_tool(result.mcp_url, "amazon_get_orders_history", {}))
        assert "2019" not in {o["created_at"][:4] for o in before["orders"]}, before
        cfg = await _arm_gateway_clock(env, result)
        assert "amazon" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "amazon_get_orders_history", {}))
        assert len(after["orders"]) == 5, after
        assert {o["created_at"][:4] for o in after["orders"]} == {"2019"}, after
    finally:
        await env.close()


_FRESH_DIRECT_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "John Doe", "email": "john.doe@email.com", "phone": "+1-415-555-0123", "address": "123 Main Street, San Francisco, CA 94102", "is_user": True, "password": "test-password"}],
    "products": [{"id": "fd_milk", "name": "Organic Whole Milk", "description": "1 gallon organic whole milk", "price": 5.49, "currency": "USD", "category": "Dairy", "unit": "gallon", "inventory_count": 50, "created_at": "2018-01-01T00:00:00Z"}],
    "product_variants": [{"id": "fdv_milk_1gal", "product_id": "fd_milk", "name": "1 Gallon", "price": 5.49, "sku": "MILK-1GAL", "inventory_count": 50, "attributes": "{}"}],
    "policies": [], "faqs": [], "carts": [], "cart_lines": [], "orders": [], "order_items": [], "payment_methods": [],
    "subscriptions": [
        {"id": "sub_milk_past", "user_id": "persona_001", "product_id": "fd_milk", "quantity": 1, "frequency": "monthly", "status": "active", "created_at": "2019-01-01T00:00:00Z"},
        {"id": "sub_milk_future", "user_id": "persona_001", "product_id": "fd_milk", "quantity": 1, "frequency": "monthly", "status": "active", "created_at": "2019-11-01T00:00:00Z"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_fresh_direct_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("fresh_direct", "fresh_direct", "fresh_direct.json", json.dumps(_FRESH_DIRECT_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-fresh_direct")
        before = json.loads(await _clock_call_tool(result.mcp_url, "fresh_direct_get_purchase_history", {"limit": 100}))
        assert "2019" not in {o["created_at"][:4] for o in before["orders"]}, before
        cfg = await _arm_gateway_clock(env, result)
        assert "fresh_direct" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "fresh_direct_get_purchase_history", {"limit": 100}))
        assert len(after["orders"]) == 5, after
        assert {o["created_at"][:4] for o in after["orders"]} == {"2019"}, after
    finally:
        await env.close()


_INSTACART_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "email": "john.doe@email.com", "name": "John Doe", "phone": "+1-555-0100", "address": "1 Market St, San Francisco, CA", "is_user": True, "password": "test-password"}],
    "products": [{"id": "prod_milk", "name": "Organic Whole Milk", "description": "1 gal organic whole milk", "price": 5.99, "currency": "USD", "category": "dairy", "unit": "gallon", "inventory_count": 100, "created_at": "2019-01-01T00:00:00Z"}],
    "product_variants": [], "policies": [], "faqs": [], "carts": [], "cart_lines": [], "orders": [], "order_items": [], "payment_methods": [],
    "subscription": {"id": "sub_seed01", "user_id": "persona_001", "product_id": "prod_milk", "quantity": 2, "frequency": "weekly", "payment_method_id": None, "status": "active", "created_at": "2019-06-01T12:00:00Z"},
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_instacart_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("instacart", "instacart", "instacart.json", json.dumps(_INSTACART_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-instacart")
        before = json.loads(await _clock_call_tool(result.mcp_url, "instacart_get_purchase_history", {"limit": 50}))
        assert "2019" not in {o["created_at"][:4] for o in before["orders"]}, before
        cfg = await _arm_gateway_clock(env, result)
        assert "instacart" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "instacart_get_purchase_history", {"limit": 50}))
        assert len(after["orders"]) == 2, after
        assert all(o["created_at"].startswith("2019-06") for o in after["orders"]), after
    finally:
        await env.close()


# ── gofundme: get_campaign days_left = end_date - now — 0 (past) on wall, ~30 (future) at T. ──
_GOFUNDME_CLOCK_DATA = {
    "users": [{"user_id": "usr_1", "name": "Ada Donor", "is_user": True, "email": "ada@example.com"}],
    "oauth_tokens": [{"token": "tok_ada", "user_id": "usr_1"}],
    "payment_methods": [],
    "campaigns": [{"campaign_id": "cmp_test", "organizer_id": "usr_2", "title": "Help Fund X", "slug": "help-fund-x", "story": "A synthetic cause straddling the virtual clock anchor.", "category": "medical", "goal_amount": 10000.0, "amount_raised": 100.0, "donor_count": 1, "status": "active", "created_at": "2019-06-01T00:00:00Z", "end_date": "2019-07-15T12:00:00Z"}],
    "donations": [], "comments": [], "campaign_updates": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_gofundme_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("gofundme", "gofundme", "gofundme.json", json.dumps(_GOFUNDME_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-gofundme")
        before = json.loads(await _clock_call_tool(result.mcp_url, "gofundme_get_campaign", {"campaign_id": "cmp_test"}))
        assert before["days_left"] == 0, before
        cfg = await _arm_gateway_clock(env, result)
        assert "gofundme" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "gofundme_get_campaign", {"campaign_id": "cmp_test"}))
        assert after["days_left"] >= 1, after
    finally:
        await env.close()


# ── jira: get_sla — an open issue is SLA-breached under the frozen 2026 anchor, not breached at virtual T (~60 min after creation). ──
_JIRA_CLOCK_DATA = {
    "organizations": [{"id": "cli_acme", "name": "Acme Retail"}],
    "projects": [{"id": "proj_ops", "key": "OPS", "name": "3PL Operations Support", "project_type": "service_desk"}],
    "users": [{"id": "user_cust1", "display_name": "Acme Contact", "email": "ops@acme.example", "role": "customer", "organization_id": "cli_acme"}],
    "issues": [{"id": "OPS-1", "project_id": "proj_ops", "summary": "EDI 850 silently rejected for Acme PO", "description": "Open incident awaiting resolution.", "issue_type": "Incident", "status": "Open", "priority": "High", "reporter_id": "user_cust1", "organization_id": "cli_acme", "labels": ["edi"], "ttfr_goal_minutes": 180, "ttr_goal_minutes": 480, "first_response_at": None, "resolved_at": None, "created_at": "2019-06-15T11:00:00Z", "updated_at": "2019-06-15T11:00:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_jira_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("jira", "jira", "jira.json", json.dumps(_JIRA_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-jira")
        # Off-clock uses jira's FROZEN JIRA_UNIVERSE_NOW anchor (~2026), so the 2019 issue reads breached.
        before = json.loads(await _clock_call_tool(result.mcp_url, "jira_get_sla", {"issue_key": "OPS-1"}))
        assert before["breached"] is True, before
        cfg = await _arm_gateway_clock(env, result)
        assert "jira" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "jira_get_sla", {"issue_key": "OPS-1"}))
        assert after["breached"] is False, after
        assert after["sla"]["time_to_resolution"]["elapsed_minutes"] < 480, after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet PHASE B batch 3 (MEDIUM write-stamp readbacks): cab, stripe, linear, clickup, confluence.
# ─────────────────────────────────────────────────────────────────────────────

def _data_json_zip(d: dict) -> bytes:
    """A universe zip whose sole member is data.json (the linear/gdrive-style inner-data.json layout)."""
    buf = _io.BytesIO()
    with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as z:
        z.writestr("data.json", json.dumps(d))
    return buf.getvalue()


# ── cab: order_ride mints time_stamp = now — wall (~2026) vs T (order_ride empties to one active ride, cancel between). ──
_CAB_CLOCK_DATA = {
    "ride_history": [
        {"ride_id": "ride_pre_t", "status": "COMPLETED", "service_type": "Default", "start_location": "Downtown", "end_location": "Airport", "price": 25.0, "duration": 20.0, "time_stamp": 1560513600.0, "distance_km": 10.0, "delay": 2.0, "delay_history": [{"delay": 2.0, "time_stamp": 1560513600.0}]},
        {"ride_id": "ride_post_t", "status": "COMPLETED", "service_type": "Premium", "start_location": "Airport", "end_location": "Downtown", "price": 45.0, "duration": 25.0, "time_stamp": 1560686400.0, "distance_km": 12.0, "delay": 5.0, "delay_history": [{"delay": 5.0, "time_stamp": 1560686400.0}]},
    ],
    "on_going_ride": None, "payment_methods": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_cab_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("cab", "cab", "cab.json", json.dumps(_CAB_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-cab")
        args = {"start_location": "Downtown", "end_location": "Airport", "service_type": "Default"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "cab_order_ride", args))
        assert not str(before["time_stamp"]).startswith("2019"), before
        await _clock_call_tool(result.mcp_url, "cab_user_cancel_ride", {})  # only one active ride allowed
        cfg = await _arm_gateway_clock(env, result)
        assert "cab" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "cab_order_ride", args))
        assert str(after["time_stamp"]).startswith("2019-06-15"), after
    finally:
        await env.close()


# ── stripe: create_customer mints `created` — wall (~2026) vs T. ──
_STRIPE_CLOCK_DATA = {
    "customers": [{"id": "cus_seedbefore0001", "name": "Acme Before", "email": "before@acme.test", "created": 1556668800}, {"id": "cus_seedafter00001", "name": "Acme After", "email": "after@acme.test", "created": 1561939200}],
    "products": [{"id": "prod_seed00000001", "name": "Seed Widget", "created": 1556668800}],
    "prices": [{"id": "price_seed0000001", "product_id": "prod_seed00000001", "unit_amount": 1000, "currency": "usd", "created": 1556668800}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_stripe_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("stripe", "stripe", "stripe.json", json.dumps(_STRIPE_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-stripe")
        args = {"name": "Clock Probe", "email": "probe@example.com"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "stripe_create_customer", args))
        assert not before["created"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "stripe" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "stripe_create_customer", args))
        assert after["created"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── linear: create_issue mints created_at — wall vs T. Uses a ZIP universe (inner data.json). ──
_LINEAR_CLOCK_DATA = {
    "current_user_id": "usr_me",
    "users": [{"id": "usr_me", "email": "me@example.com", "name": "Ada Lovelace", "display_name": "ada", "is_active": True, "created_at": "2019-01-01T00:00:00Z", "updated_at": "2019-01-01T00:00:00Z"}],
    "teams": [{"id": "team_eng", "name": "Engineering", "key": "ENG", "created_at": "2019-01-01T00:00:00Z", "updated_at": "2019-01-01T00:00:00Z"}],
    "projects": [],
    "issues": [{"id": "issue_seed001", "team_id": "team_eng", "title": "Seed issue (pre-T control)", "description": "Static control stamped before T.", "priority": 0, "labels": [], "links": [], "is_archived": False, "created_at": "2019-01-10T09:00:00Z", "updated_at": "2019-01-10T09:00:00Z"}],
    "comments": [], "team_memberships": [], "attachments": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_linear_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("linear", "linear", "linear.zip", _data_json_zip(_LINEAR_CLOCK_DATA))
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-linear")
        args = {"team": "Engineering", "title": "clock-probe"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "linear_create_issue", args))
        assert not before["created_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "linear" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "linear_create_issue", args))
        assert after["created_at"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── clickup: create_task mints date_created — wall vs T. ──
_CLICKUP_CLOCK_DATA = {
    "workspaces": [{"id": "ws_1", "name": "Clock WS", "members": ["u_ada"]}],
    "members": [{"id": "u_ada", "username": "ada", "email": "ada@acme.test", "role": 1}],
    "spaces": [{"id": "space_eng", "name": "Engineering", "workspace_id": "ws_1", "statuses": [{"status": "backlog", "type": "open", "orderindex": 0}, {"status": "in dev", "type": "custom", "orderindex": 1}]}],
    "lists": [{"id": "list_main", "name": "Main", "space_id": "space_eng", "folder_id": None}],
    "tasks": [{"id": "t_seed_pre_T", "name": "Seed before T", "list_id": "list_main", "space_id": "space_eng", "status": "backlog", "date_created": "2019-01-01T00:00:00Z"}, {"id": "t_seed_post_T", "name": "Seed after T", "list_id": "list_main", "space_id": "space_eng", "status": "in dev", "date_created": "2019-12-31T00:00:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_clickup_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("clickup", "clickup", "clickup.json", json.dumps(_CLICKUP_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-clickup")
        args = {"listId": "list_main", "name": "Clock probe task", "status": "in dev"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "clickup_create_task", args))
        assert not before["date_created"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "clickup" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "clickup_create_task", args))
        assert after["date_created"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── confluence: create_page mints created_at — wall vs T. ──
_CONFLUENCE_CLOCK_DATA = {
    "users": [{"id": "user_priya", "account_id": "acct_priya", "email": "priya.nair@northwind-analytics.com", "display_name": "Priya Nair", "is_active": True, "created_at": "2019-01-06T09:00:00Z", "updated_at": "2019-01-06T09:00:00Z"}],
    "spaces": [{"id": "space_eng", "key": "ENG", "name": "Engineering", "type": "global", "status": "current", "description": "Engineering KB", "created_at": "2019-01-08T10:00:00Z", "updated_at": "2019-05-20T14:30:00Z"}],
    "pages": [{"id": "page_eng_home", "space_id": "space_eng", "parent_id": None, "title": "Engineering Home", "body": "<p>Welcome</p>", "body_format": "storage", "status": "current", "version": 1, "author_id": "user_priya", "created_at": "2019-01-08T10:05:00Z", "updated_at": "2019-05-20T14:30:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_confluence_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("confluence", "confluence", "confluence.json", json.dumps(_CONFLUENCE_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-confluence")
        args = {"space": "ENG", "title": "Clock Probe", "body": "probe"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "confluence_create_page", args))
        assert not before["created_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "confluence" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "confluence_create_page", args))
        assert after["created_at"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet PHASE B batch 4 (MEDIUM write-stamp readbacks): crm, email_mcp, erp, figma, github.
# ─────────────────────────────────────────────────────────────────────────────

# ── crm: create_contact mints createdate — wall vs T (zip universe). ──
_CRM_CLOCK_DATA = {
    "companies": [{"id": "comp_seed_001", "name": "Northwind Traders", "domain": "northwind.example.com", "industry": "Retail", "createdate": "2019-06-01T09:00:00Z"}],
    "contacts": [{"id": "cont_seed_001", "full_name": "Ada Seed", "email": "ada.seed@northwind.example.com", "company_id": "comp_seed_001", "createdate": "2019-06-01T09:00:00Z"}, {"id": "cont_seed_002", "full_name": "Ben After", "email": "ben.after@northwind.example.com", "company_id": "comp_seed_001", "createdate": "2019-07-01T09:00:00Z"}],
    "deals": [], "leads": [], "engagements": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_crm_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("crm", "crm", "crm.zip", _data_json_zip(_CRM_CLOCK_DATA))
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-crm")
        args = {"full_name": "Grace Hopper", "email": "grace.hopper@northwind.example.com", "company_id": "comp_seed_001"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "crm_create_contact", args))
        assert not before["createdate"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "crm" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "crm_create_contact", args))
        assert after["createdate"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── email_mcp (env=email): send_email mints timestamp — read back via get_email_by_id(SENT). ──
_EMAIL_CLOCK_DATA = {
    "user_email": "alice@example.com",
    "emails": [
        {"email_id": "email_seed_before_t", "folder": "INBOX", "sender": "carol@example.com", "recipients": ["alice@example.com"], "subject": "Before T", "content": "Seeded before the canonical virtual time.", "timestamp": "2019-06-10T09:00:00Z", "is_read": False},
        {"email_id": "email_seed_after_t", "folder": "INBOX", "sender": "dan@example.com", "recipients": ["alice@example.com"], "subject": "After T", "content": "Seeded after the canonical virtual time.", "timestamp": "2019-06-20T09:00:00Z", "is_read": False},
    ],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_email_mcp_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("email_mcp", "email", "email.json", json.dumps(_EMAIL_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-email")
        send = {"sender": "alice@example.com", "recipients": ["bob@example.com"], "subject": "Clock probe", "content": "Probe email for clock verification."}
        eid_b = json.loads(await _clock_call_tool(result.mcp_url, "email_send_email", send))["email_id"]
        before = json.loads(await _clock_call_tool(result.mcp_url, "email_get_email_by_id", {"email_id": eid_b, "folder_name": "SENT"}))
        assert not before["timestamp"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "email" in {s["service"] for s in cfg["synced"]}, cfg
        eid_a = json.loads(await _clock_call_tool(result.mcp_url, "email_send_email", send))["email_id"]
        after = json.loads(await _clock_call_tool(result.mcp_url, "email_get_email_by_id", {"email_id": eid_a, "folder_name": "SENT"}))
        assert after["timestamp"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── erp: create_invoice mints created_at — wall vs T. ──
_ERP_CLOCK_DATA = {
    "parties": [{"id": "cli_acme", "name": "Acme Retail", "party_type": "client", "email": "ap@acme-retail.com", "terms": "net_30", "is_active": True, "created_at": "2019-06-01T00:00:00Z"}],
    "gl_accounts": [{"id": "acct_1200", "number": "1200", "name": "Accounts Receivable", "account_type": "asset", "normal_balance": "debit", "subsidiary_id": None, "is_active": True, "current_balance": 0.0, "created_at": "2019-06-01T00:00:00Z"}, {"id": "acct_4000", "number": "4000", "name": "Storage & Handling Revenue", "account_type": "revenue", "normal_balance": "credit", "subsidiary_id": None, "is_active": True, "current_balance": 0.0, "created_at": "2019-06-01T00:00:00Z"}],
    "periods": [{"id": "per_2019_06", "name": "Jun 2019", "start_date": "2019-06-01", "end_date": "2019-06-30", "status": "open", "created_at": "2019-06-01T00:00:00Z"}],
}


def _erp_created(r: dict) -> str:
    return (r.get("invoice", r))["created_at"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_erp_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("erp", "erp", "erp.json", json.dumps(_ERP_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-erp")
        args = {"customer_id": "cli_acme", "amount": 100.0, "period_id": "per_2019_06", "invoice_date": "2019-06-15"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "erp_create_invoice", args))
        assert not _erp_created(before).startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "erp" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "erp_create_invoice", args))
        assert _erp_created(after).startswith("2019-06-15"), after
    finally:
        await env.close()


# ── figma: post_comment mints created_at — wall vs T. ──
_FIGMA_CLOCK_DATA = {
    "projects": [{"id": "proj_product", "name": "Product Design"}],
    "files": [{"file_key": "FILEKEY001", "name": "Checkout Redesign", "project_id": "proj_product", "last_modified": "2019-06-14T09:00:00Z", "version": "100", "editor_type": "figma", "pages": [{"id": "1:0", "name": "Flows"}]}],
    "comments": [{"id": "c1", "file_key": "FILEKEY001", "message": "Initial review note.", "user": "grace", "created_at": "2019-06-14T10:00:00Z"}],
    "nodes": [{"file_key": "FILEKEY001", "id": "0:0", "type": "DOCUMENT", "name": "Document", "depth": 0, "order": 0}, {"file_key": "FILEKEY001", "id": "1:0", "type": "CANVAS", "name": "Flows", "parent_id": "0:0", "page_id": "1:0", "depth": 1, "order": 0}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_figma_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("figma", "figma", "figma.json", json.dumps(_FIGMA_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-figma")
        args = {"fileKey": "FILEKEY001", "message": "Clock check: please tighten the CTA spacing.", "nodeId": "1:0"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "figma_post_comment", args))
        assert not before["created_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "figma" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "figma_post_comment", args))
        assert after["created_at"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── github: create_issue mints created_at — wall vs T (zip universe). ──
_GITHUB_CLOCK_DATA = {
    "users": [{"login": "acme-org", "type": "Organization"}, {"login": "agent", "type": "User"}],
    "repositories": [{"id": "acme-org/webapp", "name": "webapp", "full_name": "acme-org/webapp", "owner_login": "acme-org", "default_branch": "main", "visibility": "private", "created_at": "2019-01-10T09:00:00Z", "updated_at": "2019-02-01T09:00:00Z"}],
    "issues": [{"id": "acme-org/webapp#1", "repo_id": "acme-org/webapp", "number": 1, "title": "Seed issue (pre-T)", "state": "open", "author_login": "acme-org", "created_at": "2019-01-20T09:00:00Z", "updated_at": "2019-01-20T09:00:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_github_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("github", "github", "github.zip", _data_json_zip(_GITHUB_CLOCK_DATA))
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-github")
        args = {"owner": "acme-org", "repo": "webapp", "title": "Clock probe issue", "body": "probe"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "github_create_issue", args))
        assert not before["created_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "github" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "github_create_issue", args))
        assert after["created_at"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet PHASE B batch 5 (MEDIUM write-stamp readbacks): messaging, notion, oms, quickbooks, shopping.
# ─────────────────────────────────────────────────────────────────────────────

# ── messaging: send_message overwrites conversation last_updated — wall vs T (read back via get_conversation). ──
_MESSAGING_CLOCK_DATA = {
    "conversations": [{
        "conversation_id": "c10cc0de00000000000000a1", "title": "Clock straddle thread",
        "participant_ids": ["user001", "user002"],
        "messages": [
            {"message_id": "msgc10cc0de0001", "sender_id": "user002", "timestamp": "2019-06-01 09:00:00", "content": "seed: before T", "attachment_name": None},
            {"message_id": "msgc10cc0de0002", "sender_id": "user001", "timestamp": "2019-07-01 09:00:00", "content": "seed: after T", "attachment_name": None},
        ],
        "last_updated": "2019-07-01 09:00:00",
    }],
    "current_user_id": "user001",
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_messaging_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("messaging", "messaging", "data.json", json.dumps(_MESSAGING_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-messaging")
        send = {"user_id": "user002", "content": "clock probe at T"}
        cid_b = json.loads(await _clock_call_tool(result.mcp_url, "messaging_send_message", send))["conversation_id"]
        before = json.loads(await _clock_call_tool(result.mcp_url, "messaging_get_conversation", {"conversation_id": cid_b}))
        assert not before["last_updated"].startswith("2019"), before  # read last_updated, NOT messages[-1] (get_messages sorts chronologically)
        cfg = await _arm_gateway_clock(env, result)
        assert "messaging" in {s["service"] for s in cfg["synced"]}, cfg
        cid_a = json.loads(await _clock_call_tool(result.mcp_url, "messaging_send_message", send))["conversation_id"]
        after = json.loads(await _clock_call_tool(result.mcp_url, "messaging_get_conversation", {"conversation_id": cid_a}))
        assert after["last_updated"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── notion: create_page mints created_time — wall vs T (zip universe, inner data.json). ──
_NOTION_CLOCK_DATA = {
    "users": [
        {"id": "da1ca67b-6e6a-4458-b8ff-fdc389a3c992", "type": "bot", "name": "Notion Bot", "bot": {}},
        {"id": "11111111-1111-4111-8111-111111111111", "type": "person", "name": "Alice Straddle", "person": {"email": "alice@example.com"}},
    ],
    "pages": [{
        "id": "5eed0000-0000-4000-8000-000000000001",
        "created_time": "2019-06-10T09:00:00Z", "last_edited_time": "2019-06-10T09:00:00Z",
        "created_by_id": "da1ca67b-6e6a-4458-b8ff-fdc389a3c992", "last_edited_by_id": "da1ca67b-6e6a-4458-b8ff-fdc389a3c992",
        "parent": {"type": "workspace", "workspace": True},
    }],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_notion_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("notion", "notion", "notion.zip", _data_json_zip(_NOTION_CLOCK_DATA))
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-notion")
        args = {"parent": {"type": "page_id", "page_id": "5eed0000-0000-4000-8000-000000000001"}}
        before = json.loads(await _clock_call_tool(result.mcp_url, "notion_create_page", args))
        assert not before["created_time"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "notion" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "notion_create_page", args))
        assert after["created_time"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── oms: create_order mints created_at — wall vs T. ──
_OMS_CLOCK_DATA = {
    "clients": [{"id": "cli_acme", "name": "Acme Corp", "segment": "apparel", "credit_status": "ok", "credit_limit": 100000.0, "fulfillment_rule": "allow_partial", "created_at": "2019-06-01T09:00:00Z", "updated_at": "2019-06-01T09:00:00Z"}],
    "skus": [{"id": "sku_acme_tee", "client_id": "cli_acme", "description": "Acme Tee", "unit_price": 19.99, "weight_lb": 0.3, "is_active": True, "created_at": "2019-06-01T09:00:00Z", "updated_at": "2019-06-01T09:00:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_oms_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("oms", "oms", "oms.json", json.dumps(_OMS_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-oms")
        args = {"client_id": "cli_acme", "lines": [{"sku_id": "sku_acme_tee", "qty": 2}], "channel": "ecommerce", "ship_to_street": "1 Test St", "ship_to_city": "Testville", "ship_to_zip": "10001"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "oms_create_order", args))
        assert not before["created_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "oms" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "oms_create_order", args))
        assert after["created_at"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── quickbooks: create_customer mints MetaData.CreateTime — wall vs T (distinct DisplayNames; uniqueness-guarded). ──
_QUICKBOOKS_CLOCK_DATA = {
    "customers": [{"Id": "1", "DisplayName": "Globex Partners", "CompanyName": "Globex Partners LLC", "Active": True, "MetaData": {"CreateTime": "2019-06-01T00:00:00Z", "LastUpdatedTime": "2019-06-10T09:00:00Z"}}],
    "vendors": [{"Id": "1", "DisplayName": "Initech Supplies", "Active": True, "MetaData": {"CreateTime": "2019-06-05T00:00:00Z", "LastUpdatedTime": "2019-06-05T00:00:00Z"}}],
    "accounts": [{"Id": "1", "Name": "Sales Income", "AccountType": "Income", "Classification": "Revenue", "MetaData": {"CreateTime": "2019-06-01T00:00:00Z", "LastUpdatedTime": "2019-06-01T00:00:00Z"}}],
    "items": [], "invoices": [], "bills": [],
}


def _qb_create_time(r: dict) -> str:
    return (r.get("customer", r))["MetaData"]["CreateTime"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_quickbooks_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("quickbooks", "quickbooks", "data.json", json.dumps(_QUICKBOOKS_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-quickbooks")
        before = json.loads(await _clock_call_tool(result.mcp_url, "quickbooks_create_customer", {"DisplayName": "Clock Probe Before"}))
        assert not _qb_create_time(before).startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "quickbooks" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "quickbooks_create_customer", {"DisplayName": "Clock Probe After"}))
        assert _qb_create_time(after).startswith("2019-06-15"), after
    finally:
        await env.close()


# ── shopping: checkout mints order_date — wall vs T (cart consumed/reset, so re-add before each checkout). ──
_SHOPPING_CLOCK_DATA = {
    "products": [{"product_id": "prod001", "name": "Wireless Mouse", "variants": {"item001": {"item_id": "item001", "price": 29.99, "available": True, "options": {"color": "Black"}}}}],
    "discount_codes": {},
    "orders": [{"order_id": "order_seed_pre_t", "order_status": "delivered", "order_date": "2019-06-10 09:00:00", "order_total": 29.99, "order_items": {"item001": {"item_id": "item001", "quantity": 1, "price": 29.99, "available": True, "options": {"color": "Black"}}}}],
    "payment_methods": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_shopping_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("shopping", "shopping", "shopping.json", json.dumps(_SHOPPING_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-shopping")
        await _clock_call_tool(result.mcp_url, "shopping_add_to_cart", {"item_id": "item001", "quantity": 1})
        oid_b = json.loads(await _clock_call_tool(result.mcp_url, "shopping_checkout", {}))["order_id"]
        before = json.loads(await _clock_call_tool(result.mcp_url, "shopping_get_order_details", {"order_id": oid_b}))
        assert not before["order_date"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "shopping" in {s["service"] for s in cfg["synced"]}, cfg
        await _clock_call_tool(result.mcp_url, "shopping_add_to_cart", {"item_id": "item001", "quantity": 1})
        oid_a = json.loads(await _clock_call_tool(result.mcp_url, "shopping_checkout", {}))["order_id"]
        after = json.loads(await _clock_call_tool(result.mcp_url, "shopping_get_order_details", {"order_id": oid_a}))
        assert after["order_date"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet PHASE B batch 6 (MEDIUM write-stamp readbacks): snowflake, ticketmaster, tms, trello, wms.
# ─────────────────────────────────────────────────────────────────────────────

# ── snowflake: execute_query appends a query_history row stamped executed_at — wall vs T. ──
# query_history is DESC (newest first), so after arming, the ~2026 before-row still sorts first;
# assert set-membership of a 2019-06-15 row (not the top row). Seed history predates T (2019-06-14).
_SNOWFLAKE_CLOCK_DATA = {
    "databases": [{"name": "ANALYTICS", "comment": "Primary analytics DB"}],
    "schemas": [{"database": "ANALYTICS", "name": "FINANCE", "comment": "Finance marts"}],
    "tables": [{"database": "ANALYTICS", "schema": "FINANCE", "name": "REVENUE", "table_type": "TABLE", "comment": "Monthly recognized revenue by region", "columns": [{"name": "region", "data_type": "VARCHAR"}, {"name": "arr", "data_type": "NUMBER", "is_nullable": False}], "rows": [{"region": "AMER", "arr": 120000}, {"region": "EMEA", "arr": 84000}]}],
    "query_history": [{"query_text": "SELECT 1", "status": "SUCCESS", "rows_produced": 1, "executed_at": "2019-06-14T09:00:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_snowflake_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("snowflake", "snowflake", "snowflake.json", json.dumps(_SNOWFLAKE_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-snowflake")
        sql = {"sql": "SELECT region, arr FROM ANALYTICS.FINANCE.REVENUE"}
        await _clock_call_tool(result.mcp_url, "snowflake_execute_query", sql)
        before = json.loads(await _clock_call_tool(result.mcp_url, "snowflake_query_history", {"limit": 20}))
        assert not any(r["executed_at"].startswith("2019-06-15") for r in before), before
        cfg = await _arm_gateway_clock(env, result)
        assert "snowflake" in {s["service"] for s in cfg["synced"]}, cfg
        await _clock_call_tool(result.mcp_url, "snowflake_execute_query", sql)
        after = json.loads(await _clock_call_tool(result.mcp_url, "snowflake_query_history", {"limit": 20}))
        assert any(r["executed_at"].startswith("2019-06-15") for r in after), after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Async submit/poll maturity is measured on the axis pinned AT SUBMIT —
# the clock/v1 virtual clock on a synced server, real wall time otherwise. Snowflake
# has the async twins (snowflake_submit_query + get_job_status/get_job_result) and a
# clock/v1 consumer, so one deploy exercises every clock-state row:
#   1 UNSYNCED  — no clock armed -> real-time maturity (2s wall job).
#   2 ARMED r60 — 60 VIRTUAL-second job matures in ~1 real s (virtual scaling).
#   3 DISARM    — clock cleared mid-job -> disarm guard holds the job pending (no false maturation).
#   4 RATE=0    — frozen virtual clock never crosses the deadline (backstop not yet hit) -> pending.
# ─────────────────────────────────────────────────────────────────────────────
_ASYNC_WAIT_URI = "urn:agentenv:set-async-wait/v1"
_SNOWFLAKE_SUBMIT = "snowflake_submit_query"
_SNOWFLAKE_SQL = {"sql": "SELECT region, arr FROM ANALYTICS.FINANCE.REVENUE"}


async def _set_async_wait(result, tool_name, min_seconds, max_seconds=None, scale_to_virtual=True) -> dict:
    """Arm a submit tool's async wait range via the set-async-wait/v1 gateway ext proxy (harness control path)."""
    base = f"{result.gateway_url}/svc/mcp-snowflake"
    card = await protocol_v1.get_card(base)
    params = {"tool_name": tool_name, "min_seconds": min_seconds, "scale_to_virtual": scale_to_virtual}
    if max_seconds is not None:
        params["max_seconds"] = max_seconds
    return await protocol_v1.invoke_extension(base, card, _ASYNC_WAIT_URI, params=params)


async def _clock_clear(result) -> None:
    """Disarm the gateway clock (POST /clock/clear); it then 404s on /clock/time."""
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{result.gateway_url}/clock/clear", timeout=30)
        assert r.status_code == 200 and r.json()["armed"] is False, r.text


async def _submit_snowflake_job(result) -> str:
    """Submit an async snowflake query; assert the pending handle shape and return its job_id."""
    out = json.loads(await _clock_call_tool(result.mcp_url, _SNOWFLAKE_SUBMIT, _SNOWFLAKE_SQL))
    assert out.get("status") == "pending" and out.get("job_id"), f"unexpected submit shape: {out}"
    return out["job_id"]


async def _poll_status(result, job_id) -> str:
    """Poll snowflake_get_job_status; assert the shape and return status (pending|ready|failed|error)."""
    out = json.loads(await _clock_call_tool(result.mcp_url, "snowflake_get_job_status", {"job_id": job_id}))
    assert out.get("job_id") == job_id, f"unexpected status shape: {out}"
    return out["status"]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_snowflake_async_virtual_clock(sandbox_provider):
    """Async submit/poll maturity follows the clock/v1 virtual axis pinned at submit."""
    spec = _ClockServerSpec("snowflake", "snowflake", "snowflake.json", json.dumps(_SNOWFLAKE_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    rows: dict[str, dict] = {}
    try:
        # Card advertises BOTH clock/v1 and set-async-wait/v1.
        card = await protocol_v1.get_card(f"{result.gateway_url}/svc/mcp-snowflake")
        uris = {e["uri"] for e in card["capabilities"]["extensions"]}
        assert "urn:agentenv:clock/v1" in uris, uris
        assert _ASYNC_WAIT_URI in uris, uris
        logger.info("CARD OK: mcp-snowflake advertises urn:agentenv:clock/v1 + urn:agentenv:set-async-wait/v1")

        # Warm the MCP path so the timing-sensitive immediate polls aren't paying first-call latency.
        await _clock_call_tool(result.mcp_url, "snowflake_list_databases", {})

        # ── ROW 1: UNSYNCED (clock not armed) — real-time maturity, zero-delta ──
        # scale_to_virtual defaults True, but with no clock the server is unsynced
        # (has_now_provider()==False at submit) so the wait is measured on real wall time.
        await _set_async_wait(result, _SNOWFLAKE_SUBMIT, 2, 2)
        t0 = time.monotonic()
        job1 = await _submit_snowflake_job(result)
        imm1 = await _poll_status(result, job1)
        e1 = time.monotonic() - t0
        if e1 < 1.5:
            assert imm1 == "pending", f"ROW1 immediate expected pending (elapsed={e1:.2f}s), got {imm1}"
        await asyncio.sleep(4)
        after1 = await _poll_status(result, job1)
        assert after1 == "ready", f"ROW1 after 4s wall expected ready, got {after1}"
        rows["1_UNSYNCED"] = {"expected": "pending->ready(wall)", "immediate": imm1, "after_4s": after1, "verdict": "PASS"}
        logger.info(f"ROW 1 UNSYNCED PASS: immediate={imm1} (elapsed {e1:.2f}s); after 4s real={after1} — 2s wall job matured on real time")

        # ── ROW 2: ARMED rate=60 — 60 VIRTUAL-second job matures in ~1 real s (key result) ──
        cfg = await _arm_gateway_clock(env, result, rate=60)
        assert "snowflake" in {s["service"] for s in cfg["synced"]}, cfg
        await _set_async_wait(result, _SNOWFLAKE_SUBMIT, 60, 60, scale_to_virtual=True)
        t0 = time.monotonic()
        job2 = await _submit_snowflake_job(result)
        imm2 = await _poll_status(result, job2)
        gap2 = time.monotonic() - t0
        await asyncio.sleep(3)
        after2 = await _poll_status(result, job2)
        real_to_ready = time.monotonic() - t0
        assert after2 == "ready", f"ROW2 after 3s expected READY, got {after2}"
        assert real_to_ready < 20, f"ROW2 60-VIRTUAL-s job must ready in « 60 real s, took {real_to_ready:.1f}s"
        rows["2_ARMED_r60"] = {"expected": "pending->READY(virtual)", "immediate": imm2,
                               "after_3s": after2, "real_s_to_ready": round(real_to_ready, 1), "verdict": "PASS"}
        logger.info(f"ROW 2 ARMED rate=60 PASS: immediate={imm2} (submit+poll gap {gap2:.2f}s); "
                    f"60-VIRTUAL-s job READY in {real_to_ready:.1f} real s (vs 2s wall job in row 1) — virtual scaling proven")

        # ── ROW 3: DISARM mid-job — the disarm guard holds the job pending (no false maturation) ──
        # Clock still armed rate=60 from row 2; submit stamps a ~2019 virtual start, then clear the
        # gateway clock. Polls now read off the pinned axis (utcnow() walls to ~2026); the implausible-
        # elapsed guard must hold PENDING rather than mature the job "finished years ago".
        await _set_async_wait(result, _SNOWFLAKE_SUBMIT, 60, 60, scale_to_virtual=True)
        job3 = await _submit_snowflake_job(result)
        await _clock_clear(result)
        polls3 = []
        for _ in range(3):
            polls3.append(await _poll_status(result, job3))
            await asyncio.sleep(1)
        assert all(s == "pending" for s in polls3), f"ROW3 disarm guard: expected all pending, got {polls3}"
        rows["3_DISARM_midjob"] = {"expected": "pending(disarm guard)", "polls": polls3, "verdict": "PASS"}
        logger.info(f"ROW 3 DISARM mid-job PASS: post-clear polls={polls3} — disarm guard prevented false maturation")

        # ── ROW 4: RATE=0 frozen — a frozen virtual clock never crosses the 5-virtual-s deadline ──
        # (the 300s wall backstop is NOT yet hit at 8s, so it stays pending; backstop maturation is
        # covered by unit tests and not waited out here.)
        cfg0 = await _arm_gateway_clock(env, result, rate=0)
        assert "snowflake" in {s["service"] for s in cfg0["synced"]}, cfg0
        await _set_async_wait(result, _SNOWFLAKE_SUBMIT, 5, 5, scale_to_virtual=True)
        job4 = await _submit_snowflake_job(result)
        await asyncio.sleep(8)
        after4 = await _poll_status(result, job4)
        assert after4 == "pending", f"ROW4 frozen clock: expected pending, got {after4}"
        rows["4_RATE0_frozen"] = {"expected": "pending(frozen)", "after_8s": after4, "verdict": "PASS"}
        logger.info(f"ROW 4 RATE=0 frozen PASS: after 8s status={after4} — frozen virtual clock never crossed the 5-virtual-s deadline")

        logger.info(f"ASYNC VIRTUAL CLOCK SUMMARY (all rows PASS): {json.dumps(rows)}")
    finally:
        await env.close()


# ── ticketmaster: checkout mints order.purchased_at — wall vs T (zip universe; cart emptied, re-add each phase). ──
_TICKETMASTER_CLOCK_DATA = {
    "users": [{"user_id": "persona_001", "name": "Alex Rivera", "email": "alex.rivera@email.com", "address": "12 Bay St", "city": "San Francisco", "state": "CA", "zip": "94105", "is_user": True}],
    "venues": [{"id": "VEN001", "name": "Chase Center", "city": "San Francisco", "state": "CA", "capacity": 18064, "venue_type": "arena", "address": "1 Warriors Way", "zip": "94158"}],
    "events": [{"id": "EVT001", "name": "Chappell Roan | Midwest Princess Tour", "date": "2019-08-01", "time": "20:00", "venue_id": "VEN001", "status": "on_sale", "min_price": 60.0, "max_price": 180.0, "segment": "Music", "genre": "Pop", "sub_genre": "Pop", "ticket_limit": 4, "on_sale_date": "2019-01-15", "presale_date": "2019-01-10"}],
    "ticket_types": [{"id": "TKT001", "event_id": "EVT001", "name": "General Admission", "price": 60.0, "quantity_total": 500, "quantity_available": 500, "section": "Floor GA"}],
    "orders": [{"id": "ORD001", "user_id": "persona_001", "event_id": "EVT001", "ticket_type_id": "TKT001", "quantity": 2, "subtotal": 120.0, "service_fee": 30.0, "order_processing_fee": 5.95, "total_price": 155.95, "confirmation_code": "TM-SEED01", "purchased_at": "2019-06-01T10:00:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ticketmaster_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("ticketmaster", "ticketmaster", "ticketmaster.zip", _data_json_zip(_TICKETMASTER_CLOCK_DATA))
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-ticketmaster")
        cart = {"event_id": "EVT001", "ticket_type_id": "TKT001", "quantity": 1}
        await _clock_call_tool(result.mcp_url, "ticketmaster_add_to_cart", cart)
        code_b = json.loads(await _clock_call_tool(result.mcp_url, "ticketmaster_checkout", {}))["orders"][0]["confirmation_code"]
        before = json.loads(await _clock_call_tool(result.mcp_url, "ticketmaster_get_order_details", {"identifier": code_b}))
        assert not before["order"]["purchased_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "ticketmaster" in {s["service"] for s in cfg["synced"]}, cfg
        await _clock_call_tool(result.mcp_url, "ticketmaster_add_to_cart", cart)
        code_a = json.loads(await _clock_call_tool(result.mcp_url, "ticketmaster_checkout", {}))["orders"][0]["confirmation_code"]
        after = json.loads(await _clock_call_tool(result.mcp_url, "ticketmaster_get_order_details", {"identifier": code_a}))
        assert after["order"]["purchased_at"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── tms: receive_shipment_request mints shipment.created_at — wall vs T (UPSERT by shipment_id, so distinct ids per phase). ──
_TMS_CLOCK_DATA = {
    "carriers": [{"id": "car_swift", "name": "Swift Freight", "scac": "SWFT", "modes": ["ltl", "ftl"], "service_levels": ["ground"], "contact_email": "ops@swift.example", "is_active": True, "created_at": "2019-01-01T00:00:00Z", "updated_at": "2019-01-01T00:00:00Z"}],
    "lanes": [{"id": "rate_atl_sfo", "carrier_id": "car_swift", "origin_zip": "30301", "dest_zip": "94102", "mode": "ltl", "service_level": "ground", "base_rate": 100.0, "per_lb_rate": 1.5, "fuel_surcharge_pct": 0.1, "min_charge": 50.0, "accessorials": {}, "transit_days": 4, "is_active": True, "created_at": "2019-01-01T00:00:00Z", "updated_at": "2019-01-01T00:00:00Z"}],
    "shipments": [{"id": "shp_before_t", "client_id": "cli_acme", "weight_lb": 500.0, "origin_zip": "30301", "dest_zip": "94102", "requested_service_level": "ground", "status": "received", "received_at": "2019-06-10T08:00:00Z", "created_at": "2019-06-10T08:00:00Z", "updated_at": "2019-06-10T08:00:00Z"}, {"id": "shp_after_t", "client_id": "cli_acme", "weight_lb": 750.0, "origin_zip": "30301", "dest_zip": "94102", "requested_service_level": "ground", "status": "received", "received_at": "2019-06-20T08:00:00Z", "created_at": "2019-06-20T08:00:00Z", "updated_at": "2019-06-20T08:00:00Z"}],
    "tenders": [], "bols": [], "tracking_events": [], "pods": [], "dock_appointments": [], "freight_invoices": [], "passthrough_invoices": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_tms_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("tms", "tms", "data.json", json.dumps(_TMS_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-tms")
        base = {"client_id": "cli_acme", "weight_lb": 42.0, "origin_zip": "30301", "dest_zip": "94102", "requested_service_level": "ground"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "tms_receive_shipment_request", {**base, "shipment_id": "shp_clock_before"}))
        assert not before["shipment"]["created_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "tms" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "tms_receive_shipment_request", {**base, "shipment_id": "shp_clock_after"}))
        assert after["shipment"]["created_at"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── trello: create_card mints dateLastActivity — wall vs T (zip universe, inner data.json). ──
_TRELLO_CLOCK_DATA = {
    "organizations": [{"id": "5cf000000000000000000001", "name": "acmegames", "displayName": "Acme Games", "desc": "", "website": None, "memberIds": ["5cf0000000000000000000a1"], "boardIds": ["5cf000000000000000000b01"]}],
    "boards": [{"id": "5cf000000000000000000b01", "name": "Product Roadmap", "desc": "Track features", "closed": False, "idOrganization": "5cf000000000000000000001", "url": "https://trello.com/b/roadmap", "shortUrl": "https://trello.com/b/roadmap", "nodeId": None, "dateLastActivity": "2019-01-10T09:00:00Z", "prefs": {}, "labelNames": {}}],
    "lists": [{"id": "5cf000000000000000000c01", "name": "Backlog", "idBoard": "5cf000000000000000000b01", "pos": 1024.0, "closed": False}],
    "cards": [{"id": "5cf000000000000000000d01", "name": "Seed card", "idBoard": "5cf000000000000000000b01", "idList": "5cf000000000000000000c01", "desc": "pre-existing", "pos": 1024.0, "closed": False, "due": None, "dueComplete": False, "idMembers": [], "idLabels": [], "idShort": 1, "shortLink": "seed0001", "shortUrl": "https://trello.com/c/seed0001", "url": "https://trello.com/c/seed0001", "dateLastActivity": "2019-01-10T09:00:00Z"}],
    "members": [{"id": "5cf0000000000000000000a1", "username": "jdoe", "fullName": "Jane Doe", "initials": "JD", "memberType": "normal", "confirmed": True}],
    "labels": [], "checklists": [], "check_items": [], "attachments": [], "actions": [],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_trello_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("trello", "trello", "trello.zip", _data_json_zip(_TRELLO_CLOCK_DATA))
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-trello")
        args = {"idList": "5cf000000000000000000c01", "name": "Clock probe card"}
        before = json.loads(await _clock_call_tool(result.mcp_url, "trello_create_card", args))
        assert not before["dateLastActivity"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "trello" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "trello_create_card", args))
        assert after["dateLastActivity"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── wms: receive_asn mints asn.created_at — wall vs T. ──
_WMS_CLOCK_DATA = {
    "clients": [{"id": "cli_acme", "name": "Acme Retail", "segment": "apparel", "billing_terms": "net30", "is_active": True, "created_at": "2019-06-01T00:00:00Z", "updated_at": "2019-06-01T00:00:00Z"}],
    "skus": [{"id": "sku_tee", "client_id": "cli_acme", "description": "Acme Cotton Tee", "uom": "each", "created_at": "2019-06-01T00:00:00Z", "updated_at": "2019-06-01T00:00:00Z"}],
}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_wms_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("wms", "wms", "data.json", json.dumps(_WMS_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-wms")
        args = {"client_id": "cli_acme", "lines": [{"sku_id": "sku_tee", "qty_expected": 5}]}
        before = json.loads(await _clock_call_tool(result.mcp_url, "wms_receive_asn", args))
        assert not before["asn"]["created_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "wms" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "wms_receive_asn", args))
        assert after["asn"]["created_at"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fleet PHASE B batch 7 (FINAL): zendesk (MED write-stamp) + weather,
# eight_sleep, whoop (NONE — no now-observable tool; advertisement + sync-participation only).
# ─────────────────────────────────────────────────────────────────────────────

# ── zendesk: create_ticket mints created_at — wall vs T. ──
_ZENDESK_CLOCK_DATA = {"users": [{"id": 101, "name": "Alice Requester", "email": "alice@example.com", "role": "end-user"}]}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_zendesk_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("zendesk", "zendesk", "data.json", json.dumps(_ZENDESK_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-zendesk")
        args = {"subject": "Clock probe ticket", "description": "Write-stamp probe for clock/v1", "requester_id": 101}
        before = json.loads(await _clock_call_tool(result.mcp_url, "zendesk_create_ticket", args))
        assert not before["created_at"].startswith("2019"), before
        cfg = await _arm_gateway_clock(env, result)
        assert "zendesk" in {s["service"] for s in cfg["synced"]}, cfg
        after = json.loads(await _clock_call_tool(result.mcp_url, "zendesk_create_ticket", args))
        assert after["created_at"].startswith("2019-06-15"), after
    finally:
        await env.close()


# ── NONE servers: no now-observable tool. Assert the card advertises clock/v1 and the
#    gateway can sync it (cfg['synced'] includes the env) — off-by-default until armed. ──
_EIGHT_SLEEP_CLOCK_DATA = {"users": [{"persona_id": "persona_001", "email": "john.doe@email.com", "first_name": "John", "last_name": "Doe", "timezone": "America/Los_Angeles", "temperature_unit": "f", "bed_side": "left", "is_user": True, "password": "test-password"}]}


@pytest.mark.asyncio
@pytest.mark.integration
async def test_weather_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("weather", "weather", "weather.json", json.dumps({}).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-weather")
        cfg = await _arm_gateway_clock(env, result)
        assert "weather" in {s["service"] for s in cfg["synced"]}, cfg
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_eight_sleep_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("eight_sleep", "eight_sleep", "eight_sleep.json", json.dumps(_EIGHT_SLEEP_CLOCK_DATA).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-eight_sleep")
        cfg = await _arm_gateway_clock(env, result)
        assert "eight_sleep" in {s["service"] for s in cfg["synced"]}, cfg
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_whoop_clock_consumer(sandbox_provider):
    spec = _ClockServerSpec("whoop", "whoop", "data.json", json.dumps({}).encode())
    env, result = await _deploy_clock_env(spec)
    try:
        await _assert_advertises_clock(result, "mcp-whoop")
        cfg = await _arm_gateway_clock(env, result)
        assert "whoop" in {s["service"] for s in cfg["synced"]}, cfg
    finally:
        await env.close()
