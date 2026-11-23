"""One gateway topology with every kind of service a compose file renders, for the default-stack golden and the slot
compose tests: MCP servers on and off the database, a website browser on the sites network, websites with and without
a frontend, a live site, a sidecar, an external child, pgweb and db-mcp."""

from __future__ import annotations

from unittest.mock import patch

from agent_env.env.envs.service_db import ServiceDBConfig
from agent_env.env.gateway import ExternalChild
from agent_env.providers.env_providers import env_gateway_provider
from agent_env.providers.env_providers.env_gateway_provider import (
    EnvironmentGatewayProvider,
    MCPServerConfig,
    SidecarConfig,
    WebsiteConfig,
)
from agent_env.providers.env_state import LocalPostgresStateProvider

# A website backend's session secret is drawn per render; the golden holds this one.
FIXED_SESSION_SECRET = "0" * 64


class _Image:
    def __init__(self, image_name: str):
        self.image_name = image_name


def render(**overrides) -> str:
    """The fixture topology's compose, as ``create_docker_compose`` renders it with ``overrides``."""
    kwargs = dict(
        mcp_servers=[
            MCPServerConfig(image="mcp-slack", environment_name="slack", extra_env_vars={"SLACK_MODE": "mock"}),
            MCPServerConfig(image="mcp-calc", environment_name="calc"),
            MCPServerConfig(image="playwright-mcp", environment_name="browser", networks=("website-network",)),
        ],
        gateway_image="agent-gateway",
        gateway_port=18765,
        website_configs=[
            WebsiteConfig(backend_image="shop-be", frontend_image="shop-fe", environment_name="shop",
                          domains=("shop.example.com",), listed_by_name=True),
            WebsiteConfig(backend_image="blog-be", frontend_image=None, environment_name="blog", shm_mb=256),
            WebsiteConfig(backend_image="feed-be", frontend_image=None, environment_name="feed", live=True,
                          settings={"mode": "live"}),
        ],
        sidecars=[SidecarConfig(environment_name="relay", image_artifact=_Image("ecr/relay:1.0.4"),
                                container_port=8000, host_port=18768, env={"RELAY_MODE": "remote"})],
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig(db_web_image="agent-env-db-web", db_mcp_image="agent-env-db-mcp")),
        state_instance=LocalPostgresStateProvider.default_instance(),
        host_port=lambda port: port + 1000,
        host_ips=("10.0.0.5",),
        extra_hosts=("host.docker.internal:host-gateway",),
        mcp_server_name="crm1234",
        external_children=(ExternalChild(name="desk", mcp_url="https://desk.example.com/mcp",
                                         base_url="https://desk.example.com"),),
        control_token="ctl-token",
    )
    kwargs.update(overrides)
    with patch.object(env_gateway_provider.secrets, "token_hex", return_value=FIXED_SESSION_SECRET):
        return EnvironmentGatewayProvider().create_docker_compose(**kwargs)
