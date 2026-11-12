"""The gateway's client-facing vocabulary. The server class lives in ``.gateway`` and is
container code: it needs the Postgres driver, so it is deliberately not re-exported here."""

from dataclasses import dataclass

from .constants import (
    AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY,
    AGENT_ENV_ROLE_HEADER,
    AGENT_ENV_ROLE_META_KEY,
    AGENT_ENV_SESSION_META_KEY,
    DEFAULT_ROLE,
    GATEWAY_TRAJECTORY_FILE,
    TOOL_DISABLE_ACTION,
    TOOL_ENABLE_ACTION,
    WILDCARD,
    GatewayMode,
)

AGENT_ENV_GATEWAY_MCP_PORT = 18765 # Reserved port for MCP gateway communications


@dataclass
class InternalMCPServer:
    """Configuration for an internal MCP server to proxy tools from."""

    name: str
    mcp_url: str


@dataclass(frozen=True)
class ExternalChild:
    """A child env on a sandbox of its own, which the gateway fronts as it does an MCP server: its name, and its MCP
    endpoint and base URL as the process holding it reaches them. A child that is a clock follower has the read token its
    clock watch carries, and the gateway's clock writes wait for it."""

    name: str
    mcp_url: str
    base_url: str
    clock_read_token: str | None = None


__all__ = [
    "AGENT_ENV_GATEWAY_MCP_PORT",
    "AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY",
    "AGENT_ENV_ROLE_HEADER",
    "AGENT_ENV_ROLE_META_KEY",
    "AGENT_ENV_SESSION_META_KEY",
    "DEFAULT_ROLE",
    "ExternalChild",
    "GATEWAY_TRAJECTORY_FILE",
    "GatewayMode",
    "InternalMCPServer",
    "TOOL_DISABLE_ACTION",
    "TOOL_ENABLE_ACTION",
    "WILDCARD",
]
