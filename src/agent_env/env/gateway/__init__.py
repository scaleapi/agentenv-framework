"""The gateway's client-facing vocabulary. The server class lives in ``.gateway`` and is
container code: it needs the Postgres driver, so it is deliberately not re-exported here."""

from dataclasses import dataclass

from .constants import (
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


__all__ = [
    "AGENT_ENV_GATEWAY_MCP_PORT",
    "AGENT_ENV_ROLE_HEADER",
    "AGENT_ENV_ROLE_META_KEY",
    "AGENT_ENV_SESSION_META_KEY",
    "DEFAULT_ROLE",
    "GATEWAY_TRAJECTORY_FILE",
    "GatewayMode",
    "InternalMCPServer",
    "TOOL_DISABLE_ACTION",
    "TOOL_ENABLE_ACTION",
    "WILDCARD",
]
