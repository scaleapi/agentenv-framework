"""Gateway constants shared with the data-plane client call sites.

Lives inside the gateway package so gateway.py can reach it with a relative
import — the gateway runs as a standalone container with no `agent_env` package.
Client call sites (MCPServerEnv/WebsiteEnv) import it by absolute path. One
source of truth keeps the layers ordered: client add_data >= gateway REST
proxy >= actual load time.
"""

import os
from enum import Enum
from uuid import uuid4
import secrets

# Max end-to-end time for a single service's data load (seconds).
#
# The previous value of 600 was sized above a then-slowest prod load of ~430s.
# That is no longer the ceiling: a large universe's gmail component now takes 498s and slack
# 499s, leaving under 20% headroom on a number chosen to have plenty.
#
# The margin matters more than the average because of how a load fails. A
# component that exceeds this does not retry — the environment comes up, reports
# healthy, and serves zero rows, which is indistinguishable from a component that
# is legitimately empty. Paying a few extra minutes on a genuine hang is much
# cheaper than that.
#
# The real fix is capacity, not patience: a universe loads all its components
# through one asyncio.gather onto a VM defaulting to a single vCPU, and the
# loaders are CPU-bound. This value should come back down once that is sized
# properly.
DATA_PLANE_LOAD_TIMEOUT_S = 900

# Extra seconds granted per GB of payload, on top of the floor.
#
# Explicitly a heuristic, not a model. Measured load cost tracks RECORD COUNT, not bytes:
# github moved 3.5GB in 89s while gmail took 206s for 409MB and slack 207s for 86.6MB (~20x
# and ~94x worse per byte). So size is a weak proxy -- it over-grants to the big-blob
# services that never needed it and under-grants to the small record-heavy ones that do.
# It is still strictly better than a flat cap, because the flat cap's failure mode was
# killing healthy work. A real fix is a stall timeout (fail on no progress rather than on
# elapsed time), which needs the server to report progress mid-ingest.
DATA_PLANE_LOAD_TIMEOUT_PER_GB_S = 1200

# Ceiling, so a genuinely hung load on a huge payload still surfaces in bounded time
# instead of holding a worker slot and a billable VM for hours.
DATA_PLANE_LOAD_TIMEOUT_MAX_S = 3600


def data_plane_load_timeout_s(size_bytes: int | None) -> int:
    """Read timeout for one service's data load, scaled by payload size.

    ``None``/0 (size unknown) yields the floor, which is today's behaviour -- an
    unmeasurable payload must not silently get an unbounded wait.
    """
    if not size_bytes or size_bytes <= 0:
        return DATA_PLANE_LOAD_TIMEOUT_S
    gb = size_bytes / 1_000_000_000
    return int(min(
        DATA_PLANE_LOAD_TIMEOUT_MAX_S,
        DATA_PLANE_LOAD_TIMEOUT_S + gb * DATA_PLANE_LOAD_TIMEOUT_PER_GB_S,
    ))

# Hand-rolled mirror of agentenv_protocol.types — the gateway container has no
# agentenv_protocol dependency. Keep byte-identical to that module.
WELL_KNOWN_PATH = "/.well-known/agent-env.json"
RPC_PATH = "/agentenv"
PROTOCOL_VERSION = "1.0"
METHOD_RESET = "data/reset"
METHOD_ADD = "data/add"
METHOD_GET = "data/get"
MCP_TRANSPORT = "mcp"

# Env/gateway-level extensions the gateway itself owns (tool gating + triggers).
EXT_DISABLE_TOOL_URI = "urn:agentenv:disable-tool/v1"
EXT_ENABLE_TOOL_URI = "urn:agentenv:enable-tool/v1"
EXT_TRIGGERS_URI = "urn:agentenv:triggers/v1"
EXT_CLOCK_URI = "urn:agentenv:clock/v1"
EXT_TRAJECTORY_URI = "urn:agentenv:trajectory/v1"
EXT_STEP_URI = "urn:agentenv:step/v1"
EXT_STATE_URI = "urn:agentenv:state/v1"

# Gateway's compose service name — how backing servers reach it to build env_get_time_url (compose path).
GATEWAY_COMPOSE_HOST = "gateway"

# Per-child card fetch budget (seconds) when composing the env card.
CARD_FETCH_TIMEOUT_S = 5.0

# Trigger status vocabulary, part of the GET /triggers/state contract; `state` asserts against it.
TRIGGER_STATUSES = ("armed", "firing", "queued", "fired", "failed")

# A fire is in flight; `queued` also has further provoking calls already waiting to be drained.
TRIGGER_IN_FLIGHT_STATUSES = ("firing", "queued")

GATEWAY_TRAJECTORY_FILE = os.environ.get(
    "GATEWAY_TRAJECTORY_FILE",
    f"/tmp/agentenv/{uuid4().hex[:8]}-trajectory.jsonl",
)

AGENT_ENV_ROLE_HEADER = "AgentEnv-Role"
DEFAULT_ROLE = "default"
WILDCARD = "*"
TOOL_DISABLE_ACTION = "disable"
TOOL_ENABLE_ACTION = "enable"


class GatewayMode(str, Enum):
    PERFORMANCE = "performance"
    CONSISTENT = "consistent"


DEFAULT_MCP_SERVER_NAME = "env"


def random_mcp_server_name() -> str:
    """`env` + 4 random digits: undeclared envs on one agent never collide; declare a MultiEnv name for a stable one."""
    return f"env{secrets.randbelow(10000):04d}"


# The gateway's own extensions, as its env card advertises them; here so kernel code can read them without importing gateway.py.
_TOOLS_REQUEST = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "tools": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"const": WILDCARD}]},
    },
    "required": ["role", "tools"],
}
# call_tool also needs a tool_name; if/then rather than oneOf keeps the fields top-level, where card viewers read them
_STEP_REQUEST = {
    "type": "object",
    "properties": {"action": {"enum": ["list_tools", "call_tool"]}, "tool_name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"}},
    "required": ["action"],
    "if": {"properties": {"action": {"const": "call_tool"}}},
    "then": {"required": ["tool_name"]},
}
GATEWAY_EXTENSIONS = [
    {
        "uri": EXT_DISABLE_TOOL_URI,
        "description": "Disable tools for a role.",
        "params": {"endpoint": f"/tools/{TOOL_DISABLE_ACTION}", "methods": {TOOL_DISABLE_ACTION: {"method": "POST", "request": _TOOLS_REQUEST}}},
        "required": False,
    },
    {
        "uri": EXT_ENABLE_TOOL_URI,
        "description": "Enable tools for a role.",
        "params": {"endpoint": f"/tools/{TOOL_ENABLE_ACTION}", "methods": {TOOL_ENABLE_ACTION: {"method": "POST", "request": _TOOLS_REQUEST}}},
        "required": False,
    },
    {
        "uri": EXT_TRIGGERS_URI,
        "description": "Register (additive add-by-id), remove, clear, or read (state) typed Action/State triggers + the firing log (harness control path).",
        "params": {"endpoint": "/triggers/register", "methods": {
            "register": {"method": "POST", "endpoint": "/triggers/register", "request": {"type": "object", "properties": {"watch_roles": {"type": "array", "items": {"type": "string"}}, "executor": {"type": "object"}, "triggers": {"type": "array"}}, "required": ["triggers"]}},
            "remove": {"method": "POST", "endpoint": "/triggers/remove", "request": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}}, "required": ["ids"]}},
            "clear": {"method": "POST", "endpoint": "/triggers/clear", "request": {"type": "object", "properties": {}}},
            "state": {"method": "GET", "endpoint": "/triggers/state", "request": {"type": "object", "properties": {}}}}},
        "required": False,
    },
    {
        "uri": EXT_CLOCK_URI,
        "description": "Virtual clock (urn:agentenv:clock/v1): read the gateway's virtual time (wall-clock-derived, rate-scaled) via get_time; set-time/clear are the harness control path.",
        "params": {"endpoint": "/clock/time", "methods": {
            "get_time": {"method": "GET", "endpoint": "/clock/time"},
            "set_time": {"method": "PUT", "endpoint": "/clock/set-time", "request": {"type": "object", "properties": {"virtual_time": {"type": "string"}, "virtual_seconds_per_real_second": {"type": "number"}}, "required": ["virtual_time"]}},
            "clear": {"method": "POST", "endpoint": "/clock/clear", "request": {"type": "object", "properties": {}}},
            "state": {"method": "GET", "endpoint": "/clock/state", "request": {"type": "object", "properties": {}}}}},
        "required": False,
    },
    {
        "uri": EXT_TRAJECTORY_URI,
        "description": "Env trajectory (urn:agentenv:trajectory/v1): the env's append-only JSONL record of tool calls, results, and trigger firings, virtual_time-stamped while the clock is armed.",
        "params": {"endpoint": "/trajectory", "methods": {
            "get": {"method": "GET", "endpoint": "/trajectory"}}},
        "required": False,
    },
    {
        "uri": EXT_STEP_URI,
        "description": "Step (urn:agentenv:step/v1): list or call the env's tools over REST, filtered by the AgentEnv-Role header.",
        "params": {"endpoint": "/step", "methods": {
            "step": {"method": "POST", "endpoint": "/step", "request": _STEP_REQUEST}}},
        "required": False,
    },
    {
        "uri": EXT_STATE_URI,
        "description": "State (urn:agentenv:state/v1): the env's MCP servers and their tools, its changelog id and its role rules.",
        "params": {"endpoint": "/state", "methods": {
            "get": {"method": "GET", "endpoint": "/state"}}},
        "required": False,
    },
]
