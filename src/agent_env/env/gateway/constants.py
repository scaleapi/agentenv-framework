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
# A single component of a large universe can take close to 500s, so this leaves
# wide headroom.
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
# a blob-heavy service can move GBs in about a minute while a record-heavy one takes minutes
# for well under 1GB. So size is a weak proxy -- it over-grants to the big-blob
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
CONTROL_TOKEN_ENV = "AGENTENV_CONTROL_TOKEN"
# The computers that follow the gateway's clock, each as `name=read token`, comma-separated; TOKEN in the name keeps
# it out of logged compose files.
CLOCK_FOLLOWERS_ENV = "CLOCK_FOLLOWER_TOKENS"
# How long /clock/watch holds a watch naming the current version before answering it unchanged; a follower's request
# timeout must be longer. The card's watch method advertises it as hold_s.
CLOCK_WATCH_HOLD_S = 25.0
# The longest a clock write's wait_followers_s may ask it to wait for the followers.
CLOCK_MAX_WAIT_FOLLOWERS_S = 30.0

# Env/gateway-level extensions the gateway itself owns (tool gating + triggers).
EXT_DISABLE_TOOL_URI = "urn:agentenv:disable-tool/v1"
EXT_ENABLE_TOOL_URI = "urn:agentenv:enable-tool/v1"
EXT_TRIGGERS_URI = "urn:agentenv:triggers/v1"
EXT_CLOCK_URI = "urn:agentenv:clock/v1"
EXT_TRAJECTORY_URI = "urn:agentenv:trajectory/v1"
EXT_STEP_URI = "urn:agentenv:step/v1"
EXT_STATE_URI = "urn:agentenv:state/v1"
EXT_CHANGES_URI = "urn:agentenv:changes/v1"
EXT_NOTIFICATIONS_URI = "urn:agentenv:notifications/v1"
EXT_USER_CHAT_URI = "urn:agentenv:user-chat/v1"
EXT_WEBSITE_URI = "urn:agentenv:website/v1"
EXT_BROWSER_URI = "urn:agentenv:browser/v1"
EXT_COMPUTER_APP_URI = "urn:agentenv:computer-app/v1"
WEBSITE_READY_OP = "ready"
WEBSITE_BROWSE_URL_PARAM = "browse_url"

# Rows per GET /changes page: the default, and the most a client may ask for.
CHANGES_PAGE_DEFAULT = 1000
CHANGES_PAGE_MAX = 10000

# Gateway's compose service name — how backing servers reach it to build env_get_time_url (compose path).
GATEWAY_COMPOSE_HOST = "gateway"

# Per-child card fetch budget (seconds) when composing the env card.
CARD_FETCH_TIMEOUT_S = 5.0

# Per-child tools/list budget (seconds) during discovery. Wider than a card fetch: the child answers on the
# connection every caller shares, so a large service busy with another call can take a while, and a timeout
# here fails the whole discovery, to be retried on the next connect.
CHILD_DISCOVERY_TIMEOUT_S = 30.0

# The GET /state field naming each tool name more than one source serves, with its sources: backing servers by name,
# `gateway` for the gateway's own tools. Empty unless discovery found such a clash, which leaves the env serving none
# of its servers' tools; a deploy reads it to fail with the clash instead.
STATE_TOOL_NAME_CLASHES_KEY = "tool_name_clashes"


def tool_name_clashes_text(clashes: dict[str, list[str]]) -> str:
    """Each clashing name and its sources, e.g. ``'list_items' from 'items' and 'stock'``."""
    def sources(names: list[str]) -> str:
        quoted = [f"'{n}'" for n in names]
        return f"{', '.join(quoted[:-1])} and {quoted[-1]}" if len(quoted) > 1 else quoted[0]
    return "; ".join(f"'{name}' from {sources(names)}" for name, names in clashes.items())

# Trigger status vocabulary, part of the GET /triggers/state contract; `state` asserts against it.
TRIGGER_STATUSES = ("armed", "firing", "queued", "fired", "failed", "cancelled")

# A fire is in flight; `queued` also has further provoking calls already waiting to be drained.
TRIGGER_IN_FLIGHT_STATUSES = ("firing", "queued")

# What triggers/v1 takes beyond its first release, declared on the card so a client can refuse an older gateway.
TRIGGER_NAMED_EXECUTORS = "named_executors"
TRIGGER_PROVOKING_ROLES = "provoking_roles"
TRIGGER_ACTION_DELAY = "action_delay"
TRIGGER_EXECUTOR_HOURS = "executor_hours"
TRIGGER_UNLESS = "unless"
TRIGGER_WEBSITE = "website"
TRIGGER_EXECUTOR_HOURS_TIME_ZONE = "executor_hours_time_zone"
TRIGGER_NEXT_EVENT_AT = "next_event_at"
TRIGGER_NOTIFY = "notify"
TRIGGER_WEBSITE_OP = "website_op"
# state reports each fire in flight (running, waits) and each executor call in progress (executor_calls).
TRIGGER_FIRES_IN_STATE = "fires_in_state"
TRIGGER_FEATURES = (TRIGGER_NAMED_EXECUTORS, TRIGGER_PROVOKING_ROLES, TRIGGER_ACTION_DELAY, TRIGGER_EXECUTOR_HOURS,
                    TRIGGER_UNLESS, TRIGGER_WEBSITE, TRIGGER_EXECUTOR_HOURS_TIME_ZONE, TRIGGER_NEXT_EVENT_AT,
                    TRIGGER_NOTIFY, TRIGGER_WEBSITE_OP, TRIGGER_FIRES_IN_STATE)

# The one `source` a /step call may name instead of step_api, with the id of the hook artifact that made it.
STEP_SOURCE_POST_TURN_HOOK = "post_turn_hook"
# What step/v1 takes beyond its first release, declared on the card as triggers/v1 declares its features.
STEP_SOURCE = "source"
STEP_FEATURES = (STEP_SOURCE,)

GATEWAY_TRAJECTORY_FILE = os.environ.get(
    "GATEWAY_TRAJECTORY_FILE",
    f"/tmp/agentenv/{uuid4().hex[:8]}-trajectory.jsonl",
)

AGENT_ENV_ROLE_HEADER = "AgentEnv-Role"
# Sent by a computer's site proxy, which serves a website's route at the root of the site's own name.
AGENT_ENV_SITE_ROOT_HEADER = "AgentEnv-Site-Root"
GATEWAY_PROVIDER_HEADERS_ENV = "AGENTENV_PROVIDER_REQUEST_HEADERS"
# Each synthetic site's names as its env stores them, `shop=shop.test|www.shop.test,...`, one of which a site takes as
# its Host when a client reaches it there; and the sites list_website_urls gives at their first name, `shop,docs`.
WEBSITE_NAMES_ENV = "WEBSITE_NAMES"
WEBSITE_LISTED_BY_NAME_ENV = "WEBSITE_LISTED_BY_NAME"
# MCP request `_meta` keys the gateway stamps on every child tools/call: the caller's role, and on the MCP path a
# key naming the inbound session, so a child can keep per-caller state behind the gateway's one shared session.
AGENT_ENV_ROLE_META_KEY = "agentenv.io/role"
AGENT_ENV_SESSION_META_KEY = "agentenv.io/session"
# MCP tools/list `_meta` key marking a tool hidden from every role until a rule names it; "*"-tool rules skip it.
AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY = "agentenv.io/hiddenByDefault"
DEFAULT_ROLE = "default"
WILDCARD = "*"
TOOL_DISABLE_ACTION = "disable"
TOOL_ENABLE_ACTION = "enable"
# The harness's routes that take the deploy's control token: the ones that change tool access, triggers and
# notifications. Every other route stays open for now, the clock's writes included: callers outside agent-env set the
# clock over plain HTTP and don't send the token yet.
CONTROL_ROUTES = frozenset({
    f"/tools/{TOOL_DISABLE_ACTION}", f"/tools/{TOOL_ENABLE_ACTION}",
    "/triggers/register", "/triggers/remove", "/triggers/clear",
    "/notifications/config", "/notifications/take", "/notifications/clear",
})


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
    "properties": {"action": {"enum": ["list_tools", "call_tool"]}, "tool_name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"},
                   "source": {"enum": [STEP_SOURCE_POST_TURN_HOOK]}, "hook_artifact_id": {"type": "string", "minLength": 1}},
    "required": ["action"],
    "if": {"properties": {"action": {"const": "call_tool"}}},
    "then": {"required": ["tool_name"]},
}
_WAIT_FOLLOWERS = {"wait_followers_s": {"type": "number", "minimum": 0, "maximum": CLOCK_MAX_WAIT_FOLLOWERS_S}}

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
        "description": "Register (additive add-by-id), remove, clear, or read (state) typed Action/State triggers + the firing log (harness control path). config sets the engine's opt-in setting concurrent_fires (a repeat trigger whose actions are all contact replies (nl) or notifications (notify) runs its fires side by side instead of queueing).",
        "params": {"endpoint": "/triggers/register", "features": list(TRIGGER_FEATURES), "methods": {
            "register": {"method": "POST", "endpoint": "/triggers/register", "request": {"type": "object", "properties": {"watch_roles": {"type": "array", "items": {"type": "string"}}, "executor": {"type": "object"}, "executors": {"type": "object"}, "triggers": {"type": "array"}}, "required": ["triggers"]}},
            "config": {"method": "POST", "endpoint": "/triggers/config", "request": {"type": "object", "properties": {"concurrent_fires": {"type": "boolean"}}}},
            "remove": {"method": "POST", "endpoint": "/triggers/remove", "request": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}}, "required": ["ids"]}},
            "clear": {"method": "POST", "endpoint": "/triggers/clear", "request": {"type": "object", "properties": {}}},
            "state": {"method": "GET", "endpoint": "/triggers/state", "request": {"type": "object", "properties": {}}}}},
        "required": False,
    },
    {
        "uri": EXT_CLOCK_URI,
        "description": "Virtual clock (urn:agentenv:clock/v1): read the gateway's virtual time (wall-clock-derived, rate-scaled) via get_time; set_time/set_rate/advance/clear are the harness control path, each changing the clock's opaque version and, given wait_followers_s, answering 504 naming the followers that don't show it in time (set_rate keeps t0, as advance does, so time triggers keep their marks; rate 0 freezes the clock); a follower follows it through watch, a long poll naming the version it shows and held up to hold_s; state's followers block is diagnostic, not part of the contract.",
        "params": {"endpoint": "/clock/time", "methods": {
            "get_time": {"method": "GET", "endpoint": "/clock/time"},
            "set_time": {"method": "PUT", "endpoint": "/clock/set-time", "request": {"type": "object", "properties": {"virtual_time": {"type": "string"}, "virtual_seconds_per_real_second": {"type": "number"}, **_WAIT_FOLLOWERS}, "required": ["virtual_time"]}},
            "set_rate": {"method": "PUT", "endpoint": "/clock/set-rate", "request": {"type": "object", "properties": {"virtual_seconds_per_real_second": {"type": "number", "minimum": 0}, **_WAIT_FOLLOWERS}, "required": ["virtual_seconds_per_real_second"]}},
            "advance": {"method": "PUT", "endpoint": "/clock/advance", "request": {"type": "object", "properties": {"virtual_time": {"type": "string"}, **_WAIT_FOLLOWERS}, "required": ["virtual_time"]}},
            "clear": {"method": "POST", "endpoint": "/clock/clear", "request": {"type": "object", "properties": {**_WAIT_FOLLOWERS}}},
            "state": {"method": "GET", "endpoint": "/clock/state", "request": {"type": "object", "properties": {}}},
            "watch": {"method": "GET", "endpoint": "/clock/watch", "hold_s": CLOCK_WATCH_HOLD_S, "request": {"type": "object", "properties": {"after": {"type": "string"}, "follower": {"type": "string"}}}}}},
        "required": False,
    },
    {
        "uri": EXT_TRAJECTORY_URI,
        "description": "Env trajectory (urn:agentenv:trajectory/v1): the env's append-only JSONL record of tool calls, results, trigger firings and the requests that may write through its REST proxy, each write with the changelog ids it spanned, virtual_time-stamped while the clock is armed.",
        "params": {"endpoint": "/trajectory", "methods": {
            "get": {"method": "GET", "endpoint": "/trajectory"}}},
        "required": False,
    },
    {
        "uri": EXT_STEP_URI,
        "description": "Step (urn:agentenv:step/v1): list or call the env's tools over REST, filtered by the AgentEnv-Role header; a call a post-turn hook makes names it, and the trajectory records it as the hook's.",
        "params": {"endpoint": "/step", "features": list(STEP_FEATURES), "methods": {
            "step": {"method": "POST", "endpoint": "/step", "request": _STEP_REQUEST}}},
        "required": False,
    },
    {
        "uri": EXT_STATE_URI,
        "description": "State (urn:agentenv:state/v1): the env's MCP servers and their tools, any tool name more than one of them serves, its changelog id and its role rules.",
        "params": {"endpoint": "/state", "methods": {
            "get": {"method": "GET", "endpoint": "/state"}}},
        "required": False,
    },
    {
        "uri": EXT_CHANGES_URI,
        "description": "Changes (urn:agentenv:changes/v1): the env's changelog rows after after_id, oldest first, each with the actors whose write window holds it: the role, the tool call, trigger action or request, and the trigger fires it was made for. None for a write the gateway did not serve; several for one written while calls overlapped.",
        "params": {"endpoint": "/changes", "methods": {
            "get": {"method": "GET", "endpoint": "/changes", "request": {"type": "object", "properties": {
                "after_id": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": CHANGES_PAGE_MAX}}}}}},
        "required": False,
    },
    {
        "uri": EXT_NOTIFICATIONS_URI,
        "description": "Notifications (urn:agentenv:notifications/v1): what comes due for a role between its turns, on the virtual clock, each for one recipient (`to`): checks a role scheduled for itself (schedule_check, which only the roles config names see) and notifications triggers sent (the notify action, to the default role unless it names another). state says when the next is due; take hands a recipient each due one once (harness control path), every item one shape: id, to, at, times, source (scheduled_check or trigger), source_id, channel (the label the agent sees) and parts. config replaces all settings (omitted fields reset to defaults, so one without roles turns checks off) and leaves existing checks as they are unless it drops their role: lowering max_active leaves extra active checks in place, and raising min_interval leaves existing repeats unchanged. clear resets the settings to their defaults and drops every check and notification. Unrelated to A2A push notifications.",
        "params": {"endpoint": "/notifications/state", "methods": {
            "config": {"method": "POST", "endpoint": "/notifications/config", "request": {"type": "object", "properties": {
                "roles": {"type": "array", "items": {"type": "string"}}, "max_active": {"type": "integer", "minimum": 1},
                "min_interval": {"type": "string"}}}},
            "state": {"method": "GET", "endpoint": "/notifications/state", "request": {"type": "object", "properties": {
                "to": {"type": "string"}}}},
            "take": {"method": "POST", "endpoint": "/notifications/take", "request": {"type": "object", "properties": {
                "to": {"type": "string"}, "until": {"type": "string"}}}},
            "clear": {"method": "POST", "endpoint": "/notifications/clear", "request": {"type": "object", "properties": {}}}}},
        "required": False,
    },
    {
        "uri": EXT_USER_CHAT_URI,
        "description": "User chat (urn:agentenv:user-chat/v1): the message_user tool, for the roles config names, is how an always-on agent reaches its user; each message is queued with its virtual time until the harness takes it. With stay_quiet the roles also get stay_quiet, which sends nothing and is recorded for the next take. Built in: role rules don't hide them from a configured role.",
        "params": {"endpoint": "/user_chat/state", "methods": {
            "config": {"method": "POST", "endpoint": "/user_chat/config", "request": {"type": "object", "properties": {
                "roles": {"type": "array", "items": {"type": "string"}}, "user_name": {"type": "string"},
                "stay_quiet": {"type": "boolean"}}}},
            "state": {"method": "GET", "endpoint": "/user_chat/state", "request": {"type": "object", "properties": {}}},
            "take": {"method": "POST", "endpoint": "/user_chat/take", "request": {"type": "object", "properties": {
                "take_id": {"type": "string"}, "from": {"type": "string"}}}},
            "clear": {"method": "POST", "endpoint": "/user_chat/clear", "request": {"type": "object", "properties": {}}}}},
        "required": False,
    },
]
