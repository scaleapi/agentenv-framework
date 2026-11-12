"""The user chat (urn:agentenv:user-chat/v1): how an always-on agent reaches its user.

The roles a task names in `config` get `message_user`, and with `stay_quiet: true` also `stay_quiet`. Each message is
queued with the virtual time it was sent at; nothing here reaches the user by itself: the always_on step `take`s what
was sent after each agent turn and hands it to the user sim when the user would read it. Whatever else the agent writes
(its plain reply) never reaches the user. The tools are built in: once a role is named, role rules from
modify_env_tool_access don't hide them.

`take` hands each message out once: {"id", "from", "at", "text"}, plus, with stay_quiet on, the `quiet` calls since the
last take. A `take_id` makes a retry safe. With `from` (a role) it takes only that role's messages and quiet calls, and
leaves the others queued.
"""
from __future__ import annotations

import threading
from typing import Any, Callable

from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from .caller import Caller
from .clock import Clock, _iso

MESSAGE_USER_TOOL_NAME = "message_user"
STAY_QUIET_TOOL_NAME = "stay_quiet"
_MAX_TEXT_CHARS = 20000
_MAX_TAKE_ID_CHARS = 128
_KEPT_TAKES = 64
_CONFIG_FIELDS = ("roles", "user_name", "stay_quiet")

MESSAGE_USER_DESCRIPTION = (
    "Send a chat message to {user}, the person you work for. This is the only way they see anything from you: your "
    "normal replies are kept in a log they never read. Send a message when there is something they need to know or "
    "decide; send nothing when there isn't. You can send several messages in one turn. They read messages when "
    "they're free, so a reply can take a while; you are woken when it arrives."
)
STAY_QUIET_DESCRIPTION = (
    "Call this when there is nothing worth telling {user} right now. It sends nothing; it records that you chose to "
    "stay quiet this time."
)
_QUIET_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "The message, as you'd text it."}},
    "required": ["text"],
    "additionalProperties": False,
}


# Read-only for the world: they queue a message (or a note) for the harness and change no app's data, so, like get_time,
# they don't make the trigger engine re-evaluate every armed state trigger (TriggerEngine._is_readonly). An action
# trigger on either still fires.
_ANNOTATIONS = ToolAnnotations(readOnlyHint=True)


class UserChatError(ValueError):
    """A bad config, take or tool argument."""


class UserChat:
    """The queue of messages roles sent their user; threadsafe."""

    def __init__(self, clock: Callable[[], Clock]):
        self._clock_of = clock
        self._lock = threading.Lock()
        self._roles: tuple[str, ...] = ()
        self._user_name = "your user"
        self._queue: list[dict] = []
        self._sent = 0
        self._takes: dict[str, dict] = {}
        self._stay_quiet = False
        self._quiet: list[dict] = []

    @property
    def roles(self) -> tuple[str, ...]:
        return self._roles

    @property
    def user_name(self) -> str:
        return self._user_name

    @property
    def enabled(self) -> bool:
        return bool(self._roles)

    @property
    def stay_quiet(self) -> bool:
        """Whether the configured roles also get stay_quiet (off unless a config turns it on)."""
        return self.enabled and self._stay_quiet

    def configure(self, body: Any) -> dict:
        """Name the roles that get message_user (`roles: []` turns it off) and how the tool names the user."""
        if not isinstance(body, dict):
            raise UserChatError("body must be a JSON object")
        unknown = sorted(set(body) - set(_CONFIG_FIELDS))
        if unknown:
            raise UserChatError(f"config takes {list(_CONFIG_FIELDS)}, not {unknown}")
        roles = body.get("roles", [])
        if not isinstance(roles, list) or not all(isinstance(r, str) and r.strip() for r in roles):
            raise UserChatError("roles must be a list of role names")
        name = body.get("user_name") or "your user"
        if not isinstance(name, str):
            raise UserChatError("user_name must be a string")
        stay_quiet = body.get("stay_quiet", False)
        if not isinstance(stay_quiet, bool):
            raise UserChatError("stay_quiet must be true or false")
        with self._lock:
            self._roles = tuple(dict.fromkeys(r.strip() for r in roles))
            self._user_name = name.strip() or "your user"
            self._stay_quiet = stay_quiet
        return self.state()

    def clear(self) -> dict:
        with self._lock:
            self._roles, self._user_name = (), "your user"
            self._queue.clear()
            self._takes.clear()
            self._stay_quiet = False
            self._quiet.clear()
        return {"ok": True}

    def send(self, text: Any, *, role: str) -> dict:
        if role not in self._roles:
            raise UserChatError("message_user is not available here")
        if not isinstance(text, str) or not text.strip():
            raise UserChatError("text must be the message to send")
        if len(text) > _MAX_TEXT_CHARS:
            raise UserChatError(f"text must be at most {_MAX_TEXT_CHARS} characters")
        now = self._clock_of().now()
        with self._lock:
            self._sent += 1
            item = {"id": f"user-message-{self._sent}", "from": role, "at": _iso(now) if now else None,
                    "text": text, "status": "pending"}
            self._queue.append(item)
        return {"sent": True, "id": item["id"], "at": item["at"]}

    def quiet(self, *, role: str) -> dict:
        """stay_quiet: nothing is sent; the call is kept (with its virtual time) for the next take's `quiet`."""
        if role not in self._roles or not self._stay_quiet:
            raise UserChatError("stay_quiet is not available here")
        now = self._clock_of().now()
        with self._lock:
            self._quiet.append({"from": role, "at": _iso(now) if now else None})
        return {"ok": True, "sent_to_user": False}

    def state(self) -> dict:
        with self._lock:
            out = {"roles": list(self._roles), "user_name": self._user_name, "sent": self._sent,
                   "pending": [_public(m) for m in self._queue if m["status"] == "pending"]}
            if self._stay_quiet:
                out["stay_quiet"] = True
            return out

    def take(self, body: Any = None) -> dict:
        body = body if isinstance(body, dict) else {}
        take_id = body.get("take_id")
        if take_id is not None and (not isinstance(take_id, str) or not take_id or len(take_id) > _MAX_TAKE_ID_CHARS):
            raise UserChatError(f"take_id must be a non-empty string of at most {_MAX_TAKE_ID_CHARS} characters")
        sender = body.get("from")
        if sender is not None and (not isinstance(sender, str) or not sender.strip()):
            raise UserChatError("from must be a role name")
        with self._lock:
            if take_id is not None and take_id in self._takes:
                return self._takes[take_id]
            items = [m for m in self._queue if m["status"] == "pending" and sender in (None, m["from"])]
            for m in items:
                m["status"] = "taken"
            self._queue = [m for m in self._queue if m["status"] == "pending"]
            result = {"items": [_public(m) for m in items]}
            if self._stay_quiet:
                result["quiet"] = [q for q in self._quiet if sender in (None, q["from"])]
                self._quiet = [q for q in self._quiet if sender not in (None, q["from"])]
            if take_id is not None:
                self._takes[take_id] = result
                while len(self._takes) > _KEPT_TAKES:
                    del self._takes[next(iter(self._takes))]
            return result


def _public(m: dict) -> dict:
    return {k: v for k, v in m.items() if k != "status"}


def build_user_chat_tool(chat: UserChat, make_arg_model: Callable[[str, dict], Any]) -> FastMCPTool:
    """message_user, built by hand like get_time so a backing server's tool can't silently shadow it."""

    async def message_user(text: str) -> dict | CallToolResult:
        try:
            return chat.send(text, role=Caller.current().role)
        except UserChatError as e:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=str(e))])

    return FastMCPTool(fn=message_user, name=MESSAGE_USER_TOOL_NAME,
                       description=MESSAGE_USER_DESCRIPTION.format(user=chat.user_name), parameters=_SCHEMA,
                       fn_metadata=FuncMetadata(arg_model=make_arg_model(MESSAGE_USER_TOOL_NAME, _SCHEMA)),
                       is_async=True, annotations=_ANNOTATIONS)


def build_stay_quiet_tool(chat: UserChat, make_arg_model: Callable[[str, dict], Any]) -> FastMCPTool:
    """stay_quiet, built in like message_user; only registered when the config turns it on."""

    async def stay_quiet() -> dict | CallToolResult:
        try:
            return chat.quiet(role=Caller.current().role)
        except UserChatError as e:
            return CallToolResult(isError=True, content=[TextContent(type="text", text=str(e))])

    return FastMCPTool(fn=stay_quiet, name=STAY_QUIET_TOOL_NAME,
                       description=STAY_QUIET_DESCRIPTION.format(user=chat.user_name), parameters=_QUIET_SCHEMA,
                       fn_metadata=FuncMetadata(arg_model=make_arg_model(STAY_QUIET_TOOL_NAME, _QUIET_SCHEMA)),
                       is_async=True, annotations=_ANNOTATIONS)
