"""Per-role tool visibility for an environment agents reach with no gateway in front, under the gateway's rules and
at its paths, so ``modify_env_tool_access`` and an env's ``[tools]`` behave the same with or without one."""
from __future__ import annotations

from typing import Iterable

from .errors import AgentEnvError
from .tool_context import DEFAULT_ROLE, ToolContext
from .types import EnvironmentExtension

WILDCARD = "*"
DISABLE_TOOL_EXTENSION_URI = "urn:agentenv:disable-tool/v1"
ENABLE_TOOL_EXTENSION_URI = "urn:agentenv:enable-tool/v1"
DISABLE = "disable"
ENABLE = "enable"
PATHS = {DISABLE: "/tools/disable", ENABLE: "/tools/enable"}

_REQUEST = {
    "type": "object",
    "properties": {
        "role": {"type": "string"},
        "tools": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"const": WILDCARD}]},
    },
    "required": ["role", "tools"],
}


class ToolAccess:
    """Rules per role, each a tool name or ``*`` mapped to disabled or not. A call's role is looked up first, then
    ``*``; the first rule naming the tool decides, then a ``*`` rule, except that a tool hidden by default skips
    ``*`` rules and stays hidden until a rule names it."""

    def __init__(self, hidden_by_default: Iterable[str] = ()) -> None:
        self.rules: dict[str, dict[str, bool]] = {}
        self.hidden_by_default = set(hidden_by_default)

    def is_disabled(self, role: str, name: str) -> bool:
        hidden = name in self.hidden_by_default
        for r in (role, WILDCARD):
            rules = self.rules.get(r, {})
            if name in rules:
                return rules[name]
            if WILDCARD in rules and not hidden:
                return rules[WILDCARD]
        return hidden

    def apply(self, role: str, tool: str, disabled: bool) -> None:
        if role == WILDCARD and tool == WILDCARD:
            self.rules.clear()
            self.rules[WILDCARD] = {WILDCARD: disabled}
        elif role == WILDCARD:
            for rules in self.rules.values():
                rules[tool] = disabled
            self.rules.setdefault(WILDCARD, {})[tool] = disabled
        elif tool == WILDCARD:
            self.rules[role] = {WILDCARD: disabled}
        else:
            self.rules.setdefault(role, {})[tool] = disabled

    def change(self, action: str, body: object) -> dict:
        """A ``disable-tool/v1`` or ``enable-tool/v1`` request applied: ``{role, tools: [...] | "*"}`` to
        ``{role, disabled, allowed}``, the role's own rules after the change."""
        role = body.get("role") if isinstance(body, dict) else None
        if not isinstance(role, str) or not role:
            raise AgentEnvError("invalid_params", "missing or invalid 'role' field")
        tools = body.get("tools")
        if tools == WILDCARD:
            names = [WILDCARD]
        elif isinstance(tools, list) and all(isinstance(t, str) for t in tools):
            names = tools
        else:
            raise AgentEnvError("invalid_params", "'tools' must be a list of strings or the string '*'")
        for name in names:
            self.apply(role, name, action == DISABLE)
        rules = self.rules.get(role, {})
        return {"role": role, "disabled": sorted(k for k, v in rules.items() if v),
                "allowed": sorted(k for k, v in rules.items() if not v)}

    def check_call(self, context: ToolContext) -> None:
        role = caller_role(context)
        if self.is_disabled(role, context.tool):
            raise AgentEnvError("permission_denied", f"Tool '{context.tool}' is disabled for role '{role}'", status=403)


def caller_role(context: ToolContext) -> str:
    """The role a rule is keyed by: the caller's, or the gateway's unforwarded ``default``."""
    return context.caller.role or DEFAULT_ROLE


def extensions() -> list[EnvironmentExtension]:
    return [
        EnvironmentExtension(uri=uri, description=description, required=False, params={
            "endpoint": PATHS[action], "methods": {action: {"method": "POST", "request": _REQUEST}}})
        for uri, action, description in (
            (DISABLE_TOOL_EXTENSION_URI, DISABLE, "Disable tools for a role."),
            (ENABLE_TOOL_EXTENSION_URI, ENABLE, "Enable tools for a role."),
        )
    ]
