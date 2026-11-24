"""The gateway's roles (urn:agentenv:roles/v1): roles a caller may name with the AgentEnv-Role header alone (``none``,
every role nobody declared included) and roles that answer only to a member's agent token (``token``), and the one
resolution of who a request acts as (``Roles.resolve``), which every route and the MCP sessions share.

A role is one role however its name is cased (``role_key``), since the services behind the gateway match a persona's
address case-insensitively. A member is kept as the sha256 of its agent token, never the token, and named by an opaque
id. Roles live in the gateway's memory, so a restart keeps only the startup roles its environment declares, without
members."""
from __future__ import annotations

import hashlib
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Relative import — gateway runs as a standalone container with no agent_env.*
from .constants import DEFAULT_ROLE, WILDCARD

AUTH_NONE, AUTH_TOKEN = "none", "token"
AUTH_TYPES = (AUTH_NONE, AUTH_TOKEN)
MAX_ROLE_CHARS = 100
# How the gateway established the role a request acts as, as its trajectory events record it (role_auth): a member's
# agent token, the env's control token, a computer's attribution token, the AgentEnv-Role header (or, without computer
# digests, AgentEnv-Computer) alone, or nothing at all.
AGENT_TOKEN, CONTROL_TOKEN, COMPUTER_TOKEN, HEADER, DEFAULT = (
    "agent_token", "control_token", "computer_token", "header", "default")
_NEVER_REFUSED = (CONTROL_TOKEN, COMPUTER_TOKEN)


class RoleError(ValueError):
    """A roles request the gateway refuses: the HTTP status to answer and any fields to answer with besides the
    message, which never holds a token or a digest."""

    def __init__(self, message: str, status: int = 400, **fields: Any):
        super().__init__(message)
        self.status = status
        self.fields = fields


def role_name(value: Any) -> str:
    """A role's name as the AgentEnv-Role header carries it: stripped, 1 to 100 printable ASCII characters, and not
    the wildcard, which names every role in tool rules."""
    name = value.strip() if isinstance(value, str) else ""
    if not name or len(name) > MAX_ROLE_CHARS or not (name.isascii() and name.isprintable()) or name == WILDCARD:
        raise RoleError(f"role must name a role: 1 to {MAX_ROLE_CHARS} printable ASCII characters, not '{WILDCARD}'")
    return name


def role_key(name: str) -> str:
    """What a role is known by whatever the case of its name: the services behind the gateway match a persona's address
    case-insensitively, so ``BROOKE@EXAMPLE.COM`` is the role ``brooke@example.com`` is."""
    return name.casefold()


def auth_type(value: Any) -> str:
    """The auth a role is declared with, ``{"type": "none"}`` or ``{"type": "token"}``; ``none`` when absent."""
    if value is None:
        return AUTH_NONE
    if not isinstance(value, dict) or set(value) != {"type"} or value["type"] not in AUTH_TYPES:
        raise RoleError('auth must be {"type": "none"} or {"type": "token"}')
    return value["type"]


def digest(token: str | bytes) -> str:
    """The sha256 of a token, hex: all the gateway keeps of one."""
    return hashlib.sha256(token.encode() if isinstance(token, str) else token).hexdigest()


@dataclass(frozen=True)
class Resolution:
    """Who one request acts as: its role, how that was established (``source``, a role_auth value), why it may not
    act as it (``refusal``, None when it may), and the digest of the agent token it carried (``member``), known to the
    gateway or not, so a session can be checked again as memberships change."""

    role: str
    source: str
    refusal: str | None = None
    member: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Computer:
    """A computer a request says it comes from: the role driving it now (None before any input call to its tools),
    and whether that computer's attribution token vouched for the request."""

    driver: str | None
    verified: bool


@dataclass
class _Role:
    name: str  # as declared
    auth: str
    startup: bool
    members: dict[str, str] = field(default_factory=dict)  # member id -> agent token digest


class Roles:
    """The declared roles, their members by id, and which roles each agent token digest belongs to, every role by its
    ``role_key``. Every method is synchronous, so a check and the change it guards never interleave with another
    request's."""

    def __init__(self, startup: Mapping[str, str] | None = None):
        self._roles: dict[str, _Role] = {}
        self._by_digest: dict[str, set[str]] = {}  # digest -> role keys
        for role, auth in (startup or {}).items():
            self.create(role, {"type": auth}, startup=True)

    def create(self, role: Any, auth: Any = None, *, startup: bool = False) -> dict:
        """Declare ``role``; declaring it again as it is changes nothing, and with another auth, or its name cased
        another way, is a 409 naming the role as declared and its auth."""
        role, auth = role_name(role), auth_type(auth)
        existing = self._roles.get(role_key(role))
        if existing is not None and (existing.name, existing.auth) != (role, auth):
            spelled = "" if existing.name == role else f" as {existing.name!r}"
            raise RoleError(f"role {role!r} is already declared{spelled} with auth {existing.auth!r}", 409,
                            role=existing.name, auth={"type": existing.auth})
        if existing is None:
            self._roles[role_key(role)] = _Role(role, auth, startup)
        return {"role": role, "auth": {"type": auth}, "created": existing is None}

    def delete(self, role: Any) -> dict:
        """Undeclare ``role``, with its members: it is open again, as a role nobody declared is."""
        role = role_name(role)
        entry = self._roles.pop(role_key(role), None)
        for member in (entry.members.values() if entry else ()):
            self._unindex(member, role_key(role))
        return {"role": entry.name if entry else role, "deleted": entry is not None}

    def add_member(self, role: Any, agent_token: Any) -> dict:
        """Make the agent holding ``agent_token`` a member of ``role``, a 404 when it isn't declared: its member id,
        the same for the same token, and the role's auth."""
        role = role_name(role)
        if not isinstance(agent_token, str) or not agent_token.strip():
            raise RoleError("agent_token must be a non-empty string")
        entry = self._roles.get(role_key(role))
        if entry is None:
            raise RoleError(f"role {role!r} is not declared", 404)
        member = digest(agent_token.strip())
        member_id = next((known for known, held in entry.members.items() if held == member), None)
        if member_id is None:
            member_id = f"mem_{secrets.token_hex(8)}"
            entry.members[member_id] = member
            self._by_digest.setdefault(member, set()).add(role_key(role))
        return {"member_id": member_id, "auth": {"type": entry.auth}}

    def remove_member(self, role: Any, member_id: Any) -> dict:
        """End a membership; ending one that isn't there, or one of a role that isn't declared, changes nothing."""
        role = role_name(role)
        entry = self._roles.get(role_key(role))
        member = entry.members.pop(member_id, None) if entry is not None and isinstance(member_id, str) else None
        if member is not None:
            self._unindex(member, role_key(role))
        return {"role": entry.name if entry else role, "member_id": member_id, "removed": member is not None}

    def state(self) -> dict:
        """Every declared role with its auth, as ``create`` takes it, and its member ids; never a digest."""
        return {"roles": [{"role": entry.name, "auth": {"type": entry.auth}, "members": sorted(entry.members),
                           "startup": entry.startup} for entry in sorted(self._roles.values(), key=lambda e: e.name)]}

    def resolve(self, header: str, agent_digest: str | None = None, *, control: bool = False,
                computer: Computer | None = None) -> Resolution:
        """Who a request acts as, from its AgentEnv-Role ``header`` (empty when it names none), the digest of its agent
        token, whether it carries the env's control token and the ``computer`` it comes from. An agent token the
        gateway doesn't know counts as none here, though the resolution keeps its digest for a session's later checks.
        The control token plays the role it names, else default, and so does a computer's attribution token its
        driver's; neither is refused. A role header names the role; without one, an agent token plays the one role its
        agent belongs to, else default. A computer without its token claims its driver only while that role is open,
        and nothing else names default."""
        member = agent_digest if agent_digest in self._by_digest else None
        if control:
            return Resolution(header or DEFAULT_ROLE, CONTROL_TOKEN, None, agent_digest)
        if header:
            role = header
            source = AGENT_TOKEN if self._is_member(role, member) else HEADER
        elif member is not None:
            joined = self._by_digest[member]
            role = self._roles[next(iter(joined))].name if len(joined) == 1 else DEFAULT_ROLE
            source = AGENT_TOKEN if self._is_member(role, member) else DEFAULT
        elif computer is not None and computer.verified:
            return Resolution(computer.driver or DEFAULT_ROLE, COMPUTER_TOKEN, None, agent_digest)
        elif computer is not None and computer.driver and self.auth_of(computer.driver) == AUTH_NONE:
            role, source = computer.driver, HEADER
        else:
            role, source = DEFAULT_ROLE, DEFAULT
        return Resolution(role, source, self.refusal(role, source, member), agent_digest)

    def may_play(self, role: str, source: str, member: str | None = None) -> bool:
        """Whether a caller established by ``source``, holding the agent token with digest ``member``, may act as
        ``role`` now: an open role, a token role it is a member of, or any role with the control token or a
        computer's."""
        return source in _NEVER_REFUSED or self.auth_of(role) == AUTH_NONE or self._is_member(role, member)

    def source_now(self, role: str, source: str, member: str | None = None) -> str:
        """How such a caller acts as ``role`` now, as its MCP session's next call records it (role_auth): a member's
        agent token while it is one, the header's once the membership it was established by has ended, else
        ``source``."""
        if source in _NEVER_REFUSED:
            return source
        if self._is_member(role, member):
            return AGENT_TOKEN
        return HEADER if source == AGENT_TOKEN else source

    def refusal(self, role: str, source: str, member: str | None = None) -> str | None:
        """Why such a caller may not act as ``role``; None when it may (``may_play``)."""
        if self.may_play(role, source, member):
            return None
        if member is None:
            return f"role {role!r} takes the agent token of one of its members"
        return f"this agent token isn't a member of role {role!r}"

    def auth_of(self, role: str) -> str:
        entry = self._roles.get(role_key(role))
        return entry.auth if entry is not None else AUTH_NONE

    def roles_of(self, member: str) -> frozenset[str]:
        """The roles the agent token with digest ``member`` belongs to, as declared."""
        return frozenset(self._roles[key].name for key in self._by_digest.get(member, ()))

    def _is_member(self, role: str, member: str | None) -> bool:
        return member is not None and role_key(role) in self._by_digest.get(member, ())

    def _unindex(self, member: str, key: str) -> None:
        held = self._by_digest.get(member)
        if held is not None:
            held.discard(key)
            if not held:
                del self._by_digest[member]
