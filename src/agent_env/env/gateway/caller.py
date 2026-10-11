"""Who the gateway acts for on one inbound request.

A ``Caller`` is the role the request's ``AgentEnv-Role`` header named (``default`` when it named none) and the key of
the MCP session the request belongs to (empty on the REST and harness paths, which have no session). It is bound in
three scopes, which read in order in ``gateway.py``: per HTTP request by the served app (the role), per inbound MCP
session by the session lifespan (the key, kept for every tools/call of that session), and never per process. Every
child ``tools/call`` carries ``Caller.meta()``, so the three places that stamp one (the MCP proxy tool, ``/step`` and
the trigger engine) cannot disagree about the wire shape.
"""

from __future__ import annotations

import dataclasses
from contextvars import ContextVar, Token

from .constants import AGENT_ENV_ROLE_META_KEY, AGENT_ENV_SESSION_META_KEY, DEFAULT_ROLE


@dataclasses.dataclass(frozen=True)
class Caller:
    role: str = DEFAULT_ROLE
    session: str = ""

    @staticmethod
    def current() -> "Caller":
        """The caller bound for the running task: an unforwarded one outside any request."""
        return _current.get()

    def bind(self) -> Token:
        """Make this the current caller; ``Caller.reset`` the returned token to restore the previous one."""
        return _current.set(self)

    @staticmethod
    def reset(token: Token) -> None:
        _current.reset(token)

    def meta(self) -> dict[str, str]:
        """The ``_meta`` stamp for a child ``tools/call``.

        The role entry is always present, ``default`` included, and that is load-bearing: under the protocol a
        ``_meta`` role entry of any value marks the call as proxied, so the child reads no header of the gateway's
        connection and takes the session from ``_meta`` or none. Stamping ``default`` is how an unforwarded caller
        stays unforwarded behind a gateway whose own connection carries no identity. The session entry rides along
        only when the call came in on an MCP session, so a child keying per-session state keeps one owner per agent
        across the gateway's one shared connection to it.
        """
        meta = {AGENT_ENV_ROLE_META_KEY: self.role}
        if self.session:
            meta[AGENT_ENV_SESSION_META_KEY] = self.session
        return meta


_current: ContextVar[Caller] = ContextVar("agent_env_gateway_caller", default=Caller())
