"""The header a slot's door takes (``gateway.door``), picked out of what a call is granted.

A slot's grant (``DeployedGatewayEnv.grants``) gives the door's header on every address of the slot, beside the
control token on the gateway's. A call that today sends no credential sends only the door's header, so the door lets
it in and the service behind it sees nothing it didn't before; outside a slot that is nothing at all.
"""

from __future__ import annotations

from agentenv_protocol import client as protocol_v1

from agent_env.env.gateway.constants import SLOT_TOKEN_HEADER


def slot_token_headers(token: str | None) -> dict[str, str]:
    """The header that carries a slot's door token; none without one, as for the default stack."""
    return {SLOT_TOKEN_HEADER: token} if token else {}


def door_headers(headers: dict[str, str]) -> dict[str, str]:
    """The slot door's header among ``headers``, alone; none when they don't carry it."""
    return {SLOT_TOKEN_HEADER: headers[SLOT_TOKEN_HEADER]} if SLOT_TOKEN_HEADER in headers else {}


def granted_door_headers(url: str) -> dict[str, str]:
    """The slot door's header the running step's grant gives a call to ``url`` (``protocol_v1.authorized``); none
    outside a slot or a grant."""
    return door_headers(protocol_v1.request_headers(url))
