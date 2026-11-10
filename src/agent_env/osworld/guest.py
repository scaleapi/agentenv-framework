"""Where a computer serves the OSWorld guest server: the endpoint its card advertises for the ``osworld-guest/v1``
extension, on the card itself or one of its ``children_environments``, resolved against the computer's address.
``osworld_deploy_controller`` resolves it once and records it on the env, with the controller's address, under the
metadata keys here."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from agentenv_protocol import client as protocol_v1

from agent_env.osworld import GUEST_EXTENSION

logger = logging.getLogger(__name__)

#: The ``DeployedEnv.metadata`` keys ``osworld_deploy_controller`` records on a computer env.
CONTROLLER_URL_KEY = "osworld_controller_url"
CONTROLLER_SANDBOX_ID_KEY = "osworld_controller_sandbox_id"
GUEST_URL_KEY = "osworld_guest_url"
#: A computer's card lists an app without its extensions until the app's server answers: how long resolving waits for
#: the guest extension, and how often it reads the card again.
CARD_WAIT_S = 120.0
CARD_POLL_S = 5.0


def guest_endpoint(card: dict | None) -> str | None:
    """The endpoint ``card`` advertises for the guest extension, its own or a child's; None when nothing does."""
    if not isinstance(card, dict):
        return None
    for candidate in [card, *(child.card for child in protocol_v1.children(card))]:
        params = (protocol_v1.find_extension(candidate, GUEST_EXTENSION) or {}).get("params") or {}
        if isinstance(params.get("endpoint"), str):
            return params["endpoint"]
    return None


def _joined(environment_url: str, endpoint: str) -> str:
    if "://" in endpoint:
        return endpoint.rstrip("/")
    return f"{environment_url.rstrip('/')}/{endpoint.lstrip('/')}".rstrip("/")


async def resolve_guest_url(deployed: Any, *, wait_s: float | None = None, poll_s: float | None = None) -> str:
    """The guest server's base URL on ``deployed``, from the card it was deployed with, else the card it serves now,
    read again until the app's server answers or ``wait_s`` passes. Raises RuntimeError for an env with no address or
    no app advertising the guest extension."""
    environment_url = deployed.environment_url
    if not environment_url:
        raise RuntimeError(f"env '{deployed.env_id}' has no address to read its card at")
    endpoint = guest_endpoint(deployed.environment_card)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (CARD_WAIT_S if wait_s is None else wait_s)
    while endpoint is None:
        try:
            endpoint = guest_endpoint(await protocol_v1.get_card(environment_url))
        except (httpx.HTTPError, ValueError) as e:
            logger.info(f"env '{deployed.env_id}': card not readable yet: {type(e).__name__}")
        if endpoint is not None or loop.time() >= deadline:
            break
        await asyncio.sleep(CARD_POLL_S if poll_s is None else poll_s)
    if endpoint is None:
        raise RuntimeError(f"env '{deployed.env_id}' advertises no {GUEST_EXTENSION} extension on its card: "
                           f"compose the osworld-server computer app into its computer")
    return _joined(environment_url, endpoint)


def controller_url(deployed: Any) -> str:
    """The controller ``osworld_deploy_controller`` recorded on ``deployed``."""
    if url := (deployed.metadata or {}).get(CONTROLLER_URL_KEY):
        return str(url)
    raise RuntimeError(f"env '{deployed.env_id}' has no {CONTROLLER_URL_KEY}: run osworld_deploy_controller on it first")
