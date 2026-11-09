"""How a website env's health is determined, the one definition everything that judges it reads: a site says whether
it is usable now through its card's ``urn:agentenv:website/v1`` ``ready`` (``WEBSITE_READY_OP``), answering a
``WebsiteReadiness``. ``website_readiness`` asks one site as its card advertises it; ``env_readiness`` asks every site
of an env, a composed card's children or a single site's own card, and answers for the env as a whole; asking one
site may take ``READY_TIMEOUT_S`` by default. A site may say how long it may take to be ready (``timeout_s``,
``SITE_READY_TIMEOUT_S`` when it says nothing). A server whose card has no ``ready`` is asked at its health path,
``DEFAULT_HEALTH_PATH`` unless it names one, which is deprecated. A composer serving a site to a browser at a route of
its own, as the env gateway serves ``/website/<name>/``, says where on the site's ``website/v1`` params, as
``WEBSITE_BROWSE_URL_PARAM``: a path joined onto the composed card's address, or a full URL."""
from __future__ import annotations

import asyncio
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict

from ..client import find_extension_method, invoke_extension

WEBSITE_EXTENSION_URI = "urn:agentenv:website/v1"
WEBSITE_READY_OP = "ready"
WEBSITE_BROWSE_URL_PARAM = "browse_url"
#: A live site's ``website/v1`` params: that it is one, and the other hosts its pages reach.
WEBSITE_LIVE_PARAM = "live"
WEBSITE_ALLOW_HOSTS_PARAM = "allow_hosts"
READY_TIMEOUT_S = 10.0
SITE_READY_TIMEOUT_S = 300.0
DEFAULT_HEALTH_PATH = "/api/health"


class WebsiteReadiness(BaseModel):
    """What a site says of its readiness, ``website/v1``'s ``ready``: whether it is ``ready``, the ``problems`` that
    keep it from being so, each a line a person reads, such as a ``@ready_check`` method's and what it returned, and,
    for a site that says, the ``waited_s`` since its server started against the ``timeout_s`` it may take. Fields it
    doesn't name are ignored rather than refused, so a reader keeps reading a newer site that says more, as ``ready``
    and ``problems`` keep their meaning. Its types are strict, as JSON's are: ``"yes"`` or ``1`` is no ``ready``."""

    model_config = ConfigDict(extra="ignore", strict=True)

    ready: bool
    problems: list[str] = []
    waited_s: Optional[float] = None
    timeout_s: Optional[float] = None


def _advertises_ready(card: dict) -> bool:
    return find_extension_method(card, WEBSITE_EXTENSION_URI, WEBSITE_READY_OP) is not None


async def website_readiness(base_url: str, card: dict, *, timeout: float = READY_TIMEOUT_S,
                            headers: dict[str, str] | None = None) -> WebsiteReadiness | None:
    """What the site whose ``card`` is read from ``base_url`` says of its readiness, None when the card doesn't
    advertise ``website/v1``'s ``ready``. Raises httpx's error when it can't be asked, ``ExtensionCallError`` for a
    non-2xx answer, and ``ValueError`` (pydantic's ``ValidationError`` among them) for an answer that isn't a
    ``WebsiteReadiness``."""
    if not _advertises_ready(card):
        return None
    answer = await invoke_extension(base_url, card, WEBSITE_EXTENSION_URI, timeout=timeout, method=WEBSITE_READY_OP,
                                    headers=headers)
    return WebsiteReadiness.model_validate(answer)


async def env_readiness(base_url: str, card: dict, *, timeout: float = READY_TIMEOUT_S,
                        headers: dict[str, str] | None = None) -> WebsiteReadiness:
    """What the env whose ``card`` is read from ``base_url`` says of its readiness, from every site it has, asked at
    once: the card itself when it advertises ``website/v1``'s ``ready``, as a single site's does, its problems as the
    site names them, and each child under a composed card's ``children_environments`` that advertises it, its endpoints
    rooted at ``base_url`` as a gateway's are, its problems prefixed with its name. A site that can't be asked, or
    answers something that isn't a ``WebsiteReadiness``, is a problem of its own, and one not ready that names no
    problem is one too. A child, or a card, that doesn't advertise ``ready`` isn't counted, so an env with no site
    serving it, as one of MCP servers alone, is ready with no problems."""
    sites = [("", card)] if _advertises_ready(card) else []
    sites += [(f"{child.get('name') or 'child'}: ", child) for child in card.get("children_environments") or []
              if isinstance(child, dict) and _advertises_ready(child)]
    answers = await asyncio.gather(*(_problems(base_url, site, timeout, headers) for _prefix, site in sites))
    problems = [prefix + problem for (prefix, _site), found in zip(sites, answers) for problem in found]
    return WebsiteReadiness(ready=not problems, problems=problems)


async def _problems(base_url: str, card: dict, timeout: float, headers: dict[str, str] | None) -> list[str]:
    try:
        readiness = await website_readiness(base_url, card, timeout=timeout, headers=headers)
    except (httpx.HTTPError, ValueError) as e:
        return [f"{WEBSITE_READY_OP} could not be read: {type(e).__name__}: {e}"]
    if readiness.ready:
        return []
    return readiness.problems or [f"{WEBSITE_READY_OP} says not ready"]
