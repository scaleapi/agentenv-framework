"""What a website env's task state is, the one definition everything that grades a site reads: a site gives the state a
task left it in through its card's ``urn:agentenv:website/v1`` ``state`` (``WEBSITE_STATE_OP``), answering a
``WebsiteState``, its state when the task started, its state now and what changed between them. ``website_state``
asks one site as its card advertises it, of one session when the site keeps several; ``website_sites`` names every
site of an env, a single site's own card or a composed card's website children. The op is one of agentenv's own
routes, under the data plane's path, so a browser on the site's route never reads what grades it."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from ..client import ChildEnv, children, find_extension, find_extension_method, invoke_extension
from .readiness import WEBSITE_EXTENSION_URI

WEBSITE_STATE_OP = "state"
STATE_TIMEOUT_S = 30.0


class WebsiteState(BaseModel):
    """What a site says of its task state, ``website/v1``'s ``state``: ``initial_state``, the state the task started
    from, ``current_state``, the state now, and ``state_diff``, what changed between them, each a JSON object or None
    when the site gives none. Fields it doesn't name are ignored rather than refused, so a reader keeps reading a newer
    site that says more."""

    model_config = ConfigDict(extra="ignore")

    initial_state: Optional[dict[str, Any]] = None
    current_state: Optional[dict[str, Any]] = None
    state_diff: Optional[dict[str, Any]] = None


def advertises_state(card: dict) -> bool:
    """Whether ``card`` advertises ``website/v1``'s ``state``."""
    return find_extension_method(card, WEBSITE_EXTENSION_URI, WEBSITE_STATE_OP) is not None


def website_sites(card: dict) -> list[ChildEnv]:
    """Every site of the env whose card is ``card``: the card itself when it carries ``website/v1``, as a single site's
    does, else each child of a composed card that carries it, in its order; none for an env with no site."""
    if find_extension(card, WEBSITE_EXTENSION_URI) is not None:
        return [ChildEnv(name=card.get("name"), url=card.get("url"), card=card)]
    return [child for child in children(card) if child.find_extension(WEBSITE_EXTENSION_URI) is not None]


async def website_state(base_url: str, card: dict, *, sid: str | None = None, timeout: float = STATE_TIMEOUT_S,
                        headers: dict[str, str] | None = None) -> WebsiteState | None:
    """What the site whose ``card`` is read from ``base_url`` says of its task state, of session ``sid`` when it's
    given, for a site that keeps several; None when the card doesn't advertise ``website/v1``'s ``state``. Raises
    httpx's error when it can't be asked, ``ExtensionCallError`` for a non-2xx answer, and ``ValueError`` (pydantic's
    ``ValidationError`` among them) for an answer that isn't a ``WebsiteState``."""
    if not advertises_state(card):
        return None
    answer = await invoke_extension(base_url, card, WEBSITE_EXTENSION_URI, {"sid": sid} if sid is not None else None,
                                    timeout=timeout, method=WEBSITE_STATE_OP, headers=headers)
    return WebsiteState.model_validate(answer)
