"""The apps composed onto a computer, as its driver finds them, whether each is ready, and how the driver fronts them.
Each app's own server answers on its folder's socket (``serve_app``): its card says how the app launches and which
paths it owns, and its ``computer-app/v1`` ``ready`` whether it's ready. Every app's server must answer before the
computer is ready; an app that starts with the computer must also be ready, and a ``demand`` app is checked once
something launches it. The driver lists each app as a child environment named for its folder, its card's paths under
``/svc/<name>/``, and forwards each request there to the app's server."""
from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Optional

import httpx

from .. import client
from ..types import RPC_PATH, WELL_KNOWN_PATH, EnvironmentCard
from .app import COMPUTER_APP, FAILURE, SOCKET
from .app_runtime import CHECK_TIMEOUT_S, AppReadiness
from .operations import ReadyCheck

#: Where a composed computer installs its apps, a folder each.
APPS = Path("/opt/agentenv/apps")
#: How long an app's server may take to answer, its checks run at once, before it counts as not answering.
ANSWER_TIMEOUT_S = CHECK_TIMEOUT_S + 5


def expected() -> list[Path]:
    """The folders of the apps composed onto this computer."""
    return sorted(path for path in APPS.iterdir() if path.is_dir()) if APPS.is_dir() else []


async def check(folder: Path) -> ReadyCheck:
    """Whether the app in ``folder`` is ready, as its own server says, or why its server last failed to start; named by
    its card once the server answers."""
    name = folder.name
    try:
        async with _client(folder / SOCKET) as http:
            card = (await http.get(WELL_KNOWN_PATH)).raise_for_status().json()
            name = card["name"]
            launch = client.extension_params(card, COMPUTER_APP).get("launch")
            answer = AppReadiness.model_validate((await http.get(endpoint(card, "ready"))).raise_for_status().json())
    except (httpx.HTTPError, OSError):
        return ReadyCheck(f"app {name}", False, _unanswered(folder))
    if launch is not None and launch["at"] == "demand" and not answer.launched:
        return ReadyCheck(f"app {name}", True, "its server answers; it launches on demand")
    if answer.ready:
        return ReadyCheck(f"app {name}", True, "ready")
    when = (f"not ready after {answer.timeout_s:.0f} s" if answer.waited_s > answer.timeout_s
            else f"not ready yet, {answer.waited_s:.0f} s in")
    return ReadyCheck(f"app {name}", False, f"{when}: {'; '.join(answer.problems)}")


async def children(folders: list[Path]) -> list[EnvironmentCard]:
    """Each app in ``folders`` as a child environment of its computer: its server's card, rooted at ``/svc/<name>/``,
    or, while its server isn't answering, its name and route alone."""
    cards = await asyncio.gather(*(_card(folder) for folder in folders))
    return [EnvironmentCard.model_validate(child_card(folder.name, card)) if card is not None
            else EnvironmentCard(name=folder.name, url=f"/svc/{folder.name}{RPC_PATH}")
            for folder, card in zip(folders, cards)]


def child_card(name: str, card: dict) -> dict:
    """``card`` with every path it names, its ``url`` and each extension's and method's endpoint, under
    ``/svc/<name>``, so it resolves against the computer's address; and none of its interfaces, which agents reach
    through the computer."""
    card = copy.deepcopy(card)

    def rooted(path: object) -> object:
        return f"/svc/{name}{path}" if isinstance(path, str) and path.startswith("/") else path

    card["url"] = rooted(card.get("url") or RPC_PATH)
    card["additionalInterfaces"], card["children_environments"] = [], None
    for extension in (card.get("capabilities") or {}).get("extensions") or []:
        params = extension.get("params")
        if not isinstance(params, dict):
            continue
        if "endpoint" in params:
            params["endpoint"] = rooted(params["endpoint"])
        for method in (params.get("methods") or {}).values():
            if isinstance(method, dict) and "endpoint" in method:
                method["endpoint"] = rooted(method["endpoint"])
    return card


async def owned_paths(folders: list[Path]) -> list[str]:
    """The paths under the session user's home that the apps in ``folders`` keep their data in, as their cards say."""
    cards = await asyncio.gather(*(_card(folder) for folder in folders))
    return [path for card in cards if card is not None
            for path in client.extension_params(card, COMPUTER_APP).get("paths") or []]


async def forward(folder: Path, method: str, path: str, query: str, body: bytes,
                  content_type: Optional[str]) -> httpx.Response:
    """The app's server's answer to the request, sent on as it came; it takes as long as the app does."""
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(folder / SOCKET)), base_url="http://app",
                                 timeout=None) as http:
        return await http.request(method, f"/{path}" + (f"?{query}" if query else ""), content=body,
                                  headers={"content-type": content_type} if content_type else None)


def endpoint(card: dict, method: str) -> str:
    """The path of a ``computer-app/v1`` method, as the app's card advertises it."""
    return client.find_extension_method(card, COMPUTER_APP, method)["endpoint"]


async def _card(folder: Path) -> Optional[dict]:
    """The card the app in ``folder`` serves, or None while its server isn't answering."""
    try:
        async with _client(folder / SOCKET) as http:
            return (await http.get(WELL_KNOWN_PATH)).raise_for_status().json()
    except (httpx.HTTPError, OSError, ValueError):
        return None


def _unanswered(folder: Path) -> str:
    try:
        return f"its server failed to start: {(folder / FAILURE).read_text()}"
    except FileNotFoundError:
        return "its server isn't answering yet"


def _client(socket: Path) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(socket)), base_url="http://app",
                             timeout=ANSWER_TIMEOUT_S)
