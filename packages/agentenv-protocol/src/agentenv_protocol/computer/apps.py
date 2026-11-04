"""The apps composed onto a computer, as its driver finds them, whether each is ready, and how the driver fronts them.
Each app's own server answers on its folder's socket (``serve_app``): its card says how the app launches and which
paths it owns, and its ``computer-app/v1`` ``ready`` whether it's ready. Every app's server must answer before the
computer is ready; an app that starts with the computer must also be ready, and a ``demand`` app is checked once
something launches it. The driver lists each app as a child environment named for its folder, its card's paths under
``/svc/<name>/``, and forwards each request there to the app's server."""
from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from pathlib import Path
from typing import Optional

import httpx

from .. import client
from ..types import RPC_PATH, WELL_KNOWN_PATH, EnvironmentCard
from .app import COMPUTER_APP, FAILURE, SOCKET
from .app_runtime import CHECK_TIMEOUT_S, LAUNCHES, AppReadiness
from .operations import ReadyCheck

#: Where a composed computer installs its apps, a folder each.
APPS = Path("/opt/agentenv/apps")
#: How long an app's server may take to answer, its checks run at once, before it counts as not answering.
ANSWER_TIMEOUT_S = CHECK_TIMEOUT_S + 5
#: The headers that belong to one connection, which a proxy doesn't pass on.
HOP_HEADERS = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection",
                         "te", "trailer", "transfer-encoding", "upgrade"})
#: The address a request reached the driver at, which the driver sets for the app in place of any the caller sent.
FORWARDED = ("x-forwarded-host", "x-forwarded-proto", "x-forwarded-prefix")


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
    cards = await asyncio.gather(*(served_card(folder) for folder in folders))
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
    cards = await asyncio.gather(*(served_card(folder) for folder in folders))
    return [path for card in cards if card is not None
            for path in client.extension_params(card, COMPUTER_APP).get("paths") or []]


def passed_on(headers: Iterable[tuple[str, str]], *dropped: str) -> list[tuple[str, str]]:
    """``headers`` as a proxy passes them on: without the hop-by-hop ones, those a ``Connection`` header names, and
    ``dropped``."""
    headers = list(headers)
    named = {token.strip().lower() for name, value in headers if name.lower() == "connection"
             for token in value.split(",")}
    skipped = HOP_HEADERS | named | set(dropped)
    return [(name, value) for name, value in headers if name.lower() not in skipped]


def to_app(headers: Iterable[tuple[str, str]], host: str, scheme: str, prefix: str) -> list[tuple[str, str]]:
    """A caller's headers as the app gets them: passed on without the control token's ``Authorization`` or any address
    the caller forwarded, and with the one it reached the driver at, its ``host``, ``scheme`` and the app's
    ``prefix``."""
    return passed_on(headers, "authorization", *FORWARDED) + [
        ("x-forwarded-host", host), ("x-forwarded-proto", scheme), ("x-forwarded-prefix", prefix)]


async def forward(folder: Path, method: str, target: str, headers: list[tuple[str, str]],
                  body: Optional[AsyncIterator[bytes]]) -> tuple[httpx.Response, Callable[[], Awaitable[None]]]:
    """The app's server's answer to the request, sent with ``headers`` and none of the client's own, its body still to
    be read, and what closes it once read; it takes as long as the app does."""
    http = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(folder / SOCKET)), base_url="http://app",
                             timeout=None)
    http.headers.clear()
    try:
        answer = await http.send(http.build_request(method, target, headers=headers, content=body), stream=True)
    except BaseException:
        await http.aclose()
        raise

    async def close() -> None:
        await answer.aclose()
        await http.aclose()
    return answer, close


def endpoint(card: dict, method: str) -> str:
    """The path of a ``computer-app/v1`` method, as the app's card advertises it."""
    return client.find_extension_method(card, COMPUTER_APP, method)["endpoint"]


async def served_card(folder: Path) -> Optional[dict]:
    """The card the app in ``folder`` serves, or None while its server isn't answering."""
    try:
        async with _client(folder / SOCKET) as http:
            return (await http.get(WELL_KNOWN_PATH)).raise_for_status().json()
    except (httpx.HTTPError, OSError, ValueError):
        return None


def ended_on_its_own(folder: Path) -> Optional[str]:
    """Why the app in ``folder`` can't be ready again where nothing launches it but its server: its launch log says it
    ended on its own, an infrastructure failure; None while it hasn't."""
    for entry in launch_log(folder):
        if entry.get("event") == "exit":
            how = (f"by signal {entry['signal']}" if "signal" in entry else
                   f"with code {entry['code']}" if "code" in entry else "in a way that wasn't recorded")
            return (f"it ended on its own at {entry.get('at')}, {how}, and nothing launches it again: an "
                    "infrastructure failure")
    return None


def launch_log(folder: Path) -> list[dict]:
    """Each entry of the launch log of the app in ``folder`` that's whole, in order."""
    try:
        lines = (folder / LAUNCHES).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return entries


def _unanswered(folder: Path) -> str:
    try:
        return f"its server failed to start: {(folder / FAILURE).read_text()}"
    except FileNotFoundError:
        return "its server isn't answering yet"


def _client(socket: Path) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(socket)), base_url="http://app",
                             timeout=ANSWER_TIMEOUT_S)
