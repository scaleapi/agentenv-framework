"""The apps composed onto a computer, as its driver finds them, and whether each is ready. Each app's own server
answers on its folder's socket (``serve_app``): its card says how the app launches, and its ``computer-app/v1``
``ready`` whether it's ready. Every app's server must answer before the computer is ready; an app that starts with the
computer must also be ready, and a ``demand`` app is checked once something launches it."""
from __future__ import annotations

from pathlib import Path

import httpx

from .. import client
from ..types import RPC_PATH, WELL_KNOWN_PATH
from .app import CHECK_TIMEOUT_S, COMPUTER_APP, FAILURE, SOCKET
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
            answer = (await http.get(f"{RPC_PATH}/ext/computer-app/ready")).raise_for_status().json()
    except (httpx.HTTPError, OSError):
        return ReadyCheck(f"app {name}", False, _unanswered(folder))
    if launch is not None and launch["at"] == "demand" and not answer["launched"]:
        return ReadyCheck(f"app {name}", True, "its server answers; it launches on demand")
    if answer["ready"]:
        return ReadyCheck(f"app {name}", True, "ready")
    when = (f"not ready after {answer['timeout_s']:.0f} s" if answer["waited_s"] > answer["timeout_s"]
            else f"not ready yet, {answer['waited_s']:.0f} s in")
    return ReadyCheck(f"app {name}", False, f"{when}: {'; '.join(answer['problems'])}")


def _unanswered(folder: Path) -> str:
    try:
        return f"its server failed to start: {(folder / FAILURE).read_text()}"
    except FileNotFoundError:
        return "its server isn't answering yet"


def _client(socket: Path) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(socket)), base_url="http://app",
                             timeout=ANSWER_TIMEOUT_S)
