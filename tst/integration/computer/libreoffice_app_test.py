"""The apps bundle's libreoffice on the ubuntu-24.04 base, as the office example's desktop lists it: gate 3 makes this
machine's arch ready with it; deployed, it isn't launched until it's opened, and the fonts with Office's metrics stand
in for Office's; a .docx, a .pptx, an .xlsx and a .csv opened by their type, as the file manager opens them, each open
in their app in the one copy its server launched, asking nothing, with no lock file beside them, and a .docx saves in
its own format without asking; its menu launcher, run while it runs, goes to that copy, starting no other; and a reset
of its data stops it, takes its profile back to none and launches it again, as quiet. Needs Docker and the local
registry on localhost:5000."""

import asyncio
import platform
import re

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import apps
from agentenv_protocol.types import WELL_KNOWN_PATH

from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
FOLDER = "/opt/agentenv/apps/libreoffice"
LAUNCHER = f"{FOLDER}/venv/bin/python -m agentenv_protocol.computer.launch {FOLDER}"
#: The fonts each of Office's resolves to.
FONTS = {"Calibri": "Carlito", "Cambria": "Caladea", "Arial": "Liberation Sans", "Times New Roman": "Liberation Serif",
         "Courier New": "Liberation Mono"}
#: Office files made in the computer with LibreOffice itself, headless, before its app is launched: a .docx and an
#: .xlsx converted from text, a .pptx imported from a PDF of that text, and the CSV itself.
MAKE_FILES = ("cd ~/Documents && printf 'Agenda\\nFirst item\\n' > memo.txt && printf 'name,amount\\nrent,1200\\n' > "
              "costs.csv && soffice --headless --convert-to docx memo.txt && soffice --headless --convert-to xlsx "
              "costs.csv && soffice --headless --convert-to pdf memo.txt && soffice --headless "
              "--infilter=impress_pdf_import --convert-to pptx memo.pdf && rm memo.txt memo.pdf && ls")
#: Each file opened, and the class of the window it opens in: the memo last, so its window is the one on top when it's
#: typed in.
OPENS = {"costs.csv": "libreoffice-calc", "costs.xlsx": "libreoffice-calc", "memo.pptx": "libreoffice-impress",
         "memo.docx": "libreoffice-writer"}
#: Each top-level window: its WM_CLASS instance and class, and its title.
WINDOWS = ('for w in $(xprop -root _NET_CLIENT_LIST | sed "s/.*# //; s/,//g"); do '
           'xprop -id "$w" WM_CLASS _NET_WM_NAME | tr "\\n" " "; echo; done')
WINDOW_LINE = r'WM_CLASS\(STRING\) = "([^"]*)", "([^"]*)"\s*_NET_WM_NAME\(UTF8_STRING\) = "(.*)"'
PROCESSES = "pgrep -c -x soffice.bin || true"
#: A file in its profile, which a reset of its data takes away.
MARKER = ".config/libreoffice/agentenv-marker"


def test_gate_3_makes_this_machines_arch_ready_with_libreoffice(office_desktop):
    results = office_desktop.metadata["composed"]["container"]

    assert results[ARCH]["status"] == "ready" and results[ARCH]["probe"]["passed"], results[ARCH]
    assert [app.environment_name for app in office_desktop.composed_apps()] == ["libreoffice"]


@pytest.mark.asyncio
async def test_office_files_open_by_type_in_the_copy_its_server_launched_asking_nothing_and_a_reset_brings_it_back(
        office_desktop):
    with namespace_routing():
        deployed = await Env.get(office_desktop.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            card = await _app(deployed, WELL_KNOWN_PATH)
            before = await _app(deployed, apps.endpoint(card, "ready"))
            fonts = {office: (await _run(deployed, ["fc-match", "-f", "%{family}", office])).split(",")[0]
                     for office in FONTS}
            made = (await _run(deployed, ["sh", "-c", MAKE_FILES])).split()
            opened = {}
            for name, cls in OPENS.items():
                await _run(deployed, ["exo-open", f"Documents/{name}"], timeout_s=120)
                opened[name] = await _window_of(deployed, cls, name)
            asked = await _dialogs(deployed)
            copies = await _run(deployed, ["sh", "-c", PROCESSES])
            ready = await _app(deployed, apps.endpoint(card, "ready"))
            locks = await _run(deployed, ["sh", "-c", "ls -a Documents | grep -c '^\\.~lock' || true"])
            saved_before = await _run(deployed, ["stat", "-c", "%Y", "Documents/memo.docx"])
            await _typed_and_saved(deployed, "libreoffice-writer", " and more")
            saved_after = await _run(deployed, ["stat", "-c", "%Y", "Documents/memo.docx"])
            asked_on_save = await _dialogs(deployed)
            await _run(deployed, ["sh", "-c", LAUNCHER])
            copies_after_launcher = await _run(deployed, ["sh", "-c", PROCESSES])
            ready_after_launcher = await _app(deployed, apps.endpoint(card, "ready"))
            await _run(deployed, ["touch", MARKER])
            await protocol_v1.reset_data(f"{deployed.environment_url}/svc/libreoffice",
                                         headers=deployed.control_headers)
            relaunched = await _ready(deployed, card)
            marker = await _run(deployed, ["sh", "-c", f"test -e {MARKER} && echo kept || echo gone"])
            copies_after_reset = await _run(deployed, ["sh", "-c", PROCESSES])
            await _run(deployed, ["exo-open", "Documents/memo.docx"], timeout_s=120)
            reopened = await _window_of(deployed, "libreoffice-writer", "memo.docx")
            asked_after_reset = await _dialogs(deployed)
        finally:
            report = await teardown_run(context)

    assert before["launched"] is False and "not launched yet" in before["problems"], before
    assert fonts == FONTS, fonts
    assert {"memo.docx", "memo.pptx", "costs.xlsx", "costs.csv"} <= set(made), made
    assert all(opened.values()), opened
    assert asked == [] and asked_on_save == [] and asked_after_reset == [], (asked, asked_on_save, asked_after_reset)
    assert copies == "1" and copies_after_launcher == "1" and ready["ready"], (copies, copies_after_launcher, ready)
    assert locks == "0" and int(saved_after) > int(saved_before)
    assert ready_after_launcher["ready"] and relaunched["ready"] and reopened, (ready_after_launcher, relaunched,
                                                                                reopened)
    assert marker == "gone" and copies_after_reset == "1", (marker, copies_after_reset)
    assert not report.failed


async def _run(deployed, argv: list[str], timeout_s: float = 60) -> str:
    ran = await inside.run(deployed, argv, timeout_s=timeout_s)
    assert ran["exit_code"] == 0, ran
    return ran["stdout"].strip()


async def _windows(deployed) -> list[tuple[str, str, str]]:
    """Each top-level window's WM_CLASS instance and class, and its title."""
    listed = await _run(deployed, ["sh", "-c", WINDOWS])
    found = []
    for line in listed.splitlines():
        if match := re.search(WINDOW_LINE, line):
            found.append(match.groups())
    return found


async def _window_of(deployed, cls: str, title: str, within_s: float = 60) -> str | None:
    """The title of a window of ``cls`` whose title holds ``title``, once one shows."""
    for _ in range(int(within_s)):
        if found := [name for _, klass, name in await _windows(deployed) if klass == cls and title in name]:
            return found[0]
        await asyncio.sleep(1)
    return None


async def _dialogs(deployed) -> list[str]:
    """The titles of LibreOffice's dialogs showing, a second after the last action: its dialogs are soffice's windows,
    its documents libreoffice's."""
    await asyncio.sleep(1)
    return [name for instance, _, name in await _windows(deployed) if instance == "soffice"]


async def _typed_and_saved(deployed, cls: str, text: str) -> None:
    """``text`` typed at the end of the largest window of ``cls`` and Ctrl+S pressed, through the computer's tools, as
    an agent types and saves."""
    tree = await _run(deployed, ["xwininfo", "-root", "-tree"])
    windows = [(int(w) * int(h), int(x) + int(w) // 2, int(y) + int(h) // 2) for w, h, x, y in re.findall(
        rf'\("libreoffice" "{cls}"\)\s+(\d+)x(\d+)\+-?\d+\+-?\d+\s+\+(-?\d+)\+(-?\d+)', tree)]
    assert windows, tree
    _, x, y = max(windows)
    async with streamablehttp_client(f"{deployed.environment_url}/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("computer_screenshot", {})
            for call, args in (("computer_click", {"x": x, "y": y}), ("computer_key", {"keys": ["Control", "End"]}),
                               ("computer_type", {"text": text}), ("computer_key", {"keys": ["Control", "s"]})):
                done = await session.call_tool(call, args)
                assert not done.isError, done
                await asyncio.sleep(0.5)
    await asyncio.sleep(2)


async def _app(deployed, path: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.get(f"{deployed.environment_url}/svc/libreoffice{path}", headers=deployed.control_headers)
    response.raise_for_status()
    return response.json()


async def _ready(deployed, card: dict) -> dict:
    """The app's readiness once it says it's ready, or as it last said within a minute."""
    for _ in range(60):
        readiness = await _app(deployed, apps.endpoint(card, "ready"))
        if readiness["ready"]:
            return readiness
        await asyncio.sleep(1)
    return readiness

