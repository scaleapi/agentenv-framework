"""The OpenCode CLI the agent's package-lock.json pins, against what its bridge relies on to tell the model the env's
date while the CLI runs on real time: the CLI's own date line and websearch's year are where the turn's relay rewrites
them, so the model is sent the env's date and no copy of the machine's; its bin is the native binary, which the
real-time marker reaches; and a turn runs in that one process, with no second CLI that would follow the env's clock.
Bumping the CLI runs this against the new version, so a version that moves or rewrites any of these fails here."""

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tst.unit.agents.opencode.agent import AGENT_DIR, opencode_bridge, opencode_date
from tst.util import fake_openai

pytestmark = [pytest.mark.integration]

VERSION = json.loads((AGENT_DIR / "package.json").read_text())["dependencies"]["opencode-ai"]
ENV_TODAY, ENV_YEAR = "Sun Jan 01 1950", 1950
_DAY_S = 86400
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@pytest.fixture(scope="module")
def opencode_bin(tmp_path_factory) -> Path:
    """The CLI installed as the image installs it: npm ci from the lockfile, then opencode-ai's own postinstall."""
    npm, node = shutil.which("npm"), shutil.which("node")
    assert npm and node, "node and npm install the pinned OpenCode CLI"
    folder = tmp_path_factory.mktemp("opencode")
    for name in ("package.json", "package-lock.json"):
        shutil.copy(AGENT_DIR / name, folder / name)
    subprocess.run([npm, "ci", "--ignore-scripts", "--omit=dev", "--no-audit", "--no-fund", "--loglevel=error"],
                   cwd=folder, check=True, timeout=600)
    subprocess.run([node, "node_modules/opencode-ai/postinstall.mjs"], cwd=folder, check=True, timeout=120)
    return folder / "node_modules" / ".bin" / "opencode"


def test_the_cli_is_its_native_binary_so_the_real_time_marker_reaches_it(opencode_bin):
    """A launcher script would be what the marker starts, and libfaketime takes the marker out of its environment, so
    the CLI it starts would follow the env's clock."""
    with open(opencode_bin.resolve(), "rb") as binary:
        assert binary.read(4) in (b"\x7fELF", b"\xcf\xfa\xed\xfe"), f"OpenCode {VERSION}: {opencode_bin.resolve()}"


def _system_texts(body: dict) -> list[str]:
    texts = []
    for message in body.get("messages") or []:
        if message.get("role") not in ("system", "developer"):
            continue
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"text": content}]
        texts += [part["text"] for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
    return texts


def _descriptions(body: dict) -> dict[str, str]:
    return {tool["function"]["name"]: tool["function"].get("description") or "" for tool in body.get("tools") or []}


def _machine_dates() -> set[str]:
    """This machine's date, as the CLI on real time writes one (``Date.toDateString()``) and as ISO, for yesterday,
    today and tomorrow, so a turn across midnight is still caught."""
    dates = set()
    for offset in (-_DAY_S, 0, _DAY_S):
        day = time.localtime(time.time() + offset)
        dates |= {f"{_DAYS[day.tm_wday]} {_MONTHS[day.tm_mon - 1]} {day.tm_mday:02d} {day.tm_year:04d}",
                  time.strftime("%Y-%m-%d", day)}
    return dates


def _opencode_processes_under(pid: int) -> list[str]:
    rows = [line.split(None, 2) for line in subprocess.run(["ps", "-eo", "pid=,ppid=,comm="], capture_output=True,
                                                           text=True, check=True).stdout.splitlines()]
    children: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        if len(row) == 3:
            children.setdefault(row[1], []).append((row[0], row[2]))
    found, pending = [], [str(pid)]
    while pending:
        for child, command in children.get(pending.pop(), ()):
            pending.append(child)
            if os.path.basename(command).startswith("opencode"):
                found.append(f"{child} {command}")
    return found


def test_a_turn_on_an_armed_clock_tells_the_model_the_env_date_and_no_other(opencode_bin, tmp_path, monkeypatch):
    seen_clis: list[list[str]] = []

    def respond(body: dict) -> fake_openai.Text:
        seen_clis.append(_opencode_processes_under(os.getpid()))
        return fake_openai.Text("done")

    async def env_today() -> tuple[str, int]:
        return ENV_TODAY, ENV_YEAR

    for name in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("PATH", f"{opencode_bin.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-fake")
    monkeypatch.setenv("OPENCODE_ENABLE_EXA", "1")
    monkeypatch.setattr(opencode_bridge, "SESSIONS_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(opencode_date, "clock_armed", lambda: True)
    monkeypatch.setattr(opencode_date, "_env_today", env_today)
    (tmp_path / "work").mkdir()

    with fake_openai.FakeOpenAI(respond=respond) as endpoint:
        monkeypatch.setenv("LITELLM_BASE_URL", endpoint.url)
        result = asyncio.run(opencode_bridge.run_opencode("Say done.", "ctx", model="m", timeout_s=300,
                                                          workspace=str(tmp_path / "work")))

    assert not result.error, result.error
    sent = [seen.body for seen in endpoint.completions()]
    turns = [body for body in sent if any("<env>" in text for text in _system_texts(body))]
    assert turns, (f"OpenCode {VERSION} sent its <env> block in no system message, so the relay can't tell the model "
                   "the env's date: find where it puts its date now, and fix _ENV_DATE in opencode_date.py")
    for body in turns:
        assert f"  Today's date: {ENV_TODAY}\n" in "\n".join(_system_texts(body)), (
            f"OpenCode {VERSION}'s <env> block no longer writes its date where opencode_date's _ENV_DATE finds it")
        assert f"The current year is {ENV_YEAR}." in _descriptions(body).get("websearch", ""), (
            f"OpenCode {VERSION}'s websearch tool no longer names its year where opencode_date's _SEARCH_YEAR finds it")
    machine = _machine_dates()
    for body in sent:
        harness = "\n".join([*_system_texts(body), *_descriptions(body).values()])
        assert not [date for date in machine if date in harness], (
            f"OpenCode {VERSION} wrote this machine's date where the relay doesn't rewrite it")
    assert seen_clis and all(len(clis) == 1 for clis in seen_clis), (
        f"OpenCode {VERSION} ran a turn in other than one process, and one it starts follows the env's clock: "
        f"{seen_clis}")
