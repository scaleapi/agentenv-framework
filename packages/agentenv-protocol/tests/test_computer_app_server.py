"""A computer app's own server: it adds the app's launcher, launches a boot app once per boot and says whether the app
is ready (``computer-app/v1``); its computer's driver finds it on the socket in the app's folder and asks it, lists it
as a child environment and forwards its requests to it, and leaves the paths it owns out of its own data/get."""
from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp.server.fastmcp")
uvicorn = pytest.importorskip("uvicorn")

import base64  # noqa: E402

import httpx  # noqa: E402

import agentenv_protocol.computer.launch as launcher  # noqa: E402
from agentenv_protocol import client  # noqa: E402
from agentenv_protocol.computer import (ComputerAppEnvironment, Launch, ReadyCheck, Ready, linux,  # noqa: E402
                                        ready_check)
from agentenv_protocol.computer import app, app_runtime, apps, session  # noqa: E402
from agentenv_protocol.computer.app import COMPUTER_APP, FAILURE, SOCKET  # noqa: E402
from agentenv_protocol.computer.platform import Platform  # noqa: E402
from agentenv_protocol.types import DATA_OBJECTS_EXTENSION_URI, WELL_KNOWN_PATH  # noqa: E402



@linux(launch=Launch(command=["mail"], window={"class": "Mail"}, ready=Ready(window=True, timeout_s=30)))
class Mail(ComputerAppEnvironment):
    synced = False

    @ready_check
    def inbox_synced(self):
        return self.synced


@linux(launch=Launch(at="demand", command=["music"], window={"class": "Music"}, ready=Ready(window=True)))
class Music(ComputerAppEnvironment):
    pass


class Sync(ComputerAppEnvironment):
    @ready_check
    async def caught_up(self):
        return True


@pytest.fixture(autouse=True)
def on_linux(monkeypatch):
    """Every app here runs as on Linux, which its launch is declared for."""
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux"))


@pytest.fixture
def desktop(monkeypatch, tmp_path):
    """A session with no windows yet, recording what's launched into it, its launches marked under ``tmp_path``."""
    shown, launched = [], []

    async def windows():
        return shown

    async def launch(argv, cwd, env):
        launched.append((argv, cwd, env))
        return 4242

    async def up():
        pass

    monkeypatch.setattr(session, "windows", windows)
    monkeypatch.setattr(session, "launch", launch)
    monkeypatch.setattr(session, "wait_until_up", up)
    monkeypatch.setattr(app_runtime, "LAUNCHED", tmp_path / "launched")
    monkeypatch.setenv("ENVIRONMENT_NAME", "mail")
    return shown, launched


@pytest.mark.asyncio
async def test_an_apps_server_is_ready_once_it_launched_the_app_and_each_check_passes(desktop):
    shown, launched = desktop
    mail = Mail()
    async with _http(mail) as http:
        card = (await http.get("/.well-known/agent-env.json")).json()
        ready = apps.endpoint(card, "ready")
        before = (await http.get(ready)).json()
        assert (await http.post(apps.endpoint(card, "launch"), json={})).json() == {"pid": 4242}
        shown.append({"instance": "mail", "class": "Mail", "title": "Inbox"})
        unsynced = (await http.get(ready)).json()
        mail.synced = True
        after = (await http.get(ready)).json()

    assert client.extension_params(card, COMPUTER_APP)["launch"] == {
        "at": "boot", "command": ["mail"], "window": {"class": "Mail"}, "ready": {"window": True, "timeout_s": 30.0}}
    assert {key: value for key, value in before.items() if key != "waited_s"} == {
        "ready": False, "launched": False, "timeout_s": 30.0,
        "problems": ["not launched yet", "no window with class 'Mail'", "@ready_check inbox_synced returned False"]}
    assert launched == [(["mail"], Path.cwd(), {"ENVIRONMENT_NAME": "mail"})]
    assert unsynced["problems"] == ["@ready_check inbox_synced returned False"] and unsynced["launched"] is True
    assert after["ready"] is True and after["problems"] == []


@pytest.mark.asyncio
async def test_checks_that_fail_hang_or_raise_say_so_at_once_without_holding_up_the_server(desktop, monkeypatch):
    monkeypatch.setattr(app_runtime, "CHECK_TIMEOUT_S", 1.0)
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    exits = [sys.executable, "-c", "import sys; sys.exit('no database yet')"]

    @linux(launch=Launch(command=["slow"], ready=Ready(port=port, command=exits)))
    class Slow(ComputerAppEnvironment):

        @ready_check
        def stuck(self):
            time.sleep(3)
            return True

        @ready_check
        async def waiting(self):
            await asyncio.sleep(3)
            return True

        @ready_check
        async def broken(self):
            raise RuntimeError("mailbox locked")

    started = time.monotonic()
    answer = await Slow()._computer_app_ready()

    assert answer["problems"] == ["not launched yet", f"nothing answers on port {port}",
                                  f"{shlex.join(exits)} exited 1: no database yet", "@ready_check stuck ran past 1 s",
                                  "@ready_check waiting ran past 1 s", "@ready_check broken: RuntimeError: mailbox locked"]
    assert time.monotonic() - started < 1.8


@pytest.mark.asyncio
async def test_a_boot_app_is_launched_once_per_boot_its_clock_from_that_launch_across_restarts(desktop, tmp_path, monkeypatch):
    _, launched = desktop
    launchers = []
    monkeypatch.setattr(session, "add_launcher", lambda name, folder: launchers.append((name, folder)))
    monkeypatch.chdir(tmp_path)

    first, restarted = Mail(), Mail()
    await first._computer_app.start()
    os.utime(tmp_path / "launched" / tmp_path.name, (time.time() - 100,) * 2)
    await restarted._computer_app.start()

    assert len(launched) == 1 and launchers == [("mail", tmp_path)] * 2
    assert (tmp_path / "launched" / tmp_path.name).exists()
    answer = await restarted._computer_app_ready()
    assert answer["launched"] is True and answer["waited_s"] >= 100


@pytest.mark.asyncio
async def test_a_restarted_server_finds_the_demand_app_it_launched_and_leaves_it_be(desktop, tmp_path, monkeypatch):
    _, launched = desktop
    monkeypatch.setattr(session, "add_launcher", lambda name, folder: None)
    monkeypatch.setenv("ENVIRONMENT_NAME", "music")

    opened, restarted = Music(), Music()
    await opened._computer_app.start()
    await opened._computer_app_launch()
    await restarted._computer_app.start()

    assert len(launched) == 1 and (await restarted._computer_app_ready())["launched"] is True


@pytest.mark.asyncio
async def test_a_sync_check_past_its_bound_runs_once_however_often_readiness_is_asked(desktop, monkeypatch):
    monkeypatch.setattr(app_runtime, "CHECK_TIMEOUT_S", 0.2)
    released, calls = threading.Event(), []

    class Syncing(ComputerAppEnvironment):
        @ready_check
        def synced(self):
            calls.append(len(calls))
            return released.wait(5)

    syncing = Syncing()
    stuck = [(await syncing._computer_app_ready())["problems"] for _ in range(3)]
    released.set()
    while not syncing._computer_app.checking["synced"].done():
        await asyncio.sleep(0.01)
    after = await syncing._computer_app_ready()

    assert stuck == [["@ready_check synced ran past 0.2 s"]] * 3
    assert after["ready"] is True and calls == [0, 1]


@pytest.mark.asyncio
async def test_a_boot_app_whose_launch_fails_says_why_and_is_launched_again_by_its_next_server(desktop, tmp_path,
                                                                                              monkeypatch):
    async def missing(argv, cwd, env):
        raise FileNotFoundError(f"No such file or directory: {argv[0]!r}")

    monkeypatch.setattr(session, "add_launcher", lambda name, folder: None)
    monkeypatch.setattr(session, "launch", missing)
    mail = Mail()
    await mail._computer_app.start()

    assert (await mail._computer_app_ready())["problems"][0] == (
        "its launch failed: FileNotFoundError: No such file or directory: 'mail'")
    assert not (tmp_path / "launched").exists()


@pytest.mark.asyncio
async def test_the_driver_asks_each_apps_server_on_the_socket_in_its_folder(desktop, monkeypatch):
    shown, launched = desktop
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="apps-") as root:
        monkeypatch.setattr(apps, "APPS", Path(root))
        (Path(root) / "gone").mkdir()
        (Path(root) / "broken" / SOCKET).parent.mkdir(parents=True)
        (Path(root) / "broken" / FAILURE).write_text("RuntimeError: no mailbox")
        async with contextlib.AsyncExitStack() as stack:
            served = {name: await stack.enter_async_context(_serving(cls(), Path(root) / name, monkeypatch))
                      for name, cls in [("mail", Mail), ("music", Music), ("sync", Sync)]}
            first = [await apps.check(folder) for folder in apps.expected()]
            served["mail"]._computer_app.started -= 120
            late = await apps.check(Path(root) / "mail")
            monkeypatch.setenv("ENVIRONMENT_NAME", "music")
            await asyncio.to_thread(launcher.main, [str(Path(root) / "music")])
            opened = await apps.check(Path(root) / "music")
            shown.append({"instance": "music", "class": "Music", "title": "Library"})
            playing = await apps.check(Path(root) / "music")

    assert first == [
        ReadyCheck("app broken", False, "its server failed to start: RuntimeError: no mailbox"),
        ReadyCheck("app gone", False, "its server isn't answering yet"),
        ReadyCheck("app mail", False, "not ready yet, 0 s in: not launched yet; no window with class 'Mail'; "
                                      "@ready_check inbox_synced returned False"),
        ReadyCheck("app music", True, "its server answers; it launches on demand"),
        ReadyCheck("app sync", True, "ready"),
    ]
    assert late.detail.startswith("not ready after 30 s: not launched yet")
    assert launched == [(["music"], Path.cwd(), {"ENVIRONMENT_NAME": "music"})]
    assert opened == ReadyCheck("app music", False, "not ready yet, 0 s in: no window with class 'Music'")
    assert playing == ReadyCheck("app music", True, "ready")


@pytest.mark.asyncio
async def test_the_drivers_readiness_asks_every_apps_server_at_once(desktop, monkeypatch):
    pytest.importorskip("PIL")
    from computer_fake import FakeDriver, http_client

    class Settling(ComputerAppEnvironment):
        @ready_check
        async def settled(self):
            await asyncio.sleep(1)
            return True

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="apps-") as root:
        monkeypatch.setattr(apps, "APPS", Path(root))
        async with contextlib.AsyncExitStack() as stack:
            for name in ("mail", "music", "sync"):
                await stack.enter_async_context(_serving(Settling(), Path(root) / name, monkeypatch))
            started = time.monotonic()
            async with http_client(FakeDriver()) as http:
                status = (await http.get("/agentenv/ext/computer-ready/status")).json()
            took = time.monotonic() - started

    assert [check for check in status["checks"] if check["name"].startswith("app ")] == [
        {"name": f"app {name}", "ok": True, "detail": "ready"} for name in ("mail", "music", "sync")]
    assert status["ready"] is True and took < 2.5


@linux(paths=["Mail"])
class Inbox(ComputerAppEnvironment):
    pass


@pytest.mark.asyncio
async def test_the_driver_lists_its_apps_as_children_forwards_each_ones_requests_and_leaves_their_paths_out(
        tmp_path, monkeypatch):
    from computer_fake import FakeDriver, http_client

    monkeypatch.setenv("HOME", str(tmp_path))
    for path in ("Mail/inbox.txt", "Documents/note.txt"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(path)
    get = {"jsonrpc": "2.0", "id": 1, "method": "data/get", "params": {}}
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="apps-") as root:
        monkeypatch.setattr(apps, "APPS", Path(root))
        (Path(root) / "music").mkdir()
        async with _serving(Inbox(), Path(root) / "mail", monkeypatch):
            async with http_client(FakeDriver()) as http:
                card = (await http.get(WELL_KNOWN_PATH)).json()
                mail = client.find_child(card, "mail")
                forwarded = (await http.post(mail["url"], json=get)).json()["result"]["parts"]
                unknown = await http.post("/svc/notes/agentenv", json=get)
                silent = await http.post("/svc/music/agentenv", json=get)
                own = (await http.post("/agentenv", json={**get, "params": {"paths": ["Documents", "Mail"]}})).json()

    assert [(child["name"], child["url"]) for child in card["children_environments"]] == [
        ("mail", "/svc/mail/agentenv"), ("music", "/svc/music/agentenv")]
    assert client.extension_params(mail, COMPUTER_APP)["paths"] == ["Mail"]
    assert client.find_extension_method(mail, COMPUTER_APP, "ready")["endpoint"].startswith("/svc/mail/")
    assert [(part["metadata"][DATA_OBJECTS_EXTENSION_URI]["path"], base64.b64decode(part["file"]["bytes"]))
            for part in forwarded] == [("Mail/inbox.txt", b"Mail/inbox.txt")]
    assert (unknown.status_code, unknown.json()["error"]["code"]) == (404, "target_unavailable")
    assert (silent.status_code, silent.json()["error"]["code"]) == (502, "delivery_failed")
    assert [part["metadata"][DATA_OBJECTS_EXTENSION_URI]["path"] for part in own["result"]["parts"]] == [
        "Documents/note.txt"]


@pytest.mark.asyncio
async def test_the_session_launches_an_app_in_a_session_of_its_own_from_its_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    script = "import os; open('seen', 'w').write(f\"{os.getcwd()} {os.environ['ENVIRONMENT_NAME']} " \
             "{os.environ['DISPLAY']} {os.getsid(0) == os.getpid()}\")"

    await session.launch([sys.executable, "-c", script], tmp_path, {"ENVIRONMENT_NAME": "notes"})
    await asyncio.gather(*session._running)

    assert (tmp_path / "seen").read_text() == f"{os.path.realpath(tmp_path)} notes :99 True"


@pytest.mark.asyncio
async def test_a_stop_ends_what_the_app_started_though_its_first_process_exited(tmp_path):
    sleeper = f"{sys.executable} -c 'import time; time.sleep(60)' & exit 0"
    pid = await session.launch(["sh", "-c", sleeper], tmp_path, {})
    await asyncio.gather(*session._running)

    running = session.alive(pid)
    await session.stop(pid, 5)

    assert running and not session.alive(pid)


def test_the_launcher_runs_the_app_through_its_own_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = Path("/opt/agentenv/apps/notes")

    entry = session.add_launcher("Notes", folder)

    assert entry == tmp_path / ".local/share/applications/agentenv-app-notes.desktop"
    assert entry.read_text() == ("[Desktop Entry]\nType=Application\nName=Notes\nExec=/opt/agentenv/apps/notes/venv/bin/"
                                 "python -m agentenv_protocol.computer.launch /opt/agentenv/apps/notes\nTerminal=false\n")


@pytest.mark.asyncio
async def test_the_sessions_windows_are_listed_by_class_instance_and_title_from_either_name(monkeypatch):
    replies = {("-root", "_NET_CLIENT_LIST"): "_NET_CLIENT_LIST(WINDOW): window id # 0x1e00004, 0x2000003, 0x2200001\n",
               ("-id", "0x1e00004", "WM_CLASS", "_NET_WM_NAME", "WM_NAME"):
                   '_NET_WM_NAME(UTF8_STRING) = "Inbox \\"work\\""\nWM_CLASS(STRING) = "mail", "Mail"\n'
                   'WM_NAME(STRING) = "Inbox"\n',
               ("-id", "0x2000003", "WM_CLASS", "_NET_WM_NAME", "WM_NAME"): "",
               ("-id", "0x2200001", "WM_CLASS", "_NET_WM_NAME", "WM_NAME"):
                   'WM_CLASS(STRING) = "xcalc", "XCalc"\n_NET_WM_NAME:  not found.\nWM_NAME(STRING) = "Calculator"\n'}

    async def xprop(*args):
        return replies[args]

    monkeypatch.setattr(session, "_xprop", xprop)
    assert await session.windows() == [{"instance": "mail", "class": "Mail", "title": 'Inbox "work"'},
                                       {"instance": "xcalc", "class": "XCalc", "title": "Calculator"}]


@pytest.mark.asyncio
async def test_a_window_listing_cut_short_kills_its_xprop(monkeypatch):
    started, spawn = [], asyncio.create_subprocess_exec

    async def hanging(*args, **kwargs):
        started.append(await spawn("sleep", "30", **kwargs))
        return started[-1]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", hanging)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(session.windows(), 0.2)

    assert await asyncio.wait_for(started[0].wait(), 5) == -signal.SIGKILL


@contextlib.asynccontextmanager
async def _http(app: ComputerAppEnvironment):
    asgi = app.create_app().streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://t") as http:
        yield http


@contextlib.asynccontextmanager
async def _serving(app: ComputerAppEnvironment, folder: Path, monkeypatch):
    """``app`` served on the socket in ``folder``, named for the folder, as its computer's program serves it."""
    (folder / SOCKET).parent.mkdir(parents=True)
    monkeypatch.setenv("ENVIRONMENT_NAME", folder.name)
    server = uvicorn.Server(uvicorn.Config(app.create_app().streamable_http_app(), uds=str(folder / SOCKET),
                                           log_level="warning", lifespan="off", ws="none"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        if serving.done():
            raise RuntimeError(f"{folder.name}'s server stopped before it served: {serving.exception()!r}")
        await asyncio.sleep(0.01)
    try:
        yield app
    finally:
        server.should_exit = True
        await serving
