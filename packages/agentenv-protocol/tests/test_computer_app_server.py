"""A computer app's own server: it adds the app's launcher, launches a boot app once per boot and says whether the app
is ready (``computer-app/v1``); its computer's driver finds it on the socket in the app's folder and asks it."""
from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import socket
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp.server.fastmcp")
uvicorn = pytest.importorskip("uvicorn")

import httpx  # noqa: E402

import agentenv_protocol.computer.launch as launcher  # noqa: E402
from agentenv_protocol import client  # noqa: E402
from agentenv_protocol.computer import ComputerAppEnvironment, Launch, ReadyCheck, Ready, ready_check  # noqa: E402
from agentenv_protocol.computer import app as app_module, apps, serve_app, session  # noqa: E402
from agentenv_protocol.computer.app import COMPUTER_APP, FAILURE, SOCKET  # noqa: E402

READY = "/agentenv/ext/computer-app/ready"
APP = '''from agentenv_protocol.computer import ComputerAppEnvironment


class Helper:
    pass


class Broken(ComputerAppEnvironment):
    def __init__(self):
        raise RuntimeError("no mailbox")
'''


class Mail(ComputerAppEnvironment):
    launch = Launch(command=["mail"], window={"class": "Mail"}, ready=Ready(window=True, timeout_s=30))
    synced = False

    @ready_check
    def inbox_synced(self):
        return self.synced


class Music(ComputerAppEnvironment):
    launch = Launch(at="demand", command=["music"], window={"class": "Music"}, ready=Ready(window=True))


class Sync(ComputerAppEnvironment):
    @ready_check
    async def caught_up(self):
        return True


@pytest.fixture
def desktop(monkeypatch):
    """A session with no windows yet, recording what's launched into it."""
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
    monkeypatch.setenv("ENVIRONMENT_NAME", "mail")
    return shown, launched


@pytest.mark.asyncio
async def test_an_apps_server_is_ready_once_it_launched_the_app_and_each_check_passes(desktop):
    shown, launched = desktop
    mail = Mail()
    async with _http(mail) as http:
        card = (await http.get("/.well-known/agent-env.json")).json()
        before = (await http.get(READY)).json()
        assert (await http.post("/agentenv/ext/computer-app/launch", json={})).json() == {"pid": 4242}
        shown.append({"instance": "mail", "class": "Mail", "title": "Inbox"})
        unsynced = (await http.get(READY)).json()
        mail.synced = True
        after = (await http.get(READY)).json()

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
    monkeypatch.setattr(app_module, "CHECK_TIMEOUT_S", 1.0)
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    exits = [sys.executable, "-c", "import sys; sys.exit('no database yet')"]

    class Slow(ComputerAppEnvironment):
        launch = Launch(command=["slow"], ready=Ready(port=port, command=exits))

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
async def test_a_boot_app_is_launched_once_per_boot_with_its_launcher_added_each_start(desktop, tmp_path, monkeypatch):
    _, launched = desktop
    launchers = []
    monkeypatch.setattr(app_module, "LAUNCHED", tmp_path / "launched")
    monkeypatch.setattr(session, "add_launcher", lambda name, folder: launchers.append((name, folder)))
    monkeypatch.chdir(tmp_path)

    first, restarted = Mail(), Mail()
    await first._computer_app_start()
    await restarted._computer_app_start()

    assert len(launched) == 1 and launchers == [("mail", tmp_path)] * 2
    assert (tmp_path / "launched" / tmp_path.name).exists()
    assert (await restarted._computer_app_ready())["launched"] is True


@pytest.mark.asyncio
async def test_a_boot_app_whose_launch_fails_says_why_and_is_launched_again_by_its_next_server(desktop, tmp_path,
                                                                                              monkeypatch):
    async def missing(argv, cwd, env):
        raise FileNotFoundError(f"No such file or directory: {argv[0]!r}")

    monkeypatch.setattr(app_module, "LAUNCHED", tmp_path / "launched")
    monkeypatch.setattr(session, "add_launcher", lambda name, folder: None)
    monkeypatch.setattr(session, "launch", missing)
    mail = Mail()
    await mail._computer_app_start()

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
            served["mail"]._computer_app_started_at -= 120
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
async def test_serving_an_app_starts_it_once_the_server_serves_on_the_socket_in_its_folder(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("from agentenv_protocol.computer import ComputerAppEnvironment\n\n\n"
                                     "class Notes(ComputerAppEnvironment):\n    pass\n")
    (tmp_path / SOCKET).parent.mkdir()
    (tmp_path / SOCKET).write_text("left by the last server")
    (tmp_path / FAILURE).write_text("RuntimeError: the last start failed")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
    started, ran = [], []
    monkeypatch.setattr(ComputerAppEnvironment, "_computer_app_start", lambda self: _record(started, type(self)))
    monkeypatch.setattr(serve_app.uvicorn, "run", lambda asgi, uds, log_level: ran.append((asgi, uds)))

    serve_app.main(["app.py:Notes"])
    (asgi, uds), = ran
    async with asgi.router.lifespan_context(asgi):
        await asyncio.sleep(0)

    assert uds == str(SOCKET) and not (tmp_path / SOCKET).exists() and not (tmp_path / FAILURE).exists()
    assert [cls.__name__ for cls in started] == ["Notes"]


@pytest.mark.parametrize("argv, failure", [
    ([], f"SystemExit: {serve_app.USAGE}"), (["app.py"], f"SystemExit: {serve_app.USAGE}"),
    (["missing.py:Notes"], "SystemExit: missing.py isn't a file"),
    (["app.py:Helper"], "SystemExit: app.py has no ComputerAppEnvironment subclass named Helper"),
    (["app.py:Broken"], "RuntimeError: no mailbox"),
], ids=["nothing", "no-class", "no-file", "not-an-app", "fails-to-construct"])
def test_a_server_that_cant_start_says_why_in_the_apps_folder(tmp_path, monkeypatch, argv, failure):
    (tmp_path / "app.py").write_text(APP)
    (tmp_path / SOCKET).parent.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))

    with pytest.raises((SystemExit, RuntimeError)) as stopped:
        serve_app.main(argv)

    assert (tmp_path / FAILURE).read_text() == failure == f"{type(stopped.value).__name__}: {stopped.value}"


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


@pytest.mark.asyncio
async def test_the_session_launches_an_app_in_a_session_of_its_own_from_its_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":99")
    script = "import os; open('seen', 'w').write(f\"{os.getcwd()} {os.environ['ENVIRONMENT_NAME']} " \
             "{os.environ['DISPLAY']} {os.getsid(0) == os.getpid()}\")"

    await session.launch([sys.executable, "-c", script], tmp_path, {"ENVIRONMENT_NAME": "notes"})
    await asyncio.gather(*session._running)

    assert (tmp_path / "seen").read_text() == f"{os.path.realpath(tmp_path)} notes :99 True"


def test_the_launcher_runs_the_app_through_its_own_server(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    folder = Path("/opt/agentenv/apps/notes")

    entry = session.add_launcher("Notes", folder)

    assert entry == tmp_path / ".local/share/applications/agentenv-app-notes.desktop"
    assert entry.read_text() == ("[Desktop Entry]\nType=Application\nName=Notes\nExec=/opt/agentenv/apps/notes/venv/bin/"
                                 "python -m agentenv_protocol.computer.launch /opt/agentenv/apps/notes\nTerminal=false\n")


@pytest.mark.asyncio
async def test_the_sessions_windows_are_listed_by_class_instance_and_title(monkeypatch):
    replies = {("-root", "_NET_CLIENT_LIST"): "_NET_CLIENT_LIST(WINDOW): window id # 0x1e00004, 0x2000003\n",
               ("-id", "0x1e00004", "WM_CLASS", "_NET_WM_NAME"):
                   '_NET_WM_NAME(UTF8_STRING) = "Inbox \\"work\\""\nWM_CLASS(STRING) = "mail", "Mail"\n',
               ("-id", "0x2000003", "WM_CLASS", "_NET_WM_NAME"): ""}

    async def xprop(*args):
        return replies[args]

    monkeypatch.setattr(session, "_xprop", xprop)
    assert await session.windows() == [{"instance": "mail", "class": "Mail", "title": 'Inbox \\"work\\"'}]


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


async def _record(started: list, cls: type) -> None:
    started.append(cls)
