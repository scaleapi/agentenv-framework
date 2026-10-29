"""Serving a driver or an app from its file: the class is imported with its folder importable and served; anything
that isn't such a class exits saying why, and an app's server that can't start says why in the app's folder."""
from __future__ import annotations

import asyncio
import sys

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")
pytest.importorskip("uvicorn")

from agentenv_protocol.computer import ComputerDriverEnvironment, serve_app  # noqa: E402
from agentenv_protocol.computer.app import FAILURE, SOCKET  # noqa: E402
from agentenv_protocol.computer.app_runtime import AppRuntime  # noqa: E402
from agentenv_protocol.computer.serve import USAGE, main  # noqa: E402

DRIVER = '''from desk_helpers import SIZE

from agentenv_protocol.computer import (ComputerDriverEnvironment, Display, capture, click, computer_driver, display,
                                        drag, press, scroll, type_text)


@computer_driver(name="desk", supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64"]}])
class Desk(ComputerDriverEnvironment):
    @display
    def size(self): return Display(*SIZE)

    @capture
    def grab(self): ...

    @click
    def click(self, x, y, button, count): ...

    @drag
    def drag(self, path, button): ...

    @scroll
    def scroll(self, x, y, dx, dy): ...

    @type_text
    def type(self, text): ...

    @press
    def press(self, keys): ...
'''


APP = '''from agentenv_protocol.computer import ComputerAppEnvironment


class Helper:
    pass


class Notes(ComputerAppEnvironment):
    pass


class Broken(ComputerAppEnvironment):
    def __init__(self):
        raise RuntimeError("no mailbox")


class Skipping(ComputerAppEnvironment):
    def __init__(self):
        pass
'''


@pytest.fixture
def isolated(monkeypatch):
    """Imports that a class's file makes, undone after the test."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))


@pytest.fixture
def served(monkeypatch, isolated):
    calls = []
    monkeypatch.setattr(ComputerDriverEnvironment, "serve", lambda self: calls.append(type(self).__name__))
    return calls


def test_the_class_is_imported_from_its_file_with_its_folder_importable_and_served(tmp_path, served):
    (tmp_path / "driver.py").write_text(DRIVER)
    (tmp_path / "desk_helpers.py").write_text("SIZE = (10, 10)\n")

    main([f"{tmp_path / 'driver.py'}:Desk"])

    assert served == ["Desk"]


@pytest.mark.parametrize("argv, message", [
    ([], USAGE), (["driver.py"], USAGE), (["missing.py:Desk"], "missing.py isn't a file"),
    (["driver.py:Helper"], "has no ComputerDriverEnvironment subclass named Helper"),
], ids=["nothing", "no-class", "no-file", "not-a-driver"])
def test_what_isnt_a_driver_class_exits_saying_why(tmp_path, served, monkeypatch, argv, message):
    (tmp_path / "driver.py").write_text("class Helper:\n    pass\n")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as stopped:
        main(argv)

    assert message in str(stopped.value) and served == []


@pytest.mark.asyncio
async def test_an_apps_server_serves_on_the_socket_in_its_folder_and_starts_the_app_once_it_serves(tmp_path,
                                                                                                    monkeypatch,
                                                                                                    isolated):
    (tmp_path / "app.py").write_text(APP)
    (tmp_path / SOCKET).parent.mkdir()
    (tmp_path / SOCKET).write_text("left by the last server")
    (tmp_path / FAILURE).write_text("RuntimeError: the last start failed")
    monkeypatch.chdir(tmp_path)
    started, ran = [], []

    async def start(runtime):
        started.append(runtime)

    monkeypatch.setattr(AppRuntime, "start", start)
    monkeypatch.setattr(serve_app.uvicorn, "run", lambda asgi, uds, log_level: ran.append((asgi, uds)))

    serve_app.main(["app.py:Notes"])
    (asgi, uds), = ran
    async with asgi.router.lifespan_context(asgi):
        await asyncio.sleep(0)

    assert uds == str(SOCKET) and not (tmp_path / SOCKET).exists() and not (tmp_path / FAILURE).exists()
    assert len(started) == 1 and isinstance(started[0], AppRuntime)


@pytest.mark.parametrize("argv, failure", [
    ([], f"SystemExit: {serve_app.USAGE}"), (["app.py"], f"SystemExit: {serve_app.USAGE}"),
    (["missing.py:Notes"], "SystemExit: missing.py isn't a file"),
    (["app.py:Helper"], "SystemExit: app.py has no ComputerAppEnvironment subclass named Helper"),
    (["app.py:Broken"], "RuntimeError: no mailbox"),
    (["app.py:Skipping"], "TypeError: Skipping.__init__ must call super().__init__()"),
], ids=["nothing", "no-class", "no-file", "not-an-app", "fails-to-construct", "skips-super-init"])
def test_an_apps_server_that_cant_start_says_why_in_the_apps_folder(tmp_path, monkeypatch, isolated, argv, failure):
    (tmp_path / "app.py").write_text(APP)
    (tmp_path / SOCKET).parent.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises((SystemExit, RuntimeError, TypeError)) as stopped:
        serve_app.main(argv)

    assert (tmp_path / FAILURE).read_text() == failure == f"{type(stopped.value).__name__}: {stopped.value}"
