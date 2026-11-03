"""Serving a driver or an app from its file: the class is imported with its folder importable and served; anything
that isn't such a class exits saying why, and an app's server that can't start says why in the app's folder."""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")
pytest.importorskip("uvicorn")

from agentenv_protocol import CONTROL_TOKEN_ENV  # noqa: E402
from agentenv_protocol.computer import ComputerDriverEnvironment, run_install_step, serve_app  # noqa: E402
from agentenv_protocol.computer.app import BUILD_SETTINGS, FAILURE, SOCKET  # noqa: E402
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


STEPS = '''from pathlib import Path

from pydantic import BaseModel

from agentenv_protocol.computer import ComputerAppEnvironment, install_step


class Notes(ComputerAppEnvironment):
    class Settings(BaseModel):
        token: str

    class BuildSettings(BaseModel):
        folder: str = "notes"
        edition: str

    def __init__(self):
        raise RuntimeError("no deploy's settings at build")

    @install_step()
    def make_folder(self):
        Path(self.build_settings.folder).mkdir()

    @install_step(scope="session")
    async def write_index(self):
        Path(self.build_settings.folder, "index").write_text(self.build_settings.edition)

    @install_step()
    def read_token(self):
        Path("token").write_text(self.settings.token)

    def helper(self):
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


@pytest.fixture
def app_folder(tmp_path, monkeypatch, isolated):
    """An app's folder, holding ``APP`` and what its last server left, whose server only runs its app's lifespan, as
    uvicorn runs it around serving; each server is recorded."""
    (tmp_path / "app.py").write_text(APP)
    (tmp_path / SOCKET).parent.mkdir()
    (tmp_path / SOCKET).write_text("left by the last server")
    (tmp_path / FAILURE).write_text("RuntimeError: the last start failed")
    monkeypatch.chdir(tmp_path)
    ran = []

    def run(server):
        async def lifespan():
            async with server.config.app.router.lifespan_context(server.config.app):
                await asyncio.sleep(0.05)

        asyncio.run(lifespan())
        ran.append(server)

    monkeypatch.setattr(serve_app.uvicorn.Server, "run", run)
    return tmp_path, ran


def test_an_apps_server_serves_on_the_socket_in_its_folder_with_no_control_token_and_starts_the_app_once_it_serves(
        app_folder, monkeypatch):
    folder, ran = app_folder
    started = []
    monkeypatch.setenv(CONTROL_TOKEN_ENV, "s3cret")

    async def start(runtime):
        started.append(runtime)

    monkeypatch.setattr(AppRuntime, "start", start)

    serve_app.main(["app.py:Notes"])

    (server,) = ran
    assert CONTROL_TOKEN_ENV not in os.environ
    assert server.config.uds == str(SOCKET) and not server.should_exit
    assert not (folder / SOCKET).exists() and not (folder / FAILURE).exists()
    assert len(started) == 1 and isinstance(started[0], AppRuntime)


def test_an_apps_server_whose_start_fails_stops_saying_why_in_the_apps_folder(app_folder, monkeypatch):
    folder, ran = app_folder

    async def start(runtime):
        raise PermissionError("~/.local/share/applications isn't writable")

    monkeypatch.setattr(AppRuntime, "start", start)

    with pytest.raises(PermissionError):
        serve_app.main(["app.py:Notes"])

    assert ran[0].should_exit is True
    assert (folder / FAILURE).read_text() == "PermissionError: ~/.local/share/applications isn't writable"


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


def test_an_apps_install_step_runs_without_its_init_on_the_build_settings_its_env_fixed(tmp_path, monkeypatch,
                                                                                        isolated):
    (tmp_path / "app.py").write_text(STEPS)
    (tmp_path / BUILD_SETTINGS).parent.mkdir()
    (tmp_path / BUILD_SETTINGS).write_text('{"edition": "pro"}')
    monkeypatch.chdir(tmp_path)

    run_install_step.main(["app.py:Notes", "make_folder"])
    run_install_step.main(["app.py:Notes", "write_index"])

    assert (tmp_path / "notes" / "index").read_text() == "pro"
    with pytest.raises(AttributeError, match="self.settings.token: an install step runs at build, before any deploy "
                                             "gives the app its settings; it reads self.build_settings"):
        run_install_step.main(["app.py:Notes", "read_token"])
    with pytest.raises(SystemExit, match="Notes has no @install_step method 'helper'"):
        run_install_step.main(["app.py:Notes", "helper"])
    with pytest.raises(SystemExit, match="usage: "):
        run_install_step.main(["app.py:Notes"])
