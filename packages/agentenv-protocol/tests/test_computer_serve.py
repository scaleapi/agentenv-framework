"""Serving a driver from its file: the class is imported with its folder importable and served; anything that isn't
a driver class exits saying why."""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from agentenv_protocol.computer import ComputerDriverEnvironment  # noqa: E402
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


@pytest.fixture
def served(monkeypatch):
    calls = []
    monkeypatch.setattr(ComputerDriverEnvironment, "serve", lambda self: calls.append(type(self).__name__))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "modules", dict(sys.modules))
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
