"""Reading a driver class where its own dependencies aren't installed: a module no finder finds imports as an inert
stub, the class is read without running it, and what it declares for each platform it supports comes back."""
from __future__ import annotations

import json
import re
import sys

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from agentenv_protocol.computer._declare import _Stub, _StubModule, declare, main  # noqa: E402

DRIVER = '''import not_installed_here
from not_installed_here.input import Base, Keyboard

from agentenv_protocol.computer import (ComputerDriverEnvironment, Display, capture, click, computer_driver, display,
                                        drag, press, scroll, type_text)

SHIFT = Keyboard.keysym("Shift_L")[0]
print("loading the desk driver")


class Connection(Base):
    pass


@computer_driver(name="desk", supports={supports})
class Desk(ComputerDriverEnvironment):
    buttons = {buttons}

    @display
    def size(self): return Display(10, 10)

    @capture
    def grab(self): return not_installed_here.grab()

    @click
    def click_at(self, x, y, button, count, modifiers): pass

    @drag
    def drag_along(self, path, button, duration_ms, modifiers): pass

    @scroll
    def scroll_at(self, x, y, direction, amount, modifiers): pass

    @type_text
    def type_in(self, text): pass

    @press
    def press_keys(self, keys): pass
'''
LINUX = [{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}]


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """A driver's folder; the import path, the finders and the stub modules a declaration adds are taken back."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    yield tmp_path
    for name in [name for name, module in sys.modules.items() if isinstance(module, _StubModule)]:
        del sys.modules[name]


def _driver(folder, supports=LINUX, buttons=("left", "right")):
    (folder / "driver.py").write_text(DRIVER.format(supports=supports, buttons=buttons))


def test_a_class_whose_imports_arent_installed_declares_what_it_serves_on_each_platform(folder):
    _driver(folder)

    declared = declare(folder, "driver.py", "Desk")

    assert declared["name"] == "desk" and declared["supports"] == LINUX
    assert [p["platform"] for p in declared["platforms"]] == [{"os": "linux", "arch": "amd64", "display": "x11"},
                                                              {"os": "linux", "arch": "arm64", "display": "x11"}]
    linux = declared["platforms"][0]
    assert {"urn:agentenv:computer/v1", "urn:agentenv:computer-exec/v1"} <= set(linux["provides"])
    assert len(linux["tools"]) == 7 and linux["features"]["buttons"] == ["left", "right"]


def test_a_stub_is_itself_again_except_for_dunders(folder):
    _driver(folder)
    declare(folder, "driver.py", "Desk")
    stub = sys.modules["not_installed_here"]

    assert isinstance(stub, _StubModule) and stub.anything is _Stub
    assert _Stub.a.b(1, k=2)[0] is _Stub and isinstance(_Stub, type)
    assert not hasattr(stub, "__version__") and not hasattr(_Stub, "__wrapped__")


@pytest.mark.parametrize("name, supports, problem", [
    ("Connection", LINUX, "driver.py has no ComputerDriverEnvironment subclass named Connection"),
    ("Missing", LINUX, "driver.py has no ComputerDriverEnvironment subclass named Missing"),
    ("Desk", [], "Desk declares no platform: name them with @computer_driver(supports=[...])"),
])
def test_a_name_that_isnt_a_driver_or_declares_no_platform_is_refused(folder, name, supports, problem):
    _driver(folder, supports=supports)

    with pytest.raises(ValueError, match=re.escape(problem)):
        declare(folder, "driver.py", name)


def test_main_prints_only_the_declaration_or_exits_1_with_why_it_cant(folder, capsys):
    _driver(folder)
    assert main([str(folder), "driver.py", "Desk"]) == 0
    printed = capsys.readouterr()
    assert json.loads(printed.out)["name"] == "desk" and printed.err == "loading the desk driver\n"

    _driver(folder, buttons=("thumb",))
    assert main([str(folder), "driver.py", "Desk"]) == 1
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err.splitlines()[-1].startswith("TypeError: Desk: buttons include left")
