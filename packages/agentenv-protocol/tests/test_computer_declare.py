"""Reading a driver or app class where its own dependencies aren't installed: a module no finder finds imports as an
inert stub, the class is read without running it, and what it declares comes back: a driver's for each platform it
supports, an app's as ComputerAppEnvironment says. An app's code needs only the plain package."""
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from agentenv_protocol.computer import AppDeclaration  # noqa: E402
from agentenv_protocol.computer._declare import _Stub, _StubModule, declare, main  # noqa: E402
from agentenv_protocol.computer import (APP_SETTINGS_ENV, ComputerAppEnvironment, Launch, Linux,  # noqa: E402
                                        install, linux)
from agentenv_protocol.computer import app as app_module  # noqa: E402
from agentenv_protocol.computer.app import BUILD_SETTINGS  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from agentenv_protocol.computer.platform import Platform  # noqa: E402

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
APP = '''import not_installed_here
from pydantic import BaseModel, SecretStr

from agentenv_protocol import add_data, agent_skill, reset_data
from agentenv_protocol.computer import (AptSource, ComputerAppEnvironment, Download, Launch, Packages, Ready,
                                        RunScript, Service, computer_app, install, install_step, linux, ready_check)


class FreeBSD(BaseModel):
    ports: list[str] = []


class LinuxSettings(BaseModel):
    font: str = "DejaVu Sans"


class LinuxBuildSettings(BaseModel):
    fonts: list[str] = []


class LeakyBuildSettings(BaseModel):
    key: SecretStr = SecretStr("")


@computer_app(supports={supports}, permissions=["camera"])
@linux(files="linux", python="requirements.txt", packages=Packages(apt=["dovecot-imapd"]),
       sources=[AptSource(apt="https://ppa.example/ubuntu", suite="noble", key="ppa.asc")],
       downloads=[Download(url="https://example.com/mbox", sha256="{sha}", to="data/mbox", arch="arm64")],
       run=[RunScript(script="setup.sh"), RunScript(script="profile.sh", scope="session")],
       services=[Service(name="imap", command="bin/imap")], ports=[143],
       launch=Launch(at="demand", command=["thunderbird"], window={{"class": "Thunderbird"}},
                     ready=Ready(window=True, timeout_s=30)),
       settings=LinuxSettings, build_settings=LinuxBuildSettings)
@install("freebsd", FreeBSD(ports=["mail/thunderbird"]), launch=Launch(command=["thunderbird"]))
class Mail(ComputerAppEnvironment):
    class Settings(BaseModel):
        account: str = "user@example.com"
        password: SecretStr = SecretStr("")

    class BuildSettings(BaseModel):
        locale: str = "en_US"
        writeOnly: bool = False
        flags: dict = {{"writeOnly": True}}

    @install_step()
    def packages(self): not_installed_here.apt()

    @install_step(scope="session", os="linux")
    def profile(self): pass

    @ready_check
    def inbox_open(self): return True

    @ready_check(os=["freebsd"])
    def ports_built(self): return True

    @agent_skill
    def guide(self): return "Mail", "Read mail."

    @add_data
    async def load(self, parts): pass

    @reset_data
    async def reset(self): pass


class Twice(Mail):
    skill = "skill.md"
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

    assert declared.name == "desk" and declared.supports == LINUX
    assert [p.platform for p in declared.platforms] == [Platform(os="linux", arch="amd64", display="x11"),
                                                        Platform(os="linux", arch="arm64", display="x11")]
    linux = declared.platforms[0]
    assert {"urn:agentenv:computer/v1", "urn:agentenv:computer-exec/v1"} <= set(linux.provides)
    assert len(linux.tools) == 7 and linux.features["buttons"] == ["left", "right"]


def test_a_stub_is_itself_again_except_for_dunders(folder):
    _driver(folder)
    declare(folder, "driver.py", "Desk")
    stub = sys.modules["not_installed_here"]

    assert isinstance(stub, _StubModule) and stub.anything is _Stub
    assert _Stub.a.b(1, k=2)[0] is _Stub and isinstance(_Stub, type)
    assert not hasattr(stub, "__version__") and not hasattr(_Stub, "__wrapped__")


@pytest.mark.parametrize("name, supports, problem", [
    ("Connection", LINUX, "driver.py has no ComputerDriverEnvironment or ComputerAppEnvironment subclass named Connection"),
    ("Missing", LINUX, "driver.py has no ComputerDriverEnvironment or ComputerAppEnvironment subclass named Missing"),
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
    printed_declaration = json.loads(printed.out)
    assert (printed_declaration["type"], printed_declaration["name"]) == ("computer_driver", "desk")
    assert printed_declaration["platforms"][0]["platform"] == {"os": "linux", "arch": "amd64", "display": "x11"}
    assert printed.err == "loading the desk driver\n"

    _driver(folder, buttons=("thumb",))
    assert main([str(folder), "driver.py", "Desk"]) == 1
    printed = capsys.readouterr()
    assert printed.out == "" and printed.err.splitlines()[-1].startswith("TypeError: Desk: buttons include left")


def test_an_app_class_whose_imports_arent_installed_declares_what_it_is_to_a_computer(folder):
    (folder / "app.py").write_text(APP.format(supports=LINUX, sha='a' * 64))

    declared = declare(folder, "app.py", "Mail")

    assert isinstance(declared, AppDeclaration)
    settings = declared.settings
    schemas = {"settings", "build_settings"}
    assert declared.model_dump(mode="json", exclude_none=True, exclude={**dict.fromkeys(schemas, True), "os": {
            "__all__": schemas}}) == {
        "supports": LINUX, "permissions": ["camera"],
        "os": {
            "linux": {
                "install": {
                    "files": "linux", "packages": {"apt": ["dovecot-imapd"]}, "python": "requirements.txt",
                    "sources": [{"apt": "https://ppa.example/ubuntu", "suite": "noble", "components": [],
                                 "key": "ppa.asc"}],
                    "downloads": [{"url": "https://example.com/mbox", "sha256": "a" * 64, "to": "data/mbox",
                                   "arch": "arm64"}],
                    "run": [{"script": "setup.sh", "scope": "system"}, {"script": "profile.sh", "scope": "session"}],
                    "services": [{"name": "imap", "command": "bin/imap"}], "ports": [143]},
                "launch": {"at": "demand", "command": ["thunderbird"], "window": {"class": "Thunderbird"},
                           "ready": {"window": True, "timeout_s": 30.0}}},
            "freebsd": {"install": {"ports": ["mail/thunderbird"]},
                        "launch": {"at": "boot", "command": ["thunderbird"]}}},
        "skill": {"method": "guide"}, "data": ["data/add", "data/reset"],
        "install_steps": [{"name": "packages", "scope": "system"},
                          {"name": "profile", "scope": "session", "os": ["linux"]}],
        "ready_checks": [{"name": "inbox_open"}, {"name": "ports_built", "os": ["freebsd"]}]}
    assert list(settings["properties"]) == ["account", "password"] and settings["properties"]["password"]["writeOnly"]
    assert list(declared.build_settings["properties"]) == ["locale", "writeOnly", "flags"]
    linux = declared.os["linux"]
    assert set(linux.settings["properties"]) == {"account", "password", "font"}
    assert set(linux.build_settings["properties"]) == {"locale", "writeOnly", "flags", "fonts"}
    assert (declared.os["freebsd"].settings, declared.os["freebsd"].build_settings) == (None, None)


def test_an_app_declaring_its_skill_twice_is_refused(folder):
    (folder / "app.py").write_text(APP.format(supports=LINUX, sha='a' * 64))

    with pytest.raises(ValueError, match=re.escape("Twice declares its skill twice: skill = 'skill.md' and "
                                                   "@agent_skill guide")):
        declare(folder, "app.py", "Twice")


@pytest.mark.parametrize("change, problem", [
    ('@linux(launch=Launch(command=["mail"], ready=Ready(timeout_s=5)))',
     "Ready waits on nothing: set window, port or command"),
    ('@linux(launch=Launch(command=["mail"], ready=Ready(window=True)))',
     "ready waits on a window: name it with Launch(window="),
    ('@linux(launch=Launch(command=["mail"], window={"role": "main"}))',
     "window matches by class, instance, title, not"),
    ('@linux(downloads=[Download(url="https://example.com/t", sha256="abc", to="t")])',
     "Download.sha256: String should match pattern '^[0-9a-f]{64}$'"),
    ('@linux(run=[RunScript(script="../setup.sh")])',
     "RunScript.script: '../setup.sh' must be a path inside the layer's folder"),
    ('@linux(skill="skill-linux.md")',
     "Changed declares its skill twice on linux: skill = 'skill-linux.md' there and @agent_skill guide, which its "
     "server serves on every OS"),
    ('@linux(build_settings=LeakyBuildSettings)',
     "LeakyBuildSettings holds a secret at properties.key, which its computer's image would keep"),
    ('@install_step(os="windows")\n    def fonts(self): pass',
     "Changed.fonts runs on 'windows', which Changed declares no install for; it declares freebsd, linux"),
    ('@ready_check(os="macos")\n    def window_shown(self): return True',
     "Changed.window_shown runs on 'macos', which Changed declares no install for"),
    ('class BuildSettings(BaseModel): key: SecretStr = SecretStr("")',
     "Changed.BuildSettings holds a secret at properties.key, which its computer's image would keep; declare it in "
     "Settings"),
    ('class BuildSettings(BaseModel): keys: list[SecretStr] = []',
     "Changed.BuildSettings holds a secret at properties.keys.items, which"),
    ('class BuildSettings(BaseModel): mail: Mail.Settings = Mail.Settings()',
     "Changed.BuildSettings holds a secret at $defs.Settings.properties.password, which"),
], ids=["no-wait", "no-window", "window-key", "download-sha", "script-escape", "os-skill-twice", "os-build-secret",
        "step-os",
        "check-os", "build-secret", "build-secrets-listed", "build-secret-nested"])
def test_an_app_whose_declaration_cant_be_checked_is_refused_saying_why(folder, capsys, change, problem):
    decorator, body = (change, "pass") if change.startswith("@linux") else ("", change)
    (folder / "app.py").write_text(APP.format(supports=LINUX, sha='a' * 64) + f"""

@computer_app(supports={LINUX})
{decorator}
class Changed(Mail):
    {body}
""")

    assert main([str(folder), "app.py", "Changed"]) == 1
    assert problem in capsys.readouterr().err.splitlines()[-1]


@pytest.mark.parametrize("os_type", ["linux", "macos"])
def test_an_app_reads_its_settings_build_settings_and_launch_as_declared_for_the_os_it_runs_on(monkeypatch, tmp_path,
                                                                                               os_type):
    class LinuxSettings(BaseModel):
        font: str = "DejaVu Sans"

    class LinuxBuildSettings(BaseModel):
        edition: str = "linux"

    @install("linux", Linux(), launch=Launch(command=["notes"]), settings=LinuxSettings,
             build_settings=LinuxBuildSettings)
    class Notes(ComputerAppEnvironment):
        class Settings(BaseModel):
            signature: str = ""
            lines: int = 1

        class BuildSettings(BaseModel):
            edition: str = "home"
            locale: str = "en_US"

    monkeypatch.setattr(app_module, "detect", lambda: Platform(os=os_type))
    monkeypatch.setenv("ENVIRONMENT_NAME", "notes")
    monkeypatch.setenv(APP_SETTINGS_ENV, json.dumps({"notes": {"signature": "-- Ada"}, "mail": {"lines": 9}}))
    monkeypatch.chdir(tmp_path)
    assert Notes().build_settings.edition == ("linux" if os_type == "linux" else "home")
    (tmp_path / BUILD_SETTINGS).parent.mkdir()
    (tmp_path / BUILD_SETTINGS).write_text(json.dumps({"locale": "fr_FR"}))

    notes = Notes()

    assert (notes.settings.signature, notes.settings.lines, notes.build_settings.locale) == ("-- Ada", 1, "fr_FR")
    if os_type == "linux":
        assert (notes.settings.font, notes.build_settings.edition, notes.launch.command) == (
            "DejaVu Sans", "linux", ["notes"])
    else:
        assert (type(notes.settings), notes.build_settings.edition, notes.launch) == (Notes.Settings, "home", None)


def test_a_declaration_stored_before_each_os_had_its_own_part_reads_as_its_linux_part():
    stored = {"supports": LINUX, "permissions": [], "settings": {"type": "object"}, "data": [], "install_steps": [],
              "install": {"linux": {"files": ".", "services": [{"name": "imap", "command": "bin/imap"}]}},
              "launch": {"at": "demand", "command": ["mail"]}, "ready_checks": ["inbox_open"]}

    declared = AppDeclaration.model_validate(stored)

    assert declared.os["linux"].install == stored["install"]["linux"]
    assert declared.os["linux"].launch == Launch(at="demand", command=["mail"])
    assert [check.model_dump(exclude_none=True) for check in declared.ready_checks] == [{"name": "inbox_open"}]
    assert AppDeclaration.model_validate(declared.model_dump(mode="json")) == declared


def test_linux_takes_every_field_of_a_linux_install_and_what_install_adds_for_an_os():
    added = set(inspect.signature(install).parameters) - {"os_type", "spec"}

    assert set(inspect.signature(linux).parameters) == set(Linux.model_fields) | added


def test_an_apps_code_needs_only_the_plain_package():
    blocked = """
import importlib.abc, sys
class Blocked(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("mcp", "av", "PIL", "uvicorn"):
            raise ImportError(name)
sys.meta_path.insert(0, Blocked())
from agentenv_protocol.computer import ComputerAppEnvironment, Linux, computer_app
print(sorted(name for name in sys.modules if name.startswith("agentenv_protocol.computer.")))
"""
    done = subprocess.run([sys.executable, "-c", blocked], capture_output=True, text=True, env=os.environ)

    assert done.returncode == 0, done.stderr
    assert "agentenv_protocol.computer.driver" not in done.stdout
