"""The apps bundle's libreoffice as a put reads it: the installed ``apps`` bundle names it ``libreoffice``; its class
declares Writer, Impress and Calc from apt with GTK3 and the fonts with Office's metrics, its profile as its data, its
skill, and a launch on demand of the Start Center, ready once any LibreOffice window shows; its install step writes its
defaults at the series installed and each LibreOffice desktop entry in its place, opening through ``open.py``; and
``open.py`` leaves what it opens for the server's launch, a CSV with its own filter, launching again once the app is
ready when the launch had to start it, and the server's launch hands what waits to the copy that runs."""
import asyncio
import json
import os
import sys
from contextlib import contextmanager

import pytest

from agentenv_protocol.computer._load import load_class
from agentenv_protocol.computer.app import ComputerAppEnvironment

from agent_env.bundle import parse_bundle
from agent_env.bundle.installed import find_bundle

FOLDER = find_bundle("apps").root / "envs" / "libreoffice"


@pytest.fixture
def libreoffice():
    sys.modules.pop("libreoffice", None)
    cls = load_class(FOLDER / "libreoffice.py", "LibreOffice", ComputerAppEnvironment)
    yield sys.modules[cls.__module__]
    sys.modules.pop("libreoffice", None)


@pytest.fixture
def opener():
    sys.modules.pop("libreoffice_open", None)
    import importlib.util
    spec = importlib.util.spec_from_file_location("libreoffice_open", FOLDER / "open.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module


def test_the_apps_bundle_names_its_libreoffice_app_libreoffice():
    bundle = find_bundle("apps")
    parsed = parse_bundle(bundle.root, id_root=bundle.id_root)

    assert [(entry.id, entry.type) for entry in parsed.entries] == [("@local/agentenv-framework/apps/libreoffice",
                                                                     "computer_app")]
    assert parsed.aliases == {"envs": {"libreoffice": ("libreoffice",)}}


def test_it_declares_its_packages_profile_skill_and_a_launch_on_demand_ready_on_any_of_its_windows(libreoffice):
    declared = libreoffice.LibreOffice.declaration()
    linux = declared.os["linux"]

    assert {"libreoffice-writer", "libreoffice-impress", "libreoffice-calc", "libreoffice-gtk3",
            "fonts-crosextra-carlito", "fonts-crosextra-caladea", "fonts-liberation2"} == set(
        linux.install["packages"]["apt"])
    assert linux.paths == [".config/libreoffice"] and linux.skill.file == "SKILL.md"
    assert (linux.launch.at, linux.launch.command, linux.launch.window) == (
        "demand", ["/usr/lib/libreoffice/program/soffice", "--nologo"], {"instance": "libreoffice"})
    assert linux.launch.ready.window
    assert [(step.name, step.scope) for step in declared.install_steps] == [("configure", "system")]
    assert declared.supports == [{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}]


def test_its_install_step_writes_its_defaults_at_the_series_installed_and_points_each_launcher_at_open(
        libreoffice, tmp_path, monkeypatch):
    entries, overrides, registry = tmp_path / "share", tmp_path / "local", tmp_path / "registry"
    entries.mkdir()
    registry.mkdir()
    (entries / "libreoffice-writer.desktop").write_text(
        "[Desktop Entry]\nName=LibreOffice Writer\nExec=libreoffice --writer %U\nMimeType=application/msword;\n"
        "[Desktop Action NewDocument]\nExec=libreoffice --writer\n")
    (entries / "other.desktop").write_text("[Desktop Entry]\nExec=other\n")
    ran = []

    def run(argv, **kwargs):
        ran.append(argv)
        return type("Ran", (), {"stdout": "4:25.8.1-0ubuntu1"})()

    monkeypatch.setattr(libreoffice, "DESKTOP_ENTRIES", entries)
    monkeypatch.setattr(libreoffice, "OVERRIDES", overrides)
    monkeypatch.setattr(libreoffice, "REGISTRY", registry)
    monkeypatch.setattr(libreoffice.subprocess, "run", run)
    monkeypatch.chdir(FOLDER)

    libreoffice.LibreOffice.configure(object.__new__(libreoffice.LibreOffice))

    opener = f"{FOLDER}/venv/bin/python {FOLDER}/open.py"
    assert (overrides / "libreoffice-writer.desktop").read_text() == (
        f"[Desktop Entry]\nName=LibreOffice Writer\nExec={opener} --writer %U\nMimeType=application/msword;\n"
        f"[Desktop Action NewDocument]\nExec={opener} --writer\n")
    assert sorted(path.name for path in overrides.iterdir()) == ["libreoffice-writer.desktop"]
    defaults = (registry / "agentenv.xcd").read_text()
    shipped = (FOLDER / "agentenv.xcd").read_text()
    assert '<prop oor:name="ooSetupLastVersion" oor:op="fuse"><value>25.8</value></prop>' in defaults
    assert defaults == shipped.replace('"ooSetupLastVersion" oor:op="fuse"><value>24.2<',
                                       '"ooSetupLastVersion" oor:op="fuse"><value>25.8<') != shipped
    assert ran[-1] == ["update-desktop-database", str(overrides)]


@pytest.mark.parametrize("argv, calls", [
    ([], [[]]),
    (["--writer"], [["--writer"]]),
    (["a.docx", "file:///b.pptx"], [["{cwd}/a.docx", "file:///b.pptx"]]),
    (["costs.CSV"], [["--infilter=CSV:44,34,76,1", "{cwd}/costs.CSV"]]),
    (["a.docx", "/data/costs.csv"], [["{cwd}/a.docx"], ["--infilter=CSV:44,34,76,1", "/data/costs.csv"]]),
])
def test_each_file_opens_by_its_absolute_path_a_csv_on_its_own_with_its_filter_after_the_rest(opener, tmp_path,
                                                                                               monkeypatch, argv,
                                                                                               calls):
    monkeypatch.chdir(tmp_path)

    assert opener.invocations(argv) == [[arg.format(cwd=os.getcwd()) for arg in call] for call in calls]


@pytest.mark.parametrize("argv, left", [(["/d/a.docx", "/d/costs.csv"], [["/d/a.docx"], [
    "--infilter=CSV:44,34,76,1", "/d/costs.csv"]]), ([], None)])
def test_it_leaves_what_it_opens_for_the_servers_launch_and_takes_it_back_after(opener, tmp_path, monkeypatch, argv,
                                                                                left):
    handed = []
    monkeypatch.setattr(opener, "OPENS", tmp_path / "opens")
    monkeypatch.setattr(opener, "handed_on", lambda request: handed.append(
        request and (request.parent, json.loads(request.read_text()))))

    opener.main(argv)

    assert handed == [left and (tmp_path / "opens", left)]
    assert not list((tmp_path / "opens").glob("*")) if left else not (tmp_path / "opens").exists()


@pytest.mark.parametrize("running, posts", [(True, 1), (False, 2)])
def test_it_launches_again_once_the_app_is_ready_when_the_launch_had_to_start_it(opener, tmp_path, monkeypatch,
                                                                                 running, posts):
    request = tmp_path / "1-a.json"
    request.write_text("[]")
    asked = []

    class Http:
        def get(self, path):
            asked.append(path)
            return _answer({"capabilities": {}} if path == "/.well-known/agent-env.json" else {"ready": True})

        def post(self, path, json):
            asked.append(f"POST {path}")
            if running or asked.count(f"POST {path}") == 2:
                request.unlink()
            return _answer({"pid": 7})

    @contextmanager
    def client(timeout):
        yield Http()

    monkeypatch.setattr(opener, "served", lambda folder: type("Served", (), {"sync_client": staticmethod(client)})())
    monkeypatch.setattr(opener, "endpoint", lambda card, op: f"/{op}")

    opener.handed_on(request)

    assert [path for path in asked if path.startswith("POST")] == ["POST /launch"] * posts
    assert ("/ready" in asked) is (not running)


def test_the_servers_launch_hands_what_waits_oldest_first_to_the_copy_that_runs_or_launches_it_bare(
        libreoffice, opener, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / libreoffice.OPENS).mkdir(parents=True)
    (tmp_path / libreoffice.OPENS / "2-b.json").write_text(json.dumps([["--infilter=CSV:44,34,76,1", "/d/c.csv"]]))
    (tmp_path / libreoffice.OPENS / "1-a.json").write_text(json.dumps([["/d/a.docx"]]))
    (tmp_path / libreoffice.OPENS / "3-c.staged").write_text("[")
    ran = []

    async def exec_(*argv):
        ran.append(list(argv))
        return type("Handed", (), {"wait": staticmethod(lambda: asyncio.sleep(0))})()

    monkeypatch.setattr(libreoffice.asyncio, "create_subprocess_exec", exec_)
    hand_on = object.__new__(libreoffice.LibreOffice)._launch_running()

    asyncio.run(hand_on())
    asyncio.run(hand_on())

    soffice = [libreoffice.SOFFICE, "--nologo"]
    assert ran == [[*soffice, "/d/a.docx"], [*soffice, "--infilter=CSV:44,34,76,1", "/d/c.csv"], soffice]
    assert [path.name for path in (tmp_path / libreoffice.OPENS).iterdir()] == ["3-c.staged"]
    assert opener.OPENS == FOLDER / libreoffice.OPENS


def _answer(body: dict):
    return type("Answer", (), {"raise_for_status": lambda self: self, "json": lambda self: body})()
