"""The apps bundle's libreoffice as a put reads it: the installed ``apps`` bundle names it ``libreoffice``; its class
declares Writer, Impress and Calc from apt with GTK3 and the fonts with Office's metrics, its profile as its data, its
skill, and a launch on demand of the Start Center, ready once any LibreOffice window shows; its install step writes its
defaults at the series installed and each LibreOffice desktop entry in its place, opening through ``open.py``; and
``open.py`` has the server launch the app when none runs, before handing it the files, a CSV with its own filter."""
import sys
from pathlib import Path

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
        return type("Ran", (), {"stdout": "4:24.2.7-0ubuntu0.24.04.7"})()

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
    assert '<prop oor:name="ooSetupLastVersion" oor:op="fuse"><value>24.2</value></prop>' in defaults
    assert defaults == (FOLDER / "agentenv.xcd").read_text().replace(
        "<value>24.2</value></prop>", "<value>24.2</value></prop>")
    assert ran[-1] == ["update-desktop-database", str(overrides)]


@pytest.mark.parametrize("argv, calls", [
    ([], [[]]),
    (["--writer"], [["--writer"]]),
    (["a.docx", "b.pptx"], [["a.docx", "b.pptx"]]),
    (["costs.CSV"], [["--infilter=CSV:44,34,76,1", "costs.CSV"]]),
    (["a.docx", "costs.csv"], [["a.docx"], ["--infilter=CSV:44,34,76,1", "costs.csv"]]),
])
def test_a_csv_opens_on_its_own_with_its_filter_after_the_rest(opener, argv, calls):
    assert opener.invocations(argv) == calls


@pytest.mark.parametrize("running, argv, launched, opened", [
    (True, ["a.docx"], False, [["a.docx"]]),
    (False, ["a.docx"], True, [["a.docx"]]),
    (False, [], True, []),
    (True, [], False, [[]]),
])
def test_it_has_the_server_launch_the_app_when_none_runs_then_hands_it_the_files(opener, monkeypatch, running, argv,
                                                                                 launched, opened):
    calls = []
    monkeypatch.setattr(opener, "running", lambda: running)
    monkeypatch.setattr(opener, "launched_by_its_server", lambda: calls.append("launch"))
    monkeypatch.setattr(opener.subprocess, "run", lambda argv, **kwargs: calls.append(argv[1:]))
    monkeypatch.setattr(opener.os, "execv", lambda path, argv: calls.append(argv[1:]))

    opener.main(argv)

    assert calls == (["launch"] if launched else []) + opened
    assert all(Path(opener.SOFFICE).name == "soffice" for _ in calls)
