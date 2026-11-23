"""``bundle init`` writes a computer env into the folder, as an import that leaves nothing in the store must: its base
computer with the image it's built from, its driver and apps with their files, and the multi env holding it, which a run
of the folder writes back as they were stored; a shipped base or driver stays named by its installed id. The composer is
replaced by a recorder."""

import json

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.bundle import parse_bundle
from agent_env.bundle.importer import import_task
from agent_env.bundle.installed import installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.env import Env
from agent_env.env.envs.computer.env import ComputerEnv
from tst.unit.bundle._support import layout, local_builds
from tst.unit.bundle.computer_app_env_test import APP, TOML as APP_TOML
from tst.unit.bundle.computer_driver_env_test import DRIVER, INSTALLED, LINUX, REQUIREMENTS
from tst.unit.bundle.computer_env_folder_test import DESKTOP, REFERENCE, UBUNTU

BASE = '''type = "computer"
description = "a desk"
arch = ["amd64", "arm64"]
os = { type = "linux", distribution = "ubuntu", version = "24.04", display = "x11" }
session = { user = "user" }

[forms.container]
image = { dockerfile = "container/Dockerfile" }
init = "supervisord"
input_channel = "env"
start_as = "root"

[config.resolution]
default = "1280x800"
phase = "deploy"

[config.theme]
default = "light"
phase = "deploy"
'''
OWN = '''type = "computer"
computer = "base"
driver = "desk"
apps = ["notes", { env = "mail", install = "build", settings = { signature = "-- Ada" } }]
display = { width = 1600, height = 900 }
tools = "both"

[computer_settings]
theme = "dark"
'''
KNOWN = '\n[[known_failures.linux]]\ncheck = "P3"\nname = "drag"\nreason = "xdotool drops the release"\n'
APPS = {f"envs/{name}/{file}": text for name in ("mail", "notes") for file, text in (
    ("env.toml", f'{APP_TOML}build_settings = {{ edition = "pro" }}\n' if name == "mail" else APP_TOML),
    ("app.py", APP.format(supports=LINUX, skill="skill.md", linux_skill="skill-linux.md")),
    ("requirements.txt", "agentenv-framework-protocol\n"), ("skill.md", f"# {name}\n"),
    ("skill-linux.md", f"# {name} on Linux\n"))}
OWN_FILES = {
    "artifacts/base/artifact.toml": BASE,
    "artifacts/base/container/Dockerfile": "FROM scratch\nCOPY container/start /start\n",
    "artifacts/base/container/start": "#!/bin/sh\n",
    "envs/desk/env.toml": INSTALLED + KNOWN,
    "envs/desk/driver.py": DRIVER.format(supports=LINUX, buttons=("left", "right")),
    "envs/desk/linux/requirements.txt": REQUIREMENTS, "envs/desk/linux/ppa.asc": "key",
    "envs/desk/linux/setup.sh": "#!/bin/sh\n", "envs/desk/linux/profile.sh": "#!/bin/sh\n",
    **APPS,
}


def _stored(root, desktop, extra=None):
    """Write a bundle whose task deploys the multi env ``office`` holding the computer env ``desktop``, and return the
    stored task's id."""
    layout(root, {"envs/desktop/env.toml": desktop, "envs/office/env.toml": 'type = "multi"\ncomputer_envs = ["desktop"]\n',
                  "tasks/t.json": json.dumps([{"id": "env", "type": "deploy_env", "env_id": "office"}]), **(extra or {})})
    plan = plan_bundle(resolve_bundle(parse_bundle(root)), installed=installed_bundles)
    materialize(plan, build_here=local_builds(plan))
    return f"{plan.bundle.bundle.id_root}/t"


def _written_back(folder):
    """Run the imported ``folder`` as ``agent-env run`` writes it, with no store, and return its computer env."""
    plan = plan_bundle(resolve_bundle(parse_bundle(folder)), installed=installed_bundles)
    materialize(plan, build_here=local_builds(plan))
    office = Env.get(f"{plan.bundle.bundle.id_root}/office")
    (desktop,) = office.computer_envs
    return desktop


def _files(universe) -> dict[str, bytes]:
    return {path: file.load() for path, file in universe.get_file_artifacts().items()}


def _same(read, stored, *fields):
    assert {field: getattr(read, field) for field in fields} == {field: getattr(stored, field) for field in fields}


def test_a_computer_env_with_its_own_base_driver_and_apps_reads_back_from_the_folder_as_stored(tmp_path, composed):
    task = _stored(tmp_path / "desk", OWN, OWN_FILES)
    stored = Env.get(task.rsplit("/", 1)[0] + "/desktop")

    imported = import_task(task, tmp_path / "out")

    assert imported.pinned == ()
    assert sorted(entity.folder for entity in imported.written) == [
        "artifacts/base", "envs/desk", "envs/desktop", "envs/mail", "envs/notes", "envs/office"]
    read = _written_back(tmp_path / "out")
    assert isinstance(read, ComputerEnv) and read.id != stored.id
    _same(read, stored, "environment_name", "display", "network", "computer_settings", "tools", "app_settings")
    _same(read.computer, stored.computer, "description", "arch", "os", "session", "config")
    assert read.computer.forms.container.start_as == "root"
    assert read.computer.image().dockerfile_path == "container/Dockerfile"
    assert [failure.check for failure in stored.driver.known_failures["linux"]] == ["P3"]
    _same(read.driver, stored.driver, "module", "port", "install", "known_failures", "declaration")
    assert _files(read.driver.files) == _files(stored.driver.files)
    assert [app.environment_name for app in read.apps] == ["notes", "mail"]
    for app, original in zip(read.apps, stored.apps):
        _same(app, original, "module", "declaration", "build_settings", "settings")
        assert _files(app.files) == _files(original.files)
    assert composed[-1][0] == read.id


def test_a_shipped_base_and_driver_stay_named_by_their_installed_ids(tmp_path, composed):
    task = _stored(tmp_path / "desk", DESKTOP)

    imported = import_task(task, tmp_path / "out")

    assert sorted(entity.folder for entity in imported.written) == ["envs/desktop", "envs/office"]
    assert {(floating.where, floating.id) for floating in imported.floating} >= {
        ("envs/desktop", UBUNTU), ("envs/desktop", REFERENCE)}
    toml = (tmp_path / "out" / "envs" / "desktop" / "env.toml").read_text()
    assert f'computer = "{UBUNTU}"' in toml and f'driver = "{REFERENCE}"' in toml
    read = _written_back(tmp_path / "out")
    assert (read.computer.id, read.driver.id) == (UBUNTU, REFERENCE)
    assert (read.computer_settings, read.display) == ({"timezone": "Europe/Paris"}, {"width": 1600, "height": 900})


def test_a_computer_artifact_reads_back_with_its_forms_and_settings(tmp_path, composed):
    task = _stored(tmp_path / "desk", OWN, OWN_FILES)
    stored = ComputerArtifact.get(task.rsplit("/", 1)[0] + "/base")

    import_task(task, tmp_path / "out")

    toml = (tmp_path / "out" / "artifacts" / "base" / "artifact.toml").read_text()
    assert "validated_images" not in toml
    assert sorted(path.relative_to(tmp_path / "out" / "artifacts" / "base").as_posix()
                  for path in (tmp_path / "out" / "artifacts" / "base").rglob("*") if path.is_file()) == [
        "artifact.toml", "container/Dockerfile", "container/start"]
    read = _written_back(tmp_path / "out").computer
    assert read.model_dump(include={"arch", "os", "session", "config"}) == stored.model_dump(
        include={"arch", "os", "session", "config"})
