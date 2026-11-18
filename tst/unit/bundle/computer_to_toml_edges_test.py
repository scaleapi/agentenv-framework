"""The edges of writing a computer env into a folder: a folder a run wrote back exports as the same folder, odd setting
values, distribution installs, binary and unicode files keep what they hold, a base named from a registry with a disk
built from a source, an app deployed alone, a driver two computers share, and a stored context that's gone."""

import json

import pytest

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle.importer import import_task
from agent_env.bundle.installed import installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.config import get_config
from agent_env.env import Env
from tst.unit.bundle._support import DIGEST, layout, local_builds
from tst.unit.bundle.computer_driver_env_test import DRIVER, SERVE
from tst.unit.bundle.computer_to_toml_test import BASE, OWN, OWN_FILES
from tst.util.oci_registry import UBUNTU_LAYER, Registry

OFFICE = 'type = "multi"\ncomputer_envs = ["desktop"]\n'
UBUNTU_ONLY = [{"os": {"type": "linux", "distribution": "ubuntu", "display": ["x11"]}, "arch": ["amd64", "arm64"]}]


def _store(root, files, steps):
    """Write a bundle of ``files`` whose task ``t`` is ``steps``, and return its id root."""
    layout(root, {**files, "tasks/t.json": json.dumps(steps)})
    plan = plan_bundle(resolve_bundle(parse_bundle(root)), installed=installed_bundles)
    materialize(plan, build_here=local_builds(plan))
    return plan.bundle.bundle.id_root


def _run(folder):
    """Write ``folder`` back as a run of it does, and return its id root."""
    plan = plan_bundle(resolve_bundle(parse_bundle(folder)), installed=installed_bundles)
    materialize(plan, build_here=local_builds(plan))
    return plan.bundle.bundle.id_root


def _tree(folder):
    """Every file of an imported folder by path, but the two that name where it came from."""
    return {path.relative_to(folder).as_posix(): path.read_bytes() for path in folder.rglob("*")
            if path.is_file() and path.name not in ("provenance.json", "README.md")}


DEPLOY_OFFICE = [{"id": "env", "type": "deploy_env", "env_id": "office"}]


def test_the_folder_a_run_wrote_back_exports_as_the_same_folder(tmp_path, composed):
    root = _store(tmp_path / "desk", {**OWN_FILES, "envs/desktop/env.toml": OWN, "envs/office/env.toml": OFFICE},
                  DEPLOY_OFFICE)
    import_task(f"{root}/t", tmp_path / "once")
    once = _tree(tmp_path / "once")

    again = _run(tmp_path / "once")
    import_task(f"{again}/t", tmp_path / "twice")

    assert _tree(tmp_path / "twice") == once


def test_odd_values_distribution_installs_and_binary_and_unicode_files_come_back_as_they_were(tmp_path, composed):
    base = BASE + ('\n[config.depth]\ndefault = 24\nphase = "deploy"\n\n[config.scale]\ndefault = 1.5\nphase = "deploy"\n'
                   '\n[config.dark]\ndefault = false\nphase = "deploy"\n')
    desktop = OWN.replace('theme = "dark"', 'depth = 30\nscale = 2.25\ndark = true').replace(
        'settings = { signature = "-- Ada" }', 'settings = { signature = "-- Ada", password = "env:MAIL_PASSWORD" }')
    driver = ('type = "computer_driver"\nmodule = "driver.py:Desk"\n\n[install.linux]\n'
              f'{SERVE}\n\n[install."linux/ubuntu"]\npython = "requirements.txt"\n{SERVE}\n')
    files = {**OWN_FILES, "artifacts/base/artifact.toml": base, "envs/desktop/env.toml": desktop,
             "envs/office/env.toml": OFFICE, "envs/desk/env.toml": driver,
             "envs/desk/driver.py": DRIVER.format(supports=UBUNTU_ONLY, buttons=("left", "right")),
             "envs/desk/requirements.txt": "mss>=9,<11\n", "envs/mail/docs/naïve note.md": "résumé\n"}
    blob = bytes(range(256)) * 4
    (tmp_path / "desk" / "envs" / "mail" / "icons").mkdir(parents=True)
    (tmp_path / "desk" / "envs" / "mail" / "icons" / "app.png").write_bytes(blob)
    root = _store(tmp_path / "desk", files, DEPLOY_OFFICE)
    stored = Env.get(f"{root}/desktop")

    import_task(f"{root}/t", tmp_path / "out")

    (read,) = Env.get(f"{_run(tmp_path / 'out')}/office").computer_envs
    assert read.computer_settings == stored.computer_settings == {"depth": 30, "scale": 2.25, "dark": True}
    assert [type(read.computer_settings[name]) for name in ("depth", "scale", "dark")] == [int, float, bool]
    assert read.computer.config == stored.computer.config
    assert read.app_settings["mail"]["password"] == "env:MAIL_PASSWORD"
    assert set(read.driver.install) == {"linux", "linux/ubuntu"} and read.driver.install == stored.driver.install
    mail = next(app for app in read.apps if app.environment_name == "mail")
    files = {path: file.load() for path, file in mail.files.get_file_artifacts().items()}
    assert files["icons/app.png"] == blob and files["docs/naïve note.md"] == "résumé\n".encode()


def test_a_base_named_from_a_registry_with_a_disk_built_from_a_source_is_written_as_both(tmp_path, composed, registry,
                                                                                       monkeypatch):
    served = Registry().serve(monkeypatch)
    amd64, arm64 = served.image(UBUNTU_LAYER), served.image(UBUNTU_LAYER, arch="arm64")
    served.manifests[DIGEST] = served.manifests[served.index(amd64=amd64, arm64=arm64)]
    source = f'{{ url = "https://images.example/noble.img", sha256 = "{"b" * 64}" }}'
    base = BASE.replace('image = { dockerfile = "container/Dockerfile" }', 'image = { ref = "ghcr.io/acme/desk:1" }') + (
        f'\n[forms.disk]\nsource = {source}\ninit = "supervisord"\ninput_channel = "nocloud"\nmin_disk_gb = 16\n')
    files = {**OWN_FILES, "artifacts/base/artifact.toml": base, "envs/desktop/env.toml": OWN,
             "envs/office/env.toml": OFFICE}
    del files["artifacts/base/container/Dockerfile"], files["artifacts/base/container/start"]
    root = _store(tmp_path / "desk", files, DEPLOY_OFFICE)
    stored = ComputerArtifact.get(f"{root}/base")

    imported = import_task(f"{root}/t", tmp_path / "out")

    assert sorted(path.name for path in (tmp_path / "out" / "artifacts" / "base").iterdir()) == ["artifact.toml"]
    assert [(pulled.folder, pulled.image) for pulled in imported.pulled] == [
        ("artifacts/base", f"ghcr.io/acme/desk:1@{DIGEST}")]
    read = ComputerArtifact.get(f"{_run(tmp_path / 'out')}/base")
    assert read.image().image_name == stored.image().image_name == f"ghcr.io/acme/desk:1@{DIGEST}"
    assert read.forms.disk == stored.forms.disk and read.forms.disk.source.url == "https://images.example/noble.img"


def test_an_app_deployed_alone_is_written_with_the_computer_and_driver_it_deploys_on(tmp_path, composed):
    files = {**OWN_FILES, "envs/mail/env.toml": OWN_FILES["envs/mail/env.toml"] + 'computer = "base"\ndriver = "desk"\n'}
    root = _store(tmp_path / "desk", files, [{"id": "env", "type": "deploy_env", "env_id": "mail"}])

    imported = import_task(f"{root}/t", tmp_path / "out")

    assert sorted(entity.folder for entity in imported.written) == ["artifacts/base", "envs/desk", "envs/mail"]
    read = Env.get(f"{_run(tmp_path / 'out')}/mail")
    assert (read.environment_name, read.build_settings) == ("mail", {"edition": "pro"})
    assert (read.computer.description, read.driver.module) == ("a desk", "driver.py:Desk")


def test_a_driver_and_base_two_computers_share_are_written_once(tmp_path, composed):
    laptop = 'type = "computer"\ncomputer = "base"\ndriver = "desk"\n'
    files = {**OWN_FILES, "envs/desktop/env.toml": OWN, "envs/laptop/env.toml": laptop,
             "envs/office/env.toml": 'type = "multi"\ncomputer_envs = ["desktop", "laptop"]\n'}
    root = _store(tmp_path / "desk", files, DEPLOY_OFFICE)

    imported = import_task(f"{root}/t", tmp_path / "out")

    folders = [entity.folder for entity in imported.written]
    assert sorted(folders) == sorted(set(folders)) and {"envs/desk", "artifacts/base"} <= set(folders)
    computers = Env.get(f"{_run(tmp_path / 'out')}/office").computer_envs
    assert [env.environment_name for env in computers] == ["desktop", "laptop"]
    assert computers[0].driver.id == computers[1].driver.id and computers[0].computer.id == computers[1].computer.id


def test_a_base_whose_stored_context_is_gone_fails_the_import_naming_it_and_writes_nothing(tmp_path, composed):
    root = _store(tmp_path / "desk", {**OWN_FILES, "envs/desktop/env.toml": OWN, "envs/office/env.toml": OFFICE},
                  DEPLOY_OFFICE)
    url = ComputerArtifact.get(f"{root}/base").image().build_context_object_url
    store = get_config().get_object_store_at(url)
    store._resolve(store.get_object_key(url)).unlink()

    with pytest.raises(BundleError) as caught:
        import_task(f"{root}/t", tmp_path / "out")

    assert any("artifacts/base" in problem for problem in caught.value.problems), caught.value.problems
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
