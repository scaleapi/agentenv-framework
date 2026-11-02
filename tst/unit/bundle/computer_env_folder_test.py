"""Computer env folders: an ``envs/<name>/env.toml`` of type computer names its base and driver as another bundle's
aliases, ``ubuntu-24.04`` and ``reference``, which the run pulls in and writes first; the env is written over their ids,
its computer_settings the base's by name, and the run says it builds with docker. The composer is replaced by a recorder."""

import json

import pytest

from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle.installed import installed_bundles
from agent_env.plugins._bundles import ALIASES_FILE, InstalledBundle
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle, plan_entities
from agent_env.bundle.resolve import resolve_bundle
from agent_env.bundle.run import dry_run_bundle
from agent_env.env import Env
from agent_env.env.envs.computer.utils import composer
from agent_env.env.envs.computer.utils.composer import Composing
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from tst.unit.bundle._support import layout, local_builds
from tst.unit.bundle.computer_app_env_test import APP, TOML as APP_TOML
from tst.unit.bundle.computer_driver_env_test import LINUX

UBUNTU = "@local/agentenv-framework/computers/ubuntu-24.04"
REFERENCE = "@local/agentenv-framework/drivers/reference"
DESKTOP = '''type = "computer"
computer = "ubuntu-24.04"
driver = "reference"
display = { width = 1600, height = 900 }

[computer_settings]
timezone = "Europe/Paris"
'''
WITH_APPS = DESKTOP.replace("display =", 'apps = ["notes", { env = "mail", install = "build" }]\ndisplay =')
#: Two app folders, mail and notes, of one app class.
APPS = {f"envs/{name}/{file}": text for name in ("mail", "notes") for file, text in (
    ("env.toml", APP_TOML), ("app.py", APP.format(supports=LINUX, skill="skill.md", linux_skill="skill-linux.md")),
    ("requirements.txt", "agentenv-framework-protocol\n"), ("skill.md", "# Mail\n"),
    ("skill-linux.md", "# Mail on Linux\n"))}


@pytest.fixture
def composed(tmp_path, monkeypatch, local_stores, cli_routing, builds):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("agent_env.env.published_images._registry_digest", lambda ref: None)
    calls = []

    def compose(env, pairs=None, composing=Composing()):
        if composing.on_progress is not None:
            composing.on_progress("composing")
        calls.append((env.id, env.computer.id, env.driver.id, composing.sandbox_provider)
                     + ((pairs,) if pairs is not None else ()))

    monkeypatch.setattr(composer, "compose", compose)
    return calls


def _bundle(root, toml=DESKTOP, extra=None):
    return layout(root, {"envs/desktop/env.toml": toml, "tasks/t.json": json.dumps([
        {"id": "desktop", "type": "load_artifact", "env_id": "desktop", "artifact_id": "notes"}]),
        "artifacts/notes/notes.txt": "a note\n", **(extra or {})})


def _installed(tmp_path, package, name):
    """An installed bundle of ``package`` whose aliases name an agent ``helper``."""
    root = tmp_path / package / name
    root.mkdir(parents=True)
    (root / ALIASES_FILE).write_text('[agents]\nagent = ["helper"]\n')
    return InstalledBundle(name, package, "1.0", package.replace("-", "_"), root, None, None)


def _plan(root):
    return plan_bundle(resolve_bundle(parse_bundle(root)), installed=installed_bundles)


def _plan_env(root):
    """The env alone, as ``write_installed`` writes an installed bundle's env another bundle names."""
    resolved = resolve_bundle(parse_bundle(root))
    return plan_entities(resolved, [f"{resolved.bundle.id_root}/desktop"], installed=installed_bundles)


@pytest.mark.parametrize("planned", [_plan, _plan_env], ids=["a-run", "the-env-alone"])
def test_a_computer_env_naming_shipped_aliases_is_written_over_them_and_composed(tmp_path, composed, planned):
    plan, provider = planned(_bundle(tmp_path / "desk")), LocalSandboxProvider()
    told = []

    done = materialize(plan, build_here=local_builds(plan), composing=Composing(provider),
                       on_progress=lambda write, line: told.append((write.id, line)))

    env = Env.get(f"{plan.bundle.bundle.id_root}/desktop")
    written = {item.write.id for item in done.writes}
    assert isinstance(env, ComputerEnv) and {UBUNTU, REFERENCE} <= written
    assert (env.computer.id, env.driver.id, env.environment_name) == (UBUNTU, REFERENCE, "desktop")
    assert env.computer_settings == {"timezone": "Europe/Paris"} and env.display == {"width": 1600, "height": 900}
    assert composed == [(env.id, UBUNTU, REFERENCE, provider)] and told == [(env.id, "composing")]


def test_a_reused_computer_env_composes_what_it_has_left_on_the_runs_provider_and_nothing_once_its_built(tmp_path,
                                                                                                         composed):
    root, provider = _bundle(tmp_path / "desk"), LocalSandboxProvider()
    materialize(_plan(root), build_here=local_builds(_plan(root)))
    env = Env.get(f"{_plan(root).bundle.bundle.id_root}/desktop")
    told = []

    checked = materialize(_plan(root), dry_run=True, composing=Composing(provider),
                          on_progress=lambda write, line: told.append((write.id, line)))
    again = materialize(_plan(root), build_here=local_builds(_plan(root)), composing=Composing(provider),
                        on_progress=lambda write, line: told.append((write.id, line)))
    env.update_metadata({**env.metadata, "computer": {"id": "c", "version": 1, "images": {"amd64": "a", "arm64": "b"}},
                         "composed": {"container": {"amd64": {"status": "ready", "image": "a"},
                                                    "arm64": {"status": "ready", "image": "b"}}}})
    built = materialize(_plan(root), build_here=local_builds(_plan(root)), composing=Composing(provider))

    assert built.composing == () and checked.composing == again.composing == (env.id,) and all(item.reused for item in again.writes)
    assert told == [(env.id, "would compose container/amd64, container/arm64"), (env.id, "composing")]
    assert composed == [(env.id, UBUNTU, REFERENCE, None),
                        (env.id, UBUNTU, REFERENCE, provider, [("container", "amd64"), ("container", "arm64")])]


def test_a_dry_run_says_what_it_would_compose_of_a_reused_computer_env(tmp_path, composed):
    root = _bundle(tmp_path / "desk")
    materialize(_plan(root), build_here=local_builds(_plan(root)))
    told = []

    dry = dry_run_bundle(root, sandbox="local", on_progress=told.append)

    env_id = f"{_plan(root).bundle.bundle.id_root}/desktop"
    assert dry.materialization.composing == (env_id,) and len(composed) == 1
    assert "envs/desktop: would compose container/amd64, container/arm64" in told


def test_the_env_alone_is_planned_whatever_the_bundles_tasks_would_deploy(tmp_path, composed, monkeypatch):
    """A default agent two installed bundles both alias names neither, which blocks a run of a task that deploys it,
    but not writing an env that no task of the bundle runs for."""
    claims = [_installed(tmp_path, package, name) for package, name in (("acme-kit", "kit"), ("zeta-tools", "tools"))]
    monkeypatch.setattr("agent_env.config.runtime.Config.get_default_a2a_agent_id", lambda self: "helper")
    root = _bundle(tmp_path / "desk")
    (root / "tasks" / "agent.json").write_text(json.dumps([{"id": "agent", "type": "deploy_agent", "env_ids": []}]))
    resolved = resolve_bundle(parse_bundle(root))

    plan = plan_entities(resolved, [f"{resolved.bundle.id_root}/desktop"],
                         installed=lambda: (*installed_bundles(), *claims))

    assert f"{resolved.bundle.id_root}/desktop" in {write.id for write in plan.writes}


def test_a_computer_env_folder_names_its_apps_in_order_each_written_before_it(tmp_path, composed):
    plan = _plan(_bundle(tmp_path / "desk", WITH_APPS, APPS))

    materialize(plan, build_here=local_builds(plan))

    root = plan.bundle.bundle.id_root
    env = Env.get(f"{root}/desktop")
    assert [(app.id, app.type, app.environment_name) for app in env.composed_apps()] == [
        (f"{root}/notes", "computer_app", "notes"), (f"{root}/mail", "computer_app", "mail")]
    assert composed == [(env.id, UBUNTU, REFERENCE, None)]


@pytest.mark.parametrize("stored", [False, True], ids=["in-the-bundle", "in-the-store"])
def test_a_universe_loaded_into_a_computer_env_seeds_its_files_and_each_of_its_apps_by_their_folders_names(
        tmp_path, composed, stored):
    if stored:
        plan = _plan(_bundle(tmp_path / "desk", WITH_APPS, APPS))
        materialize(plan, build_here=local_builds(plan))
        env_id, folders = f"{plan.bundle.bundle.id_root}/desktop", {}
        into = f"env '{env_id}'"
    else:
        env_id, folders, into = "desktop", {"envs/desktop/env.toml": WITH_APPS, **APPS}, "envs/desktop"

    def loading(root, *seeded):
        return layout(root, {**folders, "artifacts/world/artifact.toml": 'type = "environment_universe"\n',
                             **{f"artifacts/world/{name}/a.json": "{}" for name in ("desktop", "mail", *seeded)},
                             "tasks/t.json": json.dumps([{"id": "load", "type": "load_artifact", "env_id": env_id,
                                                          "artifact_id": "world"}])})

    _plan(loading(tmp_path / "seeds"))
    with pytest.raises(BundleError) as refused:
        _plan(loading(tmp_path / "typo", "mial"))

    assert refused.value.problems == (
        f"tasks/t.json: step 'load': artifacts/world/mial seeds an environment named 'mial', and {into} has no server "
        "by that name ('desktop', 'mail', 'notes'); name it after the server it seeds, or leave it out",)


@pytest.mark.parametrize("toml, problem", [
    ('type = "computer"\ndriver = "reference"\n', 'computer is required: the base computer, such as "ubuntu-24.04"'),
    ('type = "computer"\ncomputer = "ubuntu-24.04"\n', 'driver is required: the driver, such as "reference"'),
    (DESKTOP.replace("display =", "ports = [80]\ndisplay ="), "unknown key 'ports'"),
    (DESKTOP.replace("1600", "0"), "display is { width = <n>, height = <n> }, each 1 to 32767 pixels"),
    (WITH_APPS.replace('"build"', '"boot"'), 'apps[1]: install = \'boot\': an app installs at "build", into the '
     'computer\'s image; "boot" isn\'t supported yet'),
    (WITH_APPS.replace('["notes", ', '["notes", "notes", '), "apps: @local/~/desk/notes is listed twice"),
    (WITH_APPS.replace('install = "build"', 'mode = "build"'), 'apps[1].env: expected an id or { env = "<id>", version = '
     '<n> }, the version optional'),
], ids=["no-computer", "no-driver", "unknown-key", "display", "boot", "twice", "key-beside"])
def test_a_computer_env_folder_that_declares_what_cant_compose_is_refused_naming_the_folder(tmp_path, composed, toml,
                                                                                            problem):
    with pytest.raises(BundleError) as refused:
        plan = _plan(_bundle(tmp_path / "desk", toml, APPS))
        materialize(plan, build_here=local_builds(plan))

    assert any(line.startswith("envs/desktop") and problem in line for line in refused.value.problems), refused.value
    assert composed == []
