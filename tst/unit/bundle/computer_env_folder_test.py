"""Computer env folders: an ``envs/<name>/env.toml`` of type computer names its base and driver as another bundle's
aliases, ``ubuntu-24.04`` and ``reference``, which the run pulls in and writes first; the env is written over their ids,
its computer_settings the base's by name, and the run says it builds with docker. The composer is replaced by a recorder."""

import json

import pytest

from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle.installed import installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle, plan_entities
from agent_env.bundle.resolve import resolve_bundle
from agent_env.env import Env
from agent_env.env.envs.computer import composer
from agent_env.env.envs.computer.env import ComputerEnv
from tst.unit.bundle._support import layout, local_builds

UBUNTU = "@local/agentenv-framework/computers/ubuntu-24.04"
REFERENCE = "@local/agentenv-framework/drivers/reference"
DESKTOP = '''type = "computer"
computer = "ubuntu-24.04"
driver = "reference"
display = { width = 1600, height = 900 }

[computer_settings]
timezone = "Europe/Paris"
'''


@pytest.fixture
def composed(tmp_path, monkeypatch, local_stores, cli_routing, builds):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("agent_env.env.published_images._registry_digest", lambda ref: None)
    calls = []

    def compose(env, on_build=None):
        on_build()
        calls.append((env.id, env.computer.id, env.driver.id))

    monkeypatch.setattr(composer, "compose", compose)
    return calls


def _bundle(root, toml=DESKTOP):
    return layout(root, {"envs/desktop/env.toml": toml, "tasks/t.json": json.dumps([
        {"id": "desktop", "type": "load_artifact", "env_id": "desktop", "artifact_id": "notes"}]),
        "artifacts/notes/notes.txt": "a note\n"})


def _plan(root):
    return plan_bundle(resolve_bundle(parse_bundle(root)), installed=installed_bundles)


def _plan_env(root):
    """The env alone, as ``write_installed`` writes an installed bundle's env another bundle names."""
    resolved = resolve_bundle(parse_bundle(root))
    return plan_entities(resolved, [f"{resolved.bundle.id_root}/desktop"], installed=installed_bundles)


@pytest.mark.parametrize("planned", [_plan, _plan_env], ids=["a-run", "the-env-alone"])
def test_a_computer_env_naming_shipped_aliases_is_written_over_them_and_composed(tmp_path, composed, planned):
    plan = planned(_bundle(tmp_path / "desk"))
    building = []

    done = materialize(plan, build_here=local_builds(plan), on_build=lambda write: building.append(write.id))

    env = Env.get(f"{plan.bundle.bundle.id_root}/desktop")
    written = {item.write.id for item in done.writes}
    assert isinstance(env, ComputerEnv) and {UBUNTU, REFERENCE} <= written
    assert (env.computer.id, env.driver.id, env.environment_name) == (UBUNTU, REFERENCE, "desktop")
    assert env.computer_settings == {"timezone": "Europe/Paris"} and env.display == {"width": 1600, "height": 900}
    assert composed == [(env.id, UBUNTU, REFERENCE)] and env.id in building


@pytest.mark.parametrize("toml, problem", [
    ('type = "computer"\ndriver = "reference"\n', 'computer is required: the base computer, such as "ubuntu-24.04"'),
    ('type = "computer"\ncomputer = "ubuntu-24.04"\n', 'driver is required: the driver, such as "reference"'),
    (DESKTOP.replace("display =", "ports = [80]\ndisplay ="), "unknown key 'ports'"),
    (DESKTOP.replace("1600", "0"), "display is { width = <n>, height = <n> }, each 1 to 32767 pixels"),
], ids=["no-computer", "no-driver", "unknown-key", "display"])
def test_a_computer_env_folder_that_declares_what_cant_compose_is_refused_naming_the_folder(tmp_path, composed, toml,
                                                                                            problem):
    plan = _plan(_bundle(tmp_path / "desk", toml))

    with pytest.raises(BundleError) as refused:
        materialize(plan, build_here=local_builds(plan))

    assert any(line.startswith("envs/desktop") and problem in line for line in refused.value.problems), refused.value
    assert composed == []
