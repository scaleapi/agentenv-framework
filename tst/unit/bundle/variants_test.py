"""``--set KEY=VALUE``: a run's envs with keys set for that run alone, each env that changes, each env and task naming
one and each eval naming such a task under an id ending in the settings, with the bundle as written."""

import json

import pytest
from click.testing import CliRunner

from agent_env.bundle import BundleError, Variant, dry_run_bundle, parse_bundle
from agent_env.bundle.installed import installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.bundle.variants import with_variant
from agent_env.cli import cli
from agent_env.env import Env
from tst.unit.bundle._support import layout, local_builds
from tst.unit.bundle.agent_override_test import _store_agent
from tst.unit.bundle.computer_env_folder_test import APPS, DESKTOP, UBUNTU, WITH_APPS, composed  # noqa: F401

ROOT = "@local/~/desk"
CUA = "@local/agentenv-framework/drivers/cua-driver"
DEPLOY = {"id": "desk", "type": "deploy_env", "env_id": "desktop"}
AGENT = {"id": "agent", "type": "deploy_agent", "env_ids": ["desktop"], "a2a_agent_id": "theirs", "depends_on": ["desk"]}
PROMPT = {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "prompt": "hi", "depends_on": ["agent"]}
ALONE = [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "theirs"}, PROMPT]


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))


def _bundle(root, steps=(DEPLOY, AGENT, PROMPT), files=None):
    return layout(root, {"envs/desktop/env.toml": DESKTOP, "tasks/t.json": json.dumps(list(steps)),
                         "tasks/chat.json": json.dumps(ALONE), "evals/e.toml": 'tasks = ["t"]\n',
                         "evals/chats.toml": 'tasks = ["chat"]\n', **(files or {})})


def _varied(root, *settings):
    return with_variant(parse_bundle(root), Variant(settings))


def _entry(bundle, name):
    return next(entry for entry in bundle.entries if entry.name == name)


def _ids(bundle):
    return {entry.name: entry.id for entry in bundle.entries}


def test_a_key_set_renames_the_env_and_what_uses_it_and_leaves_the_rest(tmp_path):
    root = _bundle(tmp_path / "desk")

    bundle = _varied(root, "driver=cua-driver")

    assert _ids(bundle) == {"desktop": f"{ROOT}/desktop(driver cua-driver)", "t": f"{ROOT}/t(driver cua-driver)",
                            "chat": f"{ROOT}/chat", "e": f"{ROOT}/e(driver cua-driver)", "chats": f"{ROOT}/chats"}
    assert _entry(bundle, "desktop").config == {**_entry(parse_bundle(root), "desktop").config, "driver": "cua-driver"}
    assert _entry(bundle, "t").config == [DEPLOY, AGENT, PROMPT]


def test_a_reference_by_id_is_repointed_at_the_variant(tmp_path):
    steps = [{**DEPLOY, "env_id": f"{ROOT}/desktop"}, {**AGENT, "env_ids": [f"{ROOT}/desktop"]}, PROMPT]
    root = _bundle(tmp_path / "desk", steps, {"evals/e.toml": f'tasks = ["{ROOT}/t"]\n'})

    bundle = _varied(root, "computer=ubuntu-24.04-headless", "tools=standard")

    label = "computer ubuntu-24.04-headless, tools standard"
    assert [step.get("env_id") or step.get("env_ids") for step in _entry(bundle, "t").config] == [
        f"{ROOT}/desktop({label})", [f"{ROOT}/desktop({label})"], None]
    assert _entry(bundle, "e").config["tasks"] == [f"{ROOT}/t({label})"]
    assert _entry(bundle, "desktop").config["tools"] == "standard"


def test_a_value_is_toml_or_a_bare_word_and_a_key_may_be_dotted_into_a_table(tmp_path):
    settings = ("display={ width = 1280, height = 720 }", "computer_settings.theme=dark",
                "envs/desktop.network.mode=allow_all")

    bundle = _varied(_bundle(tmp_path / "desk"), *settings)

    config = _entry(bundle, "desktop").config
    assert (config["display"], config["computer_settings"], config["network"]) == (
        {"width": 1280, "height": 720}, {"timezone": "Europe/Paris", "theme": "dark"}, {"mode": "allow_all"})
    hashed = Variant(settings).label.split(",")[0]
    assert hashed.startswith("display ") and len(hashed) == len("display ") + 8
    assert _entry(bundle, "desktop").id == f"{ROOT}/desktop({hashed}, computer_settings.theme dark, desktop.network.mode allow_all)"


def test_one_env_is_set_by_its_folder_and_an_env_composing_an_app_set_is_renamed_too(tmp_path):
    files = {"envs/desktop/env.toml": WITH_APPS, "envs/laptop/env.toml": DESKTOP, **APPS,
             "tasks/u.json": json.dumps([{**DEPLOY, "env_id": "laptop"}])}
    setting = "envs/mail.settings.signature=Ada"

    bundle = _varied(_bundle(tmp_path / "desk", files=files), setting)

    ids, label = _ids(bundle), Variant((setting,)).label
    assert label == "mail.settings.signature Ada"
    assert (ids["mail"], ids["desktop"], ids["t"]) == (f"{ROOT}/mail({label})", f"{ROOT}/desktop({label})",
                                                       f"{ROOT}/t({label})")
    assert (ids["notes"], ids["laptop"], ids["u"]) == (f"{ROOT}/notes", f"{ROOT}/laptop", f"{ROOT}/u")
    assert _entry(bundle, "mail").config["settings"] == {"signature": "Ada"}


@pytest.mark.parametrize("setting, problem", [
    ("drivr=x", "--set drivr=x: no env this bundle's tasks deploy takes drivr; a computer env takes apps, computer, "
                "computer_settings, display, driver, environment_name, network, tools"),
    ("envs/nope.driver=x", "--set envs/nope.driver=x: envs/nope names no env of this bundle"),
    ("driver", "--set driver: expected KEY=VALUE, such as driver=cua-driver"),
    ("=x", "--set =x: expected KEY=VALUE, such as driver=cua-driver"),
])
def test_a_setting_no_env_can_take_is_refused(tmp_path, setting, problem):
    with pytest.raises(BundleError) as refused:
        _varied(_bundle(tmp_path / "desk"), setting)

    assert refused.value.problems == (problem,)


@pytest.fixture
def desk(tmp_path, local_stores, cli_routing):
    _store_agent("theirs")
    return _bundle(tmp_path / "desk")


def _plan(root, *settings):
    return plan_bundle(resolve_bundle(_varied(root, *settings)), installed=installed_bundles)


def test_the_variant_is_written_and_composed_with_its_driver_beside_the_bundles_own(desk, composed):  # noqa: F811
    plan = _plan(desk, "driver=cua-driver", "tools=driver")

    materialize(plan, build_here=local_builds(plan))

    env = Env.get(f"{ROOT}/desktop(driver cua-driver, tools driver)")
    assert (env.computer.id, env.driver.id, env.environment_name, env.tools) == (UBUNTU, CUA, "desktop", "driver")
    assert composed == [(env.id, UBUNTU, CUA, None)]
    assert (desk / "envs/desktop/env.toml").read_text() == DESKTOP


def test_a_value_the_envs_type_refuses_is_refused_before_anything_is_written(desk, composed):  # noqa: F811
    with pytest.raises(BundleError) as refused:
        dry_run_bundle(desk, variant=Variant(("tools=all",)))

    assert any(line.startswith("envs/desktop") and "tools is 'standard', 'driver' or 'both', not 'all'" in line
               for line in refused.value.problems), refused.value


def test_a_driver_that_cant_compose_with_the_computer_is_refused_as_its_written(desk, composed):  # noqa: F811
    plan = _plan(desk, "driver=macos-harness")

    with pytest.raises(BundleError) as refused:
        materialize(plan, build_here=local_builds(plan))

    assert any(line.startswith("envs/desktop") and "macos-harness" in line for line in refused.value.problems), (
        refused.value)
    assert composed == []


def test_a_run_whose_tasks_deploy_nothing_its_settings_change_is_refused(desk, composed):  # noqa: F811
    with pytest.raises(BundleError) as refused:
        dry_run_bundle(desk, tasks=["chat"], variant=Variant(("driver=cua-driver",)))

    assert refused.value.problems == ("--set driver=cua-driver: none of the tasks this run selects deploys an env it "
                                      "changes",)


def test_the_cli_runs_the_variant_and_says_what_it_writes_it_as(desk, composed, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm")

    result = CliRunner().invoke(cli, ["run", str(desk), "--task", "t", "--set", "driver=cua-driver", "--set",
                                      "tools=standard", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert ("--set driver=cua-driver --set tools=standard: 1 env and 1 task written under ids ending in "
            "(driver cua-driver, tools standard)") in result.output
