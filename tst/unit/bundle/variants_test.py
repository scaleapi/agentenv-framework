"""``--set envs/NAME.KEY=VALUE``: a run's envs with keys set for that run alone, each env that changes, each env and task naming
one and each eval naming such a task under an id ending in the settings, with the bundle as written."""

import json

import pytest
from click.testing import CliRunner

from agent_env.bundle import BundleError, BundleKind, Variant, dry_run_bundle, parse_bundle
from agent_env.bundle.installed import installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.bundle.variants import with_variant
from agent_env.cli import cli
from agent_env.env import Env
from tst.unit.bundle._support import layout, local_builds, store_agent
from tst.unit.bundle.computer_env_folder_test import APPS, DESKTOP, UBUNTU, WITH_APPS

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
    return with_variant(parse_bundle(root), Variant(settings)).bundle


def _entry(bundle, name):
    return next(entry for entry in bundle.entries if entry.name == name)


def _ids(bundle):
    return {entry.name: entry.id for entry in bundle.entries}


def test_a_key_set_renames_the_env_and_what_uses_it_and_leaves_the_rest(tmp_path):
    root = _bundle(tmp_path / "desk")

    varied = with_variant(parse_bundle(root), Variant(("envs/desktop.driver=cua-driver",)))

    label, bundle = "desktop.driver cua-driver", varied.bundle
    assert dict(varied.renamed) == {(BundleKind.ENV, f"{ROOT}/desktop"): f"{ROOT}/desktop({label})",
                                    (BundleKind.TASK, f"{ROOT}/t"): f"{ROOT}/t({label})",
                                    (BundleKind.EVAL, f"{ROOT}/e"): f"{ROOT}/e({label})"}
    assert _ids(bundle) == {"desktop": f"{ROOT}/desktop({label})", "t": f"{ROOT}/t({label})", "chat": f"{ROOT}/chat",
                            "e": f"{ROOT}/e({label})", "chats": f"{ROOT}/chats"}
    assert _entry(bundle, "desktop").config == {**_entry(parse_bundle(root), "desktop").config, "driver": "cua-driver"}
    assert _entry(bundle, "t").config == [DEPLOY, AGENT, PROMPT]


def test_an_entry_of_another_kind_with_the_envs_id_stays_as_it_is(tmp_path):
    root = _bundle(tmp_path / "desk", files={"skills/desktop/SKILL.md": "---\nname: desktop\ndescription: d\n---\n"})

    bundle = _varied(root, "envs/desktop.driver=cua-driver")

    skill = next(entry for entry in bundle.entries if entry.kind.value == "skills")
    assert skill == next(entry for entry in parse_bundle(root).entries if entry.kind.value == "skills")
    assert _entry(bundle, "t").id == f"{ROOT}/t(desktop.driver cua-driver)"


def test_an_eval_sharing_the_envs_id_keeps_its_own_toml(tmp_path):
    root = _bundle(tmp_path / "desk", files={"evals/desktop.toml": 'tasks = ["t"]\n'})

    bundle = _varied(root, "envs/desktop.driver=cua-driver")

    evals = {entry.id: entry.config for entry in bundle.entries if entry.kind is BundleKind.EVAL}
    assert evals[f"{ROOT}/desktop(desktop.driver cua-driver)"] == {"tasks": ["t"]}
    assert _entry(bundle, "desktop").config["driver"] == "cua-driver"


def test_a_reference_by_id_is_repointed_at_the_variant(tmp_path):
    steps = [{**DEPLOY, "env_id": f"{ROOT}/desktop"}, {**AGENT, "env_ids": [f"{ROOT}/desktop"]}, PROMPT]
    root = _bundle(tmp_path / "desk", steps, {"evals/e.toml": f'tasks = ["{ROOT}/t"]\n'})

    bundle = _varied(root, "envs/desktop.computer=ubuntu-24.04-headless", "envs/desktop.tools=standard")

    label = "desktop.computer ubuntu-24.04-headless, desktop.tools standard"
    assert [step.get("env_id") or step.get("env_ids") for step in _entry(bundle, "t").config] == [
        f"{ROOT}/desktop({label})", [f"{ROOT}/desktop({label})"], None]
    assert _entry(bundle, "e").config["tasks"] == [f"{ROOT}/t({label})"]
    assert _entry(bundle, "desktop").config["tools"] == "standard"


@pytest.mark.parametrize("named", ["t", f"{ROOT}/t"], ids=["name", "id"])
def test_an_eval_naming_its_task_in_a_table_is_renamed_and_repointed_too(tmp_path, named):
    root = _bundle(tmp_path / "desk", files={"evals/e.toml": f'tasks = [{{ task = "{named}" }}]\n'})

    bundle = _varied(root, "envs/desktop.driver=cua-driver")

    label = "desktop.driver cua-driver"
    assert _entry(bundle, "e").id == f"{ROOT}/e({label})"
    assert _entry(bundle, "e").config["tasks"] == [{"task": "t" if named == "t" else f"{ROOT}/t({label})"}]


def test_a_value_is_toml_or_a_bare_word_and_a_key_may_be_dotted_into_a_table(tmp_path):
    settings = ("envs/desktop.display={ width = 1280, height = 720 }", "envs/desktop.computer_settings.theme=dark",
                "envs/desktop.network.mode=allow_all")

    bundle = _varied(_bundle(tmp_path / "desk"), *settings)

    config = _entry(bundle, "desktop").config
    assert (config["display"], config["computer_settings"], config["network"]) == (
        {"width": 1280, "height": 720}, {"timezone": "Europe/Paris", "theme": "dark"}, {"mode": "allow_all"})
    hashed = Variant(settings).label.split(",")[0]
    assert hashed.startswith("desktop.display ") and len(hashed) == len("desktop.display ") + 8
    assert _entry(bundle, "desktop").id == (f"{ROOT}/desktop({hashed}, desktop.computer_settings.theme dark, "
                                            "desktop.network.mode allow_all)")


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
    ("driver=cua-driver", "--set driver=cua-driver: expected envs/NAME.KEY=VALUE, such as envs/desktop.driver=cua-driver"),
    ("envs/desktop.driver", "--set envs/desktop.driver: expected envs/NAME.KEY=VALUE, such as "
                            "envs/desktop.driver=cua-driver"),
    ("envs/desktop=x", "--set envs/desktop=x: expected envs/NAME.KEY=VALUE, such as envs/desktop.driver=cua-driver"),
    ("envs/desktop.=x", "--set envs/desktop.=x: expected envs/NAME.KEY=VALUE, such as envs/desktop.driver=cua-driver"),
    ("envs/nope.driver=x", "--set envs/nope.driver=x: envs/nope names no env of this bundle"),
    ("envs/desktop.type=website", "--set envs/desktop.type=website: an env's type says what it is, so a run can't set "
                                  "it"),
    ("envs/desktop.id=x", "--set envs/desktop.id=x: an env's id says what it is, so a run can't set it"),
])
def test_a_setting_that_names_no_envs_key_is_refused(tmp_path, setting, problem):
    with pytest.raises(BundleError) as refused:
        _varied(_bundle(tmp_path / "desk"), setting)

    assert refused.value.problems == (problem,)


@pytest.fixture
def desk(tmp_path, local_stores, cli_routing):
    store_agent("theirs")
    return _bundle(tmp_path / "desk")


def _plan(root, *settings):
    return plan_bundle(resolve_bundle(_varied(root, *settings)), installed=installed_bundles)


def test_the_variant_is_written_and_composed_with_its_driver_beside_the_bundles_own(desk, composed):
    plan = _plan(desk, "envs/desktop.driver=cua-driver", "envs/desktop.tools=driver")

    materialize(plan, build_here=local_builds(plan))

    env = Env.get(f"{ROOT}/desktop(desktop.driver cua-driver, desktop.tools driver)")
    assert (env.computer.id, env.driver.id, env.environment_name, env.tools) == (UBUNTU, CUA, "desktop", "driver")
    assert composed == [(env.id, UBUNTU, CUA, None)]
    assert (desk / "envs/desktop/env.toml").read_text() == DESKTOP


@pytest.mark.parametrize("setting, problem", [
    ("envs/desktop.tools=all", "tools is 'standard', 'driver' or 'both', not 'all'"),
    ("envs/desktop.drivr=x", "unknown key 'drivr'; a computer env takes environment_name, computer, driver, apps, "
                             "display, network, computer_settings, tools, type and id"),
])
def test_a_key_or_value_the_envs_type_refuses_is_refused_before_anything_is_written(desk, composed,
                                                                                     setting, problem):
    with pytest.raises(BundleError) as refused:
        dry_run_bundle(desk, variant=Variant((setting,)))

    assert any(line.startswith("envs/desktop") and problem in line for line in refused.value.problems), refused.value


def test_a_driver_that_cant_compose_with_the_computer_is_refused_as_its_written(desk, composed):
    plan = _plan(desk, "envs/desktop.driver=macos-harness")

    with pytest.raises(BundleError) as refused:
        materialize(plan, build_here=local_builds(plan))

    assert any(line.startswith("envs/desktop") and "macos-harness" in line for line in refused.value.problems), (
        refused.value)
    assert composed == []


def test_a_run_whose_tasks_deploy_nothing_its_settings_change_is_refused(desk, composed):
    with pytest.raises(BundleError) as refused:
        dry_run_bundle(desk, tasks=["chat"], variant=Variant(("envs/desktop.driver=cua-driver",)))

    assert refused.value.problems == ("--set envs/desktop.driver=cua-driver: none of the tasks this run selects "
                                      "deploys an env it changes",)


@pytest.mark.parametrize("selection", ["t", f"{ROOT}/t"], ids=["name", "id"])
def test_a_task_is_selected_by_its_name_or_its_id_as_the_bundle_gives_it(desk, composed, selection):
    dry = dry_run_bundle(desk, tasks=[selection], variant=Variant(("envs/desktop.driver=cua-driver",)))

    assert [entry.id for entry in dry.runs] == [f"{ROOT}/t(desktop.driver cua-driver)"]


def test_the_cli_runs_the_variant_and_says_what_it_writes_it_as(desk, composed, monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm")

    result = CliRunner().invoke(cli, ["run", str(desk), "--task", "t", "--set", "envs/desktop.driver=cua-driver",
                                      "--set", "envs/desktop.tools=standard", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert ("--set envs/desktop.driver=cua-driver --set envs/desktop.tools=standard: 1 env and 1 task written under ids "
            "ending in (desktop.driver cua-driver, desktop.tools standard)") in result.output
