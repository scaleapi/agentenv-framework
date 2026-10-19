"""A bundle using an env, agent or artifact an installed bundle defines: by id, or as the configured default agent,
it's pulled into the plan as one of the bundle's own is, named after the bundle it comes from."""

import json

import pytest

from agent_env.bundle import BundleError, BundleKind, dry_run_bundle, parse_bundle
from agent_env.bundle import preflight as preflight_module
from agent_env.bundle import run as run_module
from agent_env.bundle.installed import InstalledBundle
from agent_env.bundle.plan import check_bundle, plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from tst.unit.bundle._support import RefusingStore, layout

ROOT = "@local/~/triage"
KIT = "@local/demo-dist/kit"
SOLVER = f"{KIT}/solver"
JUDGE = {"type": "rubrics_verifier", "prompt_id": "p", "verifier_id": "v", "criteria": [{"id": "c", "description": "ok"}]}
SERVERS = {"envs/crm/Dockerfile": "FROM scratch\n", "envs/crm/env.toml": 'environment_name = "crm"\n',
           "envs/slack/Dockerfile": "FROM scratch\n", "envs/slack/env.toml": 'environment_name = "slack"\n',
           "envs/suite/env.toml": 'type = "multi"\nmcp_server_envs = ["crm", "slack"]\n'}


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "triage"


@pytest.fixture
def default_agent(monkeypatch):
    def set_default(agent_id):
        monkeypatch.setattr("agent_env.config.runtime.Config.get_default_a2a_agent_id", lambda self: agent_id)

    return set_default


def _installed(root, files, *, dist="demo-dist", name="kit", problem=None):
    layout(root, files)
    return InstalledBundle(name, dist, "1.0", "demo", None if problem else root, None, problem)


def _kit(tmp_path, files=None, **options):
    return _installed(tmp_path / "site" / "kit", {"agents/solver/Dockerfile": "FROM scratch\n", **(files or {})},
                      **options)


def _task(bundle_dir, steps, files=None):
    layout(bundle_dir, {"tasks/t.json": json.dumps(steps), **(files or {})})
    return resolve_bundle(parse_bundle(bundle_dir))


def _plan(resolved, *installed):
    return plan_bundle(resolved, installed=lambda: installed)


def _problems(resolved, *installed):
    with pytest.raises(BundleError) as caught:
        _plan(resolved, *installed)
    return caught.value.problems


def _writes(plan):
    return [(write.kind, write.id) for write in plan.writes]


def test_an_installed_bundles_agent_named_by_id_is_written_by_the_run_after_its_image(bundle_dir, tmp_path):
    kit = _kit(tmp_path)
    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": SOLVER}]), kit)

    assert _writes(plan) == [(BundleKind.ARTIFACT, f"{SOLVER}__agent_image"), (BundleKind.AGENT, SOLVER),
                             (BundleKind.TASK, f"{ROOT}/t")]
    agent = plan.writes[1].source.entry
    assert plan.writes[-1].needs == (("agent", SOLVER),) and plan.store_refs == ()
    assert (plan.bundle_of(agent).id_root, plan.path(agent)) == (KIT, "demo-dist/kit:agents/solver")


def test_without_the_installed_bundles_its_id_is_a_store_id(bundle_dir, tmp_path):
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": SOLVER}])

    with pytest.raises(BundleError) as caught:
        plan_bundle(resolved)
    assert caught.value.problems == (
        f"tasks/t.json: step 'agent': a2a_agent_id: there is no agent {SOLVER!r} in the store",)


def test_a_step_falling_back_to_an_installed_default_agent_pulls_it_in(bundle_dir, tmp_path, default_agent):
    default_agent(SOLVER)
    plan = _plan(_task(bundle_dir, [
        {"id": "agent", "type": "deploy_agent"},
        {**JUDGE, "id": "judge"},
        {**JUDGE, "id": "llm", "use_agent_judge": False, "default_model": "m"},
        {**JUDGE, "id": "own", "agent_name": "default-agent"},
    ]), _kit(tmp_path))

    assert _writes(plan) == [(BundleKind.ARTIFACT, f"{SOLVER}__agent_image"), (BundleKind.AGENT, SOLVER),
                             (BundleKind.TASK, f"{ROOT}/t")]
    task = plan.writes[-1].source
    assert [(ref.where, ref.local.id) for ref in task.references] == [
        ("step 'agent': a2a_agent_id (the default agent)", SOLVER),
        ("step 'judge': judge_a2a_agent_id (the default agent)", SOLVER)]


def test_a_default_agent_the_bundle_defines_is_written_by_the_run(bundle_dir, default_agent):
    default_agent(f"{ROOT}/solver")
    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent"}],
                       {"agents/solver/Dockerfile": "FROM scratch\n"}))

    assert plan.writes[-1].needs == (("agent", f"{ROOT}/solver"),) and plan.origins == {}


def test_a_default_agent_neither_bundle_defines_is_left_to_the_store(bundle_dir, tmp_path, default_agent):
    default_agent("house-agent")
    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent"}]), _kit(tmp_path))

    assert _writes(plan) == [(BundleKind.TASK, f"{ROOT}/t")] and plan.writes[0].source.references == ()


def test_an_installed_env_is_pulled_with_the_envs_it_names_and_its_artifacts_by_id(bundle_dir, tmp_path):
    kit = _kit(tmp_path, {**SERVERS, "artifacts/world/crm/a.json": "{}"})
    plan = _plan(_task(bundle_dir, [
        {"id": "env", "type": "deploy_env", "env_id": f"{KIT}/suite"},
        {"id": "load", "type": "load_artifact", "env_id": f"{KIT}/suite", "artifact_id": f"{KIT}/world",
         "depends_on": ["env"]},
    ]), kit)

    assert _writes(plan) == [
        (BundleKind.ARTIFACT, f"{KIT}/crm__env_image"), (BundleKind.ENV, f"{KIT}/crm"),
        (BundleKind.ARTIFACT, f"{KIT}/slack__env_image"), (BundleKind.ENV, f"{KIT}/slack"),
        (BundleKind.ENV, f"{KIT}/suite"), (BundleKind.ARTIFACT, f"{KIT}/world"), (BundleKind.TASK, f"{ROOT}/t")]


def test_a_pulled_env_is_checked_as_the_bundles_own_from_its_planned_config(bundle_dir, tmp_path):
    kit = _kit(tmp_path, SERVERS)
    resolved = _task(bundle_dir, [
        {"id": "env", "type": "deploy_env", "env_id": f"{KIT}/suite"},
        {"id": "load", "type": "load_artifact", "env_id": f"{KIT}/suite", "artifact_id": "world", "depends_on": ["env"]},
    ], {"artifacts/world/artifact.toml": 'type = "environment_universe"\n', "artifacts/world/crm/a.json": "{}",
        "artifacts/world/slakc/a.json": "{}"})
    problem = ("tasks/t.json: step 'load': artifacts/world/slakc seeds an environment named 'slakc', and "
               "demo-dist/kit:envs/suite has no server by that name ('crm', 'slack'); name it after the server it "
               "seeds, or leave it out")

    assert _problems(resolved, kit) == (problem,)


def test_an_installed_bundle_naming_another_ones_entity_pulls_that_in_too(bundle_dir, tmp_path):
    base = _installed(tmp_path / "site" / "base", {"artifacts/base/Dockerfile": "FROM scratch\n"},
                      dist="base-dist", name="base")
    kit = _kit(tmp_path, {"agents/coder/agent.toml": 'image = "@local/base-dist/base/base"\n'})
    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": f"{KIT}/coder"}]),
                 kit, base)

    assert _writes(plan) == [(BundleKind.ARTIFACT, "@local/base-dist/base/base"), (BundleKind.AGENT, f"{KIT}/coder"),
                             (BundleKind.TASK, f"{ROOT}/t")]
    assert plan.path(plan.writes[0].source.entry) == "base-dist/base:artifacts/base"


def test_the_bundles_own_entity_wins_over_an_installed_one_with_its_id(bundle_dir, tmp_path):
    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": SOLVER}], {
        "agents/mine/agent.toml": f'id = "{SOLVER}"\n', "agents/mine/Dockerfile": "FROM scratch\n"}), _kit(tmp_path))

    assert plan.origins == {} and plan.path(plan.writes[1].source.entry) == "agents/mine"


@pytest.mark.parametrize(("step", "problem"), [
    ({"a2a_agent_id": SOLVER, "a2a_agent_version": 2},
     f"{SOLVER!r} comes with the installed bundle demo-dist/kit, so it has no fixed version; drop the pin"),
    ({"a2a_agent_id": f"{KIT}/nope"}, f"there is no agent '{KIT}/nope' in the installed bundle demo-dist/kit"),
    ({"a2a_agent_id": f"{KIT}/crm"}, f"'{KIT}/crm' is the installed bundle demo-dist/kit's env, but this field takes "
                                     "an agent"),
])
def test_an_installed_id_that_names_nothing_it_can_be_is_refused_and_not_read_from_the_store(bundle_dir, tmp_path,
                                                                                            step, problem):
    kit = _kit(tmp_path, SERVERS)
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", **step}])

    assert _problems(resolved, kit) == (f"tasks/t.json: step 'agent': a2a_agent_id: {problem}",)


def test_an_installed_bundle_that_cant_be_read_is_one_problem_where_it_is_named(bundle_dir, tmp_path, default_agent):
    default_agent(SOLVER)
    broken = _kit(tmp_path, problem="its distribution's metadata can't be read")
    unparsed = _kit(tmp_path, {"tasks/t.json": "["}, name="kit")
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent"}])

    assert _problems(resolved, broken) == (
        f"tasks/t.json: step 'agent': a2a_agent_id (the default agent): {SOLVER!r} comes with the installed bundle "
        "demo-dist/kit, which can't be read: its distribution's metadata can't be read",)
    [problem] = _problems(resolved, unparsed)
    assert problem.startswith(f"tasks/t.json: step 'agent': a2a_agent_id (the default agent): {SOLVER!r} comes with "
                              "the installed bundle demo-dist/kit, which can't be read: tasks/t.json: ")


def test_a_pulled_entitys_own_problem_is_named_after_its_bundle(bundle_dir, tmp_path):
    kit = _kit(tmp_path, {"agents/solver/agent.toml": "colour = 1\n"})
    [problem] = _problems(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": SOLVER}]), kit)

    assert problem.startswith("demo-dist/kit:agents/solver: agent.toml: unknown key 'colour'")


def test_checking_a_bundle_without_a_store_pulls_and_refuses_as_planning_does(bundle_dir, tmp_path, monkeypatch):
    kit = _kit(tmp_path)
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": f"{KIT}/nope"}])
    monkeypatch.setattr("agent_env.config.runtime.Config.get_document_store", lambda self: RefusingStore())

    with pytest.raises(BundleError) as caught:
        check_bundle(resolved, installed=lambda: (kit,))
    assert caught.value.problems == (
        f"tasks/t.json: step 'agent': a2a_agent_id: there is no agent '{KIT}/nope' in the installed bundle "
        "demo-dist/kit",)
    check_bundle(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": SOLVER}]),
                 installed=lambda: (kit,))


def test_a_dry_run_preflights_a_pulled_default_agent_and_names_its_image_after_its_bundle(bundle_dir, tmp_path,
                                                                                        default_agent, monkeypatch):
    default_agent(SOLVER)
    kit = _kit(tmp_path)
    monkeypatch.setattr(run_module, "installed_bundles", lambda: (kit,))
    monkeypatch.setattr(preflight_module, "docker_unreachable", lambda: None)
    _task(bundle_dir, [{"id": "agent", "type": "deploy_agent"}])
    said = []

    dry = dry_run_bundle(bundle_dir, sandbox="local", on_progress=said.append)

    assert dry.images() == ("demo-dist/kit:agents/solver (Dockerfile image) v1: built on this machine",)
    assert [dry.path(entry) for entry in dry.runs] == ["tasks/t.json"]
    assert [line for line in said if "demo-dist/kit" in line] == [
        "demo-dist/kit:agents/solver (Dockerfile image): v1 (new)", "demo-dist/kit:agents/solver: v1 (new)"]
