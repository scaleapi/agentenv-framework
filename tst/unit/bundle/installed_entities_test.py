"""A bundle using an env, agent or artifact an installed bundle defines: by id, or as the configured default agent,
it's pulled into the plan as one of the bundle's own is, named after the bundle it comes from."""

import json
import subprocess
import sys
import textwrap
import time

import pytest

from agent_env.bundle import BundleError, BundleKind, dry_run_bundle, parse_bundle
from agent_env.bundle import preflight as preflight_module
from agent_env.bundle import run as run_module
from agent_env.bundle import installed as installed_module
from agent_env.bundle.installed import InstalledBundle, write_installed
from agent_env.bundle.ledger import locked_ids
from agent_env.bundle.materialize import materialize
from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle.plan import AliasUse, check_bundle, plan_bundle, plan_entities
from agent_env.bundle.resolve import resolve_bundle
from agent_env.plugins._bundles import ALIASES_FILE
from agent_env.entity.refs import EntityKind
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.store.routing import namespace_routing
from tst.unit.bundle._support import RefusingStore, layout, local_builds

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
        ("step 'judge': judge_a2a_agent_id (the default agent)", SOLVER),
        ("step 'llm': judge_a2a_agent_id (the default agent)", SOLVER),
        ("step 'own': judge_a2a_agent_id (the default agent)", SOLVER)]


def test_a_default_agent_the_bundle_defines_is_written_by_the_run(bundle_dir, default_agent):
    default_agent(f"{ROOT}/solver")
    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent"}],
                       {"agents/solver/Dockerfile": "FROM scratch\n"}))

    assert plan.writes[-1].needs == (("agent", f"{ROOT}/solver"),) and plan.origins == {}


@pytest.mark.parametrize("default", [SOLVER, f"{ROOT}/solver"])
def test_a_pin_on_a_default_agent_a_bundle_defines_is_refused(bundle_dir, tmp_path, default_agent, default):
    default_agent(default)
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_version": 2}],
                     {"agents/solver/Dockerfile": "FROM scratch\n"})

    [problem] = _problems(resolved, _kit(tmp_path))
    assert problem.startswith(f"tasks/t.json: step 'agent': a2a_agent_id (the default agent): {default!r} ")
    assert "so it has no fixed version; drop" in problem


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


def test_a_pulled_entity_naming_an_image_folder_of_its_bundle_pulls_the_image(bundle_dir, tmp_path):
    kit = _kit(tmp_path, {"artifacts/base/Dockerfile": "FROM scratch\n", "agents/coder/agent.toml": 'image = "base"\n'})
    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": f"{KIT}/coder"}]), kit)

    assert _writes(plan) == [(BundleKind.ARTIFACT, f"{KIT}/base"), (BundleKind.AGENT, f"{KIT}/coder"),
                             (BundleKind.TASK, f"{ROOT}/t")]



def test_an_entity_planned_alone_is_written_with_what_it_references_and_nothing_runs(tmp_path):
    kit = _kit(tmp_path, {"artifacts/base/Dockerfile": "FROM scratch\n", "agents/coder/agent.toml": 'image = "base"\n'})
    plan = plan_entities(resolve_bundle(parse_bundle(kit.root, id_root=KIT)), [f"{KIT}/coder"], installed=lambda: (kit,))

    assert _writes(plan) == [(BundleKind.ARTIFACT, f"{KIT}/base"), (BundleKind.AGENT, f"{KIT}/coder")]
    assert (plan.tasks, plan.evals) == ((), ())


def test_planning_an_entity_the_bundle_doesnt_define_is_refused(tmp_path):
    kit = _kit(tmp_path, {"tasks/t.json": json.dumps([{"id": "agent", "type": "deploy_agent", "a2a_agent_id": SOLVER}])})

    with pytest.raises(BundleError) as caught:
        plan_entities(resolve_bundle(parse_bundle(kit.root, id_root=KIT)), [f"{KIT}/coder", f"{KIT}/t"])
    assert caught.value.problems == (f"this bundle has no env, agent or artifact '{KIT}/coder'",
                                     f"this bundle has no env, agent or artifact '{KIT}/t'")


def test_an_installed_bundles_agent_is_written_by_its_id_and_anew_only_once_it_changes(
        tmp_path, monkeypatch, local_stores, cli_routing):
    kit = _kit(tmp_path)
    monkeypatch.setattr(installed_module, "installed_bundles", lambda: (kit,))

    version = write_installed(SOLVER)
    assert A2AAgent.get(SOLVER).docker_image_artifact.id == f"{SOLVER}__agent_image"
    assert write_installed(SOLVER) == version
    (kit.root / "agents/solver/Dockerfile").write_text("FROM scratch\nENV CHANGED=1\n")
    assert write_installed(SOLVER) == version + 1


@pytest.mark.parametrize(("installed", "problem"), [
    ((), "'@local/demo-dist/kit/solver' isn't an entity of an installed bundle"),
    (("broken",), "demo-dist/kit: its folder is gone"),
])
def test_an_entity_no_installed_bundle_can_write_is_refused(tmp_path, monkeypatch, installed, problem):
    bundles = tuple(_kit(tmp_path, problem="its folder is gone") for _ in installed)
    monkeypatch.setattr(installed_module, "installed_bundles", lambda: bundles)

    with pytest.raises(BundleError) as caught:
        write_installed(SOLVER)
    assert caught.value.problems == (problem,)

def test_a_pulled_entity_naming_the_running_bundles_own_entity_uses_it(tmp_path, local_stores, cli_routing):
    base = _installed(tmp_path / "site" / "base", {
        "artifacts/base/Dockerfile": "FROM scratch\n",
        "tasks/t.json": json.dumps([{"id": "agent", "type": "deploy_agent", "a2a_agent_id": f"{KIT}/coder"}])},
        dist="base-dist", name="base")
    kit = _kit(tmp_path, {"agents/coder/agent.toml": 'image = "@local/base-dist/base/base"\n'})
    plan = _plan(resolve_bundle(parse_bundle(base.root, id_root=base.id_root)), kit, base)

    assert _writes(plan) == [(BundleKind.ARTIFACT, "@local/base-dist/base/base"), (BundleKind.AGENT, f"{KIT}/coder"),
                             (BundleKind.TASK, "@local/base-dist/base/t")]
    assert [plan.path(write.source.entry) for write in plan.writes] == [
        "artifacts/base", "demo-dist/kit:agents/coder", "tasks/t.json"]


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

    assert dry.images() == ("demo-dist/kit:agents/solver (Dockerfile image) v1: a build context, built with docker on "
                            "this machine",)
    assert [dry.path(entry) for entry in dry.runs] == ["tasks/t.json"]
    assert [line for line in said if "demo-dist/kit" in line] == [
        "demo-dist/kit:agents/solver (Dockerfile image): v1 (new)", "demo-dist/kit:agents/solver: v1 (new)"]


def _user(tmp_path, name):
    return layout(tmp_path / name, {"tasks/t.json": json.dumps([
        {"id": "agent", "type": "deploy_agent", "a2a_agent_id": SOLVER}])})


def _materialized(root, *installed, id_root=None):
    plan = plan_bundle(resolve_bundle(parse_bundle(root, id_root=id_root)), installed=lambda: installed)
    return materialize(plan, build_here=local_builds(plan))


def _kits(materialization):
    return {done.write.id: (done.version, done.reused, done.reasons) for done in materialization.writes
            if done.write.id.startswith(f"{SOLVER}")}


def test_every_bundle_using_an_installed_agent_shares_one_write_of_it(bundle_dir, tmp_path, builds):
    kit = _kit(tmp_path, {"tasks/smoke.json": json.dumps([{"id": "agent", "type": "deploy_agent",
                                                            "a2a_agent_id": "solver"}])})
    image = f"{SOLVER}__agent_image"

    assert _kits(_materialized(_user(tmp_path, "first"), kit)) == {image: (1, False, ("new",)),
                                                                   SOLVER: (1, False, ("new",))}
    assert _kits(_materialized(_user(tmp_path, "second"), kit)) == {image: (1, True, ()), SOLVER: (1, True, ())}
    assert _kits(_materialized(kit.root, kit, id_root=KIT)) == {image: (1, True, ()), SOLVER: (1, True, ())}

    (kit.root / "agents/solver/Dockerfile").write_text("FROM scratch\nENV CHANGED=1\n")
    assert _kits(_materialized(_user(tmp_path, "second"), kit)) == {
        image: (2, False, ("files changed: Dockerfile",)),
        SOLVER: (2, False, (f"artifact {image} is written anew (v1 → v2)",))}


def test_a_run_locks_the_entities_it_pulls_in_as_it_does_its_own(bundle_dir, tmp_path, builds):
    plan = plan_bundle(resolve_bundle(parse_bundle(_user(tmp_path, "first"))), installed=lambda: (_kit(tmp_path),))
    held, done, waited = tmp_path / "held", tmp_path / "done", []
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import pathlib, time
        from agent_env.bundle.ledger import materializing
        with materializing([{SOLVER!r}]):
            pathlib.Path({str(held)!r}).write_text("held")
            time.sleep(1.5)
            pathlib.Path({str(done)!r}).write_text("done")
    """)])
    try:
        for _ in range(200):
            if held.exists():
                break
            time.sleep(0.05)
        materialize(plan, on_wait=lambda: waited.append(done.exists()))
        assert done.exists()
    finally:
        holder.wait()

    assert SOLVER in locked_ids(plan) and waited == [False]


# Agents named by an alias an installed bundle gives them

ALIASED = {ALIASES_FILE: '[agents]\nsolver = ["solver-x"]\n'}


def _store_agent(agent_id):
    with namespace_routing():
        image = get_artifact_store().put_document(DockerImageArtifact(
            id=f"{agent_id}-image", description="d", image_name="ghcr.io/acme/agent:1", tar_gz_s3_url="s3://b/x.tar.gz"))
        return A2AAgent.put(id=agent_id, docker_image_artifact=image)


def _step(plan, step_id="agent"):
    task = next(write for write in plan.writes if write.kind is BundleKind.TASK)
    return next(step for step in task.source.config if step["id"] == step_id)


def test_an_agent_named_by_an_installed_alias_is_pulled_in_and_the_task_names_it_by_id(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ALIASED)

    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x"}]), kit)

    assert _writes(plan) == [(BundleKind.ARTIFACT, f"{SOLVER}__agent_image"), (BundleKind.AGENT, SOLVER),
                             (BundleKind.TASK, f"{ROOT}/t")]
    assert _step(plan)["a2a_agent_id"] == SOLVER
    assert plan.aliases == {"solver-x": AliasUse("solver-x", SOLVER, "demo-dist's kit bundle")}
    assert plan.aliases["solver-x"].agent_id == SOLVER and plan.store_refs == ()


ENV_ALIASED = {"envs/crm/Dockerfile": "FROM scratch\n", "envs/crm/env.toml": 'environment_name = "crm"\n',
               ALIASES_FILE: '[envs]\ncrm = ["crm-x"]\n'}
CRM = f"{KIT}/crm"
CRM_STEPS = [{"id": "env", "type": "deploy_env", "env_id": "crm-x"},
             {"id": "agent", "type": "deploy_agent", "a2a_agent_id": SOLVER, "env_ids": ["crm-x"],
              "depends_on": ["env"]}]


def test_an_env_named_by_an_installed_alias_is_pulled_in_and_the_task_names_it_by_id(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ENV_ALIASED)

    plan = _plan(_task(bundle_dir, CRM_STEPS), kit)

    assert (BundleKind.ENV, CRM) in _writes(plan)
    assert _step(plan, "env")["env_id"] == CRM and _step(plan, "agent")["env_ids"] == [CRM]
    assert plan.env_aliases == {"crm-x": AliasUse("crm-x", CRM, "demo-dist's kit bundle", kind=EntityKind.ENV)}
    assert plan.aliases == {} and plan.store_refs == ()


def test_the_stores_env_of_an_alias_id_shadows_the_alias(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ENV_ALIASED)
    with namespace_routing():
        image = get_artifact_store().put_document(DockerImageArtifact(
            id="crm-x-image", description="d", image_name="ghcr.io/acme/crm:1", tar_gz_s3_url="s3://b/x.tar.gz"))
        MCPServerEnv.put(id="crm-x", docker_image_artifact=image, environment_name="crm")

    plan = _plan(_task(bundle_dir, CRM_STEPS), kit)

    assert (BundleKind.ENV, CRM) not in _writes(plan) and _step(plan, "env")["env_id"] == "crm-x"
    assert plan.env_aliases["crm-x"].store_version == 1 and [ref.id for ref in plan.store_refs] == ["crm-x"]



def test_an_env_definition_naming_an_env_alias_is_refused_saying_to_name_the_env_by_its_id(bundle_dir, tmp_path):
    """An env alias names an env in a task's steps; a MultiEnv's servers name theirs by id."""
    kit = _kit(tmp_path, ENV_ALIASED)

    problems = _problems(_task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "suite"}],
                               {"envs/suite/env.toml": 'type = "multi"\nmcp_server_envs = ["crm-x"]\n'}), kit)

    assert any(f"'crm-x' is an alias of {CRM}, which comes with demo-dist's kit bundle, and an env alias names an env only "
               "in a task's steps; name the env by its id" in problem for problem in problems), problems

def test_the_stores_agent_of_an_alias_id_shadows_the_alias(bundle_dir, tmp_path):
    """Someone's own agent of that id is the one they mean."""
    kit = _kit(tmp_path, ALIASED)
    _store_agent("solver-x")

    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x"}]), kit)

    assert _writes(plan) == [(BundleKind.TASK, f"{ROOT}/t")] and _step(plan)["a2a_agent_id"] == "solver-x"
    assert plan.aliases["solver-x"] == AliasUse("solver-x", SOLVER, "demo-dist's kit bundle", store_version=1)
    assert plan.aliases["solver-x"].agent_id == "solver-x" and [ref.id for ref in plan.store_refs] == ["solver-x"]


def test_a_bundles_own_agent_named_like_an_installed_alias_is_the_one_it_names(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ALIASED)

    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x"}],
                       {"agents/solver-x/Dockerfile": "FROM scratch\n"}), kit)

    assert _step(plan)["a2a_agent_id"] == f"{ROOT}/solver-x" and plan.aliases == {}


def test_every_field_naming_an_alias_in_a_task_is_renamed(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ALIASED)

    plan = _plan(_task(bundle_dir, [
        {"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x"},
        {"id": "p", "type": "prompt_agent", "prompt_id": "p", "prompt": "hi", "depends_on": ["agent"]},
        {"id": "v", **JUDGE, "judge_a2a_agent_id": "solver-x", "depends_on": ["p"]},
    ]), kit)

    assert _step(plan)["a2a_agent_id"] == _step(plan, "v")["judge_a2a_agent_id"] == SOLVER


def test_an_alias_takes_no_pin(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ALIASED)
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x",
                                   "a2a_agent_version": 2}])

    assert _problems(resolved, kit) == (
        f"tasks/t.json: step 'agent': a2a_agent_id: 'solver-x' is an alias of {SOLVER}, which comes with demo-dist's "
        "kit bundle, so it has no fixed version; drop the pin",)


def test_a_pinned_reference_to_the_stores_agent_of_an_alias_id_names_that_agent(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ALIASED)
    _store_agent("solver-x")

    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x",
                                     "a2a_agent_version": 1}]), kit)

    assert [(ref.id, ref.version) for ref in plan.store_refs] == [("solver-x", 1)] and plan.aliases == {}


def test_an_alias_two_installed_bundles_give_is_refused_naming_both(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ALIASED)
    other = _installed(tmp_path / "site2" / "tools", {"agents/helper/Dockerfile": "FROM scratch\n",
                                                      ALIASES_FILE: '[agents]\nhelper = ["solver-x"]\n'},
                       dist="zeta-tools", name="tools")
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x"}])

    [problem] = _problems(resolved, kit, other)

    assert problem == ("tasks/t.json: step 'agent': a2a_agent_id: 'solver-x' is an alias demo-dist's kit bundle and "
                       "zeta-tools's tools bundle each give an agent, so it names none of them; name the agent by its id")


def test_the_stores_agent_of_a_refused_alias_id_is_the_one_it_names(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ALIASED)
    other = _installed(tmp_path / "site2" / "tools", {"agents/helper/Dockerfile": "FROM scratch\n",
                                                      ALIASES_FILE: '[agents]\nhelper = ["solver-x"]\n'},
                       dist="zeta-tools", name="tools")
    _store_agent("solver-x")

    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x"}]), kit, other)

    assert [ref.id for ref in plan.store_refs] == ["solver-x"] and plan.aliases == {}


def test_an_agent_misspelling_an_alias_is_pointed_at_it(bundle_dir, tmp_path):
    kit = _kit(tmp_path, ALIASED)
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver_x"}])

    assert _problems(resolved, kit) == (
        "tasks/t.json: step 'agent': a2a_agent_id: there is no agent 'solver_x' in the store; did you mean the alias "
        "'solver-x'? agent-env a2a-agent list lists the agents a task can name",)


def test_an_alias_whose_agent_its_bundle_lacks_is_refused_naming_the_bundle(bundle_dir, tmp_path):
    kit = _kit(tmp_path, {ALIASES_FILE: '[agents]\nghost = ["ghost"]\n'})
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "ghost"}])

    [problem] = _problems(resolved, kit)

    assert problem.startswith(f"tasks/t.json: step 'agent': a2a_agent_id: '{KIT}/ghost' comes with the installed "
                              "bundle demo-dist/kit, which can't be read: aliases.toml: agents.ghost: this bundle has "
                              "no agent 'ghost'")


def test_a_default_agent_set_to_an_alias_is_pulled_in_for_a_step_naming_none(bundle_dir, tmp_path, default_agent):
    kit = _kit(tmp_path, ALIASED)
    default_agent("solver-x")

    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent"}]), kit)

    assert (BundleKind.AGENT, SOLVER) in _writes(plan) and "a2a_agent_id" not in _step(plan)
    assert plan.aliases == {"solver-x": AliasUse("solver-x", SOLVER, "demo-dist's kit bundle")}


def test_a_default_agent_set_to_an_alias_no_step_falls_back_to_isnt_read(bundle_dir, tmp_path, default_agent):
    kit = _kit(tmp_path, ALIASED)
    default_agent("solver-x")

    plan = _plan(_task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": SOLVER}]), kit)

    assert plan.aliases == {}


def test_checking_a_bundle_naming_an_alias_reads_no_store(bundle_dir, tmp_path, monkeypatch):
    kit = _kit(tmp_path, ALIASED)
    resolved = _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x"}])
    monkeypatch.setattr("agent_env.bundle.plan.STORE_GETTERS", RefusingStore())

    check_bundle(resolved, installed=lambda: (kit,))


def test_a_dry_run_says_which_agent_an_alias_names_and_preflights_an_aliased_default(bundle_dir, tmp_path,
                                                                                     default_agent, monkeypatch):
    default_agent("solver-x")
    kit = _kit(tmp_path, ALIASED)
    monkeypatch.setattr(run_module, "installed_bundles", lambda: (kit,))
    monkeypatch.setattr(preflight_module, "docker_unreachable", lambda: None)
    _task(bundle_dir, [{"id": "agent", "type": "deploy_agent"}])
    said = []

    dry = dry_run_bundle(bundle_dir, sandbox="local", on_progress=said.append)

    assert said[0] == f"solver-x → {SOLVER} (the alias demo-dist's kit bundle gives it)"
    assert [dry.path(entry) for entry in dry.runs] == ["tasks/t.json"]


def test_a_dry_run_says_when_the_store_shadows_an_alias(bundle_dir, tmp_path, monkeypatch):
    kit = _kit(tmp_path, ALIASED)
    _store_agent("solver-x")
    monkeypatch.setattr(run_module, "installed_bundles", lambda: (kit,))
    monkeypatch.setattr(preflight_module, "docker_unreachable", lambda: None)
    _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "a2a_agent_id": "solver-x"}])
    said = []

    dry_run_bundle(bundle_dir, sandbox="local", on_progress=said.append)

    assert said[0] == (f"solver-x → the store's agent 'solver-x' v1, which shadows the alias demo-dist's kit bundle "
                       f"gives {SOLVER}")

