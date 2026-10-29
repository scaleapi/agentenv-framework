"""``--agent``: the agent a run deploys for the solver, at each step deploying or installing the agent a step prompts,
named as a task's reference would name it, with the tasks left as written."""

import json

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError, dry_run_bundle, parse_bundle, run_bundle
from agent_env.bundle import preflight as preflight_module
from agent_env.bundle.installed import InstalledBundle
from agent_env.bundle.plan import AgentOverride, plan_bundle, solver_steps
from agent_env.bundle.resolve import resolve_bundle
from agent_env.cli import cli
from agent_env.entity.refs import EntityKind
from agent_env.plugins._bundles import ALIASES_FILE
from agent_env.store.routing import namespace_routing
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep
from tst.unit.bundle._support import layout

ROOT = "@local/~/triage"
SOLVER = "@local/demo-dist/kit/solver"
DEPLOY = {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "theirs"}
PROMPT = {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "prompt": "hi", "depends_on": ["agent"]}
INSTALL = [{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "local"},
           {"id": "agent", "type": "install_agent", "sandbox_name": "box", "agent_name": "coder", "a2a_agent_id": "theirs",
            "depends_on": ["box"]},
           {**PROMPT, "agent_name": "coder"}]


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    _store_agent("theirs")
    return tmp_path / "triage"


@pytest.fixture
def kit(tmp_path):
    """An installed bundle whose agent ``solver`` has the alias ``solver-x``."""
    root = layout(tmp_path / "site" / "kit", {"agents/solver/Dockerfile": "FROM scratch\n",
                                              ALIASES_FILE: '[agents]\nsolver = ["solver-x"]\n'})
    return InstalledBundle("kit", "demo-dist", "1.0", "demo", root, None, None)


def _store_agent(agent_id):
    with namespace_routing():
        image = get_artifact_store().put_document(DockerImageArtifact(
            id=f"{agent_id}-image", description="d", image_name="ghcr.io/acme/agent:1", tar_gz_s3_url="s3://b/x.tar.gz"))
        return A2AAgent.put(id=agent_id, docker_image_artifact=image)


def _bundle(bundle_dir, steps, files=None):
    layout(bundle_dir, {"tasks/t.json": json.dumps(steps), **(files or {})})
    return bundle_dir


def _plan(bundle_dir, agent, *installed):
    return plan_bundle(resolve_bundle(parse_bundle(bundle_dir)), installed=lambda: installed, agent=agent)


def _problems(bundle_dir, agent, *installed):
    with pytest.raises(BundleError) as caught:
        _plan(bundle_dir, agent, *installed)
    return caught.value.problems


def _step(plan, step_id):
    task = next(write for write in plan.writes if write.id == f"{ROOT}/t")
    return next(step for step in task.source.config if step["id"] == step_id)


def test_an_alias_names_the_agent_the_solver_deploys_and_the_task_stays_as_written(bundle_dir, kit):
    plan = _plan(_bundle(bundle_dir, [DEPLOY, PROMPT]), "solver-x", kit)

    assert plan.agent == AgentOverride("solver-x", SOLVER, {f"{ROOT}/t": ("agent",)})
    assert SOLVER in {write.id for write in plan.writes}
    assert _step(plan, "agent")["a2a_agent_id"] == "theirs"
    assert plan.alias_uses[EntityKind.AGENT]["solver-x"].target_id == SOLVER


def test_this_bundles_own_agent_is_named_by_its_folder(bundle_dir):
    plan = _plan(_bundle(bundle_dir, [DEPLOY, PROMPT], {"agents/helper/Dockerfile": "FROM scratch\n"}), "helper")

    assert plan.agent.agent_id == f"{ROOT}/helper" and f"{ROOT}/helper" in {write.id for write in plan.writes}


def test_the_stores_agent_is_named_by_its_id_at_the_version_the_plan_read(bundle_dir):
    _store_agent("mine")
    _store_agent("mine")

    plan = _plan(_bundle(bundle_dir, [DEPLOY, PROMPT]), "mine")

    assert (plan.agent.agent_id, plan.agent.version) == ("mine", 2)
    assert "mine" not in {write.id for write in plan.writes}


def test_a_name_that_names_no_agent_is_refused_pointing_at_a_close_alias(bundle_dir, kit):
    assert _problems(_bundle(bundle_dir, [DEPLOY, PROMPT]), "solver_x", kit) == (
        "--agent 'solver_x': there is no agent 'solver_x' in this bundle, an installed bundle or the store; did you "
        "mean the alias 'solver-x'? agent-env a2a-agent list lists the agents a task can name",)


def test_this_bundles_agent_is_named_with_its_folders_case_as_a_task_names_it(bundle_dir):
    assert _problems(_bundle(bundle_dir, [DEPLOY, PROMPT], {"agents/Helper/Dockerfile": "FROM scratch\n"}), "helper") == (
        "--agent 'helper': isn't in this bundle, but 'Helper' is; names are case-sensitive",)


def test_a_bundle_whose_tasks_prompt_no_agent_is_refused(bundle_dir):
    assert _problems(_bundle(bundle_dir, [DEPLOY]), "theirs") == (
        "--agent 'theirs': no step in this bundle's tasks prompts an agent, so none deploys a solver",)


def test_only_the_steps_deploying_or_installing_the_agent_a_step_prompts_are_the_solvers():
    steps = [
        {"id": "solver", "type": "deploy_agent"},
        {"id": "user", "type": "deploy_agent", "agent_name": "carol"},
        {"id": "judge", "type": "deploy_agent", "agent_name": "grader"},
        {"id": "installed", "type": "install_agent", "agent_name": "coder", "sandbox_name": "box"},
        {"id": "ask", "type": "prompt_agent", "user_agent_name": "carol"},
        {"id": "ask-coder", "type": "prompt_agent", "agent_name": "coder"},
        {"id": "grade", "type": "rubrics_verifier", "agent_name": "grader"},
    ]

    assert solver_steps(steps) == ("solver", "installed")


def test_preflight_checks_the_agent_the_solver_deploys(bundle_dir, monkeypatch):
    checked = []
    original = preflight_module._Walk._agent_deploy

    def record(self, where, provider, agent_id, version, *args, **kwargs):
        checked.append((agent_id, version))
        return original(self, where, provider, agent_id, version, *args, **kwargs)

    monkeypatch.setattr(preflight_module._Walk, "_agent_deploy", record)
    _bundle(bundle_dir, [{**DEPLOY, "a2a_agent_version": 1, "sandbox_type": "local"}, PROMPT],
            {"agents/helper/Dockerfile": "FROM scratch\n"})

    dry_run_bundle(bundle_dir, agent="helper")
    _store_agent("mine")
    dry_run_bundle(bundle_dir, agent="mine")

    assert checked == [(f"{ROOT}/helper", None), ("mine", 1)]


def test_an_install_is_checked_for_the_agent_the_solver_installs_not_the_one_the_task_names(bundle_dir):
    _store_agent("mine")
    _bundle(bundle_dir, INSTALL, {"agents/helper/Dockerfile": "FROM scratch\n"})

    with pytest.raises(BundleError, match="install_agent 'agent': Agent theirs v1 has no build_context_object_url"):
        dry_run_bundle(bundle_dir)
    dry_run_bundle(bundle_dir, agent="helper")
    with pytest.raises(BundleError, match="install_agent 'agent': Agent mine v1 has no build_context_object_url"):
        dry_run_bundle(bundle_dir, agent="mine")


def test_a_dry_run_says_what_the_solver_deploys(bundle_dir, monkeypatch):
    lines = []
    _store_agent("mine")

    dry_run_bundle(_bundle(bundle_dir, [{**DEPLOY, "sandbox_type": "local"}, PROMPT]), agent="mine",
                   on_progress=lines.append)

    assert "--agent mine: the solver deploys mine, in 1 of 1 task" in lines


def test_a_run_deploys_the_agent_at_the_solvers_steps_alone_at_the_version_it_wrote(bundle_dir, builds, monkeypatch):
    seen = {}

    async def run(self, *, context, **kwargs):
        seen[self.id] = context.metadata.get("user_overrides")
        return context

    monkeypatch.setattr(Task, "run", run)
    user = {"id": "user", "type": "deploy_agent", "env_ids": [], "agent_name": "carol", "a2a_agent_id": "theirs"}
    _bundle(bundle_dir, [DEPLOY, user, {**PROMPT, "user_agent_name": "carol", "depends_on": ["agent", "user"]}],
            {"agents/helper/Dockerfile": "FROM scratch\n"})

    run_bundle(bundle_dir, agent="helper", sandbox="local")

    assert seen[f"{ROOT}/t"]["step_params"] == {"agent": {"a2a_agent_id": f"{ROOT}/helper", "a2a_agent_version": 1}}


def test_a_run_deploys_the_stores_agent_at_the_version_the_plan_read_not_one_put_since(bundle_dir, monkeypatch):
    seen = {}

    async def run(self, *, context, **kwargs):
        _store_agent("mine")
        seen[self.id] = context.metadata.get("user_overrides")
        return context

    monkeypatch.setattr(Task, "run", run)
    _store_agent("mine")
    _bundle(bundle_dir, [DEPLOY, PROMPT])

    run_bundle(bundle_dir, agent="mine", sandbox="local")

    assert seen[f"{ROOT}/t"]["step_params"] == {"agent": {"a2a_agent_id": "mine", "a2a_agent_version": 1}}


def test_an_install_takes_its_steps_override_over_the_run_wide_one_and_the_stored_agent():
    step = InstallAgentTaskStep(id="agent", version=None, sandbox_name="box", a2a_agent_id="stored", a2a_agent_version=3)
    solver = {"step_params": {"agent": {"a2a_agent_id": "solver", "a2a_agent_version": None}}}

    assert step._agent_ref(TaskStepContext(metadata={"user_overrides": {"a2a_agent_id": "run-wide", **solver}})) == (
        "solver", None)
    assert step._agent_ref(TaskStepContext(metadata={"user_overrides": {"a2a_agent_id": "run-wide"}})) == ("run-wide", 3)
    assert step._agent_ref(TaskStepContext()) == ("stored", 3)


def test_the_cli_takes_the_agent_for_a_run_and_a_dry_run(bundle_dir, monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm")
    _store_agent("mine")
    _bundle(bundle_dir, [{**DEPLOY, "sandbox_type": "local"}, PROMPT])

    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--agent", "mine", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "--agent mine: the solver deploys mine, in 1 of 1 task" in result.output
