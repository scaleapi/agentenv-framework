"""Where a bundle run's tasks deploy, checked before anything is written: images only this machine has on another
provider, VMs asked of a provider that can't create one, containers on the local provider without Docker, the default
agent, and the infra envs a gateway deploy on the local provider needs."""

import asyncio
import json
import logging
from typing import ClassVar

import pytest
from click.testing import CliRunner

import agent_env.bundle.preflight as preflight_module
import agent_env.bundle.run as run_module
from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError, dry_run_bundle, run_bundle
from agent_env.cli import cli
from agent_env.config.runtime import Config
from agent_env.env import Env, GatewayEnv, MCPServerEnv, MultiEnv
from agent_env.env import bootstrap
from agent_env.env.envs.service_db import ServiceDBEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.store import get_env_store
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_state.env_state_provider import EnvStateInstance
from agent_env.providers.env_state.store import register_env_state_instance
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.store.routing import namespace_routing
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from tst.unit.bundle._support import layout, local_store

LOCAL = ("localhost:5000/local/img:v1", "file:///state/img-v1.tar.gz")
REMOTE = ("123456789012.dkr.ecr.us-west-2.amazonaws.com/img:v1", "s3://bucket/img-v1.tar.gz")


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "triage"


@pytest.fixture
def quiet_logs():
    """pytest's live logging swaps its own stdout back in to print a record, so CliRunner loses whatever is
    echoed after the first one."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _task(root, steps, name="t"):
    layout(root, {f"tasks/{name}.json": json.dumps(steps)})


def _image(id, where=LOCAL):
    with namespace_routing():
        return get_artifact_store().put_document(DockerImageArtifact(
            id=id, description=id, image_name=where[0], tar_gz_s3_url=where[1]))


def _agent(id, where=LOCAL):
    with namespace_routing():
        return A2AAgent.put(id=id, docker_image_artifact=_image(f"{id}-image", where))


def _env(id, where=LOCAL, **fields):
    with namespace_routing():
        return MCPServerEnv.put(id=id, docker_image_artifact=_image(f"{id}-image", where), environment_name=id, **fields)


def _infra(where=LOCAL, built_from=None):
    """The infra envs, as agent-env would have built them from inputs whose digest is ``built_from``."""
    def metadata(dockerfile):
        return {"dockerfile_path": str(dockerfile), bootstrap.BUILD_INPUTS_KEY: built_from} if built_from else None

    with namespace_routing():
        GatewayEnv.put(id="default", docker_image_artifact=_image("gateway-default", where),
                       metadata=metadata(bootstrap.GATEWAY_DOCKERFILE))
        ServiceDBEnv.put(id="default-db", db_docker_image_artifact=_image("service-db-default-db", where),
                         db_web_docker_image_artifact=_image("db-web-default-db", where),
                         db_mcp_docker_image_artifact=_image("db-mcp-default-db", where),
                         metadata=metadata(bootstrap.SERVICE_DB_DOCKERFILE))


def _problems(fn):
    with pytest.raises(BundleError) as e:
        fn()
    return list(e.value.problems)


AGENT = {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}


# Images only this machine has


@pytest.mark.parametrize("sandbox", ["modal", "local,modal"])
def test_an_image_only_this_machine_has_is_refused_on_another_provider_before_anything_is_written(bundle_dir, sandbox):
    _agent("solver")
    _task(bundle_dir, [AGENT])

    assert _problems(lambda: run_bundle(bundle_dir, sandbox=sandbox)) == [
        "tasks/t.json: step 'agent': deploys agent 'solver''s image on the 'modal' sandbox provider, which can't reach it: "
        "localhost:5000/local/img:v1 is in a registry on this machine; run it with --sandbox local",
    ]
    assert not local_store().path.exists()


def test_an_image_saved_only_in_this_machines_object_store_is_refused_too(bundle_dir):
    _agent("solver", (REMOTE[0], LOCAL[1]))
    _task(bundle_dir, [AGENT])

    (problem,) = _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal_vm"))
    assert problem.endswith("can't reach it: 'solver-image' is saved in this machine's object store; run it with "
                            "--sandbox local")


def test_a_remote_image_runs_anywhere_and_a_local_one_on_the_local_provider(bundle_dir):
    _agent("solver", REMOTE)
    _agent("helper")
    _task(bundle_dir, [AGENT, {**AGENT, "id": "helper", "agent_name": "helper", "a2a_agent_id": "helper"}])

    assert [entry.name for entry in dry_run_bundle(bundle_dir, sandbox="local").runs] == ["t"]
    (problem,) = _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal"))
    assert problem.startswith("tasks/t.json: step 'helper': deploys agent 'helper''s image")


def test_an_agent_built_from_the_bundle_is_refused_on_another_provider(bundle_dir):
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\n"})
    _task(bundle_dir, [AGENT])

    assert _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal")) == [
        "tasks/t.json: step 'agent': deploys agent '@local/~/triage/solver''s image on the 'modal' sandbox provider, "
        "which can't reach it: it's built on this machine from agents/solver/Dockerfile; run it with --sandbox local",
    ]


def test_a_sandbox_image_in_this_machines_registry_is_refused_on_another_provider(bundle_dir):
    sandbox = {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "container", "port": 80}
    _task(bundle_dir, [{**sandbox, "image": "localhost:5000/box:v1"}, {**sandbox, "id": "py", "sandbox_name": "py",
                                                                      "image": "python:3.12"}])

    assert _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal")) == [
        "tasks/t.json: step 'box': deploys the image localhost:5000/box:v1 on the 'modal' sandbox provider, which can't "
        "reach it: localhost:5000/box:v1 is in a registry on this machine; run it with --sandbox local",
    ]


# What a provider can create


@pytest.mark.parametrize("sandbox, refused", [("modal", True), ("local,modal_vm", True), ("modal_vm", False), (None, False)])
def test_a_vm_sandbox_is_refused_on_a_provider_that_cant_create_one(bundle_dir, sandbox, refused):
    _task(bundle_dir, [{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm",
                        "sandbox_type": "local"}])

    if refused:
        (problem,) = _problems(lambda: dry_run_bundle(bundle_dir, sandbox=sandbox))
        assert problem == (f"tasks/t.json: step 'box': deploys a VM sandbox, and the {sandbox!r} sandbox provider can't "
                           "create a VM; run it on one that can, such as --sandbox local")
    else:
        dry_run_bundle(bundle_dir, sandbox=sandbox)


def test_a_step_naming_a_provider_that_doesnt_exist_is_refused(bundle_dir):
    _agent("solver")
    _task(bundle_dir, [{**AGENT, "sandbox_type": "nosuch"}])

    (problem,) = _problems(lambda: dry_run_bundle(bundle_dir))
    assert problem.startswith("tasks/t.json: step 'agent': Unknown sandbox backend: 'nosuch'")


# Docker


def test_containers_on_the_local_provider_need_docker_and_a_vm_sandbox_doesnt(bundle_dir, monkeypatch):
    probes = []
    monkeypatch.setattr(preflight_module, "docker_unreachable", lambda: probes.append(1) or "docker isn't on PATH")
    _task(bundle_dir, [{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm",
                        "sandbox_type": "local"}], name="hello")
    _agent("solver")
    _task(bundle_dir, [AGENT], name="agent")

    dry_run_bundle(bundle_dir, tasks=["hello"])
    assert probes == []
    assert _problems(lambda: dry_run_bundle(bundle_dir, tasks=["agent", "hello"])) == [
        "the local sandbox provider runs containers for tasks/agent.json: step 'agent', and docker isn't on PATH",
    ]


class _DeploysItself(Env):
    """An env type with a deploy() of its own, as a plugin's env can have."""

    type: ClassVar[str] = "deploys_itself_preflight_test"
    description = "test"

    @classmethod
    def from_dict(cls, data):
        return cls(id=data["id"], version=data.get("version"))

    async def deploy(self, **kwargs):
        raise NotImplementedError


class _BuildsItsOwnGateway(_DeploysItself):
    """One that sets up a gateway itself, with no image of its own."""

    type: ClassVar[str] = "builds_its_own_gateway_preflight_test"

    def __init__(self, id, version, metadata=None):
        super().__init__(id, version, metadata=metadata)
        self._env_provider = EnvironmentGatewayProvider()


@pytest.mark.parametrize("cls", [_DeploysItself, _BuildsItsOwnGateway])
def test_an_env_type_that_deploys_itself_is_left_to_its_own_deploy(bundle_dir, monkeypatch, cls):
    registry = Config.env_registry
    monkeypatch.setattr(Config, "env_registry", lambda self: {**registry(self), cls.type: cls})
    with namespace_routing():
        get_env_store().put_document(cls(id="plug", version=None))
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "plug"}])

    assert dry_run_bundle(bundle_dir, sandbox="modal").infra == ()


def test_a_multi_env_on_the_server_provider_is_left_to_deploy_envs_own_refusal(bundle_dir):
    crm = _env("crm")
    with namespace_routing():
        MultiEnv.put(id="both", mcp_server_envs=[crm], env_provider_type="server")
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "both"}])

    (problem,) = _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal"))
    assert "deploys one MCP server, not a multi env" in problem


def test_a_website_on_modal_is_refused_since_its_gateway_runs_in_containers(bundle_dir):
    with namespace_routing():
        WebsiteEnv.put(id="shop", backend_docker_image_artifact=_image("shop-back", REMOTE),
                       frontend_docker_image_artifact=_image("shop-front", REMOTE), environment_name="shop")
    _infra(REMOTE)
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "shop"}])

    assert _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal")) == [
        "tasks/t.json: step 'env': deploys env 'shop', which has websites, on the 'modal' sandbox provider, whose "
        "gateway runs in containers and can't serve websites; run it on a VM provider, such as --sandbox local",
    ]


def test_an_agent_over_another_of_the_bundles_writes_gets_that_writes_own_refusal(bundle_dir):
    layout(bundle_dir, {"artifacts/base/Dockerfile": "FROM scratch\n", "agents/solver/agent.toml": 'image = "base"\n'})
    _task(bundle_dir, [AGENT])

    assert _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal")) == [
        "artifacts/base: writing a docker_image artifact isn't supported yet"]


# The default agent


def test_a_step_that_names_no_agent_needs_the_default_in_the_store(bundle_dir, monkeypatch):
    monkeypatch.setattr("agent_env.config.runtime.Config.get_default_a2a_agent_id", lambda self: "house-agent")
    _task(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": []},
                       {"id": "judge", "type": "rubrics_verifier", "prompt_id": "p", "verifier_id": "v",
                        "criteria": [{"id": "c", "description": "done"}]}])

    assert _problems(lambda: dry_run_bundle(bundle_dir)) == [
        "tasks/t.json: step 'agent': names no agent, so it deploys the default, 'house-agent', and there is no agent "
        "'house-agent' in the store; register one under that id, or point [agents] default_a2a_agent_id at an agent "
        "that is",
        "tasks/t.json: step 'judge': names no judge agent, so it deploys the default, 'house-agent', and there is no "
        "agent 'house-agent' in the store; register one under that id, or point [agents] default_a2a_agent_id at an "
        "agent that is",
    ]
    _agent("house-agent")
    dry_run_bundle(bundle_dir)


def test_a_judge_the_task_deploys_itself_or_the_direct_llm_judge_needs_no_default_agent(bundle_dir, monkeypatch):
    monkeypatch.setattr("agent_env.config.runtime.Config.get_default_a2a_agent_id", lambda self: "house-agent")
    judge = {"type": "rubrics_verifier", "prompt_id": "p", "verifier_id": "v", "criteria": [{"id": "c", "description": "done"}]}
    _agent("solver")
    _task(bundle_dir, [AGENT, {**judge, "id": "own", "agent_name": "default-agent"},
                       {**judge, "id": "llm", "use_agent_judge": False, "default_model": "m"}])

    dry_run_bundle(bundle_dir)


# Infra envs


def test_a_gateway_deploy_on_the_local_provider_names_the_infra_it_builds(bundle_dir):
    _env("crm")
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm"}])

    assert [str(build) for build in dry_run_bundle(bundle_dir).infra] == [
        "service-db env 'default-db' (missing)", "gateway env 'default' (missing)"]


def test_external_state_needs_no_service_db_and_a_website_needs_the_browser(bundle_dir):
    with namespace_routing():
        WebsiteEnv.put(id="shop", backend_docker_image_artifact=_image("shop-back"),
                       frontend_docker_image_artifact=_image("shop-front"), environment_name="shop")
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "shop", "env_state_type": "external_db"}])

    assert [build.kind for build in dry_run_bundle(bundle_dir).infra] == ["gateway", "website-browser"]


@pytest.mark.parametrize("state_type, kinds", [("local_postgres", ["service-db", "gateway"]), ("external_db", ["gateway"])])
def test_an_attached_state_instance_needs_the_service_db_by_its_own_type(bundle_dir, state_type, kinds):
    _env("crm")
    with namespace_routing():
        register_env_state_instance(EnvStateInstance(state_type=state_type, instance_id="state-1"), ttl_seconds=600)
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm", "env_state_instance_id": "state-1"}])

    assert [build.kind for build in dry_run_bundle(bundle_dir).infra] == kinds


def test_infra_built_from_other_inputs_is_rebuilt_and_infra_built_from_this_releases_isnt(bundle_dir, monkeypatch):
    _env("crm")
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm"}])
    _infra(built_from="an earlier release's")

    assert [str(build) for build in dry_run_bundle(bundle_dir).infra] == [
        "service-db env 'default-db' (its build inputs changed)",
        "gateway env 'default' (its build inputs changed)",
    ]
    monkeypatch.setattr("agent_env.env.bootstrap.build_inputs_digest", lambda kind: "an earlier release's")
    assert dry_run_bundle(bundle_dir).infra == ()


def test_missing_infra_in_stores_that_arent_local_is_refused_naming_its_put_command(bundle_dir, monkeypatch):
    monkeypatch.setattr("agent_env.env.bootstrap.stores_are_local", lambda: False)
    _env("crm")
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm"}])

    assert _problems(lambda: dry_run_bundle(bundle_dir)) == [
        "the gateway env 'default' isn't in the store, and agent-env builds infra envs only into local stores; put it "
        "with `agent-env env gateway put --id default`",
        "the service-db env 'default-db' isn't in the store, and agent-env builds infra envs only into local stores; "
        "put it with `agent-env env service-db put --id default-db`",
    ]


def test_a_gateway_deploy_on_another_provider_needs_infra_it_can_reach(bundle_dir):
    _env("crm", REMOTE)
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm"}])

    (problem, *_) = _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal_vm"))
    assert problem.startswith("tasks/t.json: step 'env': deploys on the 'modal_vm' sandbox provider, which needs the "
                              "gateway env 'default', and the store doesn't hold it")
    _infra()
    problems = _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal_vm"))
    assert problems[0] == ("tasks/t.json: step 'env': deploys the gateway env 'default''s image 'gateway-default' on the "
                           "'modal_vm' sandbox provider, which can't reach it: localhost:5000/local/img:v1 is in a "
                           "registry on this machine; run it with --sandbox local")
    assert len(problems) == 4  # the gateway's image, and the service-db's three


def test_a_problem_several_deploys_share_is_reported_once_naming_the_first(bundle_dir):
    _env("crm")
    with namespace_routing():
        GatewayEnv.put(id="default", docker_image_artifact=_image("gateway-default"))
        db = _image("db")  # one image for all three of the service-db's, so each deploy finds its problem three times
        ServiceDBEnv.put(id="default-db", db_docker_image_artifact=db, db_web_docker_image_artifact=db,
                         db_mcp_docker_image_artifact=db)
    for name in ("a", "b", "c"):
        _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm"}], name=name)

    unreachable = ("on the 'modal_vm' sandbox provider, which can't reach it: localhost:5000/local/img:v1 is in a "
                   "registry on this machine; run it with --sandbox local")
    assert _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal_vm")) == [
        f"tasks/a.json: step 'env' (and 2 more deploys): deploys env 'crm''s image 'crm-image' {unreachable}",
        f"tasks/a.json: step 'env' (and 2 more deploys): deploys the gateway env 'default''s image 'gateway-default' "
        f"{unreachable}",
        f"tasks/a.json: step 'env' (and 2 more deploys): deploys the service-db env 'default-db''s image 'db' "
        f"{unreachable}",
    ]


# Envs the bundle writes, checked from their planned env.toml

BUNDLE = "@local/~/triage"
SERVER = 'from agentenv_protocol import environment_card\n\n\n@environment_card(name="{}")\nclass S:\n    pass\n'


def _env_folder(root, name, toml="", kind="mcp_server"):
    files = ({f"envs/{name}/Dockerfile": "FROM scratch\n"} if kind == "mcp_server" else
             {f"envs/{name}/Dockerfile.backend": "FROM scratch\n", f"envs/{name}/Dockerfile.frontend": "FROM scratch\n"})
    layout(root, {**files, f"envs/{name}/server.py": SERVER.format(name),
                  f"envs/{name}/env.toml": f'type = "{kind}"\n{toml}'})


def test_an_image_the_bundle_builds_for_an_env_is_refused_on_another_provider(bundle_dir):
    _env_folder(bundle_dir, "crm")
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm"}])

    assert (f"tasks/t.json: step 'env': deploys env '{BUNDLE}/crm''s image '{BUNDLE}/crm__env_image' on the 'modal_vm' "
            "sandbox provider, which can't reach it: it's built on this machine from envs/crm/Dockerfile; run it with "
            "--sandbox local") in _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal_vm"))


def test_a_website_the_bundle_writes_is_refused_on_modal_and_a_multi_holding_one_too(bundle_dir):
    _env_folder(bundle_dir, "shop", kind="website")
    layout(bundle_dir, {"envs/suite/env.toml": 'type = "multi"\nwebsite_envs = ["shop"]\n'})
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "suite"}])

    assert (f"tasks/t.json: step 'env': deploys env '{BUNDLE}/suite', which has websites, on the 'modal' sandbox "
            "provider, whose gateway runs in containers and can't serve websites; run it on a VM provider, such as "
            "--sandbox local") in _problems(lambda: dry_run_bundle(bundle_dir, sandbox="modal"))


@pytest.mark.parametrize("folders, deployed, kinds", [
    ([("crm", "", "mcp_server")], "crm", ["service-db", "gateway"]),
    ([("shop", "", "website")], "shop", ["service-db", "gateway", "website-browser"]),
    ([("crm", "", "mcp_server"), ("shop", "", "website"),
      ("suite", 'mcp_server_envs = ["crm"]\nwebsite_envs = ["shop"]\n', "multi")], "suite",
     ["service-db", "gateway", "website-browser"]),
    ([("crm", 'env_provider_type = "server"\n', "mcp_server")], "crm", []),
], ids=["mcp-server", "website", "multi", "server-provider"])
def test_an_env_the_bundle_writes_names_the_infra_its_deploy_on_the_local_provider_builds(
    bundle_dir, folders, deployed, kinds,
):
    for name, toml, kind in folders:
        if kind == "multi":
            layout(bundle_dir, {f"envs/{name}/env.toml": f'type = "multi"\n{toml}'})
        else:
            _env_folder(bundle_dir, name, toml, kind)
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": deployed}])

    assert [build.kind for build in dry_run_bundle(bundle_dir).infra] == kinds


def test_a_store_image_an_env_names_without_a_version_is_checked_at_the_version_the_plan_read(
    bundle_dir, monkeypatch,
):
    _image("base", REMOTE)
    _infra(REMOTE)
    layout(bundle_dir, {"envs/crm/env.toml": 'image = "base"\nenvironment_name = "crm"\n'})
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm"}])
    planned = run_module.plan_bundle

    def a_local_image_lands_once_planned(*args, **kwargs):
        plan = planned(*args, **kwargs)
        _image("base", LOCAL)  # v2, after the plan read v1, which the env is written at
        return plan

    monkeypatch.setattr(run_module, "plan_bundle", a_local_image_lands_once_planned)

    assert dry_run_bundle(bundle_dir, sandbox="modal_vm").runs


def test_the_cli_dry_run_lists_the_infra_it_would_build(bundle_dir, quiet_logs):
    _env("crm")
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm"}])

    result = CliRunner().invoke(cli, ["run", str(bundle_dir), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert ("Would build first:\n  service-db env 'default-db' (missing)\n  gateway env 'default' (missing)\n"
            "Would run:\n  tasks/t.json v1\n") in result.output


def test_a_run_builds_its_infra_after_its_writes_and_before_its_tasks(bundle_dir, monkeypatch):
    _env("crm")
    _task(bundle_dir, [{"id": "env", "type": "deploy_env", "env_id": "crm"}])
    events = []
    monkeypatch.setattr("agent_env.bundle.run.ensure_default_envs",
                        lambda kinds, say: events.append(("bootstrap", sorted(kinds))))

    async def deploy(self, context):
        events.append(("deploy", self.env_id))
        return context

    monkeypatch.setattr(DeployEnvTaskStep, "execute", deploy)

    run_bundle(bundle_dir, on_progress=lambda line: events.append(("say", line)))

    assert events[:3] == [("say", "tasks/t.json: v1 (new)"), ("bootstrap", ["gateway", "service-db"]),
                          ("say", "[tasks/t.json] step 1/1 env (deploy_env)")]
    assert ("deploy", "crm") in events


# The walk resolves each deploy's provider as its step does


class _Stop(Exception):
    pass


_STEPS = {
    "deploy_env": {"id": "s", "type": "deploy_env", "env_id": "crm"},
    "deploy_agent": {"id": "s", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"},
    "deploy_sandbox": {"id": "s", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "container",
                       "image": "python:3.12", "port": 80},
    "rubrics_verifier": {"id": "s", "type": "rubrics_verifier", "prompt_id": "p", "verifier_id": "v",
                         "criteria": [{"id": "c", "description": "done"}], "judge_a2a_agent_id": "solver",
                         "use_trajectory": False},
}
_FIELDS = {"deploy_env": "sandbox_type", "deploy_agent": "sandbox_type", "deploy_sandbox": "sandbox_type",
           "rubrics_verifier": "judge_sandbox_type"}
# What a step falls back to with no provider named: deploy_env's env through deploy_through_provider, the agents through
# A2AAgent.deploy, deploy_sandbox itself.
_DEFAULT_GETTERS = {"deploy_env": "get_env_sandbox_provider", "deploy_agent": "get_agent_sandbox_provider",
                    "deploy_sandbox": "get_sandbox_provider", "rubrics_verifier": "get_agent_sandbox_provider"}
_OVERRIDE_KEYS = {"deploy_env": "env_sandbox", "deploy_agent": "agent_sandbox", "deploy_sandbox": "sandbox",
                  "rubrics_verifier": "agent_sandbox"}


def _walked_spec(bundle_dir, monkeypatch, config, sandbox):
    """The provider the walk resolves for the one deploy in ``config``: a spec, or "default"."""
    seen = []
    monkeypatch.setattr(preflight_module, "build_sandbox_provider", lambda spec: seen.append(spec) or LocalSandboxProvider())
    for getter in ("get_env_sandbox_provider", "get_agent_sandbox_provider", "get_sandbox_provider"):
        monkeypatch.setattr(preflight_module, getter, lambda getter=getter: seen.append(getter) or LocalSandboxProvider())
    _task(bundle_dir, [config])
    dry_run_bundle(bundle_dir, sandbox=sandbox)
    return seen[0]


def _executed_spec(monkeypatch, kind, config, sandbox):
    """The provider the step itself resolves when it runs, with the overrides ``run_bundle`` sets for ``sandbox``."""
    seen = []
    step = {"deploy_env": DeployEnvTaskStep, "deploy_agent": DeployAgentTaskStep, "deploy_sandbox": DeploySandboxTaskStep,
            "rubrics_verifier": RubricsVerifierTaskStep}[kind].from_dict({**config, "version": 1})

    class _Deployable:
        metadata = {}

        async def deploy(self, **kwargs):
            seen.append(kwargs.get("sandbox_type") or "default")
            raise _Stop

    monkeypatch.setattr("agent_env.env.env.Env.get", classmethod(lambda cls, *a, **k: _Deployable()))
    monkeypatch.setattr(A2AAgent, "get", classmethod(lambda cls, *a, **k: _Deployable()))
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider",
                        lambda spec: seen.append(spec) or (_ for _ in ()).throw(_Stop()))
    monkeypatch.setattr("agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider",
                        lambda: seen.append("default") or (_ for _ in ()).throw(_Stop()))
    overrides = {"env_sandbox": sandbox, "agent_sandbox": sandbox, "sandbox": sandbox} if sandbox else {}
    context = TaskStepContext(prompt_responses=[PromptResponse(prompt_id="p", response="done", prompt_text="do it")],
                              metadata={"user_overrides": {**overrides, "judge_litellm_api_key": "k",
                                                           "judge_litellm_base_url": "http://litellm"}})
    with pytest.raises(_Stop):
        asyncio.run(step.execute(context))
    return seen[0]


@pytest.mark.parametrize("kind", list(_STEPS))
@pytest.mark.parametrize("field, sandbox", [(None, None), ("modal_vm", None), ("modal_vm", "modal")],
                         ids=["default", "step", "override"])
def test_the_walk_resolves_each_deploys_provider_as_its_step_does(bundle_dir, monkeypatch, kind, field, sandbox):
    _env("crm", REMOTE)
    _agent("solver", REMOTE)
    config = {**_STEPS[kind], **({_FIELDS[kind]: field} if field else {})}

    walked = _walked_spec(bundle_dir, monkeypatch, config, sandbox)
    executed = _executed_spec(monkeypatch, kind, config, sandbox)

    assert executed == (sandbox or field or "default")
    assert walked == (sandbox or field or _DEFAULT_GETTERS[kind])
    assert _OVERRIDE_KEYS[kind]  # run_bundle sets every override key to --sandbox
