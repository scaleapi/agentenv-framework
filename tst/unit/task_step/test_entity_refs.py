"""Built-in steps declare their entity refs, and ``ref_sites`` walks what they declare."""

import inspect

import pytest

from agent_env.artifact.registry import get_artifact_registry
from agent_env.entity_refs import EntityKind, EntityRef, ref_sites
from agent_env.task_step.registry import _builtin_registry
from agent_env.task_step.snapshot_utils.snapshot_series import SnapshotConfig
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.add_skills import AddSkillsTaskStep
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.register_agent_triggers import (
    RegisterAgentTriggersStep,
    _env_trigger_conditions,
    _referenced_env_triggers,
)

BUILTINS = sorted(_builtin_registry().items())


def _params(cls: type[TaskStep]) -> dict[str, inspect.Parameter]:
    params: dict[str, inspect.Parameter] = {}
    for klass in cls.__mro__:
        if "__init__" in vars(klass) and klass is not object:
            for name, p in inspect.signature(vars(klass)["__init__"]).parameters.items():
                params.setdefault(name, p)
    params.pop("self")
    return params


@pytest.mark.parametrize("step_type,cls", BUILTINS)
def test_builtin_step_declarations_name_real_fields(step_type, cls):
    assert cls.entity_refs is not None, f"{step_type} declares no entity_refs"
    params = _params(cls)
    refs = cls.entity_refs
    paths = [r.path for r in refs]
    assert len(paths) == len(set(paths))
    for ref in refs:
        assert ref.field in params, f"{step_type}: {ref.path} names no constructor parameter"
        if ref.version_field and "." not in ref.path:
            assert ref.version_field in params, f"{step_type}: {ref.version_field} is not a parameter"
        if ref.artifact_type is not None:
            assert ref.artifact_type in get_artifact_registry()



@pytest.mark.parametrize("step_type,cls", BUILTINS)
def test_builtin_refs_pin_the_version_parameter_beside_them(step_type, cls):
    params = _params(cls)
    for ref in cls.entity_refs or ():
        sibling = ref.path.removesuffix("id") + "version"
        if ref.path.endswith("_id") and "." not in ref.path and sibling in params:
            assert ref.version_field == sibling, f"{step_type}: {ref.path} should declare version_field={sibling!r}"

def test_sites_carry_scalar_list_and_element_relative_versions():
    refs = (
        EntityRef.env("env_id", version_field="env_version"),
        EntityRef.env("env_ids[]"),
        EntityRef.artifact("artifacts[].id", version_field="version"),
    )
    data = {"env_id": "e", "env_version": 3, "env_ids": ["a", None, "b"], "artifacts": [{"id": "x"}, {"id": "y", "version": 2}]}
    assert [(s.path, s.value, s.version) for s in ref_sites(refs, data)] == [
        ("env_id", "e", 3), ("env_ids[0]", "a", None), ("env_ids[2]", "b", None),
        ("artifacts[0].id", "x", None), ("artifacts[1].id", "y", 2),
    ]


def test_absent_and_null_fields_yield_nothing():
    refs = (EntityRef.env("env_id"), EntityRef.artifact("skills[].skill_artifact_id"))
    assert list(ref_sites(refs, {"env_id": None, "skills": None})) == []
    assert list(ref_sites(refs, {})) == []


def test_a_site_rewrites_its_owner_in_place():
    data = {"artifacts": [{"id": "x", "version": None}]}
    (site,) = ref_sites(LoadArtifactTaskStep.entity_refs, data)
    site.rewrite("@local/x", version=4)
    assert data == {"artifacts": [{"id": "@local/x", "version": 4}]}


def test_a_list_element_rewrites_but_cannot_pin_a_version():
    data = {"env_ids": ["crm"]}
    (site,) = ref_sites((EntityRef.env("env_ids[]"),), data)
    with pytest.raises(ValueError):
        site.rewrite("@local/crm", version=2)
    assert data == {"env_ids": ["crm"]}
    site.rewrite("@local/crm")
    assert data == {"env_ids": ["@local/crm"]}


def test_trigger_walker_finds_nested_env_triggers_only():
    when = {"type": "all", "of": [
        {"type": "env_trigger", "env_id": "crm", "trigger_id": "t1"},
        {"type": "any", "of": [{"type": "message", "env_id": "not-a-ref"}, {"type": "env_trigger", "env_id": "mail", "trigger_id": "t2"}]},
    ]}
    data = {"triggers": [{"id": "a", "when": when}]}
    sites = list(ref_sites(RegisterAgentTriggersStep.entity_refs, data))
    assert [(s.path, s.value) for s in sites] == [
        ("triggers[0].when.of[0].env_id", "crm"), ("triggers[0].when.of[1].of[1].env_id", "mail"),
    ]


def test_declared_paths_match_what_to_dict_writes():
    steps = [
        DeployEnvTaskStep(id="d", version=None, env_id="crm", env_version=2),
        LoadArtifactTaskStep(id="l", version=None, env_id="crm", artifact_id="seed", artifact_version=1),
        AddSkillsTaskStep(
            id="s", version=None, skills=[{"skill_artifact_id": "sk", "skill_artifact_version": 5}],
            cli_artifact_ids=["cli-crm"], file_artifact_universe_ids=["fau-crm"],
        ),
        DeployAgentTaskStep(id="a", version=None, env_ids=["crm", "mail"], a2a_agent_id="triage", a2a_agent_version=3),
        PromptAgentTaskStep(id="p", version=None, prompt="hi", snapshot_config=SnapshotConfig(env_id="tickets")),
        RegisterAgentTriggersStep(
            id="t", version=None, agent_name="triage",
            triggers=[{"id": "a", "when": {"type": "any", "of": [{"type": "env_trigger", "env_id": "crm", "trigger_id": "t1"}]}}],
        ),
    ]
    found = {type(s).type: [(x.path, x.value, x.version) for x in ref_sites(type(s).entity_refs, s.to_dict())] for s in steps}
    assert found == {
        "deploy_env": [("env_id", "crm", 2)],
        "load_artifact": [("env_id", "crm", None), ("artifacts[0].id", "seed", 1)],
        "add_skills": [
            ("skills[0].skill_artifact_id", "sk", 5), ("cli_artifact_ids[0]", "cli-crm", None),
            ("file_artifact_universe_ids[0]", "fau-crm", None),
        ],
        "deploy_agent": [("env_ids[0]", "crm", None), ("env_ids[1]", "mail", None), ("a2a_agent_id", "triage", 3)],
        "prompt_agent": [("snapshot_config.env_id", "tickets", None)],
        "register_agent_triggers": [("triggers[0].when.of[0].env_id", "crm", None)],
    }


def test_a_subclass_is_undeclared_until_it_declares():
    class Variant(DeployEnvTaskStep):
        type = "deploy_env_variant"

    class Declared(DeployEnvTaskStep):
        type = "deploy_env_declared"
        entity_refs = (EntityRef.env("env_id"),)

    assert Variant.entity_refs is None
    assert Declared.entity_refs == (EntityRef.env("env_id"),)
    assert DeployEnvTaskStep.entity_refs[0].version_field == "env_version"


@pytest.mark.parametrize("bad", [
    lambda: EntityRef("env id", EntityKind.ENV),
    lambda: EntityRef("env_id", "env"),
    lambda: EntityRef("env_id", EntityKind.ENV, role="output"),
    lambda: EntityRef("env_ids[]", EntityKind.ENV, version_field="env_version"),
    lambda: EntityRef("env_id", EntityKind.ENV, artifact_type="file"),
])
def test_malformed_declarations_fail_where_they_are_written(bad):
    with pytest.raises(ValueError):
        bad()


def test_each_kind_has_a_constructor_and_only_artifacts_take_a_type():
    assert [EntityRef.env("e").kind, EntityRef.agent("a").kind, EntityRef.artifact("f").kind] == list(EntityKind)
    assert EntityRef.artifact("f", artifact_type="file").artifact_type == "file"
    with pytest.raises(ValueError):
        EntityRef.env("e", artifact_type="file")
    with pytest.raises(TypeError):
        EntityRef.agent("a", version="a_version")


def test_trigger_walker_sees_the_conditions_the_runtime_check_counts():
    when = {
        "type": "all",
        "of": [
            {"type": "env_trigger", "env_id": "tickets", "trigger_id": "t1"},
            {"type": "any", "of": [{"type": "env_trigger", "env_id": "crm", "trigger_id": "t2"}, {"type": "a2a_message"}]},
        ],
    }
    walked = {(c["env_id"], c["trigger_id"]) for _, c in _env_trigger_conditions(when)}
    assert walked == _referenced_env_triggers(when) == {("tickets", "t1"), ("crm", "t2")}



def test_a_walker_on_a_top_level_path_yields_a_concrete_path():
    ref = EntityRef.env("env_id", walk=_env_trigger_conditions)
    data = {"type": "all", "of": [{"type": "env_trigger", "env_id": "tickets", "trigger_id": "t"}]}
    assert [s.path for s in ref_sites((ref,), data)] == ["of[0].env_id"]
