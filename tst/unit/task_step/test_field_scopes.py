"""A step type's scoped fields are optional fields of its own, so a step written out without them still builds, and the
run fills them in."""

import inspect

import pytest

from agent_env.task_step import registry
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep

STEP_TYPES = sorted(get_task_step_registry().items())
# A stored step of each built-in type that scopes fields of its own, holding only what the type requires.
MINIMAL = {
    "add_skills": {"skills": [{"name": "notes", "description": "how notes are kept"}]},
    "deploy_agent": {},
    "deploy_env": {"env_id": "crm-env"},
    "deploy_human_agent": {},
    "deploy_sandbox": {"sandbox_name": "box", "sandbox_mode": "vm"},
    "install_agent": {"sandbox_name": "box"},
    "load_artifact": {"sandbox_name": "box"},
    "prompt_agent": {"prompt": "hi"},
    "rubrics_verifier": {"prompt_id": "prompt", "criteria": [{"id": "c1", "description": "says hi"}]},
    "run_docker_container": {"sandbox_name": "box"},
    "snapshot_env": {},
    "snapshot_computer_env": {},
}
# What a step without a user-scoped field that has no default says it needs, before a run starts.
UNSOURCED = {
    "add_skills": "add_skills 's': skill 'notes' names no source",
    "load_artifact": "load_artifact 's': names nothing to load",
    "run_docker_container": "run_docker_container 's': names no build context",
}


@pytest.mark.parametrize("cls", [cls for _, cls in STEP_TYPES], ids=[name for name, _ in STEP_TYPES])
def test_each_scoped_field_is_an_optional_parameter_of_its_step_type_or_one_the_base_scopes_for_every_step(cls):
    parameters = inspect.signature(cls.__init__).parameters
    for path in cls.field_scopes:
        top = path.split(".")[0].removesuffix("[]")
        if top in parameters:
            assert parameters[top].default is not inspect.Parameter.empty, f"{cls.type} scopes {top}, which it requires"
        else:
            assert path in TaskStep.field_scopes, f"{cls.type} scopes {path}, which isn't one of its fields"


@pytest.mark.parametrize("cls", [cls for _, cls in STEP_TYPES], ids=[name for name, _ in STEP_TYPES])
def test_a_step_type_keeps_the_scopes_the_base_gives_every_step(cls):
    assert TaskStep.field_scopes.items() <= cls.field_scopes.items()


def test_each_built_in_step_type_that_scopes_fields_of_its_own_has_a_minimal_step_here():
    assert {name for name, cls in registry._builtin_registry().items()
            if cls.field_scopes != TaskStep.field_scopes} == set(MINIMAL)


@pytest.mark.parametrize("name", sorted(MINIMAL))
def test_a_step_with_every_scoped_field_left_out_still_builds(name):
    step = registry._builtin_registry()[name].from_dict({"id": "s", "type": name, **MINIMAL[name]})

    assert step.type == name


@pytest.mark.parametrize("name", sorted(UNSOURCED))
def test_a_step_left_without_a_scoped_field_that_has_no_default_names_it_before_a_run(name):
    step = registry._builtin_registry()[name].from_dict({"id": "s", "type": name, **MINIMAL[name]})

    [problem] = step.config_problems()
    assert problem.startswith(UNSOURCED[name]) and "user scope, so an import leaves it out" in problem
