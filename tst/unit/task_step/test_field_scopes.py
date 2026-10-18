"""A step type's scoped fields are optional fields of its own, so a step written out without them still builds, and the
run fills them in."""

import inspect

import pytest

from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep

STEP_TYPES = sorted(get_task_step_registry().items())


@pytest.mark.parametrize("cls", [cls for _, cls in STEP_TYPES], ids=[name for name, _ in STEP_TYPES])
def test_each_scoped_field_is_an_optional_parameter_of_its_step_type_or_one_the_base_scopes_for_every_step(cls):
    parameters = inspect.signature(cls.__init__).parameters
    for path in cls.field_scopes:
        top = path.split(".")[0]
        if top in parameters:
            assert parameters[top].default is not inspect.Parameter.empty, f"{cls.type} scopes {top}, which it requires"
        else:
            assert path in TaskStep.field_scopes, f"{cls.type} scopes {path}, which isn't one of its fields"


@pytest.mark.parametrize("cls", [cls for _, cls in STEP_TYPES], ids=[name for name, _ in STEP_TYPES])
def test_a_step_type_keeps_the_scopes_the_base_gives_every_step(cls):
    assert TaskStep.field_scopes.items() <= cls.field_scopes.items()
