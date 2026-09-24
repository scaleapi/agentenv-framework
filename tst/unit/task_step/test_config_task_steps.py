"""Unit tests for config.toml-declared custom task steps.

A custom `TaskStep` named under `[task_steps]` in `.agentenv/config.toml` is
imported, ABC-guarded, and registered under its own `type` when the registry is
built — with no change to the built-in step path. Backend-agnostic; no network.
"""

import asyncio
import textwrap

import pytest

from agent_env.config import ConfigError
from agent_env.config import reset_config
from agent_env.task_step import registry
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep


class _GreetStep(TaskStep):
    """A minimal infra-free custom step: records a greeting in the context."""

    type = "greet_test_step"

    def __init__(self, id, version, message="hi", depends_on=None, fail_task_on_error=True):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.message = message

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["message"] = self.message
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "_GreetStep":
        return cls(**cls._base_from_dict(data), message=data.get("message", "hi"))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        context.metadata.setdefault("greetings", []).append(self.message)
        return context


class _CollidingStep(TaskStep):
    """Declares a `type` that a built-in already owns."""

    type = "prompt_agent"

    @classmethod
    def from_dict(cls, data: dict) -> "_CollidingStep":
        return cls(**cls._base_from_dict(data))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        return context


class _GreetStepDup(TaskStep):
    """A second class claiming the same `type` as _GreetStep."""

    type = "greet_test_step"

    @classmethod
    def from_dict(cls, data: dict) -> "_GreetStepDup":
        return cls(**cls._base_from_dict(data))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        return context


class _NoTypeStep(TaskStep):
    """Forgets to override `type`, so it inherits the base default."""

    @classmethod
    def from_dict(cls, data: dict) -> "_NoTypeStep":
        return cls(**cls._base_from_dict(data))

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        return context


class _NotAStep:
    pass


def _write_config(tmp_path, body: str):
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(textwrap.dedent(body))
    return cfg


_HERE = "tst.unit.task_step.test_config_task_steps"


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_config()
    yield
    reset_config()


def test_absent_config_leaves_builtins_only(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    reg = registry.get_task_step_registry()
    assert "greet_test_step" not in reg
    assert "prompt_agent" in reg


def test_no_task_steps_section_leaves_builtins_only(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, '[stores]\ndocument = "local"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reg = registry.get_task_step_registry()
    assert "greet_test_step" not in reg
    assert "prompt_agent" in reg


def test_config_toml_step_is_registered(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [task_steps]
        impls = ["{_HERE}:_GreetStep"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reg = registry.get_task_step_registry()
    assert reg["greet_test_step"] is _GreetStep
    assert "prompt_agent" in reg


def test_collision_with_builtin_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [task_steps]
        impls = ["{_HERE}:_CollidingStep"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="already registered"):
        registry.get_task_step_registry()


def test_two_custom_steps_same_type_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [task_steps]
        impls = ["{_HERE}:_GreetStep", "{_HERE}:_GreetStepDup"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="already registered"):
        registry.get_task_step_registry()


def test_step_without_own_type_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [task_steps]
        impls = ["{_HERE}:_NoTypeStep"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="does not define its own 'type'"):
        registry.get_task_step_registry()


def test_unimportable_impl_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [task_steps]
        impls = ["no.such.module:Thing"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="Cannot import"):
        registry.get_task_step_registry()


def test_non_taskstep_impl_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [task_steps]
        impls = ["{_HERE}:_NotAStep"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="not a subclass"):
        registry.get_task_step_registry()


def test_non_list_impls_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f'[task_steps]\nimpls = "{_HERE}:_GreetStep"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="must be a list"):
        registry.get_task_step_registry()


def test_non_string_impl_element_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, "[task_steps]\nimpls = [123]\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="must be a 'module:Class' string"):
        registry.get_task_step_registry()


def test_failed_merge_does_not_memoize_partial_registry(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [task_steps]
        impls = ["no.such.module:Thing"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError):
        registry.get_task_step_registry()
    # A bad manifest must fail loud on EVERY call, not just the first.
    with pytest.raises(ConfigError):
        registry.get_task_step_registry()


def test_custom_step_round_trips_through_task(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [task_steps]
        impls = ["{_HERE}:_GreetStep"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    from agent_env.task.task import Task

    doc = {
        "id": "t1",
        "version": 1,
        "steps": [{"id": "s1", "type": "greet_test_step", "message": "hola"}],
    }
    task = Task.from_dict(doc)
    assert isinstance(task.steps[0], _GreetStep)

    round_tripped = task.to_dict()["steps"][0]
    assert round_tripped["type"] == "greet_test_step"
    assert round_tripped["message"] == "hola"

    ctx = asyncio.run(task.steps[0].execute(TaskStepContext()))
    assert ctx.metadata["greetings"] == ["hola"]
