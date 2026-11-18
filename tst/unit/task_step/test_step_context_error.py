"""A step that finds the run's context can't satisfy it raises StepContextError, which a runner does not
retry: running the step again on the same context fails the same way."""

import ast
import re
from pathlib import Path

import pytest

import agent_env.task_step.task_steps as task_steps

_CONTEXT_FAILURE = re.compile(
    r"not found in context\.deployed_(envs|agents|sandboxes)|' is already deployed"
    r"|^PromptResponse with prompt_id=.* not found in context|^No artifact recorded by step"
)


def _message_text(node: ast.expr) -> str:
    """The literal text of a message, with each formatted value as ``{}``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_message_text(part) if isinstance(part, ast.Constant) else "{}" for part in node.values)
    return ""


def _plain_context_raises(source: str) -> list[int]:
    """The lines of ``source`` raising a context failure as a plain RuntimeError."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "RuntimeError"
        and node.exc.args
        and _CONTEXT_FAILURE.search(_message_text(node.exc.args[0]))
    ]


@pytest.mark.parametrize(
    "source",
    [
        """raise RuntimeError(f"Env '{env_id}' not found in context.deployed_envs")""",
        """raise RuntimeError(f'Agent {name!r} not found in context.deployed_agents')""",
        """raise RuntimeError(\n    f"Sandbox '{name}' not found "\n    "in context.deployed_sandboxes"\n)""",
        """raise RuntimeError("No artifact recorded by step 'collect'")""",
    ],
    ids=["double-quoted", "single-quoted", "split", "plain-string"],
)
def test_the_guard_sees_a_context_failure_however_it_is_written(source):
    assert _plain_context_raises(source) == [1]


def test_no_step_raises_a_context_failure_as_a_plain_runtime_error():
    root = Path(task_steps.__file__).parent
    found = [
        f"{path.relative_to(root)}:{line}"
        for path in sorted(root.rglob("*.py"))
        for line in _plain_context_raises(path.read_text())
    ]
    assert found == [], "raise StepContextError (agent_env.task_step.context) for these instead"
