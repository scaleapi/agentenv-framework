"""A step that finds the run's context can't satisfy it raises StepContextError, which a runner does not
retry: running the step again on the same context fails the same way."""

import re
from pathlib import Path

import agent_env.task_step.task_steps as task_steps

_PLAIN_CONTEXT_RAISES = re.compile(
    r"""raise RuntimeError\(\s*f"([^"\n]*not found in context\.deployed_|[^"\n]*' is already deployed|"""
    r"""PromptResponse with prompt_id=[^"\n]*not found in context|No artifact recorded by step)"""
)


def test_no_step_raises_a_context_failure_as_a_plain_runtime_error():
    root = Path(task_steps.__file__).parent
    found = [
        f"{path.relative_to(root)}:{source[: match.start()].count(chr(10)) + 1}"
        for path in sorted(root.rglob("*.py"))
        for source in [path.read_text()]
        for match in _PLAIN_CONTEXT_RAISES.finditer(source)
    ]
    assert found == [], "raise StepContextError (agent_env.task_step.context) for these instead"
