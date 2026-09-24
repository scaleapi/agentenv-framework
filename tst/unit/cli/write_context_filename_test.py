"""A run's context file is named from the task id, so an ``@local`` id must still give one file."""

import re

import pytest

from agent_env.cli.eval.run import _write_context as write_eval_context
from agent_env.cli.task.run import _write_context as write_task_context
from agent_env.task_step.context import TaskStepContext


@pytest.mark.parametrize("write_context", [write_task_context, write_eval_context])
@pytest.mark.parametrize("task_id, stem", [
    ("@local/~/work/triage/close-dupes", "local-work-triage-close-dupes-[0-9a-f]{12}"),
    ("legacy-task", "legacy-task"),
])
def test_the_context_file_lands_directly_in_the_output_dir(tmp_path, write_context, task_id, stem):
    write_context(TaskStepContext(), task_id, str(tmp_path))
    written = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()]
    assert len(written) == 1 and re.fullmatch(rf"{stem}_[0-9a-f]{{8}}\.json", written[0])
