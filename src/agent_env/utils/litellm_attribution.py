"""LiteLLM cost-attribution helpers.

Every LiteLLM request agent-env sends carries two attribution fields so the
proxy's spend logs can be rolled up by who ran the work:

- `user` = the `project_id`. Spend reporting joins on it to attribute a
  request to its owning project (and, downstream, to the customer); callers
  never pass a customer identifier directly.
- `metadata.tags` = secondary attribution: `projectId:<id>` (duplicates
  `user`, so tag-based reports agree with user-based ones) and `taskId:<id>`
  (the agent-env Task), so spend rolls up per project and per task.

Both fields are sourced from `TaskStepContext.metadata` so the run
group's project/task flows into every in-process LLM call (judge LLMs, etc.)
without each call site having to plumb them by hand.

Use this helper at every `litellm.completion` / `litellm.acompletion`
call site so attribution is consistent across the codebase.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_env.config.model import ModelParam

logger = logging.getLogger(__name__)


def build_litellm_cost_attribution_kwargs(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Build LiteLLM cost-attribution kwargs (`user`, `metadata`).

    Reads `project_id` and `task_id` from `metadata` (typically
    `TaskStepContext.metadata`). Missing `project_id` warns but doesn't
    block — the request is sent regardless. `task_id` is best-effort.

    Returns a kwargs dict to spread into `litellm.completion(**kwargs)` /
    `litellm.acompletion(**kwargs)`.
    """
    metadata = metadata or {}
    kwargs: dict[str, Any] = {}
    project_id = metadata.get("project_id")
    task_id = metadata.get("task_id")
    if project_id:
        kwargs[ModelParam.USER] = project_id
        tags = [f"projectId:{project_id}"]
        if task_id:
            tags.append(f"taskId:{task_id}")
        kwargs[ModelParam.METADATA] = {"tags": tags}
    else:
        logger.warning(
            "context.metadata.project_id is not set; LiteLLM cost "
            "attribution will not be tagged with a project. Request "
            "is still being sent. Fill the Project ID field in the run UI "
            "(or pass project_id at the API call site) so spend is "
            "attributed correctly."
        )
    return kwargs
