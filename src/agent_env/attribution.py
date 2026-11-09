"""Cost-attribution dimensions carried from the caller down to a sandbox provider.

Attribution is an open ``dict[str, str]``. Core threads it without reading it, and providers
carry every key (Modal as sandbox tags, E2B as sandbox metadata), so a deployment can attribute
on whatever dimensions and terminology it uses.
"""

from __future__ import annotations

Attribution = dict[str, str]

# The one key in a step's open ``metadata`` map that is forwarded to compute as attribution.
ATTRIBUTION_KEY = "attribution"


# Names the pipeline step that deployed a sandbox, as ``<task_id>_<step_id>``, and the run
# (task instance) it belongs to, so sandbox cost can be broken down per run and per step.
# Modal stamps both on each sandbox as tags.
PIPELINE_STEP_KEY = "pipeline_step"
RUN_ID_KEY = "run_id"


def attribution_of(step) -> Attribution:
    """A step's attribution: a copy of ``metadata["attribution"]`` (``{}`` when unset)."""
    return dict((getattr(step, "metadata", None) or {}).get(ATTRIBUTION_KEY) or {})


def deploy_attribution(step, context) -> Attribution:
    """``attribution_of(step)`` plus ``pipeline_step`` and ``run_id`` when known.

    ``pipeline_step`` needs ``context.metadata["task_id"]`` (set by the hub worker and the CLI);
    ``run_id`` is ``context.instance_id``. Task-authored values win.
    """
    attribution = attribution_of(step)
    task_id = (getattr(context, "metadata", None) or {}).get("task_id")
    if task_id and PIPELINE_STEP_KEY not in attribution:
        attribution[PIPELINE_STEP_KEY] = f"{task_id}_{step.id}"
    instance_id = getattr(context, "instance_id", None)
    if instance_id and RUN_ID_KEY not in attribution:
        attribution[RUN_ID_KEY] = instance_id
    return attribution
