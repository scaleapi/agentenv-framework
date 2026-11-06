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

# Where a sandbox's cpu and memory came from (``step``, ``sizer`` or ``default``; see
# ``providers.sandbox_providers.sizing``). They describe the sandbox rather than name what it is for, so a provider
# that builds names, labels or tags from attribution leaves them out (``identifying``).
CPU_SOURCE_KEY = "cpu_source"
MEMORY_SOURCE_KEY = "memory_source"
SIZE_SOURCE_KEYS = frozenset({CPU_SOURCE_KEY, MEMORY_SOURCE_KEY})


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


def identifying(attribution) -> Attribution:
    """``attribution`` without the keys that describe a sandbox's size rather than what it is for: what a provider
    builds a sandbox's name, labels or tags from, whose length and number are limited."""
    return {key: value for key, value in (attribution or {}).items() if key not in SIZE_SOURCE_KEYS}
