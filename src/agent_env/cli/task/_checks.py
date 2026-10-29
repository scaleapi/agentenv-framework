"""What `agent-env task run` checks of a stored task before its first step: that the plugin packages it requires are
installed and every env and artifact type its steps read resolves. With local stores it then sets up the infra envs
the task deploys on, as `agent-env run` does."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import click

from agent_env.artifact.artifact import Artifact
from agent_env.bundle.preflight import stored_infra
from agent_env.bundle.requires import UnmetRequirements
from agent_env.entity.refs import EntityKind, RefRole, ref_sites
from agent_env.env.bootstrap import GATEWAY, SERVICE_DB, ensure_default_envs, stores_are_local
from agent_env.env.env import Env
from agent_env.plugins._requirements import unmet_problems
from agent_env.store.base import NotFoundError
from agent_env.store.document_store import UnknownDocumentTypeError
from agent_env.task import Task
from agent_env.task.task import UnknownStepTypeError
from agent_env.task_step.task_step import TaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep

# The entities a step reads whose type a plugin may register.
_GETTERS = {EntityKind.ENV: Env.get, EntityKind.ARTIFACT: Artifact.get}


def runnable_task(task_id: str, version: int | None, metadata: Mapping[str, Any], *,
                  say: Callable[[str], None]) -> Task:
    """The stored task ``task_id`` at ``version``, or its latest, once what it requires is installed and every env and
    artifact type the steps a run with ``metadata`` runs read loads here. With local stores, the infra envs those steps
    deploy on are set up, but for one the run is given its own of (``gateway_env_id``, ``service_db_env_id``). Raises
    UnmetRequirements naming the command that installs what it requires, or a ClickException for a type nothing here
    registers."""
    try:
        task = Task.get(task_id, version=version)
    except UnknownStepTypeError as e:
        if e.requirements:
            raise UnmetRequirements([str(e)], e.requirements) from e
        raise
    owner = f"task {task.id} v{task.version}"
    overrides = metadata.get("user_overrides") or {}
    steps = task.steps[metadata.get("start_step") or 0:]  # the steps the run runs
    params = overrides.get("step_params") or {}
    problems, unmet = unmet_problems(task.requires or (), owner)
    # A step the run gives step_params reads them its own way, in place of what it names.
    problems += [f"{owner}: {problem}" for problem in _unknown_types([step for step in steps if step.id not in params])]
    if unmet:
        raise UnmetRequirements(problems, unmet)
    if problems:
        raise click.ClickException("\n".join(problems))
    given = {kind for kind, key in ((GATEWAY, "gateway_env_id"), (SERVICE_DB, "service_db_env_id")) if overrides.get(key)}
    if stores_are_local() and (kinds := stored_infra(_with_state(steps, overrides)) - given):
        ensure_default_envs(kinds, say=say)
    return task


def _with_state(steps: list[TaskStep], overrides: Mapping[str, Any]) -> list[TaskStep]:
    """``steps``, each deploy_env step with the run's env state overrides, which it reads over its own."""
    state = {key: overrides[key] for key in ("env_state_type", "env_state_instance_id") if overrides.get(key)}
    return [type(step).from_dict({**step.to_dict(), **state}) if state and isinstance(step, DeployEnvTaskStep)
            else step for step in steps]


def _unknown_types(steps: list[TaskStep]) -> list[str]:
    """Each env and artifact one of ``steps`` reads, as the store holds it, whose type nothing here registers. One the
    store doesn't hold, or whose id the run fills in, is left to its step."""
    found = []
    for step in steps:
        for site in ref_sites(type(step).entity_refs or (), step.to_dict()):
            get = _GETTERS.get(site.ref.kind)
            if site.ref.role is RefRole.OUTPUT or get is None or not isinstance(site.value, str):
                continue
            try:
                get(site.value, site.version if isinstance(site.version, int) else None)
            except UnknownDocumentTypeError as e:
                found.append(f"step {step.id!r}: {site.path} {site.value!r}: {e}")
            except (NotFoundError, ValueError):
                continue
    return found
