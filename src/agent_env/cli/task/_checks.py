"""What `agent-env task run` checks of a stored task before its first step: that the plugin packages it requires are
installed and every env and artifact type its steps read resolves. With local stores it then sets up the infra envs
the task deploys on, as `agent-env run` does."""

from __future__ import annotations

from collections.abc import Callable

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

# The entities a step reads whose type a plugin may register.
_GETTERS = {EntityKind.ENV: Env.get, EntityKind.ARTIFACT: Artifact.get}


def runnable_task(task_id: str, version: int | None, *, say: Callable[[str], None], gateway_env_id: str | None = None,
                  service_db_env_id: str | None = None) -> Task:
    """The stored task ``task_id`` at ``version``, or its latest, once what it requires is installed and every env and
    artifact type its steps read loads here. With local stores, the infra envs it deploys on are set up, but for one
    the run is given its own of (``gateway_env_id``, ``service_db_env_id``). Raises UnmetRequirements naming the
    command that installs what it requires, or a ClickException for a type nothing here registers."""
    try:
        task = Task.get(task_id, version=version)
    except UnknownStepTypeError as e:
        if e.requirements:
            raise UnmetRequirements([str(e)], e.requirements) from e
        raise
    owner = f"task {task.id} v{task.version}"
    problems, unmet = unmet_problems(task.requires or (), owner)
    problems += [f"{owner}: {problem}" for problem in _unknown_types(task)]
    if unmet:
        raise UnmetRequirements(problems, unmet)
    if problems:
        raise click.ClickException("\n".join(problems))
    overridden = {kind for kind, given in ((GATEWAY, gateway_env_id), (SERVICE_DB, service_db_env_id)) if given}
    if stores_are_local() and (kinds := stored_infra(task.steps) - overridden):
        ensure_default_envs(kinds, say=say)
    return task


def _unknown_types(task: Task) -> list[str]:
    """Each env and artifact a step of ``task`` reads, as the store holds it, whose type nothing here registers. One
    the store doesn't hold, or whose id the run fills in, is left to its step."""
    found = []
    for step in task.steps:
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
