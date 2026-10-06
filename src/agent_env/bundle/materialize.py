"""Write a planned bundle's entities, tasks and evals, reusing each version whose inputs haven't changed.

Materializing first refuses every write this release has no writer for, so nothing is written for a bundle
that can't be written whole. It needs the CLI's namespace routing, which sends ``@local`` writes to the
``@local`` namespace's store. Holding the bundle's lock, it writes the entities in the plan's order, an image
built from an entry's Dockerfile just before the entry, with ``docker build`` on this machine; then it
builds and preflights every task before writing any of them, and writes the evals last, since they name the
tasks. Each write goes through the ledger, so one whose inputs haven't changed reuses the version the bundle
last wrote. A reused task keeps whatever its steps took from config when it was first written, such as a
rubrics verifier's default judge model. A dry run takes the same path, refusals, checks and preflights
included, without the lock or a single write.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.registry import canonical_type, get_artifact_registry
from agent_env.entity_refs import EntityRef, RefRole, ref_sites
from agent_env.env.registry import get_env_registry
from agent_env.eval import Eval, EvalTask
from agent_env.store.ids import image_repository
from agent_env.store.routing import namespace_routing_enabled
from agent_env.task import Task
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_step import TaskStep
from agent_env.utils.docker_build import DockerBuildError, build_image

from ._fs import relative, with_article
from .authoring import AuthoringContext, build_context_files
from .ledger import Ledger, materializing
from .parse import BundleError, BundleKind
from .plan import Plan, Write, env_writer, folder_walk, unpinned_store_refs
from .resolve import BuiltImage, build_step

@dataclass(frozen=True)
class Materialized:
    """One write: the version it left in the store, or in a dry run would leave, and whether that is the one the
    bundle last wrote."""

    write: Write
    version: int
    reused: bool
    reasons: tuple[str, ...]  # why a new version was written; empty when reused


@dataclass(frozen=True)
class Materialization:
    """What materializing a plan left in the store, or in a dry run would leave, one ``Materialized`` per write."""

    plan: Plan
    writes: tuple[Materialized, ...]  # in the plan's order
    # A dry run's steps with a preflight that read what it would write first, so the store can't check them yet.
    not_preflighted: tuple[tuple[Write, TaskStep], ...] = ()

    def version_of(self, store: str, id: str) -> int:
        for done in self.writes:
            if (done.write.kind.store, done.write.id) == (store, id):
                return done.version
        raise KeyError(f"{store} {id!r} isn't one of this plan's writes")


def materialize(
    plan: Plan,
    *,
    dry_run: bool = False,
    on_wait: Callable[[], None] | None = None,
    on_build: Callable[[Write], None] | None = None,
    on_write: Callable[[Materialized], None] | None = None,
) -> Materialization:
    """Write ``plan``'s entities, tasks and evals. The first write that fails stops it, and every earlier
    one stays: they're in the ledger, so the next run reuses them. ``on_wait`` is called when another run
    holds a lock this one needs, ``on_build`` before an image is built, and ``on_write`` after each write,
    reused or not.

    ``dry_run`` checks and preflights what a run would, and writes nothing: no entity, ledger row or lock.
    Each write gets the version it would reuse or the store's next, which another run can take first. A step
    reading what the run would write first isn't preflighted, since the store doesn't hold it yet."""
    _refuse_unwritable(plan)
    if not namespace_routing_enabled():
        raise RuntimeError("materializing a bundle needs namespace routing, which the agent-env CLI turns on; "
                           "call it inside agent_env.store.routing.namespace_routing()")
    ledger = Ledger.for_plan(plan)
    entities = [write for write in plan.writes if write.kind not in (BundleKind.TASK, BundleKind.EVAL)]
    tasks = [write for write in plan.writes if write.kind is BundleKind.TASK]
    evals = [write for write in plan.writes if write.kind is BundleKind.EVAL]
    done: dict[tuple[str, str], Materialized] = {}

    def through_ledger(write: Write, write_fn: Callable[[], int]) -> None:
        with _noted(plan, write, "checking" if dry_run else "writing"):
            check = ledger.check(write, {need: done[need].version for need in write.needs})
            if check.unchanged:
                version = check.version
                if check.adopted and not dry_run:
                    ledger.adopt(check)
            elif dry_run:
                version = check.next_version
            else:
                version = ledger.record(check, write_fn)
        done[_key(write)] = Materialized(write, version, check.unchanged, check.reasons)
        if on_write is not None:
            on_write(done[_key(write)])

    with nullcontext() if dry_run else materializing(plan.bundle.bundle, on_wait):
        _refuse_builds_without_docker(plan, ledger)  # once another run writing these ids is done
        for write in entities:
            through_ledger(write, lambda: _write_entity(plan, write, on_build))
        unwritten = {_key(item.write) for item in done.values() if not item.reused} if dry_run else set()
        built, problems, unchecked = {}, [], []
        for write in tasks:
            with _noted(plan, write, "preflighting"):
                built[write.id] = _build(write)
                found, skipped = _preflight(write, built[write.id], unwritten)
            problems.extend(f"{_path(plan, write)}: {problem}" for problem in found)
            unchecked.extend((write, step) for step in skipped)
        if problems:
            raise BundleError(problems)
        for write in tasks:
            through_ledger(write, lambda: Task.put(id=write.id, steps=built[write.id].steps).version)
        for write in evals:
            through_ledger(write, lambda: _WRITERS[write.kind](plan, write))
    return Materialization(plan, tuple(done[_key(write)] for write in plan.writes), tuple(unchecked))


def _write_entity(plan: Plan, write: Write, on_build: Callable[[Write], None] | None) -> int:
    if not isinstance(write.source, BuiltImage):
        return _WRITERS[write.kind](plan, write)
    if on_build is not None:
        on_build(write)
    return _write_built_image(plan, write)


def _write_built_image(plan: Plan, write: Write) -> int:
    """Build the image an entry's Dockerfile describes and write it as a docker_image artifact: pushed to the
    image store, saved as a tarball, and its build context kept for installing it into a running container.
    The build context is a copy of the files the ledger hashes, ``build_context_files``, so an image the
    ledger reuses was built from what it hashed."""
    image = write.source
    tag = f"{image_repository(write.id)}:bundle"
    with tempfile.TemporaryDirectory(prefix="agent-env-build-") as staged:
        context = Path(staged)
        for key, path in build_context_files(plan.bundle.bundle, image.entry).items():
            (context / key).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, context / key)
        try:
            build_image(context / image.dockerfile, context, tag, platform=None)
        except DockerBuildError as e:
            raise BundleError([f"{_path(plan, write)}: {_tail(str(e))}"]) from None
        try:
            return DockerImageArtifact.put(
                id=write.id, description=f"built from {_path(plan, write)}/{image.dockerfile}", image_name=tag,
                build_context_path=str(context),
            ).version
        except RuntimeError as e:  # the image store, docker push or docker save
            raise BundleError([f"{_path(plan, write)}: {e}"]) from None


_BUILD_OUTPUT_TAIL_LINES = 40


def _tail(message: str) -> str:
    """A failed build's message, its first line and the end of docker's output."""
    first, _, output = message.partition("\n")
    return "\n".join([first, *output.splitlines()[-_BUILD_OUTPUT_TAIL_LINES:]])


def _write_artifact(plan: Plan, write: Write) -> int:
    entry = write.source.entry
    cls = get_artifact_registry()[canonical_type(entry.type)]
    return cls.from_toml(_pinned(plan, write, cls.toml_refs), AuthoringContext(plan.bundle.bundle, entry)).version


def _write_agent(plan: Plan, write: Write) -> int:
    entry = write.source.entry
    return A2AAgent.from_toml(_pinned(plan, write, A2AAgent.toml_refs),
                              AuthoringContext(plan.bundle.bundle, entry)).version


def _write_env(plan: Plan, write: Write) -> int:
    entry = write.source.entry
    cls = get_env_registry()[entry.type]
    return cls.from_toml(_pinned(plan, write, cls.toml_refs), AuthoringContext(plan.bundle.bundle, entry)).version


def _pinned(plan: Plan, write: Write, refs: tuple[EntityRef, ...]) -> Any:
    """A copy of ``write``'s resolved toml with each store ref that names no version pinned to the version
    the plan read, which the ledger hashed (``unpinned_store_refs``)."""
    config = copy.deepcopy(write.source.config)
    pins = unpinned_store_refs(plan, write)
    for site in ref_sites(refs, config, inline_pins=True):
        planned = pins.get((site.ref.kind, site.value)) if site.version is None else None
        if planned is None:
            continue
        if site.version_key is None:
            site.owner[site.key] = {site.ref.kind.value: site.value, "version": planned}
        else:
            site.rewrite(site.value, planned)
    return config


def _write_eval(plan: Plan, write: Write) -> int:
    """An unpinned task ref, the bundle's own or a store's, is written without a version: the eval runs
    its latest."""
    tasks = [EvalTask(ref.id, ref.version) for ref in write.source.references]
    return Eval.put(id=write.id, tasks=tasks).version


# The writers this release has, by kind; any other kind is refused before anything is written. Tasks are
# written separately, once every one of them is preflighted, and evals after them, since they name the tasks.
_WRITERS: dict[BundleKind, Callable[[Plan, Write], int]] = {
    BundleKind.ARTIFACT: _write_artifact, BundleKind.AGENT: _write_agent, BundleKind.ENV: _write_env,
    BundleKind.EVAL: _write_eval,
}


def _refuse_unwritable(plan: Plan) -> None:
    problems = [f"{_path(plan, write)}: writing {what} isn't supported yet"
                for write in plan.writes if (what := _unwritable(write))]
    if problems:
        raise BundleError(problems)


def _refuse_builds_without_docker(plan: Plan, ledger: Ledger) -> None:
    """An image the ledger will reuse needs no docker, so only the ones it would build are refused."""
    if shutil.which("docker") is not None:
        return
    problems = [f"{_path(plan, write)}: building its image from {write.source.dockerfile} needs docker, and it isn't "
                "on PATH" for write in plan.writes
                if isinstance(write.source, BuiltImage) and not ledger.check(write, {}).unchanged]
    if problems:
        raise BundleError(problems)


def _unwritable(write: Write) -> str | None:
    """What ``write`` would write, when this release has no writer for it."""
    if isinstance(write.source, BuiltImage) or write.kind is BundleKind.TASK:
        return None
    if write.kind not in _WRITERS:
        return with_article(write.kind.value.removesuffix("s"))
    type_ = write.source.entry.type
    if write.kind is BundleKind.ARTIFACT and folder_walk(get_artifact_registry().get(canonical_type(type_))) is None:
        return with_article(f"{type_} artifact")
    if write.kind is BundleKind.ENV and not env_writer(get_env_registry().get(type_)):
        return with_article(f"{type_} env")
    return None


@contextmanager
def _noted(plan: Plan, write: Write, doing: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        e.add_note(f"while {doing} {_path(plan, write)} ({write.id})")
        raise


def _build(write: Write) -> Task:
    return Task(id=write.id, version=None, steps=[build_step(step) for step in write.source.config])


def _preflight(write: Write, task: Task, unwritten: set[tuple[str, str]]) -> tuple[list[str], list[TaskStep]]:
    """``task``'s preflight problems, and the steps with a preflight left unchecked because they read a
    (store, id) in ``unwritten``, which a dry run would write first; an artifact and an agent can share an id.
    A step reading one of the task's own outputs is skipped too, and not listed: the output only exists once the
    task runs."""
    outputs = {output.id for output in write.source.outputs}
    problems, unchecked = [], []
    for config, step in zip(write.source.config, task.steps):
        reads = _reads(config)
        if any(id in outputs for _, id in reads):
            continue
        if reads & unwritten:
            if type(step).preflight is not TaskStep.preflight:
                unchecked.append(step)
            continue
        problems.extend(step.preflight())
    return problems, unchecked


def _reads(step: dict) -> set[tuple[str, str]]:
    """The (store, id) of each entity ``step`` reads."""
    refs = get_task_step_registry()[step["type"]].entity_refs or ()
    return {(site.ref.kind.value, site.value) for site in ref_sites(refs, step) if site.ref.role is RefRole.INPUT}


def _key(write: Write) -> tuple[str, str]:
    return (write.kind.store, write.id)


def _path(plan: Plan, write: Write) -> str:
    return relative(plan.bundle.bundle.root, write.source.entry.path)
