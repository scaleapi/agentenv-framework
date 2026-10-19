"""The images a bundle import writes: each stored build context, pruned to what its Dockerfile reads, in its entity's
folder; contexts stored as one object are downloaded once."""

from __future__ import annotations

import filecmp
import os
import posixpath
import shutil
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.config import get_config
from agent_env.store.base import NotFoundError
from agent_env.utils.docker.installed import installed_packages
from agent_env.utils.docker.lock import Locked, lock
from agent_env.utils.docker.prune import Pruned, write_pruned


@dataclass(eq=False)
class ImageJob:
    """One image an entity's folder is built from; ``toml`` is its toml value, completed once it's written."""

    artifact: DockerImageArtifact
    folder: str  # the entity's folder in the bundle
    key: str  # the toml key naming the image
    toml: dict
    toml_name: str  # the entity's own toml, which the folder holds beside the image's files
    size: int | None = None  # of the stored context
    pruned: Pruned | None = None
    problem: str | None = None  # why it can't be written
    locked: Locked | None = None
    unlocked: str | None = None  # why its versions weren't pinned


def context_groups(jobs: list[ImageJob]) -> list[list[ImageJob]]:
    """``jobs`` grouped by the stored context they're built from: the same object when the store tells its content
    (an ETag), else the same URL."""
    groups: dict[tuple, list[ImageJob]] = {}
    for job in jobs:
        url = job.artifact.build_context_object_url
        metadata = get_config().get_object_store_at(url).get_object_metadata_at(url)
        job.size = metadata.size if metadata is not None else None
        key = ("etag", metadata.etag, metadata.size) if metadata is not None and metadata.etag else ("url", url)
        groups.setdefault(key, []).append(job)
    return list(groups.values())


def write_images(groups: list[list[ImageJob]], staging: Path, scratch: Path,
                 on_group: Callable[[list[ImageJob]], None] | None = None) -> None:
    """Download each group's context once, write each job's pruned folder into its entity's folder in ``staging``, and
    note on a job why it can't be written; then pin each written image's installs."""
    placed: list[ImageJob] = []
    for index, group in enumerate(groups):
        if on_group is not None:
            on_group(group)
        url = group[0].artifact.build_context_object_url
        archive = scratch / f"context-{index}.tar.gz"
        try:
            get_config().get_object_store_at(url).download_to_file(url, str(archive))
        except (NotFoundError, ValueError) as e:  # gone from the store, or in one this config can't read
            for job in group:
                job.problem = ("its build context can't be downloaded: the store no longer holds it"
                               if isinstance(e, NotFoundError) else
                               "its build context can't be downloaded: the configured stores can't read it")
            continue
        folders = {job: scratch / f"image-{index}-{n}" for n, job in enumerate(group)}
        try:
            pruned = write_pruned(archive, {folders[job]: job.artifact.dockerfile_path for job in group})
        except _UNREADABLE:  # one image's Dockerfile can't be found or read: write the others, and say which
            pruned = {}
            for job in group:
                try:
                    pruned |= write_pruned(archive, {folders[job]: job.artifact.dockerfile_path})
                except _UNREADABLE as e:
                    job.problem = f"its build context can't be written as a folder: {e or type(e).__name__}"
        finally:
            archive.unlink(missing_ok=True)
        for job in group:
            if job.problem is None:
                job.pruned = pruned[folders[job]]
                _place(job, folders[job], staging / job.folder)
            shutil.rmtree(folders[job], ignore_errors=True)
            if job.problem is None:
                placed.append(job)
    # Pinning rewrites a Dockerfile, which another image placed in the same folder may hold too: so only once all are.
    locked: dict[tuple[str, str], ImageJob] = {}
    for job in placed:
        first = locked.setdefault((job.folder, job.pruned.dockerfile), job)
        if first is job:
            _lock(job, staging / job.folder)
        else:  # two images built from one Dockerfile in one folder: it's pinned once
            job.locked, job.unlocked = first.locked, first.unlocked


# What reading a stored context raises when it isn't one a folder can be written from, or is damaged.
_UNREADABLE = (ValueError, tarfile.TarError, EOFError)


def _lock(job: ImageJob, folder: Path) -> None:
    """Pin the folder's installs to the versions the stored image holds, or note on the job why it can't."""
    url = job.artifact.tar_gz_object_url
    if not url:
        job.unlocked = "the store holds no image to read its versions from, so a build resolves them afresh"
        return
    # Each reason leaves the image's url out, as everything the folder records does.
    try:
        store = get_config().get_object_store_at(url)
    except ValueError:
        job.unlocked = "the configured stores can't read its saved image, so a build resolves them afresh"
        return
    try:
        with store.open(url) as saved:
            installed = installed_packages(saved)
    except NotFoundError:
        job.unlocked = "the store no longer holds its saved image, so a build resolves them afresh"
        return
    except OSError as e:
        job.unlocked = f"its saved image can't be read ({e.strerror or type(e).__name__}), so a build resolves them afresh"
        return
    except _UNREADABLE as e:
        job.unlocked = f"its saved image can't be read ({e or type(e).__name__}), so a build resolves them afresh"
        return
    if installed is None:
        job.unlocked = "its saved image's layers can't be read, so a build resolves them afresh"
        return
    job.locked = lock(folder, job.pruned.dockerfile, installed)


def _place(job: ImageJob, built: Path, dest: Path) -> None:
    """Move ``built``'s files into ``dest``, the entity's folder, unless one can't be read from a bundle or clashes."""
    if problem := _link_problem(built) or _clash(built, dest, job.toml_name):
        job.problem = problem
        return
    for path in sorted(built.rglob("*"), key=lambda path: len(path.parts)):
        target = dest / path.relative_to(built)
        if path.is_dir() and not path.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
        elif not os.path.lexists(target):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)
    job.toml["dockerfile"] = job.pruned.dockerfile


def _link_problem(built: Path) -> str | None:
    """Why a link in ``built`` can't be read from a bundle, which follows only links staying in it, to something there."""
    for path in sorted(built.rglob("*")):
        if not path.is_symlink():
            continue
        name = path.relative_to(built).as_posix()
        target = os.readlink(path)
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
        if target.startswith("/") or resolved == ".." or resolved.startswith("../"):
            return f"its build context holds the link {name} -> {target}, which leads out of it, and a bundle can't"
        if not path.exists():
            return f"its build context holds the link {name} -> {target}, which leads nowhere, and a bundle can't"
    return None


def _clash(built: Path, dest: Path, toml_name: str) -> str | None:
    """Why ``built`` can't share ``dest`` with what's there: its own copy of the entity's toml, or a file differing."""
    if os.path.lexists(built / toml_name):
        return f"its build context holds a {toml_name}, which the bundle's own {toml_name} would replace"
    for path in sorted(built.rglob("*")):
        target = dest / path.relative_to(built)
        if path.is_file() and not path.is_symlink() and target.is_file() and not filecmp.cmp(path, target,
                                                                                             shallow=False):
            return f"its build context and another image's both hold {path.relative_to(built).as_posix()}, differently"
    return None
