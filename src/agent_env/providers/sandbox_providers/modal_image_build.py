"""Building on Modal an image that is only a build context (``DockerImageArtifact.put_context``): its context unpacked
on this machine, its Dockerfile in the forms Modal's build takes, and each source built once in a process."""

from __future__ import annotations

import asyncio
import glob
import logging
import tarfile
import tempfile
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from agent_env.config import get_config
from agent_env.utils.docker.context import extract, ignore_file
from agent_env.utils.docker.dockerfile import Instruction, copy_arguments, instructions, opens_heredoc

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.docker_image import DockerImageArtifact

logger = logging.getLogger(__name__)

# The id of each image built from a build context, by Modal token, environment, app and source digest, for every
# provider in the process to reuse; one build of each runs at a time, so deploys of one image at once share its build.
_built_image_ids: dict[tuple[str, str, str, str], str] = {}
_build_locks: weakref.WeakValueDictionary[tuple[asyncio.AbstractEventLoop, tuple[str, str, str, str]], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)


async def context_image(image: DockerImageArtifact, *, client: modal.Client, app: modal.App, token_id: str,
                        app_name: str, environment_name: str | None,
                        stale: str | None = None) -> tuple[modal.Image, str]:
    """The Modal image built from ``image``'s build context in ``app``, and its id: built now unless one built from the
    same sources is, other than ``stale``, an id Modal no longer holds."""
    key = (token_id, environment_name or "", app_name, image.source_digest or image.build_context_object_url)
    async with _build_locks.setdefault((asyncio.get_running_loop(), key), asyncio.Lock()):
        image_id = _built_image_ids.get(key)
        if image_id is None or image_id == stale:
            image_id = _built_image_ids[key] = await build(image, app)
    return modal.Image.from_id(image_id, client=client), image_id


async def build(image: DockerImageArtifact, app: modal.App) -> str:
    """Build ``image`` from its build context, with the context's own ignore file; returns the image's id."""
    logger.info(f"Building {image.image_name} on Modal from its build context...")
    with tempfile.TemporaryDirectory(prefix="agent-env-build-") as work:
        context = Path(work) / "context"
        try:
            await asyncio.to_thread(_fetch_context, image.build_context_object_url, Path(work) / "context.tar.gz",
                                    context)
            dockerfile = context / image.dockerfile_path
            ignores = ignore_file(context, dockerfile)
            # Beside the context, not in it, so a COPY of the context still copies the Dockerfile as written.
            for_modal = Path(work) / "Dockerfile"
            for_modal.write_text(modal_dockerfile(dockerfile.read_text("utf8"), context), "utf8")
            built = modal.Image.from_dockerfile(
                for_modal, context_dir=context, ignore=ignores.read_text("utf8").splitlines() if ignores else [],
            )
            await built.build.aio(app)
        except Exception as e:
            raise RuntimeError(f"Building {image.image_name} on Modal from its build context failed: "
                               f"{fmt_exc(e)}") from e
    logger.info(f"Built {image.image_name} on Modal as {built.object_id}")
    return built.object_id


def fmt_exc(e: BaseException) -> str:
    """``e`` as ``<module>.<class>: <message>``."""
    type_name = f"{type(e).__module__}.{type(e).__qualname__}"
    msg = str(e).strip()
    return f"{type_name}: {msg}" if msg else type_name


# The flags COPY takes as ADD does; ADD's own (--checksum, --keep-git-dir, --unpack) keep an ADD as it is.
_COPY_FLAGS = ("--chown=", "--chmod=", "--link")


def modal_dockerfile(text: str, context: Path) -> str:
    """``text``, a Dockerfile, with the two forms Modal's build refuses written as what ``docker build`` makes of them:
    a JSON-form COPY or ADD whose paths hold no whitespace in shell form, which Modal's context parser reads, and an
    ADD of files and folders in ``context``, none of them a tar archive, as the COPY it is, since Modal's ADD only
    fetches a URL. Every other line is left as written."""
    return "".join(_for_modal(instruction, context) for instruction in instructions(text))


def _for_modal(instruction: Instruction, context: Path) -> str:
    args = copy_arguments(instruction)
    if args is None or args.heredoc:
        return instruction.text
    paths = (*args.sources, args.destination)
    if args.json_form:
        # Written in shell form, a path with whitespace would split, and one like <<EOF would open a heredoc.
        if not all(path and not any(c.isspace() for c in path) and not opens_heredoc(path) for path in paths):
            return instruction.text
    elif args.keyword != "ADD":
        return instruction.text
    keyword = args.keyword
    if keyword == "ADD":
        if not all(flag.startswith(_COPY_FLAGS) for flag in args.flags) \
                or not all(_local_and_not_an_archive(source, context) for source in args.sources):
            return instruction.text
        keyword = "COPY"
    # The rewrite is one line, so comment lines among the continued ones go above it rather than be lost.
    comments = [line for line in instruction.text.splitlines(keepends=True)[1:] if line.lstrip().startswith("#")]
    return "".join(comments) + " ".join([keyword, *args.flags, *paths]) + "\n"


def _local_and_not_an_archive(source: str, context: Path) -> bool:
    """Whether ``source``, an ADD source, names files or folders in ``context``, none a tar archive ADD would unpack."""
    if "://" in source or source.startswith("git@"):
        return False
    matches = glob.glob(source.lstrip("/"), root_dir=context) if glob.has_magic(source) else [source.lstrip("/")]
    paths = [context / match for match in matches]
    return bool(paths) and all(path.exists() for path in paths) \
        and not any(path.is_file() and tarfile.is_tarfile(path) for path in paths)


def _fetch_context(object_url: str, archive: Path, out: Path) -> None:
    """Download the build context at ``object_url`` to ``archive`` and unpack it into ``out``."""
    get_config().get_object_store_at(object_url).download_to_file(object_url, str(archive))
    extract(archive, out)
