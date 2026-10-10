"""Building on Modal an image that is only a build context (``DockerImageArtifact.put_context``): its context unpacked
on this machine, its Dockerfile in the forms Modal's build takes, and each source built once in a process."""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import re
import tarfile
import tempfile
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from agent_env.config import get_config
from agent_env.utils.build_context import extract, ignore_file

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.docker_image import DockerImageArtifact

logger = logging.getLogger(__name__)

# The id of each image built from a build context, by Modal token, app and source digest, for every provider in the
# process to reuse; one build of each runs at a time, so deploys of one image at once share its build.
_built_image_ids: dict[tuple[str, str, str], str] = {}
_build_locks: weakref.WeakValueDictionary[tuple[asyncio.AbstractEventLoop, tuple[str, str, str]], asyncio.Lock] = (
    weakref.WeakValueDictionary()
)


async def context_image(image: DockerImageArtifact, *, client: modal.Client, app: modal.App, token_id: str,
                        app_name: str, stale: str | None = None) -> tuple[modal.Image, str]:
    """The Modal image built from ``image``'s build context in ``app``, and its id: built now unless one built from the
    same sources is, other than ``stale``, an id Modal no longer holds."""
    key = (token_id, app_name, image.source_digest or image.build_context_object_url)
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
            dockerfile = context / (image.dockerfile_path or "Dockerfile")
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


_COPY_OR_ADD = re.compile(r"\s*(COPY|ADD)\s+(.*)", re.IGNORECASE | re.DOTALL)
# The flags COPY takes as ADD does; ADD's own (--checksum, --keep-git-dir, --unpack) keep an ADD as it is.
_COPY_FLAGS = ("--chown=", "--chmod=", "--link")


def modal_dockerfile(text: str, context: Path) -> str:
    """``text``, a Dockerfile, with the two forms Modal's build refuses written as what ``docker build`` makes of them:
    a JSON-form COPY or ADD whose paths hold no whitespace in shell form, which Modal's context parser reads, and an
    ADD of files and folders in ``context``, none of them a tar archive, as the COPY it is, since Modal's ADD only
    fetches a URL. Every other line is left as written."""
    lines, instruction = [], []
    for line in text.splitlines(keepends=True):
        instruction.append(line)
        if not line.rstrip("\r\n").endswith("\\"):
            lines.append(_for_modal("".join(instruction), context))
            instruction = []
    return "".join(lines + instruction)


def _for_modal(instruction: str, context: Path) -> str:
    match = _COPY_OR_ADD.fullmatch(re.sub(r"\\\r?\n", " ", instruction).strip())
    if match is None:
        return instruction
    keyword, rest = match.group(1).upper(), match.group(2).strip()
    flags = []
    while rest.startswith("--"):
        flag, _, rest = rest.partition(" ")
        flags.append(flag)
        rest = rest.strip()
    if rest.startswith("["):
        try:
            args = json.loads(rest)
        except ValueError:
            return instruction
        if not (isinstance(args, list) and len(args) >= 2
                and all(isinstance(arg, str) and arg and not any(c.isspace() for c in arg) for arg in args)):
            return instruction
    elif keyword == "ADD" and not rest.startswith("<<") and not any(c in rest for c in "\"'"):
        args = rest.split()
    else:
        return instruction
    if keyword == "ADD":
        if len(args) < 2 or not all(flag.startswith(_COPY_FLAGS) for flag in flags) \
                or not all(_local_and_not_an_archive(source, context) for source in args[:-1]):
            return instruction
        keyword = "COPY"
    return " ".join([keyword, *flags, *args]) + "\n"


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
