"""Upstream OSWorld's setup, getters and metrics at ``OSWORLD_COMMIT``, as the ``osworld_*`` steps run them in the worker:
its source, which upstream's setup.py won't install, from GitHub's archive of that commit, checked by sha256, kept under
the cache root and put on ``sys.path``; or a folder holding the same files, named by ``SOURCE_ENV``, for an image that
bakes them in (``python -m agent_env.task_step.task_steps.computer.osworld.upstream <folder>`` fetches them there). What
they import is this folder's ``requirements.txt`` (``REQUIREMENTS``), which a worker running the steps installs.

Only ``desktop_env``'s package files, its controllers and evaluators, and the providers module its setup controller
imports for the proxy pool are kept, with upstream's LICENSE (Apache-2.0); its agents, VM providers, server and tests
are left out."""
from __future__ import annotations

import hashlib
import importlib
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_env.config.paths import cache_root
from agent_env.task_step.task_steps.computer.osworld import OSWORLD_COMMIT

logger = logging.getLogger(__name__)

#: A folder holding upstream's files at ``OSWORLD_COMMIT``, used as it is instead of the cache.
SOURCE_ENV = "AGENT_ENV_OSWORLD_SOURCE"
#: What upstream's checks import.
REQUIREMENTS = Path(__file__).parent / "requirements.txt"
ARCHIVE_URL = f"https://github.com/xlang-ai/OSWorld/archive/{OSWORLD_COMMIT}.tar.gz"
ARCHIVE_SHA256 = "1e6242c8d376d33f70f6c5dff5153e997a91c6564ef9a9bbd13ff0919e417dcb"
#: What's kept of the archive, under its top folder.
MEMBERS = ("LICENSE", "desktop_env/__init__.py", "desktop_env/actions.py", "desktop_env/controllers/",
           "desktop_env/evaluators/", "desktop_env/providers/__init__.py", "desktop_env/providers/base.py",
           "desktop_env/providers/aws/__init__.py", "desktop_env/providers/aws/proxy_pool.py")
#: A file only a complete fetch leaves, last.
_MARKER = ".complete"
_DOWNLOAD_TIMEOUT_S = 300
_lock = threading.Lock()


@dataclass(frozen=True)
class Upstream:
    """The upstream modules the steps call."""

    setup: types.ModuleType
    python: types.ModuleType
    getters: types.ModuleType
    metrics: types.ModuleType


def cache_dir() -> Path:
    return cache_root() / "osworld" / OSWORLD_COMMIT


def _wanted(name: str) -> str | None:
    """``name``, an archive member, without its top folder, when it's one ``MEMBERS`` keeps."""
    _, _, rest = name.partition("/")
    return rest if any(rest == m or (m.endswith("/") and rest.startswith(m)) for m in MEMBERS) else None


def fetch(dest: Path) -> Path:
    """Upstream's files at ``OSWORLD_COMMIT`` into ``dest``, unless a complete fetch is already there, staged beside it
    and renamed into place whole, so workers sharing the cache never see or remove a partial one. Raises RuntimeError
    when the archive's sha256 isn't ``ARCHIVE_SHA256``."""
    if (dest / _MARKER).exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as scratch:
        archive = Path(scratch) / "osworld.tar.gz"
        logger.info(f"fetching upstream OSWorld at {OSWORLD_COMMIT[:12]} from {ARCHIVE_URL}")
        with urllib.request.urlopen(ARCHIVE_URL, timeout=_DOWNLOAD_TIMEOUT_S) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != ARCHIVE_SHA256:
            raise RuntimeError(f"upstream OSWorld's archive at {OSWORLD_COMMIT} has sha256 {digest}, not {ARCHIVE_SHA256}")
        staged = Path(scratch) / "source"
        with tarfile.open(archive) as tar:
            members = []
            for member in tar.getmembers():
                if (rest := _wanted(member.name)) and (member.isfile() or member.isdir()):
                    member.name = rest
                    members.append(member)
            tar.extractall(staged, members=members, filter="data")
        (staged / _MARKER).touch()
        try:
            # Atomic, and refused when another worker sharing the cache published first: a folder there is only ever
            # a complete fetch, never removed, since that worker may be importing from it.
            staged.rename(dest)
        except OSError:
            if not (dest / _MARKER).exists():
                raise
            logger.info(f"another worker fetched upstream OSWorld into {dest} first; using its copy")
    return dest


def source() -> Path:
    """The folder upstream's files are in: ``SOURCE_ENV``'s, else the cache's, fetched once."""
    if named := os.environ.get(SOURCE_ENV, "").strip():
        if not (Path(named) / "desktop_env" / "evaluators").is_dir():
            raise RuntimeError(f"{SOURCE_ENV}={named!r} holds no desktop_env/evaluators: fetch upstream OSWorld there "
                               f"with python -m {__name__} {named}")
        return Path(named)
    return fetch(cache_dir())


def _easyocr_stub() -> None:
    """Stands in for easyocr, which upstream's ``metrics/docs.py`` imports at the top for ``compare_image_text`` alone,
    when it isn't installed: the real one pulls torch in, so ``REQUIREMENTS`` leaves it out and that metric
    fails."""
    try:
        importlib.import_module("easyocr")
        return
    except ImportError:
        pass

    class Reader:
        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError("compare_image_text needs easyocr, which the osworld steps' requirements leave out")

    stub = types.ModuleType("easyocr")
    stub.Reader = Reader  # type: ignore[attr-defined]
    sys.modules["easyocr"] = stub


def load() -> Upstream:
    """Upstream's modules, imported once. Raises RuntimeError naming ``REQUIREMENTS`` when what they import
    isn't installed."""
    with _lock:
        folder = str(source())
        if folder not in sys.path:
            sys.path.append(folder)
        _easyocr_stub()
        try:
            return Upstream(setup=importlib.import_module("desktop_env.controllers.setup"),
                            python=importlib.import_module("desktop_env.controllers.python"),
                            getters=importlib.import_module("desktop_env.evaluators.getters"),
                            metrics=importlib.import_module("desktop_env.evaluators.metrics"))
        except ImportError as e:
            raise RuntimeError(f"upstream OSWorld's checks need what they import installed where the steps run: "
                               f"pip install -r {REQUIREMENTS}: {e}") from e


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: python -m {__name__} <folder>")
    logging.basicConfig(level=logging.INFO)
    print(fetch(Path(sys.argv[1]).resolve()))
