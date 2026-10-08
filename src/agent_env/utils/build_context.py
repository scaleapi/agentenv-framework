"""A Docker build context as a tar.gz that is the same, with the same digest, wherever the same files are written
from: what ``docker build`` would send from a folder, in a fixed order, with nothing of the machine in it."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
import tarfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from modal import FilePatternMatcher

from agent_env.utils.paths import LEAVINGS

# Bumped when what source_digest covers changes, so a digest is never compared across meanings.
_DIGEST_SCHEME = 1
_READ_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class _Entry:
    name: str  # POSIX, NFC, relative to the context's root
    mode: int
    path: Path | None = None  # a file's, read for its content
    sha256: str | None = None  # a file's
    link: str | None = None  # a link's target, as written in it

    @property
    def content(self) -> str | None:
        """What the digest records for the entry beyond its name and mode: a file's sha256 or a link's target."""
        return self.sha256 if self.path is not None else (f"link:{self.link}" if self.link is not None else None)


@dataclass(frozen=True)
class BuildContext:
    """The files ``docker build`` sends from a folder: all but what its ignore file excludes, the Dockerfile and the
    ignore file always kept, and what the OS or Python leaves behind always dropped. A link is sent as a link, never
    followed, as docker sends it. Modes are git's two: 0755 when the owner may execute, else 0644."""

    dockerfile: str | None  # its path in the context, None when it's outside it
    entries: tuple[_Entry, ...]

    @classmethod
    def of(cls, root: Path, dockerfile: Path | None = None) -> BuildContext:
        """The context at ``root`` for ``dockerfile``. Raises ValueError naming every entry it can't take: a special
        file, an unreadable one, two names equal once normalized."""
        if not root.is_dir():
            raise ValueError(f"build context {root} is not a folder")
        top = Path(os.path.realpath(root))
        named = _inside(top, dockerfile) if dockerfile is not None else None
        ignore_file = _ignore_file(top, dockerfile)
        kept = {name for name in (named, _inside(top, ignore_file) if ignore_file else None) if name}
        matcher = FilePatternMatcher.from_file(ignore_file) if ignore_file else None
        walk = _Walk(top, matcher, kept)
        walk.run()
        if walk.problems:
            raise ValueError(f"build context {root}: " + "; ".join(walk.problems))
        return cls(dockerfile=named, entries=walk.entries())

    def source_digest(self, platform: str | None) -> str:
        """sha256 over every entry's name, mode and content, the Dockerfile's path and ``platform``: equal for equal
        sources, so an image built from one can stand for the other."""
        manifest = {
            "scheme": _DIGEST_SCHEME,
            "dockerfile": self.dockerfile,
            "platform": platform,
            "entries": [[entry.name, f"{entry.mode:o}", entry.content] for entry in self.entries],
        }
        canonical = json.dumps(manifest, separators=(",", ":"), ensure_ascii=False)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def write(self, out: Path) -> None:
        """Write the context to ``out`` as a tar.gz with sorted names, no times, owners or file name in it, and git's
        modes. Raises RuntimeError when a file changed since the context was read."""
        with (out.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz,
              tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar):
            for entry in self.entries:
                info = tarfile.TarInfo(entry.name)
                info.mode, info.mtime = entry.mode, 0
                if entry.path is None:
                    info.type = tarfile.SYMTYPE if entry.link is not None else tarfile.DIRTYPE
                    info.linkname = entry.link or ""
                    tar.addfile(info)
                    continue
                with entry.path.open("rb") as f:
                    info.size = os.fstat(f.fileno()).st_size
                    reader = _Hashing(f)
                    tar.addfile(info, reader)
                if reader.hexdigest() != entry.sha256:
                    raise RuntimeError(f"{entry.name} changed while its build context was being written")


class _Walk:
    """Lists a context folder in sorted order."""

    def __init__(self, top: Path, matcher: FilePatternMatcher | None, kept: set[str]):
        self.top = top
        self.matcher = matcher
        self.kept = kept
        self.found: dict[str, _Entry] = {}
        self.folders: set[str] = set()
        self.problems: list[str] = []

    def run(self) -> None:
        pending = [(self.top, ())]
        while pending:
            folder, parts = pending.pop()
            try:
                with os.scandir(folder) as listing:
                    children = sorted(listing, key=lambda child: child.name)
            except OSError as e:
                self.problems.append(f"{_show(parts)}: {e.strerror or e}")
                continue
            for child in children:
                if child.name in LEAVINGS:
                    continue
                child_parts = (*parts, unicodedata.normalize("NFC", child.name))
                name = "/".join(child_parts)
                excluded = name not in self.kept and self._excluded(name)
                if child.is_dir(follow_symlinks=False):
                    if not excluded:
                        self.folders.add(name)
                    if not excluded or not self._prunable(name):
                        pending.append((Path(child.path), child_parts))
                elif excluded:
                    continue
                elif child.is_symlink():
                    self._add(_Entry(name, 0o777, link=os.readlink(child.path)))
                elif child.is_file(follow_symlinks=False):
                    self._file(name, Path(child.path))
                else:
                    self.problems.append(f"{name}: neither a regular file, a folder nor a link")

    def entries(self) -> tuple[_Entry, ...]:
        folders = set(self.folders)
        for name in self.found:
            folders.update(str(parent) for parent in PurePosixPath(name).parents if str(parent) != ".")
        listed = [*self.found.values(), *(_Entry(name, 0o755) for name in folders - set(self.found))]
        return tuple(sorted(listed, key=lambda entry: entry.name))

    def _excluded(self, name: str) -> bool:
        return self.matcher is not None and self.matcher(Path(name))

    def _prunable(self, name: str) -> bool:
        """Whether nothing under an excluded folder can be sent: no exception pattern brings a file back, and no file
        the context always keeps is in it."""
        return self.matcher.can_prune_directories() and not any(kept.startswith(f"{name}/") for kept in self.kept)

    def _add(self, entry: _Entry) -> None:
        if entry.name in self.found:
            self.problems.append(f"{entry.name}: two entries have this name once normalized to NFC; rename one")
        else:
            self.found[entry.name] = entry

    def _file(self, name: str, path: Path) -> None:
        try:
            with path.open("rb") as f:
                digest = hashlib.sha256()
                while chunk := f.read(_READ_CHUNK_BYTES):
                    digest.update(chunk)
                executable = os.fstat(f.fileno()).st_mode & stat.S_IXUSR
        except OSError as e:
            self.problems.append(f"{name}: {e.strerror or e}")
            return
        self._add(_Entry(name, 0o755 if executable else 0o644, path=path, sha256=digest.hexdigest()))


class _Hashing:
    """A file the tar reads through, hashing what it reads."""

    def __init__(self, f):
        self._f = f
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._f.read(size)
        self._digest.update(data)
        return data

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _ignore_file(top: Path, dockerfile: Path | None) -> Path | None:
    """The ignore file ``docker build`` reads: ``<Dockerfile>.dockerignore`` beside the Dockerfile, else the
    context's ``.dockerignore``."""
    candidates = [dockerfile.with_name(f"{dockerfile.name}.dockerignore")] if dockerfile is not None else []
    return next((path for path in (*candidates, top / ".dockerignore") if path.is_file()), None)


def _inside(top: Path, path: Path) -> str | None:
    """``path``'s name in the context at ``top``, links resolved, or None when it resolves outside it."""
    resolved = Path(os.path.realpath(path))
    return unicodedata.normalize("NFC", resolved.relative_to(top).as_posix()) if resolved.is_relative_to(top) else None


def _show(parts: tuple[str, ...]) -> str:
    return "/".join(parts) or "."
