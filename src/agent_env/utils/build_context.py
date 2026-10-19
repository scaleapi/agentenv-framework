"""A Docker build context as a tar.gz that is the same, with the same digest, wherever the same files are written
from: what ``docker build`` would send from a folder, in a fixed order, with nothing of the machine in it."""

from __future__ import annotations

import bisect
import csv
import functools
import gzip
import heapq
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import tarfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from modal import FilePatternMatcher

from agent_env.utils.dockerfile import Instruction, copy_arguments, instructions, leading_flags
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
        ignores = ignore_file(top, dockerfile)
        # Sent whatever the ignore file says, and through any link to them, so a build finds them in the context.
        sent = [path for path in (dockerfile, ignores) if path is not None]
        kept = {name for path in sent for name in (_inside(top, path), _inside(top, Path(os.path.realpath(path)))) if name}
        named = None
        if dockerfile is not None and _inside(top, Path(os.path.realpath(dockerfile))):
            named = _inside(top, dockerfile)
        matcher = FilePatternMatcher.from_file(ignores) if ignores else None
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


def extract(archive: Path, out: Path) -> None:
    """Unpack a build context's tar.gz into the folder ``out``: its files, folders and links. A link that leads out
    of ``out`` is left out, as nothing there is the build's. Raises ValueError on anything else, which no context
    written by ``BuildContext.write`` holds: a hard link, a device, a name that leads out of ``out``."""
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(out, filter=_context_member)


@dataclass(frozen=True)
class ContextReads:
    """What a Dockerfile's build reads of its context: the context paths in ``paths``, each of which may hold Docker's
    wildcards, or all of it when ``whole`` says why: it copies all of it, or reads what can't be told before a build."""

    paths: tuple[str, ...] = ()
    whole: str | None = None


@dataclass(frozen=True)
class Pruned:
    """What ``write_pruned`` wrote into a folder for the Dockerfile at ``dockerfile`` in it: ``files`` files of
    ``size`` bytes, the biggest of them by bytes, and why it wrote everything the context sends when it did."""

    dockerfile: str
    files: int
    size: int
    whole: str | None
    largest: tuple[tuple[str, int], ...] = ()


# The COPY and ADD flags that change how a source is written, not which one it is.
_SOURCE_KEEPING_FLAGS = ("--chown=", "--chmod=", "--link", "--checksum=", "--keep-git-dir", "--unpack")


def context_reads(dockerfile: str) -> ContextReads:
    """What a build of ``dockerfile`` reads of its context: each local COPY and ADD source of every stage and each bind
    mount's source, but not a COPY --from, a URL or git source, or a heredoc."""
    paths: list[str] = []
    for instruction in instructions(dockerfile):
        if not instruction.complete:
            return ContextReads(whole=f"a heredoc in {instruction.keyword} never ends, so the rest can't be read")
        if instruction.keyword in ("COPY", "ADD"):
            whole = _copy_reads(instruction, paths)
        elif instruction.keyword == "RUN":
            whole = _mount_reads(instruction, paths)
        else:
            continue
        if whole:
            return ContextReads(whole=whole)
    return ContextReads(paths=tuple(dict.fromkeys(paths)))


def write_pruned(archive: Path, images: Mapping[Path, str | None]) -> dict[Path, Pruned]:
    """Write into each folder ``images`` names the part of the build context tar.gz at ``archive`` that its
    Dockerfile, at the path ``images`` gives in the context, reads: what ``context_reads`` finds, less what the
    context's ignore file leaves out and what the OS or Python leaves behind, with the Dockerfile, the ignore file
    and the targets of kept links. An image with no path given builds the context's ``Dockerfile``, or, with none at
    its root, the only one in it. Reads the archive twice, a third time for a Dockerfile found or reached through a
    link, and never unpacks the rest of it. Raises ValueError when a Dockerfile isn't in the archive or can't be told
    apart from others, or on a member ``extract`` refuses."""
    dockerfiles = {folder: posixpath.normpath(dockerfile) for folder, dockerfile in images.items() if dockerfile}
    wanted = {name for dockerfile in dockerfiles.values() for name in (dockerfile, *_ignore_files(dockerfile))}
    links: dict[str, str | None] = {}
    texts: dict[str, str] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if (name := _member_name(member)) is None:
                continue
            links[name] = member.linkname if member.issym() else None
            if name in wanted and member.isreg():
                texts[name] = tar.extractfile(member).read().decode("utf-8", "replace")
    for folder in images.keys() - dockerfiles.keys():
        dockerfiles[folder] = _found_dockerfile(links)
        wanted |= {dockerfiles[folder], *_ignore_files(dockerfiles[folder])}
    if unread := {name for name in wanted if name in links and not links[name] and name not in texts}:
        texts |= _texts(archive, unread)
    # A Dockerfile or ignore file stored as a link is read where it leads, as docker build reads it.
    linked = {name: _resolve(name, links)[0] for name in wanted if links.get(name)}
    if unread := {target for target in linked.values() if target not in texts}:
        texts |= _texts(archive, unread)
    texts |= {name: texts[target] for name, target in linked.items() if target in texts}
    plans: dict[Path, tuple[set[str], str | None]] = {}
    for folder, dockerfile in dockerfiles.items():
        if dockerfile not in texts:
            raise ValueError(f"the build context has no {dockerfile}")
        ignore = next((name for name in _ignore_files(dockerfile) if name in texts), None)
        matcher = FilePatternMatcher(*_patterns(texts[ignore])) if ignore else None
        reads, crossed = _through_links(context_reads(texts[dockerfile]), links)
        # A source the reading found that names nothing here is likelier a misreading than a build that fails.
        if missing := next((path for path in reads.paths if not any(_matches(path, name) for name in links)), None):
            reads, crossed = ContextReads(whole=f"{missing} matches nothing in the context"), set()
        plans[folder] = (_kept(links, reads, dockerfile, ignore, matcher, crossed), reads.whole)
    written: dict[Path, list[tuple[str, int]]] = {folder: [] for folder in images}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = _member_name(member)
            first: Path | None = None
            for folder in (folder for folder, (kept, _) in plans.items() if name in kept):
                if member.isreg() and first is not None:
                    (folder / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(first, folder / name)
                else:
                    folder.mkdir(parents=True, exist_ok=True)
                    tar.extract(member, folder, filter=_context_member)
                    first = folder / name if member.isreg() else None
                if member.isreg():
                    written[folder].append((name, member.size))
    return {folder: Pruned(dockerfiles[folder], len(files), sum(size for _, size in files), plans[folder][1],
                           tuple(heapq.nlargest(_LARGEST, files, key=lambda file: file[1])))
            for folder, files in written.items()}


# How many of an image's biggest files its Pruned names, for an import to show what it wrote.
_LARGEST = 5


def _found_dockerfile(links: Mapping[str, str | None]) -> str:
    """The Dockerfile an image that names none builds: the context's ``Dockerfile``, else the only one in it."""
    if "Dockerfile" in links:
        return "Dockerfile"
    found = sorted(name for name in links if posixpath.basename(name) == "Dockerfile")
    if len(found) != 1:
        raise ValueError(f"the build context names no Dockerfile and holds {len(found)}"
                         f"{': ' + ', '.join(found) if found else ''}; name the one it builds")
    return found[0]


def _texts(archive: Path, names: set[str]) -> dict[str, str]:
    """The text of each regular file ``names`` names in the tar.gz at ``archive``."""
    texts = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isreg() and (name := _member_name(member)) in names:
                texts[name] = tar.extractfile(member).read().decode("utf-8", "replace")
    return texts


def _copy_reads(instruction: Instruction, paths: list[str]) -> str | None:
    """Add the context paths a COPY or ADD reads to ``paths``; why it reads all of the context, when it does."""
    args = copy_arguments(instruction)
    if args is None:
        return f"{instruction.keyword} {instruction.arguments} can't be read"
    if any(flag.startswith("--from=") for flag in args.flags):
        return None
    if unknown := next((flag for flag in args.flags if not flag.startswith(_SOURCE_KEEPING_FLAGS)), None):
        return f"{args.keyword} {unknown} isn't followed"
    for source in args.sources:
        if args.keyword == "ADD" and ("://" in source or source.startswith("git@")):
            continue
        if "$" in source:
            return f"{args.keyword} {source} names a variable"
        if (path := _context_path(source)) is None:
            return f"{args.keyword} {source} copies all of it"
        paths.append(path)
    return None


def _mount_reads(instruction: Instruction, paths: list[str]) -> str | None:
    """Add the context paths a RUN's bind mounts read to ``paths``; why it reads all of the context, when it does. A
    bind mount names no other image or stage with ``from``, and its source defaults to the whole context."""
    for flag in leading_flags(instruction.arguments)[0]:
        if not flag.startswith("--mount="):
            continue
        # A mount's options are one CSV record, as docker build reads them, so a quoted field holds a comma or space.
        options = {}
        for field in next(csv.reader([flag.removeprefix("--mount=")])):
            key, _, value = field.partition("=")
            options[key.strip()] = value.strip().strip("\"'")
        if options.get("type", "bind") != "bind" or "from" in options:
            continue
        source = options.get("source") or options.get("src") or "."
        if "$" in source:
            return f"RUN --mount source {source} names a variable"
        if (path := _context_path(source)) is None:
            return f"RUN --mount binds {source}, all of it"
        paths.append(path)
    return None


def _context_path(source: str) -> str | None:
    """``source``'s path in the context; None for the context itself, or a path that would leave it."""
    path = posixpath.normpath(source.lstrip("/"))
    return None if path == "." or path == ".." or path.startswith("../") else path


def _through_links(reads: ContextReads, links: Mapping[str, str | None]) -> tuple[ContextReads, set[str]]:
    """``reads`` with each source that names nothing itself followed through the links its leading parts name, as
    docker build follows them, and the links it crossed. A wildcard source is left as it is."""
    if reads.whole is not None:
        return reads, set()
    paths, crossed = [], set()
    for path in reads.paths:
        if _WILDCARD.search(path) or any(_matches(path, name) for name in links):
            paths.append(path)
            continue
        resolved, hops = _resolve(path, links)
        paths.append(resolved)
        crossed |= hops
    return ContextReads(paths=tuple(paths)), crossed


def _resolve(path: str, links: Mapping[str, str | None]) -> tuple[str, set[str]]:
    """``path`` with each part that is a link in the context replaced by where it leads, and the links that took; as it
    is, crossing nothing, when a link leads out of the context or round in a loop."""
    resolved, crossed = "", set()
    for part in path.split("/"):
        resolved = posixpath.join(resolved, part)
        while (target := links.get(resolved)) is not None:
            following = posixpath.normpath(posixpath.join(posixpath.dirname(resolved), target))
            if target.startswith("/") or following == ".." or following.startswith("../") or resolved in crossed:
                return path, set()
            crossed.add(resolved)
            resolved = following
    return resolved, crossed


def _kept(links: Mapping[str, str | None], reads: ContextReads, dockerfile: str, ignore: str | None,
          matcher: FilePatternMatcher | None, crossed: set[str] = frozenset()) -> set[str]:
    """The names in ``links`` (each name's link target, None when it's no link) that a build reads, as ``write_pruned``
    keeps them, and the folders above them. What the OS or Python leaves behind is dropped, as ``BuildContext`` drops it."""
    def sent(name: str) -> bool:
        return not LEAVINGS.intersection(name.split("/")) and (matcher is None or not matcher(Path(name)))

    if reads.whole is not None:
        kept = {name for name in links if sent(name)}
    else:
        kept = {name for name in links if sent(name) and any(_matches(path, name) for path in reads.paths)}
    kept |= {name for name in (dockerfile, ignore) if name in links} | {name for name in crossed if sent(name)}
    names = sorted(links)
    pending = [name for name in kept if links[name]]
    while pending:
        name = pending.pop()
        target = posixpath.normpath(posixpath.join(posixpath.dirname(name), links[name]))
        if links[name].startswith("/") or target == ".." or target.startswith("../"):
            continue
        for other in _at_or_under(names, target):
            if other not in kept and sent(other):
                kept.add(other)
                if links[other]:
                    pending.append(other)
    return kept | {str(parent) for name in kept for parent in PurePosixPath(name).parents if str(parent) in links}


def _at_or_under(names: list[str], path: str) -> list[str]:
    """The names in ``names``, sorted, that are ``path`` or under it."""
    at = bisect.bisect_left(names, path)
    found = [path] if at < len(names) and names[at] == path else []
    start = bisect.bisect_left(names, f"{path}/")
    end = bisect.bisect_left(names, f"{path}0")  # "0" sorts right after "/"
    return found + names[start:end]


def _matches(path: str, name: str) -> bool:
    """Whether ``name`` is ``path`` or under it, each part matched as Docker matches a COPY source's."""
    want, have = path.split("/"), name.split("/")
    return len(have) >= len(want) and all(
        part == glob if not _WILDCARD.search(glob) else bool((pattern := _part_pattern(glob)) and pattern.fullmatch(part))
        for part, glob in zip(have, want))


_WILDCARD = re.compile(r"[*?\[\\]")


@functools.lru_cache(maxsize=1024)
def _part_pattern(glob: str) -> re.Pattern[str] | None:
    """A path part's wildcard pattern as Go's ``filepath.Match``, which docker build matches COPY sources with, reads
    it: ``*`` any characters and ``?`` any one, ``[...]`` a class that ``^`` negates (``!`` is a character like any
    other), and ``\\`` the next character as it is. None when the pattern is malformed, which docker build refuses."""
    out, at = [], 0
    while at < len(glob):
        char = glob[at]
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "\\":
            if at + 1 == len(glob):
                return None
            at += 1
            out.append(re.escape(glob[at]))
        elif char == "[":
            at += 1
            negated = at < len(glob) and glob[at] == "^"
            at += negated
            members = []
            while at < len(glob) and glob[at] != "]":
                if glob[at] == "\\":
                    at += 1
                    if at == len(glob):
                        return None
                    members.append(re.escape(glob[at]))
                elif glob[at] == "-" and members and at + 1 < len(glob) and glob[at + 1] != "]":
                    members.append("-")
                else:
                    members.append(re.escape(glob[at]))
                at += 1
            if at == len(glob) or not members:
                return None
            out.append(f"[{'^' if negated else ''}{''.join(members)}]")
        else:
            out.append(re.escape(char))
        at += 1
    return re.compile("".join(out))


def _ignore_files(dockerfile: str) -> tuple[str, ...]:
    """Where ``docker build`` looks for its ignore file, in order: ``<Dockerfile>.dockerignore`` beside the
    Dockerfile, then the context's ``.dockerignore``."""
    return f"{dockerfile}.dockerignore", ".dockerignore"


def _patterns(ignore_file: str) -> list[str]:
    """An ignore file's patterns, its comment lines left out as Docker leaves them out."""
    return [line for line in ignore_file.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _member_name(member: tarfile.TarInfo) -> str | None:
    """``member``'s path in the context; None for the context's own folder."""
    name = posixpath.normpath(member.name.lstrip("/"))
    return None if name == "." else name


class _Walk:
    """Lists a context folder in sorted order."""

    def __init__(self, top: Path, matcher: FilePatternMatcher | None, kept: set[str]):
        self.top = top
        self.matcher = matcher
        self.kept = kept
        self.found: dict[str, _Entry] = {}
        self.folders: set[str] = set()
        self.names: set[str] = set()
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
                    if (excluded and self._prunable(name)) or not self._claim(name):
                        continue
                    if not excluded:
                        self.folders.add(name)
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
        if self._claim(entry.name):
            self.found[entry.name] = entry

    def _claim(self, name: str) -> bool:
        """Whether ``name`` is still free: two names on disk can be one once normalized to NFC."""
        if name in self.names:
            self.problems.append(f"{name}: two entries have this name once normalized to NFC; rename one")
            return False
        self.names.add(name)
        return True

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


def _context_member(member: tarfile.TarInfo, out: str) -> tarfile.TarInfo | None:
    """``member`` as ``extract`` writes it, None to leave it out."""
    if not (member.isreg() or member.isdir() or member.issym()):
        raise ValueError(f"{member.name!r} in the build context is neither a file, a folder nor a link")
    try:
        member = tarfile.tar_filter(member, out)
    except tarfile.FilterError as e:
        raise ValueError(f"{member.name!r} in the build context: {e}") from e
    if member.issym():
        top = os.path.realpath(out)
        target = os.path.realpath(os.path.join(top, os.path.dirname(member.name), member.linkname))
        if os.path.commonpath([top, target]) != top:
            return None
    return member


def ignore_file(top: Path, dockerfile: Path | None) -> Path | None:
    """The ignore file ``docker build`` reads: ``<Dockerfile>.dockerignore`` beside the Dockerfile, else the
    context's ``.dockerignore``."""
    candidates = [dockerfile.with_name(f"{dockerfile.name}.dockerignore")] if dockerfile is not None else []
    return next((path for path in (*candidates, top / ".dockerignore") if path.is_file()), None)


def _inside(top: Path, path: Path) -> str | None:
    """``path``'s name in the context at ``top``, the folders above it resolved but not the file itself, so a link
    keeps its own name; None when it's outside the context."""
    named = Path(os.path.realpath(path.parent)) / path.name
    return unicodedata.normalize("NFC", named.relative_to(top).as_posix()) if named.is_relative_to(top) else None


def _show(parts: tuple[str, ...]) -> str:
    return "/".join(parts) or "."
