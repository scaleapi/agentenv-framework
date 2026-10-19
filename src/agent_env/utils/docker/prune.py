"""The part of a stored build context each image's build reads, written into a folder of its own."""

from __future__ import annotations

import bisect
import functools
import heapq
import os
import posixpath
import re
import shutil
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from modal import FilePatternMatcher

from agent_env.utils.docker.context import context_member, ignore_matcher, ignore_names
from agent_env.utils.docker.dockerfile import ContextReads, context_reads
from agent_env.utils.paths import LEAVINGS

_WILDCARD = re.compile(r"[*?\[]")
_LARGEST = 5


@dataclass(frozen=True)
class Pruned:
    """What ``write_pruned`` wrote into one folder, and why it wrote all the context sends when it did."""

    dockerfile: str
    files: int
    size: int
    whole: str | None
    largest: tuple[tuple[str, int], ...] = ()


def write_pruned(archive: Path, images: Mapping[Path, str | None]) -> dict[Path, Pruned]:
    """Write into each folder what its Dockerfile's build reads of the context tar.gz (None: the one it finds)."""
    links, dockerfiles, texts = _read(archive, images)
    plans = {folder: _plan(dockerfile, links, texts) for folder, dockerfile in dockerfiles.items()}
    written = _write(archive, {folder: kept for folder, (kept, _) in plans.items()})
    return {folder: Pruned(dockerfiles[folder], len(files), sum(files.values()), plans[folder][1],
                           tuple(heapq.nlargest(_LARGEST, files.items(), key=lambda file: file[1])))
            for folder, files in written.items()}


def _read(archive: Path,
          images: Mapping[Path, str | None]) -> tuple[dict[str, str | None], dict[Path, str], dict[str, str]]:
    """The archive's names to their link targets, each image's Dockerfile, and its Dockerfile and ignore file texts."""
    dockerfiles = {folder: posixpath.normpath(dockerfile) for folder, dockerfile in images.items() if dockerfile}
    wanted = {name for dockerfile in dockerfiles.values() for name in (dockerfile, *ignore_names(dockerfile))}
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
        wanted |= {dockerfiles[folder], *ignore_names(dockerfiles[folder])}
    if unread := {name for name in wanted if name in links and not links[name] and name not in texts}:
        texts |= _texts(archive, unread)
    # A Dockerfile or ignore file stored as a link is read where it leads, as docker build reads it.
    linked = {name: _resolve(name, links)[0] for name in wanted if links.get(name)}
    if unread := {target for target in linked.values() if target not in texts}:
        texts |= _texts(archive, unread)
    return links, dockerfiles, texts | {name: texts[target] for name, target in linked.items() if target in texts}


def _plan(dockerfile: str, links: Mapping[str, str | None], texts: Mapping[str, str]) -> tuple[set[str], str | None]:
    """The names a build of ``dockerfile`` keeps, and why it keeps all the context sends when it does."""
    if dockerfile not in texts:
        raise ValueError(f"the build context has no {dockerfile}")
    ignore = next((name for name in ignore_names(dockerfile) if name in texts), None)
    matcher = ignore_matcher(texts[ignore]) if ignore else None
    reads, crossed = _through_links(context_reads(texts[dockerfile]), links)
    # A source the reading found that names nothing here is likelier a misreading than a build that fails.
    if missing := next((path for path in reads.paths if not any(_matches(path, name) for name in links)), None):
        reads, crossed = ContextReads(whole=f"{missing} matches nothing in the context"), set()
    return _kept(links, reads, dockerfile, matcher, crossed), reads.whole


def _write(archive: Path, kept: Mapping[Path, set[str]]) -> dict[Path, dict[str, int]]:
    """Write each folder's kept names from the archive, and each folder's files by name with their sizes."""
    # By name, since a name the archive holds twice is written over: what's left is its last entry.
    written: dict[Path, dict[str, int]] = {folder: {} for folder in kept}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = _member_name(member)
            first: Path | None = None
            for folder in (folder for folder, names in kept.items() if name in names):
                if member.isreg() and first is not None:
                    _check_inside(folder, name)
                    (folder / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(first, folder / name)
                else:
                    folder.mkdir(parents=True, exist_ok=True)
                    tar.extract(member, folder, filter=_member)
                    first = folder / name if member.isreg() else None
                if member.isreg():
                    written[folder][name] = member.size
                else:
                    written[folder].pop(name, None)
    # A folder the archive only implies, as the parent of what it holds, has no member of its own.
    for folder, names in kept.items():
        for name in names:
            if not os.path.lexists(folder / name):
                _check_inside(folder, name)
                (folder / name).mkdir(parents=True, exist_ok=True)
    return written


def _check_inside(folder: Path, name: str) -> None:
    """Raise ValueError when ``name`` in ``folder``, through any link written there, would land outside it."""
    top = os.path.realpath(folder)
    if os.path.commonpath([top, os.path.realpath(folder / name)]) != top:
        raise ValueError(f"{name!r} in the build context leads out of {folder}")


def _member(member: tarfile.TarInfo, out: str) -> tarfile.TarInfo | None:
    """``member`` as ``write_pruned`` writes it: a link as it is, wherever it leads, and at the time of writing."""
    if (member := context_member(member, out, links_out=True)) is None:
        return None
    # A stored context's times are fixed, and docker build skips sending a file whose time and size it has seen before.
    return member.replace(mtime=None, deep=False)


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


def _through_links(reads: ContextReads, links: Mapping[str, str | None]) -> tuple[ContextReads, set[str]]:
    """``reads`` with each source that names nothing followed through the links on its path, and the links crossed."""
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
    """``path`` with its links followed as docker build follows them, and the links crossed; as it is when they loop."""
    resolved, crossed = "", set()
    for part in path.split("/"):
        resolved = posixpath.join(resolved, part)
        while links.get(resolved) is not None:
            if resolved in crossed:
                return path, set()
            crossed.add(resolved)
            resolved = _link_target(resolved, links)
    return resolved, crossed


def _link_target(name: str, links: Mapping[str, str | None]) -> str:
    """Where the link ``name`` leads, as docker build follows it in the context: ``/`` and ``..`` stop at its root."""
    target = links[name]
    parts = [] if target.startswith("/") or "/" not in name else name.split("/")[:-1]
    for part in target.split("/"):
        if part == "..":
            parts = parts[:-1]
        elif part not in ("", "."):
            parts.append(part)
    return "/".join(parts)


def _kept(links: Mapping[str, str | None], reads: ContextReads, dockerfile: str, matcher: FilePatternMatcher | None,
          crossed: set[str] = frozenset()) -> set[str]:
    """The names in ``links``, and the folders they imply, that ``write_pruned`` keeps for ``reads``."""
    def sent(name: str) -> bool:
        return not LEAVINGS.intersection(name.split("/")) and (matcher is None or not matcher(Path(name)))

    implied = {str(parent) for name in links for parent in PurePosixPath(name).parents if str(parent) != "."}
    candidates = links.keys() | implied
    if reads.whole is not None:
        kept = {name for name in candidates if sent(name)}
    else:
        kept = {name for name in candidates if sent(name) and any(_matches(path, name) for path in reads.paths)}
    # docker build reads the Dockerfile and ignore file where they lead, whatever the ignore file says. Both ignore files
    # stay, since docker build's frontends differ on which one an empty <Dockerfile>.dockerignore leaves in effect.
    for name in (dockerfile, *ignore_names(dockerfile)):
        resolved, hops = _resolve(name, links)
        kept |= {other for other in (name, resolved, *hops) if other in links}
    # A link a source only crosses is needed for its path, not for all it leads to.
    pending = [name for name in kept if links.get(name)]
    kept |= {name for name in crossed if sent(name)}
    names = sorted(links)
    while pending:
        # A target is followed through the links on its way, which stay for its path.
        target, hops = _resolve(_link_target(pending.pop(), links), links)
        kept |= {hop for hop in hops if sent(hop)}
        for other in _at_or_under(names, target) if target else names:
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


@functools.lru_cache(maxsize=1024)
def _part_pattern(glob: str) -> re.Pattern[str] | None:
    """A path part's wildcard as docker build matches it; None when docker build refuses it as malformed."""
    out, at = [], 0
    while at < len(glob):
        char = glob[at]
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            at += 1
            negated = at < len(glob) and glob[at] == "^"
            at += negated
            ranges = []
            while not (at < len(glob) and glob[at] == "]" and ranges):
                # A class's characters as Go's filepath.Match reads them: a range's ends neither - nor ], low to high.
                if at + 1 >= len(glob) or glob[at] in "-]":
                    return None
                low = high = glob[at]
                at += 1
                if glob[at] == "-":
                    if at + 2 >= len(glob) or glob[at + 1] in "-]":
                        return None
                    high, at = glob[at + 1], at + 2
                if low > high:
                    return None
                ranges.append(re.escape(low) if low == high else f"{re.escape(low)}-{re.escape(high)}")
            out.append(f"[{'^' if negated else ''}{''.join(ranges)}]")
        else:
            out.append(re.escape(char))
        at += 1
    return re.compile("".join(out))


def _member_name(member: tarfile.TarInfo) -> str | None:
    """``member``'s path in the context; None for the context's own folder. Raises ValueError for one leading out."""
    name = posixpath.normpath(member.name.lstrip("/"))
    if name == ".." or name.startswith("../"):
        raise ValueError(f"{member.name!r} in the build context leads out of it")
    return None if name == "." else name
