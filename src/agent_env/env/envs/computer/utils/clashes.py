"""Whether a part of a composed computer, its driver or an app, undid what an earlier part installed: what each part's
run scripts and install steps wrote, as ``layers_in_image`` lists it per layer, compared in build order. Only certain
losses are refused; any other file two parts wrote differently is only said, since it may be an edit that kept the
earlier part's change."""

from __future__ import annotations

import re
from typing import NamedTuple

#: Where what a run script or an install step writes may be lost, as the filesystem hierarchy standard has it:
#: temporary files, runtime state, logs and caches. Never compared.
SCRATCH = re.compile(r"^/(tmp|var/tmp|var/log|var/cache|run)/|/\.cache/|/__pycache__/")
#: What each kind of entry is, to a reader: a program is a file.
_KINDS = {"file": "file", "exec": "file", "link": "link", "dir": "folder"}


class Entry(NamedTuple):
    """A path of a layer, as ``layers_in_image`` lists it: its kind, and a file's or a program's sha256 or a link's
    target."""

    kind: str
    what: str = ""

    @property
    def known(self) -> bool:
        """Whether its content is known: not a hard link to a file of an earlier layer, which the reader doesn't read."""
        return not self.what.startswith("=> ")

    @classmethod
    def listed(cls, entry: str) -> Entry:
        kind, _, what = entry.partition(" ")
        return cls(kind, what)


def compare(layers: list[tuple[str, dict[str, str]]]) -> tuple[list[str], list[str]]:
    """What each of ``layers``, a part's id and what one of its commands wrote, in build order, did to what another part
    left, but under ``SCRATCH``: each certain loss (``_lost``), then each other file two parts wrote differently. A path
    is compared with the last other part to write it, whoever wrote it since, until it's deleted or lost."""
    entries: dict[str, Entry] = {}
    writers: dict[str, list[str]] = {}
    lost, differ = [], []
    for of, wrote in layers:
        listed = {path: Entry.listed(entry) for path, entry in wrote.items() if not SCRATCH.search(path)}
        for path, entry in sorted(listed.items()):
            if entry.kind == "opaque":
                for hidden in [hidden for hidden in sorted(entries) if hidden.startswith(f"{path}/")
                               and hidden not in listed]:
                    if entries[hidden].kind not in ("dir", "deleted") and (by := _other(writers[hidden], of)):
                        lost.append(f"{hidden}: {of} deletes {by}'s file")
                    entries[hidden], writers[hidden] = Entry("deleted"), []
                entry = Entry("dir")
            if entry.kind == "deleted":
                for gone in [gone for gone in entries if gone.startswith(f"{path}/") and gone not in listed]:
                    entries[gone], writers[gone] = Entry("deleted"), []
            reason = None
            if path in entries and (by := _other(writers[path], of)):
                if reason := _lost(path, of, entry, by, entries[path]):
                    lost.append(reason)
                elif _differs(entry, entries[path]):
                    differ.append(f"{path} ({by}, {of})")
            entries[path] = entry
            writers[path] = ([] if entry.kind == "deleted" else
                             [of] if reason else [*(writer for writer in writers.get(path, []) if writer != of), of])
    return lost, differ


def _other(writers: list[str], of: str) -> str | None:
    """The part of ``writers``, oldest write first, that wrote a path last before ``of``."""
    return next((writer for writer in reversed(writers) if writer != of), None)


def _lost(path: str, of: str, entry: Entry, by: str, was: Entry) -> str | None:
    """How ``of``'s ``entry`` at ``path`` certainly loses what ``by`` left there, ``was``: deleting it, making it
    another kind, pointing its link elsewhere, or replacing its program; None when it doesn't."""
    if was.kind == "deleted":
        return None
    if entry.kind == "deleted":
        return f"{path}: {of} deletes {by}'s {_KINDS[was.kind]}"
    if _KINDS[entry.kind] != _KINDS[was.kind]:
        return f"{path}: {of} makes {by}'s {_KINDS[was.kind]} a {_KINDS[entry.kind]}"
    if was.kind == "link" and entry.what != was.what:
        return f"{path}: {of} points {by}'s link elsewhere"
    if was.kind == "exec" and entry.what != was.what and entry.known and was.known:
        return f"{path}: {of} replaces {by}'s program"
    return None


def _differs(entry: Entry, was: Entry) -> bool:
    """Whether ``entry`` is a file that may hold other content than ``was``, a file, or a program whose content isn't
    known."""
    return (entry.kind in ("file", "exec") and entry.what != was.what
            and (was.kind == "file" or (was.kind == "exec" and not (entry.known and was.known))))
