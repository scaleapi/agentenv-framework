"""Run a program with fds it inherits moved to the numbers it expects: ``python -I -S _with_fds.py <from>=<to> ... --
<argv>``. ``session.launch`` runs an app launched with ``fds`` through it, by path, so its server runs no Python between
its fork and its exec and this loads nothing but the standard library."""
from __future__ import annotations

import fcntl
import os
import sys


def main(argv: list[str]) -> None:
    split = argv.index("--")
    moves = [tuple(int(number) for number in pair.split("=")) for pair in argv[:split]]
    # Each source copied above every number named first, so no copy, source or target is another.
    above = max((number for move in moves for number in move), default=2) + 1
    copies = [(fcntl.fcntl(source, fcntl.F_DUPFD_CLOEXEC, above), target) for source, target in moves]
    for source, _ in moves:
        os.close(source)
    for copy, target in copies:
        os.dup2(copy, target)
        os.close(copy)
    os.execvp(argv[split + 1], argv[split + 1:])


if __name__ == "__main__":
    main(sys.argv[1:])
