"""Serve a computer driver from its file, as a composed computer's program starts it:
``python -m agentenv_protocol.computer.serve <file>:<Class>``, run in the driver's folder, serves the class on
``MCP_PORT``. The file's folder is importable, so it may import its own modules."""
from __future__ import annotations

import sys
from pathlib import Path

from ._load import load_class
from .driver import ComputerDriverEnvironment

USAGE = "usage: python -m agentenv_protocol.computer.serve <file>:<Class>"


def main(argv: list[str]) -> None:
    if len(argv) != 1 or ":" not in argv[0]:
        raise SystemExit(USAGE)
    file, _, name = argv[0].partition(":")
    driver_class(Path(file), name)().serve()


def driver_class(file: Path, name: str) -> type[ComputerDriverEnvironment]:
    """The class ``name`` in ``file``, imported as a module of its own; exits saying why when it isn't a driver."""
    return load_class(file, name, ComputerDriverEnvironment)


if __name__ == "__main__":
    main(sys.argv[1:])
