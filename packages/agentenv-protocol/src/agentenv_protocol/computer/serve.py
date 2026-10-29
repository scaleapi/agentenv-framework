"""Serve a computer driver from its file, as a composed computer's program starts it:
``python -m agentenv_protocol.computer.serve <file>:<Class>``, run in the driver's folder, serves the class on
``MCP_PORT``. The file's folder is importable, so it may import its own modules."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .driver import ComputerDriverEnvironment

USAGE = "usage: python -m agentenv_protocol.computer.serve <file>:<Class>"


def main(argv: list[str]) -> None:
    if len(argv) != 1 or ":" not in argv[0]:
        raise SystemExit(USAGE)
    file, _, name = argv[0].partition(":")
    driver_class(Path(file), name)().serve()


def driver_class(file: Path, name: str) -> type[ComputerDriverEnvironment]:
    """The class ``name`` in ``file``, imported as a module of its own; exits saying why when it isn't a driver."""
    if not file.is_file():
        raise SystemExit(f"{file} isn't a file")
    sys.path.insert(0, str(file.resolve().parent))
    spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file.stem] = module
    spec.loader.exec_module(module)
    cls = getattr(module, name, None)
    if not (isinstance(cls, type) and issubclass(cls, ComputerDriverEnvironment)):
        raise SystemExit(f"{file} has no ComputerDriverEnvironment subclass named {name}")
    return cls


if __name__ == "__main__":
    main(sys.argv[1:])
