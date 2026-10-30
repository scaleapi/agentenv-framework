"""Run one of a computer app's ``@install_step`` methods, as its computer's build does:
``python -m agentenv_protocol.computer.run_install_step <file>:<Class> <method>``, run from the app's venv in its folder,
offline. The class's ``__init__`` isn't run, since no deploy's settings are known at build: ``self.settings`` holds only
the Settings' defaults. Needs only the plain package."""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

from ._load import load_class
from .app import ComputerAppEnvironment

USAGE = "usage: python -m agentenv_protocol.computer.run_install_step <file>:<Class> <method>"


def main(argv: list[str]) -> None:
    if len(argv) != 2 or ":" not in argv[0]:
        raise SystemExit(USAGE)
    file, _, name = argv[0].partition(":")
    cls = load_class(Path(file), name, ComputerAppEnvironment)
    if argv[1] not in {step.name for step in cls.declaration().install_steps}:
        raise SystemExit(f"{cls.__name__} has no @install_step method {argv[1]!r}")
    app = cls.__new__(cls)
    app.settings = cls.Settings.model_construct()
    done = getattr(app, argv[1])()
    if inspect.isawaitable(done):
        asyncio.run(done)


if __name__ == "__main__":
    main(sys.argv[1:])
