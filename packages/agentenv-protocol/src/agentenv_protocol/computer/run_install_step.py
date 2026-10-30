"""Run one of a computer app's ``@install_step`` methods, as its computer's build does:
``python -m agentenv_protocol.computer.run_install_step <file>:<Class> <method>``, run from the app's venv in its
folder, offline. The class's ``__init__`` isn't run: the step has ``self.build_settings``, the values its env fixed,
and no deploy's ``self.settings``, which aren't known at build. Needs only the plain package."""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import NoReturn

from ._load import load_class
from .app import ComputerAppEnvironment, _built_settings

USAGE = "usage: python -m agentenv_protocol.computer.run_install_step <file>:<Class> <method>"


class _NoDeploy:
    """What ``self.settings`` is to an install step: a deploy's settings, which no build knows."""

    def __getattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"self.settings.{name}: an install step runs at build, before any deploy gives the app "
                             "its settings; it reads self.build_settings")


def main(argv: list[str]) -> None:
    if len(argv) != 2 or ":" not in argv[0]:
        raise SystemExit(USAGE)
    file, _, name = argv[0].partition(":")
    cls = load_class(Path(file), name, ComputerAppEnvironment)
    if argv[1] not in {step.name for step in cls.declaration().install_steps}:
        raise SystemExit(f"{cls.__name__} has no @install_step method {argv[1]!r}")
    app = cls.__new__(cls)
    app.settings, app.build_settings = _NoDeploy(), _built_settings(cls)
    done = getattr(app, argv[1])()
    if inspect.isawaitable(done):
        asyncio.run(done)


if __name__ == "__main__":
    main(sys.argv[1:])
