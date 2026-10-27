"""Read a computer driver class where its own dependencies may not be installed, as a ``computer_driver`` env's put
does: ``python -m agentenv_protocol.computer._declare <folder> <file> <class>`` imports the file with every module
not installed here standing in as an inert stub, and prints what the class declares for each platform it supports as
JSON. Nothing runs beyond defining the module and its classes, and a class that can't be defined even so exits 1
with why."""
from __future__ import annotations

import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import sys
import types
from pathlib import Path

from .card import COMPUTER
from .driver import ComputerDriverEnvironment
from .platform import supported


class _StubMeta(type):
    def __getattr__(cls, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return cls

    def __call__(cls, *args, **kwargs):
        return cls

    def __getitem__(cls, key):
        return cls


class _Stub(metaclass=_StubMeta):
    """Whatever a stubbed module holds: itself again on any attribute, call or index, and a class to derive from."""


class _StubModule(types.ModuleType):
    __path__: list = []

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub


class _Stubs(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Last on ``sys.meta_path``: a module no other finder finds imports as a stub."""

    def find_spec(self, fullname, path, target=None):
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module) -> None:
        return None


def declare(folder: Path, file: str, name: str) -> dict:
    """What the class ``name`` in ``folder/file`` declares: its card's name, its ``supports``, and for each platform
    those name, the extensions it provides, its tools and the computer/v1 features. Raises ValueError for a class
    that isn't a driver or declares no platform."""
    sys.path.insert(0, str(folder))
    sys.meta_path.append(_Stubs())
    spec = importlib.util.spec_from_file_location("computer_driver_module", folder / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, name, None)
    if not (isinstance(cls, type) and issubclass(cls, ComputerDriverEnvironment)):
        raise ValueError(f"{file} has no ComputerDriverEnvironment subclass named {name}")
    platforms = [platform for entry in cls.supports for platform in supported(entry)]
    if not platforms:
        raise ValueError(f"{name} declares no platform: name them with @computer_driver(supports=[...])")
    cards = [(platform, cls.declaration(platform)) for platform in platforms]
    return {"name": cards[0][1].name, "supports": list(cls.supports), "platforms": [
        {"platform": platform.as_dict(), "provides": [e.uri for e in card.capabilities.extensions],
         "tools": [t.name for t in card.capabilities.tools],
         "features": next(e.params for e in card.capabilities.extensions if e.uri == COMPUTER)}
        for platform, card in cards]}


def main(argv: list[str]) -> int:
    """Print the declaration as the only output; what the module prints while it loads goes to stderr."""
    folder, file, name = argv
    try:
        with contextlib.redirect_stdout(sys.stderr):
            declared = declare(Path(folder), file, name)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(json.dumps(declared))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
