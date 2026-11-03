"""Import a class from a file of a driver's or an app's folder, as its computer's program starts it."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_class(file: Path, name: str, base: type) -> type:
    """The class ``name`` in ``file``, imported as a module of its own with the file's folder importable; exits saying
    why when it isn't a subclass of ``base``."""
    if not file.is_file():
        raise SystemExit(f"{file} isn't a file")
    sys.path.insert(0, str(file.resolve().parent))
    spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file.stem] = module
    spec.loader.exec_module(module)
    cls = getattr(module, name, None)
    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise SystemExit(f"{file} has no {base.__name__} subclass named {name}")
    return cls
