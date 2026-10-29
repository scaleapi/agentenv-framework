"""Run one of an app's ``@ready_check`` methods in the app's own venv, where its dependencies are installed:
``python -m agentenv_protocol.computer._ready <folder> <file> <class> <method>`` exits 0 when the method returns true,
1 when it returns false, and 2, saying why on stderr, when it can't run. The class reads its settings as it does when
it serves."""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    folder, file, name, method = argv
    try:
        sys.path.insert(0, folder)
        spec = importlib.util.spec_from_file_location("computer_app_module", Path(folder) / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = getattr(getattr(module, name)(), method)()
        if inspect.isawaitable(result):
            result = asyncio.run(result)
    except Exception as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
