"""Bootstrap that run_code writes into the sandbox and runs.

Imports the entry script, calls its entrypoint with the JSON input, and writes the
JSON result. Reads `run_dir`, `entrypoint` and `script_file` from argv.

Both the entry script's own directory and `run_dir` go on sys.path, so a multi-file
artifact can import its siblings whether it is laid out flat or under a subdirectory.
Stdlib only — it runs in the sandbox, which may not have agent_env installed.
"""

import importlib.machinery
import importlib.util
import json
import os
import sys

run_dir, entrypoint, script_file = sys.argv[1], sys.argv[2], sys.argv[3]

script_path = os.path.join(run_dir, script_file)
# The entry's own directory first: a nested entry (`src/run.py`) imports siblings by
# bare name (`import helpers`), which run_dir alone does not resolve. dict.fromkeys
# collapses the two for a flat layout, where they are the same directory.
sys.path[:0] = list(dict.fromkeys([os.path.dirname(script_path), run_dir]))

# `from .helpers import value` needs the entry to belong to a package, which loading it
# from a file path alone does not give it. The package is bound to a name we own rather
# than to the staged directory's own name: importing `os.run` for a member `os/run.py`
# would find the runner's own already-imported `os` and fail, and a member directory is
# under no obligation to be a legal module name in the first place.
_MODULE, _PACKAGE = "agent_env_user_script", "agent_env_user_pkg"
if os.path.dirname(script_file):
    pkg_dir = os.path.dirname(script_path)
    pkg_spec = importlib.machinery.ModuleSpec(_PACKAGE, None, is_package=True)
    pkg_spec.submodule_search_locations = [pkg_dir]
    sys.modules[_PACKAGE] = importlib.util.module_from_spec(pkg_spec)
    name = f"{_PACKAGE}.{os.path.basename(script_file)[: -len('.py')]}"
else:
    name = _MODULE

spec = importlib.util.spec_from_file_location(name, script_path)
mod = importlib.util.module_from_spec(spec)
# Registered before exec so the entry can also be reached as a sibling's import target.
sys.modules[name] = mod
spec.loader.exec_module(mod)

with open(f"{run_dir}/input.json") as f:
    _input = json.load(f)

_result = getattr(mod, entrypoint)(_input)

with open(f"{run_dir}/output.json", "w") as f:
    json.dump(_result, f)
