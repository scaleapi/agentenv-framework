"""Bootstrap that run_code writes into the sandbox and runs.

Imports the entry script, calls its entrypoint with the JSON input, and writes the
JSON result. Reads `run_dir`, `entrypoint` and `script_file` from argv.

Given a fourth argument, `timeout_seconds`, it first supervises: it reruns itself without
it in a new session and stops that whole process group at the limit, so no target needs
GNU `timeout`.

Both the entry script's own directory and `run_dir` go on sys.path, so a multi-file
artifact can import its siblings whether it is laid out flat or under a subdirectory.
Stdlib only, and Python 3.9 compatible: it runs in the sandbox, which may not have
agent_env installed, or anything newer than macOS's /usr/bin/python3.
"""

import sys

_TIMEOUT_EXIT_CODE = 124  # GNU timeout's, which run_code reports as a timeout
_GRACE_SECONDS = 5  # between SIGTERM and SIGKILL
_POLL_SECONDS = 0.1


def _signal_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except (ProcessLookupError, PermissionError):  # gone, or (macOS) only zombies are left
        pass


def _supervise(command, timeout_seconds):
    """Run `command` in a new session; return its exit status, or 124 if it hit the limit.

    At the limit, or on a signal forwarded to it, the group gets that signal (SIGTERM at the
    limit), then SIGKILL once the child exits or the grace period runs out.
    """
    child = None
    early = []  # signals that came before the child did
    deadline = time.monotonic() + timeout_seconds
    kill_at = None
    timed_out = False

    def forward(signum, _frame):
        nonlocal kill_at
        if child is None:
            early.append(signum)
        else:
            _signal_group(child.pid, signum)
        kill_at = kill_at or time.monotonic() + _GRACE_SECONDS

    # Caught before the child starts, as GNU timeout does, so it gets their defaults: one ignored here (a background
    # job's SIGINT) would stay ignored in it.
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, forward)
    child = subprocess.Popen(command, start_new_session=True)
    for signum in early:
        _signal_group(child.pid, signum)
    while child.poll() is None:
        now = time.monotonic()
        if kill_at is not None and now >= kill_at:
            break
        if not timed_out and now >= deadline:
            timed_out = True
            _signal_group(child.pid, signal.SIGTERM)
            kill_at = kill_at or now + _GRACE_SECONDS
        try:
            child.wait(_POLL_SECONDS)
        except subprocess.TimeoutExpired:
            pass
    if kill_at is not None:
        _signal_group(child.pid, signal.SIGKILL)
    status = child.wait()
    if timed_out:
        return _TIMEOUT_EXIT_CODE
    return status if status >= 0 else 128 - status


if len(sys.argv) == 5:
    # Only the supervisor imports these, and not from the run dir (sys.path[0]): an artifact member named after one
    # would otherwise run here, outside the limit. The script's own process imports just what it always has.
    import os  # loaded at startup, so never from the run dir

    # Python puts the realpath there, and the run dir is often reached through a symlink (macOS's /tmp).
    here = os.path.dirname(os.path.realpath(__file__))
    sys.path[:] = [entry for entry in sys.path if os.path.realpath(entry or ".") != here]
    import signal
    import subprocess
    import time

    sys.exit(_supervise([sys.executable] + sys.argv[:4], float(sys.argv[4])))

# Below the supervisor, which must not import these.
import importlib.machinery
import importlib.util
import json
import os

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
