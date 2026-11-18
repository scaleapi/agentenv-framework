"""Bootstrap that run_code writes into the sandbox and runs.

Imports the entry script, calls its entrypoint with the JSON input, and writes the
JSON result. Reads `run_dir`, `entrypoint` and `script_file` from argv.

Given a fourth argument, `timeout_seconds`, it first supervises: it reruns itself without
it in a new session and stops that whole process group at the limit, so no target needs
GNU `timeout`. Windows has no process groups to signal, so there it ends the rerun's process
tree with `taskkill`. While it supervises, it holds a lock on `runner.lock` in `run_dir`, its pid
in `runner.pid`.

`--clear <run_dir>` ends the supervisor holding that lock, with all it started, then removes the
folder; `--size <path>` prints a file's size. run_code runs these inside a computer, as its OS
may have no POSIX tools to do it with.

Both the entry script's own directory and `run_dir` go on sys.path, so a multi-file
artifact can import its siblings whether it is laid out flat or under a subdirectory.
Stdlib only, and Python 3.9 compatible: it runs in the sandbox, which may not have
agent_env installed, or anything newer than macOS's /usr/bin/python3.
"""

import sys

_TIMEOUT_EXIT_CODE = 124  # GNU timeout's, which run_code reports as a timeout
_GRACE_SECONDS = 5  # between SIGTERM and SIGKILL
_POLL_SECONDS = 0.1
_LOCK, _PID = "runner.lock", "runner.pid"


def _end_tree(pid):
    subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)


def _signal_group(pgid, signum):
    if os.name == "nt":
        _end_tree(pgid)
        return
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
    for name in ("SIGTERM", "SIGINT", "SIGHUP"):  # Windows has no SIGHUP
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), forward)
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
        _signal_group(child.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    status = child.wait()
    if timed_out:
        return _TIMEOUT_EXIT_CODE
    return status if status >= 0 else 128 - status


def _hold(run_dir):
    """Lock ``run_dir`` for this supervisor's life, its pid beside the lock."""
    with open(os.path.join(run_dir, _PID), "w") as pid:
        pid.write(str(os.getpid()))
    lock = open(os.path.join(run_dir, _LOCK), "wb")
    if os.name == "nt":
        msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def _held(path):
    """Whether another process holds the lock at ``path``."""
    with open(path, "r+b") as lock:
        try:
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
    return False


def _clear(run_dir):
    """End the supervisor still running from ``run_dir``, as a run whose caller died leaves one, with all it started,
    then remove the folder. A supervisor passes a SIGTERM on to its group; Windows ends the tree at once."""
    lock = os.path.join(run_dir, _LOCK)
    if os.path.exists(lock) and _held(lock):
        with open(os.path.join(run_dir, _PID)) as pid_file:
            pid = int(pid_file.read())
        if os.name == "nt":
            _end_tree(pid)
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + _GRACE_SECONDS * 2
        while _held(lock) and time.monotonic() < deadline:
            time.sleep(_POLL_SECONDS)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)


if sys.argv[1:2] in (["--clear"], ["--size"]):
    import os
    import shutil
    import signal
    import subprocess
    import time

    if os.name == "nt":
        import msvcrt
    else:
        import fcntl

    if sys.argv[1] == "--size":
        print(os.path.getsize(sys.argv[2]))
    else:
        _clear(sys.argv[2])
    sys.exit(0)

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

    if os.name == "nt":
        import msvcrt
    else:
        import fcntl

    lock = _hold(sys.argv[1])
    sys.exit(_supervise([sys.executable] + sys.argv[:4], float(sys.argv[4])))

# Below the supervisor, which must not import these.
import importlib.machinery
import importlib.util
import json
import os

# Absolute, as a computer's is not: the script may change directory before the runner writes its result.
run_dir, entrypoint, script_file = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]

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
