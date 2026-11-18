"""The integration tier's setup shared by every test."""

import os
import tempfile
from pathlib import Path

import pytest

from agent_env.store import local_state

#: The locks that guard this machine's Docker and image registry, by the start of their ids: a run's tests keep stores
#: and state roots of their own, but all of them, on every pytest worker, use the one Docker and registry.
MACHINE_LOCKS = ("computer-compose:", "linux-disk:", "windows-disk:", "omarchy-disk:", "disk-export:", "image-build:",
                 "image-pull:", "image-save:")


@pytest.fixture(scope="session", autouse=True)
def machine_locks_shared_by_the_runs_workers():
    """Each test's state root, where agent-env keeps its locks, is its own, so two pytest workers composing one
    computer took no lock in common and both built it under the same tags: the second moved them off the image the
    first had pinned, or their pushes collided. The locks guarding this machine's Docker and registry are kept in one
    folder for the run, as one user's processes keep theirs, so such work takes turns and the second reuses the first's.
    """
    run = os.environ.get("PYTEST_XDIST_TESTRUNUID") or str(os.getpid())
    folder = Path(tempfile.gettempdir()) / f"agent-env-machine-locks-{run}"
    folder.mkdir(mode=0o700, exist_ok=True)
    per_test = local_state._lock_path
    patch = pytest.MonkeyPatch()
    patch.setattr(local_state, "_lock_path",
                  lambda id: folder / per_test(id).name if id.startswith(MACHINE_LOCKS) else per_test(id))
    yield
    patch.undo()
