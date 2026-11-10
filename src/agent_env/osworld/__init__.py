"""OSWorld tasks on agent-env computers: the ``osworld-server`` computer app, which runs the OSWorld guest server on a
computer, and what the ``osworld_*`` steps share to reach it and the OSWorld controller deployed beside it.

The guest server's source isn't part of agent-env; :func:`stage_server` copies the app's folder with that source into a
folder ``agent-env env put`` takes."""
from __future__ import annotations

import shutil
from pathlib import Path

#: The app's ``environment_name``.
APP_NAME = "osworld-server"
#: The extension the app's card advertises at the prefix its server forwards to the guest server under. A computer
#: nests the app's card under its ``children_environments`` with the endpoint rooted where it serves the app, so a step
#: finds the guest server by this URI in the computer's card.
GUEST_EXTENSION = "urn:agentenv:osworld-guest/v1"
#: The prefix the app's server forwards to the guest server under.
PROXY_PREFIX = "/osworld"
#: The app's folder.
APP_DIR = Path(__file__).parent / "apps" / APP_NAME
#: Where a staged app keeps the guest server's source, and the files of that source it runs.
PAYLOAD_DIR = "payload"
PAYLOAD_FILES = ("main.py", "pyxcursor.py")


def stage_server(source: Path, dest: Path) -> Path:
    """Copy the app's folder to ``dest`` with the guest server's source from ``source``, a folder holding
    ``PAYLOAD_FILES``, under ``payload/``; returns ``dest``."""
    if missing := [name for name in PAYLOAD_FILES if not (source / name).is_file()]:
        raise FileNotFoundError(f"{source} is not the OSWorld guest server's source: no {missing}")
    if dest.exists():
        raise FileExistsError(f"{dest} exists")
    shutil.copytree(APP_DIR, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (dest / PAYLOAD_DIR).mkdir()
    for name in PAYLOAD_FILES:
        shutil.copy2(source / name, dest / PAYLOAD_DIR / name)
    return dest
