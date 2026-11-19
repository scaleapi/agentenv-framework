"""OSWorld tasks on agent-env computers, and the ``osworld`` bundle agentenv-framework ships.

``osworld_initialize`` sets a task up on a computer that runs the ``osworld-server`` app and ``osworld_evaluate`` grades
it, both in the worker, with upstream OSWorld's own setup, getters and metrics (:mod:`.runtime`, from :mod:`.upstream`)
reaching the computer's guest server through loopback forwarders (:mod:`.forwarder`) with the env's control token,
which the agent doesn't hold (:mod:`.guest`). What upstream imports is this
folder's ``requirements.txt``, installed where the steps run.

This folder is also the ``osworld`` bundle (``README.md``), whose ``osworld-server`` computer app (``envs/``), which
runs upstream OSWorld's guest server, fetched at ``OSWORLD_COMMIT`` and checked by sha256, a task names by its alias
(``aliases.toml``)."""
from __future__ import annotations

from pathlib import Path

#: The upstream OSWorld commit the guest server and the steps' checks are from (xlang-ai/OSWorld, Apache-2.0).
OSWORLD_COMMIT = "5123bfdbbc34b947bcffcff564abc62d0ab7072d"
#: The app's ``environment_name``.
APP_NAME = "osworld-server"
#: The extension the app's card advertises at the prefix its server forwards to the guest server under. A computer
#: nests the app's card under its ``children_environments`` with the endpoint rooted where it serves the app, so a step
#: finds the guest server by this URI in the computer's card.
GUEST_EXTENSION = "urn:agentenv:osworld-guest/v1"
#: The prefix the app's server forwards to the guest server under.
PROXY_PREFIX = "/osworld"
#: The bundle's folder, this package's.
BUNDLE_DIR = Path(__file__).parent
#: The app's folder.
APP_DIR = BUNDLE_DIR / "envs" / APP_NAME
