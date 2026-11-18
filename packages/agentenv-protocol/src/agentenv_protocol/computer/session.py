"""What a computer app's own server does in the desktop session, its OS's way: wait for the session, launch the app
into it in a group that holds whatever it starts, stop, pause and look for that group, add the app's menu launcher and
list the session's windows. Linux and macOS share the POSIX way (``_session_posix``); Windows has its own
(``_session_windows``). How a launched process exited is kept alike for both (``_reaping``)."""
from __future__ import annotations

import sys

from ._reaping import returncode

if sys.platform == "win32":
    from ._session_windows import add_launcher, alive, launch, paused, stop, wait_until_up, windows
else:
    from ._session_posix import add_launcher, alive, launch, paused, stop, wait_until_up, windows

__all__ = ["add_launcher", "alive", "launch", "paused", "returncode", "stop", "wait_until_up", "windows"]
