"""What cua-driver's driver asks a Hyprland desktop, through Hyprland's IPC (``hyprctl -j``): where keys go, and whether
cua-driver's Hyprland plugin can take input. driver.py imports it from beside itself, its folder importable wherever
the driver runs, and calls it only on Hyprland."""
from __future__ import annotations

import asyncio
import json
import subprocess

from agentenv_protocol.computer import DeviceUnavailable, ReadyCheck

#: How long a request to Hyprland's IPC waits for its answer.
TIMEOUT_S = 5


async def focused() -> dict:
    """Where keys go: the focused window, whose Hyprland address is its cua-driver window id, in the foreground through
    the plugin's seat, since a desktop-scope key with modifiers goes through libei, which Hyprland doesn't serve; with
    nothing focused, the desktop."""
    window = await asyncio.to_thread(hyprctl, "activewindow")
    if not window or not window.get("address"):
        return {}
    return {"pid": window["pid"], "window_id": int(window["address"], 16), "delivery_mode": "foreground"}


async def plugin() -> ReadyCheck:
    """That Hyprland has cua-driver's plugin loaded, enabled, built for it and taking input; one restarted after a
    crash runs in safe mode, with no plugins, where input is sent but never arrives, and a plugin loaded again into a
    running Hyprland answers but has no input seat until the desktop restarts. When Hyprland doesn't answer at all, as
    once it restarts under a new instance signature after the driver started, it says so instead."""
    status = await asyncio.to_thread(hyprctl, "cua:status")
    transport = (status or {}).get("transport") or {}
    if status is None and await asyncio.to_thread(hyprctl, "activeworkspace") is None:
        problem = ("Hyprland doesn't answer on its IPC; if it restarted after this driver started, restart the driver "
                   "with it")
    elif status is None:
        problem = "Hyprland hasn't loaded cua-driver's plugin"
    elif status.get("configured") is not True:
        problem = "cua-driver's plugin is loaded but not enabled"
    elif (status.get("abi") or {}).get("match") is not True:
        problem = "cua-driver's plugin was built for another Hyprland"
    elif transport.get("ready") is not True:
        problem = f"cua-driver's plugin isn't taking input: {transport.get('last_error') or 'its transport is down'}"
    else:
        return ReadyCheck("cua-driver plugin", True, "loaded, enabled, built for this Hyprland and taking input")
    return ReadyCheck("cua-driver plugin", False, problem)


async def require_plugin() -> None:
    """Input refused unless cua-driver's plugin can take it: the runtime runs ready checks only until they first pass,
    and input sent while Hyprland runs without the plugin is accepted but never arrives."""
    if not (check := await plugin()).ok:
        raise DeviceUnavailable(check.detail)


def hyprctl(request: str) -> dict | None:
    """Hyprland's JSON answer to ``request``, or None when it gives none, as for a plugin's request it doesn't know or
    when Hyprland doesn't answer."""
    try:
        out = subprocess.run(["hyprctl", "-j", request], capture_output=True, text=True, timeout=TIMEOUT_S)
        answer = json.loads(out.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    return answer if isinstance(answer, dict) else None
