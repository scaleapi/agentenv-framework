"""What cua-driver's driver asks a Hyprland desktop, through Hyprland's IPC (``hyprctl -j``): where keys go, and whether
cua-driver's Hyprland plugin can take input. driver.py imports it from beside itself, its folder importable wherever
the driver runs, and calls it only on Hyprland."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import string
import subprocess

from agentenv_protocol.computer import DeviceUnavailable, ReadyCheck

#: How long a request to Hyprland's IPC waits for its answer.
TIMEOUT_S = 5
#: Why a computer whose Hyprland restarted under its driver has failed for good: Hyprland's watchdog brings it back after
#: a crash, in safe mode with no plugins, within the same session, so nothing restarts the driver with it.
REPLACED = ("Hyprland restarted under this driver, as its watchdog does after a crash, in safe mode with no plugins, so "
            "the computer can't take input again: an infrastructure failure")
#: How keys and text go to the desktop: in the foreground, through the plugin's seat, so Hyprland's own shortcuts see
#: them first and then whatever has the keyboard gets them, a window or a layer such as a menu. Without it a key with
#: modifiers goes through libei, which Hyprland doesn't serve.
DESKTOP = {"delivery_mode": "foreground"}
#: X's keysym names for ASCII's punctuation, by which cua-driver presses those keys at the desktop: given the character
#: there, it has no evdev key for it and falls back to libei, which can hang its input worker.
PUNCTUATION = {
    "!": "exclam", '"': "quotedbl", "#": "numbersign", "$": "dollar", "%": "percent", "&": "ampersand",
    "'": "apostrophe", "(": "parenleft", ")": "parenright", "*": "asterisk", "+": "plus", ",": "comma", "-": "minus",
    ".": "period", "/": "slash", ":": "colon", ";": "semicolon", "<": "less", "=": "equal", ">": "greater",
    "?": "question", "@": "at", "[": "bracketleft", "\\": "backslash", "]": "bracketright", "^": "asciicircum",
    "_": "underscore", "`": "grave", "{": "braceleft", "|": "bar", "}": "braceright", "~": "asciitilde",
}
#: The key each character a US keyboard types with Shift is on: a key aimed at a window is one of the seat's US layout,
#: the only layout cua-driver's plugin takes, and a character it has no key for, or an uppercase letter's, it refuses or
#: types unshifted.
SHIFTED = {
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8", "(": "9", ")": "0", "_": "-",
    "+": "=", "{": "[", "}": "]", "|": "\\", ":": ";", '"': "'", "<": ",", ">": ".", "?": "/", "~": "`",
    **{letter: letter.lower() for letter in string.ascii_uppercase},
}
#: What the computer skill tells an agent of how its keys go here: the desktop's shortcuts are Super chords, which the
#: driver sends to Hyprland, and every other key goes to the focused window, past Hyprland's own binds.
NOTES = ("Keys held with Super go to the desktop, where its shortcuts take them; every other key or chord goes to the "
         "focused window, so a desktop shortcut without Super, such as Alt+Tab, doesn't fire.")
#: What cua-driver says when a key aimed at a window is refused because a layer, such as a menu, holds the keyboard.
GRABBED = "foreground_grab"
#: What it says when one is refused because the seat's keymap isn't the US layout's: an input method such as Omarchy's
#: fcitx5 takes on the keymap that desktop text, or a desktop key the layout has no key for, uploads.
UNLAID = "unsupported_layout"
#: A chord at the desktop that puts the US layout's keymap back: its keys are all the layout's, so cua-driver sends it
#: with that keymap, which an input method takes on in turn, and it's bound to nothing (RESET_BIND), so Hyprland keeps
#: it from every window.
RESET = {"modifiers": ["super", "ctrl", "alt", "shift"], "key": "f12"}
RESET_BIND = "SUPER + CTRL + ALT + SHIFT + F12"
#: RESET_BIND as Hyprland lists its binds: Super, Alt, Ctrl and Shift's mask, and its key.
RESET_LISTED = (64 | 8 | 4 | 1, "F12")
#: How long a key aimed at a window waits for the keymap RESET puts back, and how often it's sent again meanwhile.
RESET_S, RESET_POLL_S = 1.0, 0.05


def on_us_keyboard(modifiers: list[str], key: str) -> dict:
    """A key as a person presses it on a US keyboard, a character typed with Shift as Shift and the key it's on."""
    if base := SHIFTED.get(key):
        return {"modifiers": modifiers if "shift" in modifiers else [*modifiers, "shift"], "key": base}
    return {"modifiers": modifiers, "key": key}


async def focused() -> dict:
    """The focused window as a key's target: its pid, its Hyprland address, which is its cua-driver window id, in the
    foreground through the plugin's seat, where a key edits text as a person's does; nothing with nothing focused."""
    window = await asyncio.to_thread(hyprctl, "activewindow")
    if not window or not window.get("address"):
        return {}
    return {"pid": window["pid"], "window_id": int(window["address"], 16), "delivery_mode": "foreground"}


async def reset_bind(made: dict | None) -> dict | None:
    """The bind of RESET_BIND this driver may press, as Hyprland lists it: ``made``, the one it bound to nothing, while
    that bind alone has the chord; one it binds to nothing now when the chord has none, as once Hyprland's config
    reloads; None while any other bind has the chord, since one anybody else made could do anything."""
    return await asyncio.to_thread(_reset_bind, made)


def _reset_bind(made: dict | None) -> dict | None:
    if binds := _chord_binds():
        return made if made is not None and binds == [made] else None
    lua(f'hl.bind("{RESET_BIND}", function() end)')
    return binds[0] if len(binds := _chord_binds()) == 1 else None


def _chord_binds() -> list[dict]:
    """RESET_BIND's binds as Hyprland lists them, each by its mask, key, dispatcher and argument, which tell them apart:
    a bind made in Lua has the dispatcher ``__lua`` and its function's number as its argument."""
    return [{field: bind.get(field) for field in ("modmask", "key", "dispatcher", "arg")}
            for bind in hyprctl("binds") or [] if (bind.get("modmask"), bind.get("key")) == RESET_LISTED]


async def plugin() -> ReadyCheck:
    """That Hyprland has cua-driver's plugin loaded, enabled, built for it and taking input; one restarted after a
    crash runs in safe mode, with no plugins, where input is sent but never arrives, and a plugin loaded again into a
    running Hyprland answers but has no input seat until the desktop restarts. When the Hyprland the driver started with
    doesn't answer and another runs in its place, the check fails for good (``REPLACED``)."""
    status = await asyncio.to_thread(hyprctl, "cua:status")
    transport = (status or {}).get("transport") or {}
    if status is None and await asyncio.to_thread(hyprctl, "activeworkspace") is None:
        if await asyncio.to_thread(_replaced):
            return ReadyCheck("cua-driver plugin", False, REPLACED, final=True)
        problem = "Hyprland doesn't answer on its IPC"
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


def _replaced() -> bool:
    """Whether another Hyprland runs in this session in place of the one whose instance signature the driver has."""
    instances = hyprctl("instances") or []
    ours = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    return bool(instances) and all(instance.get("instance") != ours for instance in instances)


async def require_plugin() -> None:
    """Input refused unless cua-driver's plugin can take it: the runtime runs ready checks only until they first pass,
    and input sent while Hyprland runs without the plugin is accepted but never arrives."""
    if not (check := await plugin()).ok:
        raise DeviceUnavailable(check.detail)


def hyprctl(request: str) -> dict | list | None:
    """Hyprland's JSON answer to ``request``, a list for one such as ``binds``, or None when it gives none, as for a
    plugin's request it doesn't know or when Hyprland doesn't answer."""
    try:
        out = subprocess.run(["hyprctl", "-j", request], capture_output=True, text=True, timeout=TIMEOUT_S)
        answer = json.loads(out.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    return answer if isinstance(answer, (dict, list)) else None


def lua(code: str) -> None:
    """Runs ``code`` in Hyprland's Lua config (``hyprctl eval``); what it did is for the caller to ask Hyprland."""
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(["hyprctl", "eval", code], capture_output=True, text=True, timeout=TIMEOUT_S)
