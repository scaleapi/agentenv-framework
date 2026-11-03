"""The computer skill: a SKILL.md an agent reads before it uses the computer, written from the computer as it booted:
its platform, its session user, its screen, its OS's shortcuts, what its card declares, and the driver's notes. A
driver whose software ships a skill of its own (``bundled_skill``) leads with it, its files beside the SKILL.md, and
the computer's part follows. It says nothing about any task."""
from __future__ import annotations

import getpass
import os
from pathlib import Path
from typing import TYPE_CHECKING

from .keys import PRINTABLE_KEYS

if TYPE_CHECKING:
    from .driver import ComputerDriverEnvironment

#: The chords each OS's desktops keep for the everyday actions, as ``computer_key`` takes them.
SHORTCUTS = {
    "linux": """\
- Copy, cut and paste: `['Control', 'c']`, `['Control', 'x']`, `['Control', 'v']`; in a terminal, with `'Shift'` too.
- Undo `['Control', 'z']`, save `['Control', 's']`, select all `['Control', 'a']`, find `['Control', 'f']`.
- Switch windows `['Alt', 'Tab']`; close the focused one `['Alt', 'F4']`.
- In a file manager, `['Control', 'l']` takes a typed path, `['F2']` renames and `['Control', 'Shift', 'n']` makes a
  folder.
""",
}
_SESSIONS = {"x11": "an X11 session", "wayland": "a Wayland session"}


async def skill(driver: ComputerDriverEnvironment) -> tuple[str, str, dict[str, str]]:
    """The skill's description, its markdown body and its other files, as ``agent_skill`` takes them."""
    display = await driver._computer.display_now()
    platform = driver.platform
    system = " ".join(part for part in ((platform.distribution or platform.os or "").title(), platform.version,
                                        f"({platform.arch})" if platform.arch else None) if part)
    session = _SESSIONS.get(platform.display or "", "a desktop session")
    if desktop := os.environ.get("XDG_CURRENT_DESKTOP"):
        session += f" ({desktop})"
    description = (f"How to use this computer through its computer_* tools: {system}, {display.width}x{display.height}, "
                   "what its input takes, and its shortcuts.")
    bundled = await driver.bundled_skill()
    opening = ([
        "# Using this computer",
        "You see this computer through `computer_screenshot` and drive it through the other `computer_*` tools, as a "
        "person at its screen would: look, then click, type and press keys on what the screen shows.",
    ] if bundled is None else [
        "# This computer",
        "Beside the tools above, this computer serves its `computer_*` tools: `computer_screenshot` shows the screen, "
        "and the others click, type and press keys on what it shows.",
    ])
    sections = [
        *opening,
        "## The computer",
        "\n".join([f"- {system}, {session}.", f"- You act as `{getpass.getuser()}`, whose home is `{Path.home()}`.",
                   f"- The screen is {display.width}x{display.height}."]),
        "## Input",
        _input(driver),
    ]
    if shortcuts := SHORTCUTS.get(platform.os or ""):
        sections += ["## Shortcuts", shortcuts.rstrip()]
    if notes := driver.notes.strip():
        sections += ["## Notes", notes]
    body = "\n\n".join(sections) + "\n"
    if bundled is None:
        return description, body, {}
    own_description, own_body, files = bundled
    return (f"{own_description} This computer: {system}, {display.width}x{display.height}, and its computer_* tools.",
            own_body.rstrip() + "\n\n" + body, files)


def _input(driver: ComputerDriverEnvironment) -> str:
    keys = ("any W3C key name, such as `'Enter'`, `'Control'`, `'ArrowLeft'` or `'a'`" if driver.keys == "all" else
            _keys(driver.keys))
    space, scale = driver.settings.default_space, driver.settings.default_norm_scale
    points = ("pixels of the latest screenshot" if space == "frame" else
              f"0 to {scale} across and down the latest screenshot")
    return "\n".join([
        f"- Buttons: {', '.join(driver.buttons)}; up to {driver.max_click_count} clicks in one `computer_click`.",
        f"- Keys: {keys}. A chord is pressed in order and released in reverse.",
        f"- Up to {driver.max_text_length} characters in one `computer_type`; `computer_wait` waits up to "
        f"{driver.max_wait_ms} ms.",
        f"- Points are {points}, unless a call names another space.",
        "- Input is foreground: it goes to the focused window and moves the pointer, as a person's would.",
    ])


def _keys(keys) -> str:
    """The keys a driver takes, every printable ASCII character said once when it takes them all."""
    every = set(PRINTABLE_KEYS) <= set(keys)
    named = [key for key in keys if not (every and key in PRINTABLE_KEYS)]
    return "only " + ", ".join(f"`{key!r}`" for key in named) + (", and any printable ASCII character" if every else "")
