"""The checks a driver class passes when it is defined, beside its operations' (``operations.check_class``): its
members leave the base class's own alone, and its features are in range."""
from __future__ import annotations

from ..errors import AgentEnvError
from .keys import canonical
from .operations import declared

BUTTONS = ("left", "right", "middle", "back", "forward")


def check_names(cls: type, base: type) -> None:
    """Refuse a driver member that would replace one of ``base``'s own: its private members and the tools, which
    define the contract. The hooks (``caller_identity``, ``ready_checks``, ...) and the features are the driver's."""
    reserved = {name for name in dir(base)
                if (name.startswith("_") and not name.startswith("__")) or name.startswith("computer_")}
    clash = sorted(name for name in vars(cls) if name in reserved or name.startswith(("_computer_", "computer_")))
    if clash:
        raise TypeError(f"{cls.__qualname__} redefines {', '.join(clash)}, which {base.__name__} uses "
                        "itself; give the driver's own members other names")


def check_features(cls: type) -> None:
    problems = []
    unknown = [b for b in cls.buttons if b not in BUTTONS]
    if "left" not in cls.buttons or unknown:
        problems.append(f"buttons include left and come from {list(BUTTONS)}, got {list(cls.buttons)}")
    for name, low in (("max_click_count", 1), ("max_text_length", 1), ("max_wait_ms", 60_000), ("max_drag_points", 2),
                      ("max_duration_ms", 0), ("max_key_repeat", 1), ("max_scroll_amount", 1)):
        value = getattr(cls, name)
        if not isinstance(value, int) or value < low:
            problems.append(f"{name} is an integer of at least {low}, got {value!r}")
    if cls.keys != "all":
        try:
            if [canonical(k) for k in cls.keys] != list(cls.keys):
                problems.append("keys are W3C key values in their W3C case")
        except AgentEnvError as e:
            problems.append(f"keys: {e.message}")
    if ("set_display" in declared(cls)) != bool(cls.display_modes):
        problems.append("display_modes and @set_display come together")
    if problems:
        raise TypeError(f"{cls.__qualname__}: " + "; ".join(problems))
