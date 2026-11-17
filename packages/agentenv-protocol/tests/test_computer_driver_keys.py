"""The named keys X11Driver, WindowsDriver and MacOSDriver map: each a W3C key value, the two PC keyboards' the same,
and the Mac's those but the keys its keyboard lacks, with its Help key; read from the drivers' sources, so that none of
the libraries an OS driver imports is needed."""
from __future__ import annotations

import ast
from pathlib import Path

from agentenv_protocol.computer import keys

COMPUTER = Path(keys.__file__).parent
#: The PC keys a Mac keyboard has no key code for, and the Mac's own Help key, which a PC keyboard lacks.
MAC_LACKS = {"ContextMenu", "Insert", "Pause", "PrintScreen", "F21", "F22", "F23", "F24"}
MAC_ONLY = {"Help"}


def _named(module: str, *tables: str) -> set[str]:
    """The keys of the module-level dicts ``tables`` in ``computer/<module>.py``, a dict comprehension's included."""
    found = set()
    for node in ast.parse((COMPUTER / f"{module}.py").read_text()).body:
        if isinstance(node, ast.Assign) and {getattr(t, "id", None) for t in node.targets} & set(tables):
            for key, value in zip(node.value.keys, node.value.values):
                if key is None:
                    names = ast.DictComp(key=value.key, value=ast.Constant(None), generators=value.generators)
                    found |= set(eval(compile(ast.fix_missing_locations(ast.Expression(names)), module, "eval")))
                else:
                    found.add(ast.literal_eval(key))
    return found


def test_every_named_key_a_driver_maps_is_a_w3c_key_value():
    tables = {"_x11": _named("_x11", "KEYSYMS"), "_windows": _named("_windows", "VIRTUAL_KEYS"),
              "_macos": _named("_macos", "NAMED", "MODIFIERS")}

    assert all(tables.values())
    assert {module: sorted(names - set(keys.NAMED) - {" "}) for module, names in tables.items()} == {
        "_x11": [], "_windows": [], "_macos": []}


def test_the_pc_drivers_map_the_same_named_keys_and_the_mac_those_its_keyboard_has():
    x11, windows = _named("_x11", "KEYSYMS"), _named("_windows", "VIRTUAL_KEYS")

    assert windows == x11
    assert _named("_macos", "NAMED", "MODIFIERS") == x11 - MAC_LACKS | MAC_ONLY
