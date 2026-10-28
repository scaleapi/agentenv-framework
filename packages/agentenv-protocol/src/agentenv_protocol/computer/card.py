"""What a driver's card says: the extensions, the tools, extensions and data-plane methods its operations support,
the extensions' params, the extensions it declares without handlers, and the descriptions that name the default
coordinate space."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .recording import recording_params
from ..types import METHOD_ADD, EnvironmentExtension, EnvironmentTool, IntakeDeclaration, IntakeFormat, intake_extension

if TYPE_CHECKING:
    from .driver import ComputerDriverEnvironment

COMPUTER = "urn:agentenv:computer/v1"
COMPUTER_READY = "urn:agentenv:computer-ready/v1"
COMPUTER_DISPLAY = "urn:agentenv:computer-display/v1"
COMPUTER_SPACES = "urn:agentenv:computer-spaces/v1"
COMPUTER_POINTER = "urn:agentenv:computer-pointer/v1"
COMPUTER_KEYBOARD = "urn:agentenv:computer-keyboard/v1"
COMPUTER_EXEC = "urn:agentenv:computer-exec/v1"
COMPUTER_RECORDING = "urn:agentenv:computer-recording/v1"

MAX_EXEC_TIMEOUT_S = 3600
KEYBOARD_RANGES = {"tap_hold_ms": (0, 1000), "type_delay_ms": (0, 1000), "max_hold_ms": (0, 600_000)}

#: The tools a group serves.
GROUP_TOOLS = {
    "pointer": ("computer_pointer_move", "computer_pointer_down", "computer_pointer_up", "computer_pointer_position",
                "computer_pointer_list"),
    "keyboard": ("computer_keyboard_down", "computer_keyboard_up"),
}
#: The tools whose description names the default coordinate space.
POINT_TOOLS = ("computer_screenshot", "computer_click", "computer_drag", "computer_scroll", "computer_pointer_move",
               "computer_pointer_down", "computer_pointer_up")
#: The operation or group an extension needs, and one of its methods needs; without it, it is not served.
_EXTENSION_NEEDS = {COMPUTER_KEYBOARD: "keyboard", COMPUTER_EXEC: "run_in_session"}
_METHOD_NEEDS = {(COMPUTER_DISPLAY, "set"): "set_display"}


def served_tools(driver: ComputerDriverEnvironment, tools: list[EnvironmentTool]) -> list[EnvironmentTool]:
    unserved = {name for group, names in GROUP_TOOLS.items() if not driver.has(group) for name in names}
    return [t.model_copy(update={"description": point_description(driver, t.name)}) if t.name in POINT_TOOLS else t
            for t in tools if t.name not in unserved]


def served_extensions(driver: ComputerDriverEnvironment,
                      extensions: list[EnvironmentExtension]) -> list[EnvironmentExtension]:
    served = []
    for extension in extensions:
        need = _EXTENSION_NEEDS.get(extension.uri)
        if need is not None and not driver.has(need):
            continue
        if extension.uri == COMPUTER_RECORDING and driver.recording is None:
            continue
        dropped = {op for (uri, op), need in _METHOD_NEEDS.items() if uri == extension.uri and not driver.has(need)}
        if dropped:
            methods = {op: entry for op, entry in extension.params["methods"].items() if op not in dropped}
            extension = extension.model_copy(update={"params": {**extension.params, "methods": methods}})
        served.append(extension)
    return served


def served_operations(driver: ComputerDriverEnvironment, operations: list[str]) -> list[str]:
    return [op for op in operations if op != METHOD_ADD or driver.has("write_files")]


def extension_params(driver: ComputerDriverEnvironment, uri: str) -> dict:
    if uri == COMPUTER_DISPLAY:
        return {"modes": [list(m) for m in driver.display_modes]}
    if uri == COMPUTER_SPACES:
        return {"spaces": ["frame", "norm"]}
    if uri == COMPUTER_KEYBOARD:
        return {**{name: {"min": low, "max": high} for name, (low, high) in KEYBOARD_RANGES.items()},
                "layouts": list(driver.layouts)}
    if uri == COMPUTER_EXEC:
        return {"max_output_bytes": driver.settings.exec_output_bytes, "max_timeout_s": MAX_EXEC_TIMEOUT_S}
    if uri == COMPUTER_RECORDING and driver.recording is not None:
        return recording_params(driver.recording, driver.has("record"))
    return {}


def declared_extensions(driver: ComputerDriverEnvironment) -> list[EnvironmentExtension]:
    features = {
        "buttons": list(driver.buttons), "max_click_count": driver.max_click_count,
        "keys": driver.keys if driver.keys == "all" else list(driver.keys), "max_text_length": driver.max_text_length,
        "max_wait_ms": driver.max_wait_ms, "max_drag_points": driver.max_drag_points,
        "max_duration_ms": driver.max_duration_ms, "max_key_repeat": driver.max_key_repeat,
        "max_scroll_amount": driver.max_scroll_amount,
    }
    declared = [EnvironmentExtension(
        uri=COMPUTER, description="The seven computer_* tools: screenshot, click, drag, scroll, type, key, wait.",
        params={**features, "delivery": "foreground", "platform": driver.platform.as_dict(),
                "supports": list(driver.supports)})]
    if driver.has("pointer"):
        declared.append(EnvironmentExtension(
            uri=COMPUTER_POINTER, description="The computer_pointer_* tools, and click and scroll at the pointer.",
            params={"buttons": list(driver.buttons), "kinds": ["foreground"]}))
    intake = IntakeDeclaration(get=[IntakeFormat(part="file", format="file-tree")])
    if driver.has("write_files"):
        intake.add = [IntakeFormat(part="file", format="file-tree", load="additive")]
    declared.append(intake_extension(intake, "data/add writes files under the session user's home; data/get "
                                             "returns the env's data paths."))
    return declared


def point_description(driver: ComputerDriverEnvironment, name: str) -> str:
    """A point tool's docstring, then the default coordinate space."""
    space, scale = driver.settings.default_space, driver.settings.default_norm_scale
    note = (" Points are in frame space by default: pixels of the frame's image." if space == "frame" else
            f" Points are in norm space by default: 0 to {scale} over the frame, so send x and y on that grid.")
    return " ".join((getattr(type(driver), name).__doc__ or "").split()) + note
