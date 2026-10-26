"""ComputerDriverEnvironment: a driver writes screen operations, and the base class serves the computer contract
from them: the tools agents call, the framework extensions agentenv calls, the data plane and the card."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any, Callable, ClassVar

from ..agent_env_environment import AgentEnvEnvironment, environment_card
from ..errors import AgentEnvError
from ..tool_context import ToolContext
from ..types import EnvironmentExtension, EnvironmentTool, IntakeDeclaration, IntakeFormat, intake_extension
from . import linux
from .dataplane import DataPlane
from .errors import DeliveryFailed, DeviceUnavailable, InputBusy, Internal
from .extensions import (
    COMPUTER,
    COMPUTER_DISPLAY,
    COMPUTER_EXEC,
    COMPUTER_KEYBOARD,
    COMPUTER_POINTER,
    COMPUTER_SPACES,
    KEYBOARD_RANGES,
    MAX_EXEC_TIMEOUT_S,
    Extensions,
)
from .frames import Frame, FrameStore, to_pil
from .keys import canonical
from .operations import GROUPS, INPUT, Display, ReadyCheck, check_class, declared, default_for, resolve
from .platform import Platform, detect
from .queue import InputQueue
from .settings import ComputerSettings, load_settings
from .tools import GROUP_TOOLS, POINT_OPTIONAL_WITH_POINTER, POINT_TOOLS, Tools, _result, effect_data

#: The caller every call comes from until agents carry an identity of their own.
DEFAULT_IDENTITY = "default"
_BUTTONS = ("left", "right", "middle", "back", "forward")
#: The handlers each optional operation or group brings; without it they are withdrawn.
_HANDLERS = {
    "set_display": ("_computer_display_set",),
    "keyboard": ("_computer_keyboard_get", "_computer_keyboard_set"),
    "run_in_session": ("_computer_exec_run",),
    "write_files": ("_computer_data_add",),
}


class ComputerDriverEnvironment(Tools, Extensions, DataPlane, AgentEnvEnvironment):
    """Subclass it and mark the methods that capture and send input (see ``operations``); the class is checked when
    it is defined. Features are class attributes, on the card. A subclass's ``__init__`` calls ``super().__init__()``.
    """

    buttons: ClassVar[tuple[str, ...]] = ("left",)
    max_click_count: ClassVar[int] = 1
    #: ``"all"`` for every W3C key, or the key values this computer can press.
    keys: ClassVar[Any] = "all"
    max_text_length: ClassVar[int] = 10_000
    max_wait_ms: ClassVar[int] = 60_000
    max_drag_points: ClassVar[int] = 2
    max_duration_ms: ClassVar[int] = 10_000
    max_key_repeat: ClassVar[int] = 100
    max_scroll_amount: ClassVar[int] = 100
    #: The (width, height) modes computer-display/v1 set takes; with ``@set_display`` only.
    display_modes: ClassVar[tuple] = ()
    layouts: ClassVar[tuple[str, ...]] = ("us",)
    #: The platforms the driver supports, as the driver env declares them: ``[{"os": {...}, "arch": [...]}]``.
    supports: ClassVar[list] = []
    tool_access = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _check_names(cls)
        check_class(cls)
        _check_features(cls)

    def __init__(self, settings: ComputerSettings | None = None, platform: Platform | None = None) -> None:
        self.settings = settings if settings is not None else load_settings()
        self.platform = platform if platform is not None else detect()
        self._computer_operations = resolve(type(self), self.platform)
        self._computer_frames = FrameStore()
        self._computer_queue = InputQueue()
        self._computer_ready = False
        self._computer_ready_lock = asyncio.Lock()
        self._computer_exec_running: dict = {}
        self._computer_exec_finished: OrderedDict = OrderedDict()
        self._computer_held: dict[str, list[str]] = {}
        self._computer_releases: dict[str, asyncio.Task] = {}
        self._computer_served: dict[str, EnvironmentTool] = {}
        for capability, handlers in _HANDLERS.items():
            if not self.has(capability):
                for name in handlers:
                    setattr(self, name, None)

    def has(self, capability: str) -> bool:
        """Whether an operation, or a whole group (``pointer``, ``keyboard``), works on this platform."""
        if capability in GROUPS:
            return all(op in self._computer_operations for op in GROUPS[capability])
        return capability in self._computer_operations

    def caller_identity(self, context: ToolContext) -> str:
        """Who a call comes from, for each caller's latest frame and others' input since a frame."""
        return DEFAULT_IDENTITY

    async def ready_checks(self) -> list[ReadyCheck]:
        """Checks of the driver's own, run with the base class's by computer-ready/v1; never sends input."""
        return []

    # The card.

    def served_tools(self, tools: list[EnvironmentTool]) -> list[EnvironmentTool]:
        unserved = {name for group, names in GROUP_TOOLS.items() if not self.has(group) for name in names}
        self._computer_served = {t.name: t.model_copy(deep=True) for t in tools if t.name not in unserved}
        if not self.has("pointer"):
            for name in POINT_OPTIONAL_WITH_POINTER:
                schema = self._computer_served[name].inputSchema
                schema["required"] = [*schema.get("required", []), "x", "y"]
        self._computer_describe_tools()
        return list(self._computer_served.values())

    def create_app(self) -> Any:
        self._computer_require_init()
        app = super().create_app()
        self._computer_describe_tools()
        return app

    def mount(self, app: Any) -> Any:
        self._computer_require_init()
        mounted = super().mount(app)
        self._computer_describe_tools()
        return mounted

    def _computer_describe_tools(self) -> None:
        """Write the default coordinate space into the descriptions of the tools that take a point, and carry the
        served descriptions and input schemas over to mcp's FastMCP's tools/list, which reads the tool functions."""
        space, scale = self.settings.default_space, self.settings.default_norm_scale
        note = (" Points are in frame space by default: pixels of the frame's image." if space == "frame" else
                f" Points are in norm space by default: 0 to {scale} over the frame, so send x and y on that grid.")
        for name in POINT_TOOLS:
            descriptor = self._computer_served.get(name)
            if descriptor is not None:
                descriptor.description = " ".join((getattr(type(self), name).__doc__ or "").split()) + note
        manager = getattr(self.mcp, "_tool_manager", None)
        for name, descriptor in self._computer_served.items():
            registered = manager.get_tool(name) if manager is not None else None
            if registered is not None:
                registered.description = descriptor.description
                registered.parameters["required"] = sorted(
                    {*registered.parameters.get("required", []), *(descriptor.inputSchema or {}).get("required", [])})

    def extension_params(self, uri: str) -> dict:
        if uri == COMPUTER_DISPLAY:
            return {"modes": [list(m) for m in self.display_modes]}
        if uri == COMPUTER_SPACES:
            return {"spaces": ["frame", "norm"]}
        if uri == COMPUTER_KEYBOARD:
            return {**{name: {"min": low, "max": high} for name, (low, high) in KEYBOARD_RANGES.items()},
                    "layouts": list(self.layouts)}
        if uri == COMPUTER_EXEC:
            return {"max_output_bytes": self.settings.exec_output_bytes, "max_timeout_s": MAX_EXEC_TIMEOUT_S}
        return {}

    def declared_extensions(self) -> list[EnvironmentExtension]:
        features = {
            "buttons": list(self.buttons), "max_click_count": self.max_click_count,
            "keys": self.keys if self.keys == "all" else list(self.keys), "max_text_length": self.max_text_length,
            "max_wait_ms": self.max_wait_ms, "max_drag_points": self.max_drag_points,
            "max_duration_ms": self.max_duration_ms, "max_key_repeat": self.max_key_repeat,
            "max_scroll_amount": self.max_scroll_amount,
        }
        declared = [EnvironmentExtension(
            uri=COMPUTER, description="The seven computer_* tools: screenshot, click, drag, scroll, type, key, wait.",
            params={**features, "delivery": "foreground", "platform": self.platform.as_dict(),
                    "supports": list(self.supports)})]
        if self.has("pointer"):
            declared.append(EnvironmentExtension(
                uri=COMPUTER_POINTER, description="The computer_pointer_* tools, and click and scroll at the pointer.",
                params={"buttons": list(self.buttons), "kinds": ["foreground"]}))
        intake = IntakeDeclaration(get=[IntakeFormat(part="file", format="file-tree")])
        if self.has("write_files"):
            intake.add = [IntakeFormat(part="file", format="file-tree", load="additive")]
        declared.append(intake_extension(intake, "data/add writes files under the session user's home; data/get "
                                                 "returns the env's data paths."))
        return declared

    # What the tools and extensions call.

    async def _computer_call(self, operation: str, *args: Any) -> Any:
        """Run one of the driver's operations: an AgentEnvError passes through, anything else is an input that may
        have been delivered (delivery_failed) or an internal failure."""
        method = getattr(self, self._computer_operations[operation])
        try:
            result = method(*args)
            return await result if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future) else result
        except AgentEnvError:
            raise
        except Exception as e:
            error = DeliveryFailed if operation in INPUT else Internal
            raise error(f"@{operation} failed: {type(e).__name__}: {e}") from e

    async def _computer_display_now(self) -> Display:
        display = await self._computer_call("display")
        if not isinstance(display, Display) or display.width <= 0 or display.height <= 0 or display.scale <= 0:
            raise Internal(f"@display returned {display!r}, not a Display with a positive size and scale")
        return display

    async def _computer_capture(self, identity: str) -> Frame:
        display = await self._computer_display_now()
        image = to_pil(await self._computer_call("capture"), display)
        return self._computer_frames.add(image, display, self._computer_queue.snapshot(), identity=identity)

    async def _computer_pointer(self) -> tuple[int, int]:
        where = await self._computer_call("pointer_position")
        try:
            x, y = where
            return int(x), int(y)
        except (TypeError, ValueError):
            raise Internal(f"@pointer_position returned {where!r}, not (x, y)") from None

    async def _computer_act(self, ctx: ToolContext, operation: str, args: tuple, *, frame: Frame | None,
                            screenshot: bool, summary: str, mapped: bool = False, repeat: int = 1,
                            hold: tuple | None = None, move_to: tuple | None = None, then: Callable | None = None):
        """One input call: its turn in the queue, the operation (after a move, repeated or held), then the result
        with others' input since ``frame`` and, when asked, a fresh frame. A point ``mapped`` from ``frame`` is
        checked again once the call has its turn, since the display may have changed while it waited; ``then``
        runs as soon as the input is delivered."""
        identity = self.caller_identity(ctx)
        sent = False
        async with self._computer_queue.turn(self.settings.queue_wait_s):
            if mapped and frame is not None:
                self._computer_frames.check_current(frame, await self._computer_display_now())
            others = self._computer_queue.others_since(frame.inputs, identity) if frame is not None else 0
            effect = None
            try:
                if move_to is not None:
                    await self._computer_call("pointer_move", *move_to)
                    sent = True
                for _ in range(repeat):
                    effect = await self._computer_call(operation, *args)
                    sent = True
                if hold is not None:
                    await asyncio.sleep(hold[0] / 1000)
                    await self._computer_call("key_up", list(reversed(hold[1])))
            except AgentEnvError as e:
                if sent:
                    e.applied = "unknown"
                raise
            finally:
                if sent:
                    self._computer_queue.delivered(identity)
            if then is not None:
                then()
        data = {"applied": "yes", "delivery": "foreground", "others_since_frame": others, "effect": effect_data(effect)}
        if not screenshot:
            return _result(summary, data)
        try:
            new = await self._computer_capture(identity)
        except AgentEnvError as e:
            return _result(f"{summary}; the screenshot after it failed ({e.code}: {e.message})", data)
        return _result(f"{summary}, then captured {new.id}", {**data, "frame": new.data()}, new.encode())

    async def _computer_ensure_ready(self) -> None:
        """Refuse with device_unavailable until the ready checks have passed once; run them if none has yet."""
        if self._computer_ready:
            return
        async with self._computer_ready_lock:
            if self._computer_ready:
                return
            failed = [c for c in await self._computer_ready_checks() if not c.ok]
        if failed:
            raise DeviceUnavailable("the computer isn't ready: " + "; ".join(f"{c.name}: {c.detail}" for c in failed))

    async def _computer_ready_checks(self) -> list[ReadyCheck]:
        checks = []
        try:
            display = await self._computer_display_now()
            checks.append(ReadyCheck("display", True, f"{display.width}×{display.height} at scale {display.scale:g}"))
        except AgentEnvError as e:
            display = None
            checks.append(ReadyCheck("display", False, e.message))
        if display is None:
            checks.append(ReadyCheck("capture", False, "the display didn't answer"))
        else:
            try:
                image = to_pil(await self._computer_call("capture"), display)
                checks.append(ReadyCheck("capture", True, f"{image.size[0]}×{image.size[1]}"))
            except AgentEnvError as e:
                checks.append(ReadyCheck("capture", False, e.message))
        try:
            checks += list(await self.ready_checks())
        except Exception as e:
            checks.append(ReadyCheck("driver", False, f"{type(e).__name__}: {e}"))
        if all(c.ok for c in checks):
            self._computer_ready = True
        return checks

    def _computer_hold(self, identity: str, keys: list[str]) -> None:
        held = self._computer_held.setdefault(identity, [])
        held += [k for k in keys if k not in held]
        previous = self._computer_releases.pop(identity, None)
        if previous is not None:
            previous.cancel()
        self._computer_releases[identity] = asyncio.ensure_future(
            self._computer_release_after(identity, self.settings.max_hold_ms))

    def _computer_release(self, identity: str, keys: list[str]) -> None:
        held = [k for k in self._computer_held.get(identity, []) if k not in keys]
        if held:
            self._computer_held[identity] = held
            return
        self._computer_held.pop(identity, None)
        release = self._computer_releases.pop(identity, None)
        if release is not None and release is not asyncio.current_task():
            release.cancel()

    async def _computer_release_after(self, identity: str, ms: int) -> None:
        """Release what ``identity`` still holds once ``max_hold_ms`` has passed, waiting as long as other input
        keeps the queue busy."""
        await asyncio.sleep(ms / 1000)
        while keys := list(reversed(self._computer_held.get(identity, []))):
            try:
                async with self._computer_queue.turn(self.settings.queue_wait_s):
                    await self._computer_call("key_up", keys)
            except InputBusy:
                continue
            self._computer_release(identity, keys)

    def _computer_require_init(self) -> None:
        if not hasattr(self, "_computer_operations"):
            raise TypeError(f"{type(self).__qualname__}.__init__ must call super().__init__()")

    # The Linux defaults for the OS operations.

    @default_for("run_in_session", os="linux")
    async def _computer_linux_run_in_session(self, argv: list, cwd: str | None, env: dict | None, timeout_s: float):
        return await linux.run_in_session(argv, cwd, env, timeout_s, output_bytes=self.settings.exec_output_bytes)

    @default_for("write_files", os="linux")
    async def _computer_linux_write_files(self, files: dict) -> None:
        await asyncio.to_thread(linux.write_files, files)


def computer_driver(name: str | None = None, *, supports: list | None = None) -> Callable:
    """Name a driver, its card's name unless ENVIRONMENT_NAME gives one, and declare the platforms it supports."""
    def deco(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, ComputerDriverEnvironment)):
            raise TypeError("@computer_driver decorates a ComputerDriverEnvironment subclass")
        if supports is not None:
            cls.supports = list(supports)
        return environment_card(name=name)(cls) if name else cls
    return deco


def _check_names(cls: type) -> None:
    """Refuse a driver member that would replace the base class's own: its private members and the tools, which
    define the contract. The hooks (``caller_identity``, ``ready_checks``, ...) and the features are the driver's."""
    reserved = {name for name in dir(ComputerDriverEnvironment)
                if (name.startswith("_") and not name.startswith("__")) or name.startswith("computer_")}
    clash = sorted(name for name in vars(cls) if name in reserved or name.startswith(("_computer_", "computer_")))
    if clash:
        raise TypeError(f"{cls.__qualname__} redefines {', '.join(clash)}, which ComputerDriverEnvironment uses "
                        "itself; give the driver's own members other names")


def _check_features(cls: type) -> None:
    problems = []
    unknown = [b for b in cls.buttons if b not in _BUTTONS]
    if "left" not in cls.buttons or unknown:
        problems.append(f"buttons include left and come from {list(_BUTTONS)}, got {list(cls.buttons)}")
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

