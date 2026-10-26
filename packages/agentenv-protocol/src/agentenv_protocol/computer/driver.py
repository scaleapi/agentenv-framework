"""ComputerDriverEnvironment: a driver writes screen operations, and the base class serves the computer contract
from them: the tools agents call, the framework extensions agentenv calls, the data plane and the card."""
from __future__ import annotations

import asyncio
from typing import Any, Callable, ClassVar

from ..agent_env_environment import _CARD_CONFIG_ATTR, AgentEnvEnvironment, AgentEnvFastMCPApplication, environment_card
from ..tool_context import ToolContext
from ..types import EnvironmentCard, EnvironmentExtension, EnvironmentTool
from . import card, linux
from .checks import check_features, check_names
from .dataplane import DataPlane
from .extensions import Extensions
from .operations import ReadyCheck, check_class, default_for
from .platform import Platform, detect
from .runtime import ComputerRuntime
from .settings import ComputerSettings, load_settings
from .tools import Tools

#: The caller every call comes from until agents carry an identity of their own.
DEFAULT_IDENTITY = "default"


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
    #: What the base class keeps for the driver, built by ``__init__``.
    _computer: ComputerRuntime | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        check_names(cls, ComputerDriverEnvironment)
        check_class(cls)
        check_features(cls)

    def __init__(self, settings: ComputerSettings | None = None, platform: Platform | None = None) -> None:
        self.settings = settings if settings is not None else load_settings()
        self.platform = platform if platform is not None else detect()
        self._computer = ComputerRuntime(self)

    @classmethod
    def declaration(cls, platform: Platform) -> EnvironmentCard:
        """The card this driver class serves on ``platform`` with default settings, under its own name, built without
        running its ``__init__``: what a ``computer_driver`` env records for each platform it supports. The card
        hooks may read the class and the base class's state, not what a subclass's ``__init__`` sets."""
        driver = cls.__new__(cls)
        ComputerDriverEnvironment.__init__(driver, settings=ComputerSettings(), platform=platform)
        config = dict(getattr(cls, _CARD_CONFIG_ATTR, None) or {})
        card = EnvironmentCard(**{**config, "name": config.get("name") or cls.__name__})
        return AgentEnvFastMCPApplication(card, driver).environment_card

    def has(self, capability: str) -> bool:
        """Whether an operation, or a whole group (``pointer``, ``keyboard``), works on this platform."""
        return self._computer.has(capability)

    def caller_identity(self, context: ToolContext) -> str:
        """Who a call comes from, for each caller's latest frame and others' input since a frame."""
        return DEFAULT_IDENTITY

    async def ready_checks(self) -> list[ReadyCheck]:
        """Checks of the driver's own, run with the base class's by computer-ready/v1; never sends input."""
        return []

    def create_app(self) -> Any:
        self._computer_require_init()
        return super().create_app()

    def mount(self, app: Any) -> Any:
        self._computer_require_init()
        return super().mount(app)

    # The card: what the driver's operations support.

    def served_tools(self, tools: list[EnvironmentTool]) -> list[EnvironmentTool]:
        return card.served_tools(self, tools)

    def served_extensions(self, extensions: list[EnvironmentExtension]) -> list[EnvironmentExtension]:
        return card.served_extensions(self, extensions)

    def served_operations(self, operations: list[str]) -> list[str]:
        return card.served_operations(self, operations)

    def extension_params(self, uri: str) -> dict:
        return card.extension_params(self, uri)

    def declared_extensions(self) -> list[EnvironmentExtension]:
        return card.declared_extensions(self)

    def _computer_require_init(self) -> None:
        if self._computer is None:
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
