"""A computer's tool prefix: the names it serves its tools under, so several computers serve theirs side by side on
one gateway. The SDK's ``computer_*`` names in its text are rewritten to them; its driver's own and forwarded tools
keep their text as written, and a sentence tells an agent how those are named."""
from __future__ import annotations

import re
from collections.abc import Iterable

from ..types import EnvironmentTool
from .tools import COMPUTER_TOOLS

#: The SDK's tool names, and a ``computer_*`` family of them, each as a whole word.
_SDK_NAMES = re.compile(r"(?<![\w-])(?:" + "|".join([*map(re.escape, sorted(COMPUTER_TOOLS, key=len, reverse=True)),
                                                      r"computer_(?:[a-z]+_)*\*"]) + r")(?![\w-])")


class ToolPrefix:
    """``prefix``, the start of every name a computer serves a tool under; none leaves every name as it is."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def served(self, name: str) -> str:
        """The name the tool ``name`` is served under."""
        return f"{self.prefix}{name}"

    def own(self, served: str) -> str:
        """A served tool's own name."""
        return served.removeprefix(self.prefix)

    def text(self, text: str | None) -> str | None:
        """``text`` naming the SDK's tools as they are served; any other name stays as written."""
        return _SDK_NAMES.sub(lambda m: self.prefix + m.group(0), text) if self.prefix and text else text

    def tools(self, tools: list[EnvironmentTool]) -> list[EnvironmentTool]:
        """``tools`` under their served names, the SDK's own with their descriptions' names served too."""
        return [tool.model_copy(update={"name": self.served(tool.name), **(
            {"description": self.text(tool.description)} if tool.name in COMPUTER_TOOLS else {})}) for tool in tools]

    def rule(self, others: Iterable[str]) -> str | None:
        """The sentence telling an agent how ``others``, the tools whose text stays as written, are served; None
        without a prefix or any of them."""
        example = next(iter(others), None)
        if not self.prefix or example is None:
            return None
        return (f"This computer serves its tools under `{self.prefix}`: a tool this text names without `{self.prefix}` is "
                f"served with it, so `{example}` is `{self.prefix}{example}`. Names inside a tool's arguments stay as written.")

    def instructions(self, text: str, others: Iterable[str]) -> str:
        """An MCP server's instructions as the computer serves them: after the rule, if there is one, its names served."""
        return "\n\n".join(part for part in (self.rule(others), self.text(text)) if part)
