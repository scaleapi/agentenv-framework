"""A Dockerfile's instructions as ``docker build`` reads them, and the arguments of its COPY and ADD instructions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_DIRECTIVE = re.compile(r"#\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(\S+)\s*")
# ``<<<`` is a shell here-string, which docker build leaves to the shell, not a heredoc.
_HEREDOC = re.compile(r"(?<!<)<<(?!<)(-?)([\"']?)([A-Za-z_][A-Za-z0-9_]*)\2")
# The instructions whose arguments can open a heredoc.
_HEREDOC_KEYWORDS = frozenset({"RUN", "COPY", "ADD"})


@dataclass(frozen=True)
class Instruction:
    """One instruction, or one line between instructions (``keyword`` None: blank, a comment, a parser directive).
    ``text`` is what the Dockerfile holds for it, so the texts of all of them joined are the Dockerfile."""

    keyword: str | None  # upper case
    arguments: str  # on one line: escaped line breaks and the comment lines among them removed, heredoc bodies left out
    text: str
    heredocs: tuple[str, ...] = ()  # each heredoc's body, in order
    complete: bool = True  # False when a heredoc it opens runs to the end of the Dockerfile unclosed


@dataclass(frozen=True)
class CopyArguments:
    """A COPY or ADD's flags as written (``--name`` or ``--name=value``), sources and destination."""

    keyword: str
    flags: tuple[str, ...]
    sources: tuple[str, ...]  # a heredoc's ``<<NAME`` left out
    destination: str
    json_form: bool
    heredoc: bool  # a source is a heredoc


def instructions(dockerfile: str) -> list[Instruction]:
    """``dockerfile``'s instructions and the lines between them, in order: an instruction's escaped line breaks
    continue it, a comment line among them is dropped from its arguments, and a heredoc's body is part of it."""
    lines = dockerfile.splitlines(keepends=True)
    escape, at = "\\", 0
    while at < len(lines) and (directive := _DIRECTIVE.fullmatch(lines[at].strip())):
        if directive.group(1).lower() == "escape":
            escape = directive.group(2)
        at += 1
    found = [Instruction(None, "", line) for line in lines[:at]]
    while at < len(lines):
        line = lines[at]
        if not line.strip() or line.lstrip().startswith("#"):
            found.append(Instruction(None, "", line))
            at += 1
            continue
        text, parts = [], []
        while at < len(lines):
            line = lines[at]
            text.append(line)
            at += 1
            if parts and (not line.strip() or line.lstrip().startswith("#")):
                continue
            body = line.rstrip("\r\n").rstrip()
            if not body.endswith(escape):
                parts.append(body)
                break
            parts.append(body[:-len(escape)])
        keyword, _, arguments = " ".join(part.strip() for part in parts).partition(" ")
        keyword, arguments = keyword.upper(), arguments.strip()
        heredocs, complete = [], True
        if keyword in _HEREDOC_KEYWORDS:
            for strip_tabs, _, name in _HEREDOC.findall(arguments):
                body, closed = [], False
                while at < len(lines):
                    line = lines[at]
                    text.append(line)
                    at += 1
                    if (line.rstrip("\r\n").lstrip("\t") if strip_tabs else line.rstrip("\r\n")) == name:
                        closed = True
                        break
                    body.append(line)
                heredocs.append("".join(body))
                complete = complete and closed
        found.append(Instruction(keyword, arguments, "".join(text), tuple(heredocs), complete))
    return found


def copy_arguments(instruction: Instruction) -> CopyArguments | None:
    """A COPY or ADD's arguments; None for any other instruction, or one whose arguments it can't read: JSON that
    isn't a list of at least two strings, a quoted shell-form path, or fewer than a source and a destination."""
    if instruction.keyword not in ("COPY", "ADD"):
        return None
    rest, flags = instruction.arguments, []
    while rest.startswith("--"):
        flag, _, rest = rest.partition(" ")
        flags.append(flag)
        rest = rest.strip()
    json_form = rest.startswith("[")
    if json_form:
        try:
            args = json.loads(rest)
        except ValueError:
            return None
        if not (isinstance(args, list) and all(isinstance(arg, str) for arg in args)):
            return None
    else:
        if any(quote in rest for quote in "\"'"):
            return None
        args = rest.split()
    if len(args) < 2:
        return None
    sources = tuple(source for source in args[:-1] if not _HEREDOC.fullmatch(source))
    return CopyArguments(instruction.keyword, tuple(flags), sources, args[-1], json_form,
                         heredoc=len(sources) < len(args) - 1)
