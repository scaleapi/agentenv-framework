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
    arguments: str  # on one line, escaped line breaks removed as docker build removes them, heredoc bodies left out
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
    # The escape character ending a line, spaces or tabs after it, joins the next line on as it is.
    continued = re.compile(re.escape(escape) + r"[ \t]*$")
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
            body = line.rstrip("\r\n")
            if not (end := continued.search(body)):
                parts.append(body)
                break
            parts.append(body[:end.start()])
        keyword, *rest = "".join(parts).split(None, 1)
        keyword, arguments = keyword.upper(), rest[0].strip() if rest else ""
        heredocs, complete = [], True
        if keyword in _HEREDOC_KEYWORDS:
            for strip_tabs, name in _heredoc_names(arguments):
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


def leading_flags(arguments: str) -> tuple[list[str], str]:
    """The ``--flag`` words ``arguments`` begins with, split on any whitespace outside quotes, and the rest of it."""
    flags, at = [], 0
    while arguments.startswith("--", at):
        end, quote = at, None
        while end < len(arguments) and (quote or not arguments[end].isspace()):
            char = arguments[end]
            quote = (None if char == quote else quote) if quote else (char if char in "\"'" else None)
            end += 1
        flags.append(arguments[at:end])
        at = end
        while at < len(arguments) and arguments[at].isspace():
            at += 1
    return flags, arguments[at:]


def _json_strings(arguments: str) -> list[str] | None:
    """``arguments`` as the JSON list of strings of an exec-form instruction; None when it's shell form, as docker
    build reads anything that doesn't parse as one, a ``[`` it opens with included."""
    if not arguments.startswith("["):
        return None
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) and all(isinstance(arg, str) for arg in parsed) else None


def opens_heredoc(word: str) -> bool:
    """Whether ``word``, written in a shell-form instruction, would open a heredoc."""
    return _HEREDOC.fullmatch(word) is not None


def _heredoc_names(arguments: str) -> list[tuple[str, str]]:
    """Each heredoc ``arguments`` opens, as (its ``-`` if any, its name): a ``<<NAME`` outside quotes, in shell form only,
    since docker build reads a quoted one as text and an exec-form (JSON) instruction opens none."""
    if _json_strings(arguments) is not None:
        return []
    found, quote, at = [], None, 0
    while at < len(arguments):
        char = arguments[at]
        if char == "\\" and quote != "'":
            at += 2
            continue
        if quote:
            quote = None if char == quote else quote
        elif char in "\"'":
            quote = char
        elif heredoc := _HEREDOC.match(arguments, at):
            found.append((heredoc.group(1), heredoc.group(3)))
            at = heredoc.end()
            continue
        at += 1
    return found


def copy_arguments(instruction: Instruction) -> CopyArguments | None:
    """A COPY or ADD's arguments; None for any other instruction, or one whose arguments it can't read: a quoted
    shell-form path, or fewer than a source and a destination. Arguments that don't parse as a JSON list of strings are
    shell form, as docker build reads them."""
    if instruction.keyword not in ("COPY", "ADD"):
        return None
    flags, rest = leading_flags(instruction.arguments)
    args = _json_strings(rest)
    json_form = args is not None
    if not json_form:
        if any(quote in rest for quote in "\"'"):
            return None
        args = rest.split()
    if len(args) < 2:
        return None
    sources = tuple(source for source in args[:-1] if json_form or not _HEREDOC.fullmatch(source))
    return CopyArguments(instruction.keyword, tuple(flags), sources, args[-1], json_form,
                         heredoc=len(sources) < len(args) - 1)
