"""A Dockerfile's instructions as ``docker build`` reads them, its COPY and ADD arguments, and what a build of it reads
of its context."""

from __future__ import annotations

import csv
import json
import posixpath
import re
from dataclasses import dataclass

_DIRECTIVE = re.compile(r"#\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*(\S+)\s*")
# A word opening a heredoc, as BuildKit reads one: a file descriptor, <<, a -, and its delimiter, quoted or not. A
# here-string (<<<) is left to the shell.
_HEREDOC = re.compile(r"\d*<<(-?)([^<]+)")
_HEREDOC_KEYWORDS = frozenset({"RUN", "COPY", "ADD"})
# The COPY and ADD flags that change how a source is written, not which one it is.
_SOURCE_KEEPING_FLAGS = ("--chown=", "--chmod=", "--link", "--checksum=", "--keep-git-dir", "--unpack")
# The RUN --mount types that never read the build context.
_OFF_CONTEXT_MOUNTS = frozenset({"cache", "tmpfs", "secret", "ssh"})
# The Dockerfile frontend this reading follows; another may read the context its own way.
_DOCKERFILE_SYNTAX = re.compile(r"(?:docker\.io/)?docker/dockerfile(?:-upstream)?(?::[\w.-]+)?(?:@sha256:[0-9a-f]{64})?")


@dataclass(frozen=True)
class Instruction:
    """An instruction, or a line between them (``keyword`` None); all their ``text`` joined is the Dockerfile."""

    keyword: str | None  # upper case
    arguments: str  # on one line, continued lines joined, heredoc bodies left out
    text: str
    heredocs: tuple[str, ...] = ()  # each heredoc's body, in order
    complete: bool = True  # False when a heredoc it opens is never closed


@dataclass(frozen=True)
class ContextReads:
    """The context paths a build reads, each of which may hold Docker's wildcards, or why it reads all of it."""

    paths: tuple[str, ...] = ()
    whole: str | None = None


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
    """``dockerfile``'s instructions and the lines between them, in order, as ``docker build`` splits them."""
    lines = dockerfile.splitlines(keepends=True)
    parsed, at = _directives(lines)
    escape = parsed.get("escape", "\\")
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


def directives(dockerfile: str) -> dict[str, str]:
    """The parser directives ``dockerfile`` opens with, by lower-case name."""
    return _directives(dockerfile.splitlines())[0]


def _directives(lines: list[str]) -> tuple[dict[str, str], int]:
    """The parser directives ``lines`` open with, and how many lines they take."""
    found, at = {}, 0
    while at < len(lines) and (directive := _DIRECTIVE.fullmatch(lines[at].strip())):
        found[directive.group(1).lower()] = directive.group(2)
        at += 1
    return found, at


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
    """``arguments`` as an exec-form JSON list of strings; None when ``docker build`` reads it as shell form."""
    if not arguments.startswith("["):
        return None
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) and all(isinstance(arg, str) for arg in parsed) else None


def opens_heredoc(word: str) -> bool:
    """Whether ``word``, written in a shell-form instruction, would open a heredoc."""
    return _heredoc(word) is not None


def _heredoc_names(arguments: str) -> list[tuple[str, str]]:
    """Each heredoc ``arguments`` opens, as (its ``-``, its delimiter): each word that opens one, as BuildKit reads it."""
    return [heredoc for word in _words(arguments) if (heredoc := _heredoc(word))]


def _heredoc(word: str) -> tuple[str, str] | None:
    """The heredoc ``word`` opens, as (its ``-``, its delimiter, unquoted); None when it opens none."""
    if not (match := _HEREDOC.fullmatch(word)):
        return None
    delimiter, quote, at = [], None, 0
    while at < len(match.group(2)):
        char = match.group(2)[at]
        if char == "\\" and quote != "'" and at + 1 < len(match.group(2)):
            at += 1
            delimiter.append(match.group(2)[at])
        elif char == quote:
            quote = None
        elif char in "\"'" and quote is None:
            quote = char
        else:
            delimiter.append(char)
        at += 1
    return (match.group(1), "".join(delimiter)) if delimiter else None


def _words(arguments: str) -> list[str]:
    """``arguments`` split into words on whitespace outside quotes, each as written, quotes and escapes kept."""
    words, word, quote, at = [], [], None, 0
    while at < len(arguments):
        char = arguments[at]
        if char == "\\" and quote != "'" and at + 1 < len(arguments):
            word.append(arguments[at:at + 2])
            at += 2
            continue
        if quote:
            quote = None if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char.isspace():
            if word:
                words.append("".join(word))
                word = []
            at += 1
            continue
        word.append(char)
        at += 1
    return words + (["".join(word)] if word else [])


def copy_arguments(instruction: Instruction) -> CopyArguments | None:
    """A COPY or ADD's arguments; None for another instruction, a quoted shell-form path, or fewer than two paths."""
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
    sources = tuple(source for source in args[:-1] if json_form or _heredoc(source) is None)
    return CopyArguments(instruction.keyword, tuple(flags), sources, args[-1], json_form,
                         heredoc=len(sources) < len(args) - 1)


def context_reads(dockerfile: str) -> ContextReads:
    """The context paths a build of ``dockerfile`` reads: its local COPY and ADD sources and bind-mount sources."""
    if (syntax := directives(dockerfile).get("syntax")) and not _DOCKERFILE_SYNTAX.fullmatch(syntax):
        return ContextReads(whole=f"# syntax={syntax} isn't followed")
    paths: list[str] = []
    for instruction in instructions(dockerfile):
        if not instruction.complete:
            return ContextReads(whole=f"a heredoc in {instruction.keyword} never ends, so the rest can't be read")
        if instruction.keyword in ("COPY", "ADD"):
            reads = _copy_reads(instruction)
        elif instruction.keyword == "RUN":
            reads = _mount_reads(instruction)
        else:
            continue
        if reads.whole:
            return reads
        paths.extend(reads.paths)
    return ContextReads(paths=tuple(dict.fromkeys(paths)))


def _copy_reads(instruction: Instruction) -> ContextReads:
    """The context paths a COPY or ADD reads."""
    args = copy_arguments(instruction)
    if args is None:
        return ContextReads(whole=f"{instruction.keyword} {instruction.arguments} can't be read")
    if any(flag.startswith("--from=") for flag in args.flags):
        return ContextReads()
    if unknown := next((flag for flag in args.flags if not flag.startswith(_SOURCE_KEEPING_FLAGS)), None):
        return ContextReads(whole=f"{args.keyword} {unknown} isn't followed")
    paths = []
    for source in args.sources:
        if args.keyword == "ADD" and ("://" in source or source.startswith("git@")):
            continue
        if why := _unread(source):
            return ContextReads(whole=f"{args.keyword} {source} {why}")
        if (path := _context_path(source)) is None:
            return ContextReads(whole=f"{args.keyword} {source} copies all of it")
        paths.append(path)
    return ContextReads(paths=tuple(paths))


def _mount_reads(instruction: Instruction) -> ContextReads:
    """The context paths a RUN's bind mounts read."""
    paths = []
    for flag in leading_flags(instruction.arguments)[0]:
        if not flag.startswith("--mount="):
            continue
        # A mount's options are one CSV record, as docker build reads them, so a quoted field holds a comma or space.
        options = {}
        for field in next(csv.reader([flag.removeprefix("--mount=")])):
            key, _, value = field.partition("=")
            options[key.strip().lower()] = value.strip().strip("\"'")
        kind = options.get("type", "bind").lower()
        if kind in _OFF_CONTEXT_MOUNTS or (kind == "bind" and "from" in options):
            continue
        if kind != "bind":
            return ContextReads(whole=f"RUN --mount type={kind} isn't followed")
        source = options.get("source") or options.get("src") or "."
        if why := _unread(source):
            return ContextReads(whole=f"RUN --mount source {source} {why}")
        if (path := _context_path(source)) is None:
            return ContextReads(whole=f"RUN --mount binds {source}, all of it")
        paths.append(path)
    return ContextReads(paths=tuple(paths))


def _unread(source: str) -> str | None:
    """Why ``source`` can't be read before a build, as docker build expands and unescapes it first; None when it can."""
    if "$" in source:
        return "names a variable"
    if "\\" in source or "`" in source:
        return "holds an escape"
    return None


def _context_path(source: str) -> str | None:
    """``source``'s path in the context; None for the context itself, or a path that would leave it."""
    path = posixpath.normpath(source.lstrip("/"))
    return None if path == "." or path == ".." or path.startswith("../") else path
