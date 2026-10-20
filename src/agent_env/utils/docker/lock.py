"""Pin an image folder's Python and global npm installs to the versions its stored image holds, so a rebuild installs
what the stored image did."""

from __future__ import annotations

import itertools
import os
import posixpath
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from agent_env.utils.docker.context import ignore_matcher, ignore_names
from agent_env.utils.docker.dockerfile import Instruction, instructions
from agent_env.utils.docker.installed import Installed, normalized

# A command that resolves Python packages and reads PIP_CONSTRAINT or UV_CONSTRAINT.
_PIP_RESOLVES = re.compile(r"\b(?:pip3?|python3?(?:\.\d+)?\s+-m\s+pip|uv\s+pip)\s+(?:install|wheel|download)\b")
# An npm install command, up to the shell's next separator, across escaped line breaks.
_NPM_INSTALL = re.compile(r"(?P<command>\bnpm\s+(?:install|i|add)\b)(?P<arguments>(?:[^&;|\n\\]|\\\n|\\.)*)")
_GLOBAL = re.compile(r"(?:^|\s)(?:-g|--global)(?=\s|$)")
# A shell word in an npm install: bare, or wholly in single or double quotes.
_WORD = re.compile(r"""(?<=\s)(?:"(?P<double>[^"\\]*)"|'(?P<single>[^']*)'|(?P<bare>[^\s'"\\]+))(?=\s|\\\n|$)""")
# Each project lock, and the command that installs from it, which pins those packages already.
_LOCK_FILES = {"uv.lock": re.compile(r"\buv\s+sync\b"), "poetry.lock": re.compile(r"\bpoetry\s+(?:install|sync)\b"),
               "pdm.lock": re.compile(r"\bpdm\s+(?:install|sync)\b"),
               "Pipfile.lock": re.compile(r"\bpipenv\s+(?:install|sync)\b")}
# Files whose lines are requirements a pip command reads.
_REQUIREMENT_FILES = re.compile(r".*(?:requirements|constraints).*\.(?:txt|in)")
# pip applies constraints inside isolated builds too, where a package can need a newer build tool than the image holds.
_BUILD_TOOLS = frozenset({"pip", "setuptools", "wheel"})
CONSTRAINTS = "constraints.txt"
IN_IMAGE = "/tmp/agent-env-constraints.txt"
# The first line of a constraints file agent-env writes, so pinning another image in the folder doesn't read it as asked.
_HEADER = "# Pinned by agent-env to the versions of the image the store held.\n"


@dataclass(frozen=True)
class Locked:
    """What pinning an image folder pinned, and what it left as the build resolves it."""

    python: int  # packages constraints.txt pins
    constraints: str | None  # its path in the folder, when one was written
    npm: tuple[str, ...]  # package@version, each written into an npm install -g
    lock_file: str | None  # the project lock a command the build runs installs from, which pins its packages already
    not_covered: tuple[str, ...]


def lock(folder: Path, dockerfile: str, installed: Installed) -> Locked:
    """Pin the folder's Dockerfile at ``dockerfile`` to what ``installed`` holds: a constraints.txt beside it, passed to
    each stage that runs pip as PIP_CONSTRAINT and UV_CONSTRAINT build args, and exact versions in npm -g installs."""
    parts = instructions((folder / dockerfile).read_text(encoding="utf-8"))
    texts, npm, npm_left = _pin_npm_installs(parts, installed.npm)
    texts, pinned, constraints, python_left = _pin_python(folder, dockerfile, parts, texts, installed)
    (folder / dockerfile).write_text("".join(texts), encoding="utf-8")
    return Locked(pinned, constraints, tuple(npm), _lock_file(folder, parts), (*npm_left, *python_left))


def _pin_npm_installs(parts: list[Instruction], versions: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    """Each instruction's text with its npm -g installs pinned to ``versions``, the package@version pins it wrote,
    and what it left as the build resolves it."""
    pinned: list[str] = []
    unpinned: list[str] = []
    texts = [_pin_npm(part.text, versions, pinned, unpinned) if part.keyword == "RUN" else part.text for part in parts]
    left = ["npm packages' own dependencies, which a global install resolves afresh"] if pinned else []
    left += [f"npm {name}, which an npm install -g names in a form agent-env doesn't rewrite"
             for name in dict.fromkeys(unpinned)]
    return texts, pinned, left


def _pin_python(folder: Path, dockerfile: str, parts: list[Instruction], texts: list[str],
                installed: Installed) -> tuple[list[str], int, str | None, list[str]]:
    """``texts`` with each stage running pip given the constraints file it writes, how many packages that pins and
    the file's path, and what it left as the build resolves it."""
    stages = _stages_running_pip(parts)
    if not stages:
        return texts, 0, None, []
    pins, left = _pins(installed, _asked_for(folder, parts))
    constraints = _constraints_path(folder, dockerfile)
    if _ignored(folder, dockerfile, constraints):
        return texts, 0, None, [*left, f"Python packages, since the folder's ignore file leaves out {constraints}"]
    if not pins:
        return texts, 0, None, left
    (folder / constraints).write_text(
        _HEADER + "".join(f"{name}=={version}\n" for name, version in sorted(pins.items())), encoding="utf-8")
    arguments = f'COPY ["{constraints}", "{IN_IMAGE}"]\nARG PIP_CONSTRAINT={IN_IMAGE}\nARG UV_CONSTRAINT={IN_IMAGE}\n'
    texts = [text + ("" if text.endswith("\n") else "\n") + arguments if index in stages else text
             for index, text in enumerate(texts)]
    return texts, len(pins), constraints, left


def _lock_file(folder: Path, parts: list[Instruction]) -> str | None:
    """The project lock in the folder that a command the Dockerfile runs installs from."""
    runs = " ".join(" ".join((part.arguments, *part.heredocs)) for part in parts if part.keyword == "RUN")
    return next((path.relative_to(folder).as_posix() for name, command in _LOCK_FILES.items() if command.search(runs)
                 for path in sorted(folder.rglob(name))), None)


def _pins(installed: Installed, asked: dict[str, list[tuple[str, Requirement]]]) -> tuple[dict[str, str], list[str]]:
    """The version to pin each Python distribution to, and what's left as the build resolves it, and why: pinning
    it could fail a build that works unpinned."""
    pins, left = {}, []
    held = sorted(_BUILD_TOOLS & installed.python.keys())
    if held:
        left.append(f"{', '.join(held)}, which isolated builds install their own versions of")
    for name, versions in installed.python.items():
        if name in _BUILD_TOOLS:
            continue
        if len(versions) > 1:
            left.append(f"Python {name}, which the image holds at {', '.join(versions)}")
        elif name in installed.direct:
            left.append(f"Python {name}, installed from a path or URL rather than an index")
        elif not _readable(versions[0]):
            left.append(f"Python {name}, whose version {versions[0]} pip can't read")
        elif clash := next(((where, requirement) for where, requirement in asked.get(name, ())
                            if not requirement.specifier.contains(versions[0], prereleases=True)), None):
            left.append(f"Python {name}, which {clash[0]} asks for as {clash[1]} though the image ends with "
                        f"{versions[0]}")
        else:
            pins[name] = versions[0]
    return pins, left


def _readable(version: str) -> bool:
    try:
        Version(version)
    except InvalidVersion:
        return False
    return True


def _asked_for(folder: Path, parts: list[Instruction]) -> dict[str, list[tuple[str, Requirement]]]:
    """Each requirement with a version the Dockerfile's RUN commands or the folder's requirements files name, by
    distribution, with where it's named."""
    asked: dict[str, list[tuple[str, Requirement]]] = {}
    for part in parts:
        if part.keyword == "RUN":
            for text in (part.arguments, *part.heredocs):
                for requirement in _requirements(_words(text)):
                    asked.setdefault(normalized(requirement.name), []).append(("the Dockerfile", requirement))
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not path.is_symlink() and _REQUIREMENT_FILES.fullmatch(path.name):
            text = path.read_text("utf-8", "replace")
            if text.startswith(_HEADER):
                continue
            lines = (line.partition(" #")[0].partition(" --")[0].strip().rstrip("\\").strip() for line in text.splitlines())
            for requirement in _requirements(lines):
                asked.setdefault(normalized(requirement.name), []).append((path.relative_to(folder).as_posix(),
                                                                          requirement))
    return asked


def _words(text: str) -> list[str]:
    try:
        return shlex.split(text, comments=True)
    except ValueError:
        return text.split()


def _requirements(candidates) -> list[Requirement]:
    """The candidates that are requirements naming a version."""
    found = []
    for candidate in candidates:
        if candidate.startswith("-") or not any(sign in candidate for sign in "=<>~!"):
            continue
        try:
            requirement = Requirement(candidate)
        except InvalidRequirement:
            continue
        if requirement.specifier and requirement.url is None:
            found.append(requirement)
    return found


def _stages_running_pip(parts: list[Instruction]) -> set[int]:
    """The index of each FROM whose stage runs a pip or uv pip command that resolves packages."""
    stages, current = set(), None
    for index, part in enumerate(parts):
        if part.keyword == "FROM":
            current = index
        elif part.keyword == "RUN" and current is not None and _PIP_RESOLVES.search(
                " ".join((part.arguments, *part.heredocs))):
            stages.add(current)
    return stages


def _pin_npm(text: str, versions: dict[str, str], pinned: list[str], unpinned: list[str]) -> str:
    """``text`` with each package an ``npm install -g`` in it names written at the version ``versions`` holds; one it
    names in a form this doesn't rewrite goes in ``unpinned``."""
    def install(match: re.Match) -> str:
        arguments = match["arguments"]
        if not _GLOBAL.search(arguments):
            return match.group(0)
        here: list[str] = []
        arguments = _WORD.sub(lambda word: _pinned_word(word, versions, here), arguments)
        pinned.extend(f"{name}@{versions[name]}" for name in here)
        unpinned.extend(name for word in _words(arguments) if (name := _package(word)) in versions and name not in here)
        return match["command"] + arguments
    return _NPM_INSTALL.sub(install, text)


def _pinned_word(word: re.Match, versions: dict[str, str], pinned: list[str]) -> str:
    """``word`` naming the version ``versions`` holds of the package it names, in the quotes it was written in."""
    value = next(value for value in (word["double"], word["single"], word["bare"]) if value is not None)
    name = _package(value)
    if name not in versions:
        return word.group(0)
    pinned.append(name)
    quote = '"' if word["double"] is not None else "'" if word["single"] is not None else ""
    return f"{quote}{name}@{versions[name]}{quote}"


def _package(word: str) -> str | None:
    """The package an npm install's ``word`` names, or None for a flag."""
    if word.startswith("-"):
        return None
    return "@" + word[1:].partition("@")[0] if word.startswith("@") else word.partition("@")[0]


def _constraints_path(folder: Path, dockerfile: str) -> str:
    """Where the constraints go: beside the Dockerfile, under the first of the import's names nothing there has."""
    names = itertools.chain([CONSTRAINTS, f"agent-env-{CONSTRAINTS}"],
                            (f"agent-env-constraints-{n}.txt" for n in itertools.count(2)))
    return next(path for name in names
                if not os.path.lexists(folder / (path := posixpath.join(posixpath.dirname(dockerfile), name))))


def _ignored(folder: Path, dockerfile: str, path: str) -> bool:
    """Whether the ignore file the build reads leaves ``path`` out of its context."""
    ignore = next((folder / name for name in ignore_names(dockerfile) if (folder / name).is_file()), None)
    return ignore is not None and ignore_matcher(ignore.read_text(encoding="utf-8"))(Path(path))
