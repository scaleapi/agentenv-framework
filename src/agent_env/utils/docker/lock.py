"""Pin an image folder's Python and global npm installs to the versions its stored image holds, so a rebuild installs
what the stored image did."""

from __future__ import annotations

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
_NPM_INSTALL = re.compile(r"\bnpm\s+(?:install|i|add)\b(?:[^&;|\n\\]|\\\n|\\.)*")
_GLOBAL = re.compile(r"(?:^|\s)(?:-g|--global)(?=\s|$)")
_WORD = re.compile(r"(?<=\s)[^\s'\"\\]+")
# A project lock its build installs from pins its packages already.
_LOCK_FILES = ("uv.lock", "poetry.lock", "pdm.lock", "Pipfile.lock")
# Files whose lines are requirements a pip command reads.
_REQUIREMENT_FILES = re.compile(r".*(?:requirements|constraints).*\.(?:txt|in)")
# pip applies constraints inside isolated builds too, where a package can need a newer build tool than the image holds.
_BUILD_TOOLS = frozenset({"pip", "setuptools", "wheel"})
CONSTRAINTS = "constraints.txt"
_IN_IMAGE = "/tmp/agent-env-constraints.txt"


@dataclass(frozen=True)
class Locked:
    """What pinning an image folder pinned, and what it left as the build resolves it."""

    python: int  # packages constraints.txt pins
    constraints: str | None  # its path in the folder, when one was written
    npm: tuple[str, ...]  # package@version, each written into an npm install -g
    lock_file: str | None  # the project lock the build installs from, which pins its packages already
    not_covered: tuple[str, ...]


def lock(folder: Path, dockerfile: str, installed: Installed) -> Locked:
    """Pin the folder's Dockerfile at ``dockerfile`` to what ``installed`` holds: a constraints.txt beside it, passed to
    each stage that runs pip as PIP_CONSTRAINT and UV_CONSTRAINT build args, and exact versions in npm -g installs."""
    parts = instructions((folder / dockerfile).read_text(encoding="utf-8"))
    not_covered: list[str] = []
    npm: list[str] = []
    texts = [_pin_npm(part.text, installed.npm, npm) if part.keyword == "RUN" else part.text for part in parts]
    if npm:
        not_covered.append("npm packages' own dependencies, which a global install resolves afresh")
    lock_file = next((path.relative_to(folder).as_posix() for name in _LOCK_FILES for path in sorted(folder.rglob(name))),
                     None)
    stages = _stages_running_pip(parts)
    constraints, pinned = None, 0
    if stages and lock_file is None:
        pins, left = _pins(installed, _asked_for(folder, parts))
        not_covered += left
        constraints = _constraints_path(folder, dockerfile)
        if _ignored(folder, dockerfile, constraints):
            not_covered.append(f"Python packages, since the folder's ignore file leaves out {constraints}")
            constraints = None
        elif pins:
            (folder / constraints).write_text("".join(f"{name}=={version}\n" for name, version in sorted(pins.items())),
                                              encoding="utf-8")
            pinned = len(pins)
            arguments = (f'COPY ["{constraints}", "{_IN_IMAGE}"]\nARG PIP_CONSTRAINT={_IN_IMAGE}\n'
                         f"ARG UV_CONSTRAINT={_IN_IMAGE}\n")
            texts = [text + ("" if text.endswith("\n") else "\n") + arguments if index in stages else text
                     for index, text in enumerate(texts)]
        else:
            constraints = None
    (folder / dockerfile).write_text("".join(texts), encoding="utf-8")
    return Locked(pinned, constraints, tuple(npm), lock_file, tuple(not_covered))


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
            lines = (line.partition(" #")[0].partition(" --")[0].strip().rstrip("\\").strip()
                     for line in path.read_text("utf-8", "replace").splitlines())
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


def _pin_npm(text: str, versions: dict[str, str], pinned: list[str]) -> str:
    """``text`` with each package an ``npm install -g`` in it names written at the version ``versions`` holds."""
    def install(match: re.Match) -> str:
        segment = match.group(0)
        if not _GLOBAL.search(segment):
            return segment
        return _WORD.sub(lambda word: _pinned_word(word.group(0), versions, pinned), segment)
    return _NPM_INSTALL.sub(install, text)


def _pinned_word(word: str, versions: dict[str, str], pinned: list[str]) -> str:
    if word.startswith("-"):
        return word
    name = "@" + word[1:].partition("@")[0] if word.startswith("@") else word.partition("@")[0]
    if name not in versions:
        return word
    pinned.append(f"{name}@{versions[name]}")
    return f"{name}@{versions[name]}"


def _constraints_path(folder: Path, dockerfile: str) -> str:
    """Where the constraints go: beside the Dockerfile, under a name of the import's own when that one is taken."""
    path = posixpath.join(posixpath.dirname(dockerfile), CONSTRAINTS)
    return path if not (folder / path).exists() else posixpath.join(posixpath.dirname(dockerfile),
                                                                    f"agent-env-{CONSTRAINTS}")


def _ignored(folder: Path, dockerfile: str, path: str) -> bool:
    """Whether the ignore file the build reads leaves ``path`` out of its context."""
    ignore = next((folder / name for name in ignore_names(dockerfile) if (folder / name).is_file()), None)
    return ignore is not None and ignore_matcher(ignore.read_text(encoding="utf-8"))(Path(path))
