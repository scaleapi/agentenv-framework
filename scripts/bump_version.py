"""Bump the patch version in pyproject.toml.

Picks the next version as
``max(pyproject_version, highest existing v*.*.* tag) + patch`` so the
result never collides with an already-pushed tag, even if pyproject
and tag history have drifted.

Also bumps the patch version of ``packages/agentenv-protocol/pyproject.toml``
on every release so the protocol wheel is always freshly published and
agent-env can pin it exactly. Its version is its own X.Y.Z series,
independent of agent-env's tags.

Writes the new version back to ``pyproject.toml`` via ``tomlkit``
(preserves formatting) and prints the new version to **stdout** as a
single line. Progress logs go to **stderr** so callers can capture
the new version cleanly:

    NEW_VERSION=$(python scripts/bump_version.py)

It also writes both new versions into the two packages' editable entries in
``uv.lock``. That is a text edit, not ``uv lock``: a release must not
re-resolve anything, and it keeps ``uv lock --check`` passing on every bump
commit. Nothing is written until all three edits are known to apply, so a
failure leaves the files as they were.

Used by .circleci/config.yml's ``bump`` job, and by CI's ``unit`` job to check
that the lock survives a bump; can also be run locally.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import tomlkit

PYPROJECT = Path("pyproject.toml")
PROTOCOL_PYPROJECT = Path("packages/agentenv-protocol/pyproject.toml")
UV_LOCK = Path("uv.lock")
SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_semver(s: str) -> tuple[int, int, int] | None:
    m = SEMVER_RE.match(s.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def highest_tag_version() -> tuple[int, int, int] | None:
    """Return the highest semver from ``git tag -l 'v*.*.*'``, or None."""
    out = subprocess.run(
        ["git", "tag", "-l", "v*.*.*"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    versions: list[tuple[int, int, int]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("v"):
            continue
        parsed = parse_semver(line[1:])
        if parsed is not None:
            versions.append(parsed)
    return max(versions) if versions else None


def main() -> int:
    doc = tomlkit.parse(PYPROJECT.read_text())
    current_str = str(doc["project"]["version"])
    current = parse_semver(current_str)
    if current is None:
        print(f"ERROR: pyproject version '{current_str}' is not X.Y.Z", file=sys.stderr)
        return 1

    # agentenv-framework-protocol's own patch version moves on every release too.
    pdoc = tomlkit.parse(PROTOCOL_PYPROJECT.read_text())
    pcur_str = str(pdoc["project"]["version"])
    pcur = parse_semver(pcur_str)
    if pcur is None:
        print(f"ERROR: agentenv-framework-protocol version '{pcur_str}' is not X.Y.Z", file=sys.stderr)
        return 1

    tag = highest_tag_version()
    base = max(current, tag) if tag is not None else current
    new_str = f"{base[0]}.{base[1]}.{base[2] + 1}"
    pnew_str = f"{pcur[0]}.{pcur[1]}.{pcur[2] + 1}"

    lock = lock_with_versions(
        UV_LOCK.read_text(),
        {str(doc["project"]["name"]): new_str, str(pdoc["project"]["name"]): pnew_str},
    )
    if lock is None:
        return 1

    doc["project"]["version"] = new_str
    pdoc["project"]["version"] = pnew_str
    PYPROJECT.write_text(tomlkit.dumps(doc))
    PROTOCOL_PYPROJECT.write_text(tomlkit.dumps(pdoc))
    UV_LOCK.write_text(lock)

    print(
        f"Bumped version: pyproject={current_str} "
        f"latest_tag={'v' + '.'.join(map(str, tag)) if tag else 'none'} "
        f"-> {new_str}",
        file=sys.stderr,
    )
    print(f"Bumped agentenv-framework-protocol: {pcur_str} -> {pnew_str}", file=sys.stderr)
    print("Updated both editable entries in uv.lock", file=sys.stderr)
    print(new_str)
    return 0


def lock_with_versions(lock: str, versions: dict[str, str]) -> str | None:
    """``lock`` with each package's editable entry at its new version, or None unless each has exactly one."""
    for name, version in versions.items():
        entry = re.compile(rf'(name = "{re.escape(name)}"\nversion = ")[^"]+("\nsource = \{{ editable)')
        lock, n = entry.subn(rf"\g<1>{version}\g<2>", lock)
        if n != 1:
            print(f"ERROR: uv.lock has {n} editable entries for {name}, expected 1", file=sys.stderr)
            return None
    return lock


if __name__ == "__main__":
    sys.exit(main())
