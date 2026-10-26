"""Filesystem plumbing for the bundle parser: a path spelled as the filesystem stores it, the user's
home, a regular file read without surprises, names compared without case, and paths rendered for
messages."""

from __future__ import annotations

import os
import stat
import sys
import unicodedata
from pathlib import Path

from agent_env.utils.paths import os_reason

try:
    import fcntl
    import pwd
except ImportError:  # Windows, which agent-env does not target.
    fcntl = pwd = None

# Config files are small; the cap keeps a stray large file, or a link to one, from being read whole.
MAX_FILE_BYTES = 8 << 20
_UNSHOWN = frozenset({"Cc", "Cf", "Zl", "Zp"})


class Unreadable(Exception):
    pass


def on_disk(path: Path) -> Path:
    """``path`` spelled as the filesystem stores it; a strict resolve keeps the typed case on a
    case-insensitive filesystem. macOS answers with ``F_GETPATH``. Elsewhere each component's
    spelling comes from its parent's listing, where the parent can be listed."""
    if sys.platform == "darwin" and fcntl is not None:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                spelled = fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024))
            finally:
                os.close(fd)
        except OSError:
            return path
        return Path(os.fsdecode(spelled.split(b"\0", 1)[0]))
    spelled_path = Path(path.anchor)
    for part in path.parts[1:]:
        try:
            names = os.listdir(spelled_path)
        except OSError:
            names = []
        if part not in names:
            matches = [name for name in names if fold(name) == fold(part)]
            part = matches[0] if len(matches) == 1 else part
        spelled_path /= part
    return spelled_path


def home() -> Path | None:
    """The user's home: ``$HOME`` if it is an absolute folder other than ``/``, else the passwd
    entry's, else none."""
    candidates = [os.environ.get("HOME", "")]
    if pwd is not None:
        try:
            candidates.append(pwd.getpwuid(os.getuid()).pw_dir)
        except KeyError:
            pass
    for candidate in candidates:
        if not candidate.startswith("/"):
            continue
        try:
            resolved = Path(candidate).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved.is_dir() and resolved != Path(resolved.anchor):
            return on_disk(resolved)
    return None


def read_regular(path: Path) -> bytes:
    """A regular file's bytes, opened without blocking so a FIFO can't hang the parse. Raises
    Unreadable with the reason otherwise."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError as e:
        raise Unreadable(os_reason(e)) from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Unreadable("not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(MAX_FILE_BYTES + 1)
    except OSError as e:
        raise Unreadable(os_reason(e)) from None
    finally:
        os.close(fd)
    if len(data) > MAX_FILE_BYTES:
        raise Unreadable(f"larger than the {MAX_FILE_BYTES >> 20} MiB limit for a bundle's config files")
    return data


def fold(name: str) -> str:
    """The canonical caseless form: casefold output isn't always NFC, so normalize both sides."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold())


def with_article(noun: str) -> str:
    return f"{'an' if noun[:1] in 'aeiou' else 'a'} {noun}"


def relative(root: Path, path: Path) -> str:
    """``path`` relative to the bundle's ``root``, for a message."""
    return show(os.path.relpath(path, root))


def show(value) -> str:
    """Text for a message, with undecodable bytes, lone surrogates and control, format and
    line-separator characters escaped, so a problem stays one printable line."""
    text = str(value)
    try:
        text = os.fsencode(text).decode("utf-8", "backslashreplace")
    except UnicodeEncodeError:
        text = text.encode("utf-8", "backslashreplace").decode("utf-8")
    return "".join(ascii(c)[1:-1] if unicodedata.category(c) in _UNSHOWN else c for c in text)
