from __future__ import annotations

import shlex


def write_file_script(path: str, content: str) -> str:
    """A script that writes ``content`` and a newline to ``path``, the content one quoted argument so no byte of it is shell."""
    return f"printf '%s\\n' {shlex.quote(content)} > {shlex.quote(path)}"
