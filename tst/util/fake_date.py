"""A stand-in ``date`` for the tests of the clocked agents' date rewrites (opencode_date, codex_date), which read the
env's time from a ``date`` the A2A server starts: it prints ``FAKE_DATE_OUT`` (and fails without one), sleeps
``FAKE_DATE_SLEEP`` first, leaves a child holding its output when ``FAKE_DATE_KID`` names a file for its pid, and exits
``FAKE_DATE_EXIT``, recording the environment it got."""

import os
import time
from pathlib import Path

SCRIPT = ('#!/bin/sh\nenv > "$FAKE_DATE_ENV"\n[ "$1" = "+%s %N" ] || exit 2\n'
          '[ -n "$FAKE_DATE_SLEEP" ] && sleep "$FAKE_DATE_SLEEP"\n'
          '[ -n "$FAKE_DATE_KID" ] && { sleep 30 & echo $! > "$FAKE_DATE_KID"; }\n'
          '[ -n "$FAKE_DATE_OUT" ] || { echo no clock >&2; exit 1; }\necho "$FAKE_DATE_OUT"\n'
          'exit "${FAKE_DATE_EXIT:-0}"\n')


def install(tmp_path: Path, monkeypatch) -> Path:
    """Put the stand-in first on PATH, with the server's real-time marker set and the zone UTC; the file the stand-in
    records its environment in, there once it has run. The caller undoes ``monkeypatch`` and calls ``time.tzset()``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "date").write_text(SCRIPT)
    (bin_dir / "date").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DATE_ENV", str(tmp_path / "date.env"))
    monkeypatch.setenv("AGENTENV_REAL_TIME", "1")
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    return tmp_path / "date.env"
