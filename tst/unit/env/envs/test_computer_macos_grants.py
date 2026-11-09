"""The macOS VM bundle's grants, lib.sh sourced in bash with macOS's tools stood in for: a screen capture grant records
its approval ahead under the client TCC knows it by, and for an app under its bundle's file URL too, which macOS keys an
app's "bypass the system private window picker" approval by, each hinted at no sooner than it's approved."""
import subprocess
from pathlib import Path

import agent_env

LIB = Path(agent_env.__file__).parent / "computers" / "vm_bundle" / "macos" / "lib.sh"
APPROVED = ["-dict", "kScreenCaptureApprovalLastAlerted", "-date", "3024-01-01 00:00:00 +0000",
            "kScreenCaptureApprovalLastUsed", "-date", "3024-01-01 00:00:00 +0000",
            "kScreenCapturePrivacyHintDate", "-date", "3024-01-01 00:00:00 +0000",
            "kScreenCapturePrivacyHintPolicy", "-int", "3153600000"]
STUBS = {
    "sudo": 'shift 3; exec "$@"',
    "sqlite3": 'printf "sqlite3\\n--\\n" >> "$LOG"',
    "codesign": 'echo "designated => identifier \\"com.example.app\\""',
    "csreq": 'cat > /dev/null; : > "$3"',
    "xxd": "echo 00",
    "pgrep": "exit 1",
    "defaults": 'if [ "$1" = read ]; then echo com.example.app; else printf "%s\\n" "$@" "--" >> "$LOG"; fi',
}


def _granted(tmp_path, *commands):
    stubs, home, log = tmp_path / "stubs", tmp_path / "home", tmp_path / "log"
    stubs.mkdir()
    for name, body in STUBS.items():
        (stubs / name).write_text(f"#!/bin/bash\n{body}\n")
        (stubs / name).chmod(0o755)
    (tmp_path / ".password").write_text("")
    script = "\n".join(["set -euo pipefail", f'. "{LIB}"', f'export PATH="{stubs}:$PATH"', f'build="{tmp_path}"',
                        *commands])
    subprocess.run(["bash", "-c", script], check=True,
                   env={"HOME": str(home), "LOG": str(log), "PATH": "/usr/bin:/bin"})
    written = [call.strip("\n").split("\n") for call in log.read_text().split("--\n") if call.strip("\n")]
    return [call for call in written if call != ["sqlite3"]], home


def test_an_apps_screen_capture_is_approved_ahead_by_its_identifier_and_its_bundles_file_url(tmp_path):
    writes, home = _granted(tmp_path, 'grant "/opt/agentenv/codex/Codex Computer Use.app" accessibility screen_capture')

    approvals = f"{home}/Library/Group Containers/group.com.apple.replayd/ScreenCaptureApprovals.plist"
    assert writes == [["write", approvals, "com.example.app", *APPROVED],
                      ["write", approvals, "file:///opt/agentenv/codex/Codex%20Computer%20Use.app/", *APPROVED]]


def test_an_executables_screen_capture_is_approved_ahead_by_its_path_alone_and_no_other_service_is(tmp_path):
    tool = tmp_path / "tool"
    tool.write_text("")

    writes, home = _granted(tmp_path, f'grant_executable "{tool}" screen_capture',
                            f'grant_executable "{tool}" post_event')

    approvals = f"{home}/Library/Group Containers/group.com.apple.replayd/ScreenCaptureApprovals.plist"
    assert writes == [["write", approvals, str(tool), *APPROVED]]
