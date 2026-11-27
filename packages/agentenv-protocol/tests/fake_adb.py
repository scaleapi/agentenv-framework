"""A stand-in for adb that the tests of a driver beside an Android device run as ``AGENTENV_ADB``: ``/sdcard`` is the
folder ``FAKE_DEVICE`` names, ``shell`` runs its command with sh there, each call is logged beside that folder, and the
``FAKE_*`` variables make it fail, hang or start its server first."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    device = Path(os.environ["FAKE_DEVICE"])
    args = sys.argv[1:]
    if os.environ.get("FAKE_OFFLINE"):
        print("error: device offline", file=sys.stderr)
        return 1
    if os.environ.get("FAKE_DAEMON_STARTS"):
        print("* daemon not running; starting now at tcp:5037\n* daemon started successfully", file=sys.stderr)
    with open(device.parent / "calls.jsonl", "a") as out:
        out.write(json.dumps({"args": args, "serial": os.environ.get("ANDROID_SERIAL"),
                              "server": os.environ.get("ADB_SERVER_SOCKET")}) + "\n")

    def local(remote: str) -> Path:
        return device / Path(remote).relative_to("/sdcard")

    if args[0] == "pull" and os.environ.get("FAKE_PULL_HANGS"):
        (device.parent / "adb.pid").write_text(str(os.getpid()))
        time.sleep(60)
    if args[0] == "push" and os.environ.get("FAKE_PUSH_FAILS"):
        local(args[2]).write_bytes(b"half")
        print("error: connection reset", file=sys.stderr)
        return 1
    if args[0] == "push":
        shutil.copy(args[1], local(args[2]))
    elif args[0] == "pull":
        source = local(args[1])
        (shutil.copytree if source.is_dir() else shutil.copy)(source, Path(args[2]) / source.name)
    elif args[0] == "shell" and not args[1].startswith("am broadcast"):
        return subprocess.run(["sh", "-c", args[1].replace("/sdcard", str(device))]).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
