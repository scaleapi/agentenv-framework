"""Write the Claude Code ATIF fixtures (``tst/data/claude_code_atif``) from what ``record.py`` recorded: one file per
scenario, a context's turns, each its atif/1 final read's turn context and aux log, its native final read's events and
the usage the turn reported. Each request shape's tools go to ``tools/<hash16>.json`` once, the aux record naming it,
since every run on a CLI version offers the same ones. Then run the recorded test as a script for the expected
documents.

    python tst/manual/claude_code_atif/make_fixtures.py OUT"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tst" / "data" / "claude_code_atif"
KEY = re.compile(r"sk-[A-Za-z0-9_-]{18,}")


def cli_version(turns: list[dict]) -> str | None:
    return next((event.get("claude_code_version") for turn in turns for event in turn["native"]
                 if event.get("type") == "system" and event.get("subtype") == "init"), None)


def usage_of(task: dict) -> dict | None:
    return next((part["data"]["usage"] for part in task["status"]["message"]["parts"]
                 if part.get("kind") == "data" and isinstance(part.get("data"), dict) and "usage" in part["data"]),
                None)


def main() -> None:
    out = Path(sys.argv[1])
    (FIXTURES / "tools").mkdir(parents=True, exist_ok=True)
    for folder in sorted(path for path in out.iterdir() if path.is_dir()):
        turns = []
        for path in sorted(folder.glob("turn-*.json"), key=lambda p: int(p.stem.split("-")[1])):
            recorded = json.loads(path.read_text())
            atif = recorded["atif"]
            if not isinstance(atif, dict) or "turn_context" not in atif:
                print(folder.name, path.name, "has no atif/1 read:", str(atif)[:200])
                continue
            aux = []
            for entry in atif.get("aux") or []:
                record = entry["record"]
                if entry["name"] == "tools":
                    shared = f"tools/{record['shapeHash'][:16]}.json"
                    (FIXTURES / shared).write_text(json.dumps(record["tools"], indent=1, ensure_ascii=False) + "\n")
                    record = {**record, "tools": {"shared": shared}}
                aux.append({**entry, "record": record})
            turns.append({"turn": atif["turn_context"], "native": recorded["native"]["trajectory"], "aux": aux,
                          "usage": usage_of(recorded["task"])})
        if not turns:
            continue
        text = json.dumps({"scenario": folder.name, "cli": cli_version(turns), "turns": turns}, indent=1,
                          ensure_ascii=False) + "\n"
        if KEY.search(text):
            raise SystemExit(f"{folder.name}: a key-like string; not written")
        (FIXTURES / f"{folder.name}.json").write_text(text)
        print(folder.name, len(turns), "turns,", len(text) // 1024, "KB")


if __name__ == "__main__":
    main()
