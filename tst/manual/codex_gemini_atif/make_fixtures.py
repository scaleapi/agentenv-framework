"""Write the Codex or Gemini ATIF fixtures (``tst/data/codex_atif``, ``tst/data/gemini_atif``) from what ``record.py``
recorded: one file per scenario, a context's turns, each its atif/1 final read's turn context and aux log, its native
final read's events and the usage the turn reported. Then run the agent's recorded test as a script for the expected
documents.

    python tst/manual/codex_gemini_atif/make_fixtures.py OUT codex|gemini"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KEY = re.compile(r"sk-[A-Za-z0-9_-]{18,}")


def usage_of(task: dict) -> dict | None:
    return next((part["data"]["usage"] for part in task["status"]["message"]["parts"]
                 if part.get("kind") == "data" and isinstance(part.get("data"), dict) and "usage" in part["data"]),
                None)


def main() -> None:
    out, agent = Path(sys.argv[1]) / sys.argv[2], sys.argv[2]
    fixtures = REPO / "tst" / "data" / f"{agent}_atif"
    fixtures.mkdir(parents=True, exist_ok=True)
    cli = (out / "cli_version.txt").read_text().split()[-1]
    for folder in sorted(path for path in out.iterdir() if path.is_dir()):
        turns = []
        for path in sorted(folder.glob("turn-*.json"), key=lambda p: int(p.stem.split("-")[1])):
            recorded = json.loads(path.read_text())
            atif = recorded["atif"]
            if not isinstance(atif, dict) or "turn_context" not in atif:
                print(folder.name, path.name, "has no atif/1 read:", str(atif)[:200])
                continue
            turns.append({"turn": atif["turn_context"], "native": recorded["native"]["trajectory"],
                          "aux": atif.get("aux") or [], "usage": usage_of(recorded["task"])})
        if not turns:
            continue
        text = json.dumps({"scenario": folder.name, "cli": cli, "turns": turns}, indent=1, ensure_ascii=False) + "\n"
        if KEY.search(text):
            raise SystemExit(f"{folder.name}: a key-like string; not written")
        (fixtures / f"{folder.name}.json").write_text(text)
        print(folder.name, len(turns), "turns,", len(text) // 1024, "KB")


if __name__ == "__main__":
    main()
