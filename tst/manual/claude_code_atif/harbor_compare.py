"""Convert one recorded session with Harbor's own Claude Code converter, for ``harbor/COMPARISON.md`` and the
recorded test's ``from_atif`` check. Run in an environment with Harbor installed (``pip install harbor``):

    python tst/manual/claude_code_atif/harbor_compare.py OUT/subagent tst/data/claude_code_atif/harbor/subagent.harbor.json \
        [MODEL]

``MODEL`` is the model the recording ran, ``anthropic/claude-sonnet-5-5`` by default."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from harbor.agents.installed.claude_code import ClaudeCode


def main() -> None:
    scenario, out = Path(sys.argv[1]), Path(sys.argv[2])
    model = sys.argv[3] if len(sys.argv) > 3 else "anthropic/claude-sonnet-5-5"
    with tempfile.TemporaryDirectory() as folder:
        logs = Path(folder)
        shutil.copytree(scenario / "projects", logs / "sessions" / "projects")
        stdout = []
        for turn in sorted(scenario.glob("turn-*.json")):
            stdout += json.loads(turn.read_text())["native"]["trajectory"]
        (logs / "claude-code.txt").write_text("\n".join(json.dumps(event) for event in stdout) + "\n")
        trajectory = ClaudeCode(logs_dir=logs, model_name=model).convert_trajectory(logs)
        out.write_text(json.dumps(trajectory.to_json_dict(), indent=1) + "\n")
        print("steps", len(trajectory.steps))


if __name__ == "__main__":
    main()
