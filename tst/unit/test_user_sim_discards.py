"""scripts/user_sim_discards.py: the user sim judge's discards per assistant model, read from run records."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "user_sim_discards.py"
_spec = importlib.util.spec_from_file_location("user_sim_discards", SCRIPT)
discards = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(discards)


def _instance(n: int, model: str, value: str | None, because=(), skipped: str | None = None) -> dict:
    entry = {"format": "user_sim_checks", "run_step_id": "week", "user_sim_error": {"value": value,
                                                                                     "because": list(because)}}
    if skipped:
        entry = {**entry, "skipped": skipped, "user_sim_error": {"value": "no"}}
    return {"instance_id": f"run-{n}", "context": {"metadata": {
        "instance_id": f"run-{n}", "models": {"solver": {"week": model}},
        "verifications": {"usj": entry, "psi_process": {"format": "rubric_partial", "score": 0.5}}}}}


def _records(tmp_path: Path) -> Path:
    runs = [_instance(1, "claude-opus-5-5", "no"), _instance(2, "claude-opus-5-5", "yes", ["ending"]),
            _instance(3, "qwen-flash-next", "yes", ["ending", "oversharing"]),
            _instance(4, "qwen-flash-next", "unknown"), _instance(5, "qwen-flash-next", "yes", ["ending"]),
            _instance(6, "qwen-flash-next", None, skipped="the run ended with error_type=timeout")]
    for i, run in enumerate(runs):
        (tmp_path / f"r{i}.json").write_text(json.dumps(run))
    (tmp_path / "copy.json").write_text(json.dumps([runs[1]]))  # the same run exported twice counts once
    (tmp_path / "notes.json").write_text("not json")
    return tmp_path


def test_each_assistant_model_s_judged_runs_discards_unknowns_rate_and_checks(tmp_path):
    got = discards.tally([str(_records(tmp_path))])
    assert got["claude-opus-5-5"] == {"judged": 2, "discarded": 1, "unknown": 0, "rate": 0.5, "checks": {"ending": 1}}
    assert got["qwen-flash-next"] == {"judged": 3, "discarded": 2, "unknown": 1, "rate": 2 / 3,
                                      "checks": {"ending": 2, "oversharing": 1}}  # the skipped run isn't judged


def test_it_prints_a_row_per_model(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), str(_records(tmp_path))], capture_output=True, text=True,
                         check=True).stdout
    [opus] = [line for line in out.splitlines() if line.startswith("claude-opus-5-5")]
    assert opus.split()[1:5] == ["2", "1", "0", "50%"] and opus.endswith("ending 1")


def test_records_without_a_run_id_each_count(tmp_path):
    run = _instance(1, "claude-opus-5-5", "yes", ["rogue"])
    for ctx in (run["context"], run):
        ctx.pop("instance_id", None)
        ctx.get("metadata", {}).pop("instance_id", None)
    (tmp_path / "a.json").write_text(json.dumps(run))
    (tmp_path / "b.json").write_text(json.dumps([run, run]))  # three records, none with a run id
    assert discards.tally([str(tmp_path)])["claude-opus-5-5"]["judged"] == 3
