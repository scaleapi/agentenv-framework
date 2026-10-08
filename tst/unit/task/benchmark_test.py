"""A benchmark baseline must carry its own helper functions and state schema."""

from tst.benchmarks import task_graphs


def test_task_benchmark_loads_helpers_and_state_from_the_selected_revision(monkeypatch):
    source = '''
@dataclass
class _SchedulerState:
    baseline_marker: str

def _dependency_ids(steps):
    return {"baseline": set()}

def _ancestor_ids(step_id, dependencies):
    return set(dependencies)

class Task:
    @classmethod
    def _validate_dag(cls, steps):
        return _ancestor_ids("step", _dependency_ids(steps))

    def _build_scheduler_state(self, steps, start_step):
        return _SchedulerState(baseline_marker="baseline")
'''
    monkeypatch.setattr(task_graphs.subprocess, "check_output", lambda *args, **kwargs: source)

    baseline = task_graphs._baseline_methods("baseline-ref")

    assert baseline._validate_dag([]) == {"baseline"}
    assert baseline._build_scheduler_state(object.__new__(baseline), [], 0).baseline_marker == "baseline"
