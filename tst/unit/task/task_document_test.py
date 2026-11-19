import pytest

from agent_env.plugins import _requirements
from agent_env.task import Task
from agent_env.task.task import UnknownStepTypeError
from tst.unit.bundle._support import GRADED_STEPS


def test_a_stored_task_document_with_an_unread_project_id_still_loads():
    """Stored task documents may carry keys Task no longer reads, such as project_id."""
    task = Task.from_dict({"id": "t", "type": "task", "version": 3, "steps": [], "project_id": "p"})

    assert (task.id, task.version, task.steps) == ("t", 3, [])
    assert "project_id" not in task.to_dict()


def test_a_task_records_what_it_requires_only_when_given_it():
    assert "requires" not in Task("t", 1).to_dict()
    task = Task.from_dict({"id": "t", "type": "task", "version": 2, "steps": [], "requires": ["agentenv-acme>=1.2"]})

    assert task.requires == ["agentenv-acme>=1.2"]
    assert task.to_dict()["requires"] == ["agentenv-acme>=1.2"]


_UNKNOWN = {"id": "t", "type": "task", "version": 2, "steps": [{"id": "s", "type": "acme_task_document_test"}]}


def test_a_step_type_nothing_registers_names_the_package_the_task_requires(monkeypatch):
    monkeypatch.setattr(_requirements, "installed", {}.get)

    with pytest.raises(UnknownStepTypeError) as caught:
        Task.from_dict({**_UNKNOWN, "requires": ["agentenv-acme>=1.2", "agentenv-other>=1"]})

    assert str(caught.value) == (
        "Unknown task step type: acme_task_document_test; task t v2: requires 'agentenv-acme>=1.2', which isn't "
        "installed; task t v2: requires 'agentenv-other>=1', which isn't installed; task t v2: install what it requires "
        "with: agent-env plugin add 'agentenv-acme>=1.2' 'agentenv-other>=1'")
    assert caught.value.requirements == ("agentenv-acme>=1.2", "agentenv-other>=1")


@pytest.mark.parametrize("record", [{}, {"requires": ["agentenv-acme>=1.2"]}], ids=["no-record", "met"])
def test_without_an_unmet_requirement_it_names_only_the_type(monkeypatch, record):
    monkeypatch.setattr(_requirements, "installed", {"agentenv-acme": "1.3"}.get)

    with pytest.raises(UnknownStepTypeError) as caught:
        Task.from_dict({**_UNKNOWN, **record})

    assert str(caught.value) == "Unknown task step type: acme_task_document_test"
    assert caught.value.requirements == ()


@pytest.mark.parametrize("record", [["not a requirement!"], ["evil; os_name != \"\x1b[2K\ragentenv-browser\""], [7]],
                         ids=["unparseable", "terminal-escape", "not-a-string"])
def test_a_record_that_isnt_a_requirement_is_named_and_never_offered(monkeypatch, record):
    monkeypatch.setattr(_requirements, "installed", {}.get)

    with pytest.raises(UnknownStepTypeError) as caught:
        Task.from_dict({**_UNKNOWN, "requires": [*record, "agentenv-acme>=1.2"]})

    assert "isn't a pip requirement that can be shown as written" in str(caught.value)
    assert caught.value.requirements == ("agentenv-acme>=1.2",)
    assert "plugin add 'agentenv-acme>=1.2'" in str(caught.value) and repr(record[0]) in str(caught.value)


def test_a_record_that_isnt_a_list_is_read_as_none():
    assert Task.from_dict({"id": "t", "type": "task", "version": 1, "steps": [], "requires": "agentenv-acme"}).requires \
        is None


def _graded(**fields) -> dict:
    return {"id": "t", "type": "task", "version": 1, "steps": [dict(step) for step in GRADED_STEPS], **fields}


@pytest.mark.parametrize("headlines", [["final"], ["tests", "final"], ["final", "tests"]])
def test_a_tasks_headline_verifiers_round_trip_in_the_order_given(headlines):
    task = Task.from_dict(_graded(headline_verifiers=headlines))

    assert task.headline_verifiers == headlines
    assert Task.from_dict(task.to_dict()).to_dict()["headline_verifiers"] == headlines


@pytest.mark.parametrize("fields", [{}, {"headline_verifiers": []}, {"headline_verifiers": "final"}],
                         ids=["absent", "empty", "not-a-list"])
def test_a_task_naming_no_headline_verifiers_records_none(fields):
    task = Task.from_dict(_graded(**fields))

    assert task.headline_verifiers is None
    assert "headline_verifiers" not in task.to_dict()


@pytest.mark.parametrize(("headlines", "message"), [
    (["final", "verify"], "headline_verifiers names no step in this task: ['verify'] (its steps: ['box', 'final', "
                          "'tests'])"),
    (["final", 3], "headline_verifiers names no step in this task: [3]"),
    (["final", "tests", "final"], "headline_verifiers names ['final'] more than once"),
], ids=["unknown", "not-a-string", "twice"])
def test_a_headline_naming_no_step_of_the_task_or_named_twice_is_refused(headlines, message):
    with pytest.raises(ValueError) as refused:
        Task.from_dict(_graded(headline_verifiers=headlines))

    assert str(refused.value).startswith(message)


def test_each_version_of_a_task_keeps_its_own_headline_verifiers(local_stores):
    steps = Task.from_dict(_graded()).steps

    first = Task.put(id="t", steps=steps, headline_verifiers=["final"])
    second = Task.put(id="t", steps=steps, headline_verifiers=["tests", "final"])

    assert Task.get("t", first.version).headline_verifiers == ["final"]
    assert (Task.get("t").version, Task.get("t").headline_verifiers) == (second.version, ["tests", "final"])
