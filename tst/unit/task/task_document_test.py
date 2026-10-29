import pytest

from agent_env.plugins import _requirements
from agent_env.task import Task
from agent_env.task.task import UnknownStepTypeError


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
