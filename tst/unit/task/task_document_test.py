from agent_env.task import Task


def test_a_stored_task_document_with_an_unread_project_id_still_loads():
    """Stored task documents may carry keys Task no longer reads, such as project_id."""
    task = Task.from_dict({"id": "t", "type": "task", "version": 3, "steps": [], "project_id": "p"})

    assert (task.id, task.version, task.steps) == ("t", 3, [])
    assert "project_id" not in task.to_dict()
