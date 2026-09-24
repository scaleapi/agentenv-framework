"""Get a task instance by ID."""

import json

import click


@click.command("get-instance")
@click.option("--id", "instance_id", required=True, help="Task instance ID")
def get_instance(instance_id: str):
    """Look up a task run instance."""
    from agent_env.store.base import NotFoundError

    from agent_env.task.store import get_task_instance_store

    try:
        instance = get_task_instance_store().get(instance_id)
    except NotFoundError:
        click.echo(f"Error: Task instance '{instance_id}' not found", err=True)
        raise SystemExit(1)

    click.echo(f"Instance ID: {instance.instance_id}")
    click.echo(f"Task ID: {instance.task_id}")
    click.echo(f"Task Version: {instance.task_version}")
    click.echo(f"Status: {instance.status}")
    click.echo(f"Progress: {instance.current_step}/{instance.total_steps}")
    if instance.created_at_utc:
        click.echo(f"Created At (UTC): {instance.created_at_utc}")
    if instance.completed_at_utc:
        click.echo(f"Completed At (UTC): {instance.completed_at_utc}")
    if instance.error:
        click.echo(f"Error: {instance.error}")
    if instance.context:
        click.echo(f"Context: {json.dumps(instance.context, indent=2)}")
