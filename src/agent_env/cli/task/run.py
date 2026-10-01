"""Run a stored task."""

import asyncio
import copy
import csv
import json
import os
import textwrap
from uuid import uuid4

import click

from agent_env.cli.banner import print_banner
from agent_env.cli.identity import get_agent_env_client_id
from agent_env.store.ids import derive_id, fs_safe, is_local_id, validate_local_id
from agent_env.task_step.task_steps.collect_artifacts import CollectArtifactsTaskStep


_print_lock = asyncio.Lock()


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent each line of text."""
    return textwrap.indent(text, prefix)


def _format_env(env) -> str:
    from agent_env.env.env import DeployedGatewayEnv, DeployedSandboxEnv

    fronted = isinstance(env, DeployedGatewayEnv)  # a record without a gateway has none of its URLs
    lines = [
        "deployed-env:",
        f"  instance_id: {env.instance_id}",
        f"  env_id: {env.env_id}",
        f"  env_version: {env.env_version}",
        *([f"  gateway_url: {env.gateway_url}"] if fronted else []),
        f"  mcp_url: {env.mcp_url}",
        *([f"  db_web_url: {env.db_web_url}", f"  db_mcp_url: {env.db_mcp_url}"] if fronted else []),
    ]
    if isinstance(env, DeployedSandboxEnv):  # an env outside our sandboxes has no sandbox to name
        lines.append(f"  sandbox_id: {env.sandbox_id}")
        if env.sandbox_type:
            lines.append(f"  sandbox_type: {env.sandbox_type}")
    if fronted and env.vnc_url:
        lines.append(f"  vnc_url: {env.vnc_url}")
    return "\n".join(lines)


def _format_agent(agent) -> str:
    lines = [
        "deployed-agent:",
        f"  agent_name: {agent.agent_name}",
        f"  api_url: {agent.api_url}",
    ]
    if agent.a2a_url:
        lines.append(f"  a2a_url: {agent.a2a_url}")
    if agent.sandbox_id:
        lines.append(f"  sandbox_id: {agent.sandbox_id}")
    if agent.sandbox_type:
        lines.append(f"  sandbox_type: {agent.sandbox_type}")
    if agent.a2a_card:
        lines.append(f"  agent_card: {json.dumps(agent.a2a_card, indent=2)}")
    return "\n".join(lines)


def _format_tool_access_change(change: dict) -> str:
    state = change.get("role_state_after") or {}
    disabled = state.get("disabled") or []
    allowed = state.get("allowed") or []
    requested = change.get("tools_requested") or []
    lines = [
        "tool-access-change:",
        f"  env_id: {change.get('env_id')}",
        f"  action: {change.get('action')}",
        f"  role: {change.get('role')}",
        f"  tools_requested ({len(requested)}): {requested if len(requested) <= 5 else requested[:5] + ['...']}",
        f"  role_state_after.disabled ({len(disabled)}): {disabled if len(disabled) <= 5 else disabled[:5] + ['...']}",
        f"  role_state_after.allowed ({len(allowed)}): {allowed if len(allowed) <= 5 else allowed[:5] + ['...']}",
    ]
    return "\n".join(lines)


def _format_server_config_changes(changes: list[dict]) -> str:
    lines = [f"server-config-applied ({len(changes)}):"]
    for c in changes:
        lines.append(f"  service={c.get('service')} uri={c.get('uri')} args={c.get('args')}")
    return "\n".join(lines)


def _format_prompt_response(pr) -> str:
    indented_response = _indent(pr.response, "    ")
    lines = [
        "prompt-response:",
        f"  prompt_id: {pr.prompt_id}",
        "  response: |",
        indented_response,
    ]
    if pr.agent_trajectory_s3_uri:
        lines.append(f"  agent_trajectory_s3_uri: {pr.agent_trajectory_s3_uri}")
    if pr.agent_trajectory_s3_prefix:
        lines.append(f"  agent_trajectory_s3_prefix: {pr.agent_trajectory_s3_prefix}")
    if pr.compact_trajectory_s3_uri:
        lines.append(f"  compact_trajectory_s3_uri: {pr.compact_trajectory_s3_uri}")
    if pr.agent_trajectory_file_path:
        lines.append(f"  agent_trajectory_file_path: {pr.agent_trajectory_file_path}")
    if pr.tool_call_count is not None:
        lines.append(f"  tool_call_count: {pr.tool_call_count}")
    return "\n".join(lines)


def _format_verification_results(results) -> str:
    indented = _indent(json.dumps(results, indent=2), "    ")
    lines = [
        "verification-results: |",
        indented,
    ]
    return "\n".join(lines)


def _format_snapshot_capture(snap: dict) -> str:
    lines = [
        "agent-snapshot:",
        f"  artifact_id: {snap.get('id')}",
        f"  artifact_version: {snap.get('version')}",
        f"  bundle_object_url: {snap.get('bundle_object_url') or snap.get('bundle_s3_url')}",
        f"  source_agent_name: {snap.get('source_agent_name')}",
        f"  source_context_id: {snap.get('source_context_id')}",
    ]
    return "\n".join(lines)


def _format_loaded_snapshot(load: dict) -> str:
    lines = [
        "loaded-snapshot:",
        f"  context_id: {load.get('context_id')}",
        f"  source_artifact_id: {load.get('source_artifact_id')}",
        f"  source_artifact_version: {load.get('source_artifact_version')}",
    ]
    return "\n".join(lines)


# Per-step output attribution via the step type + the `[-1]` trick.
#
# Invariant: every step that appends to context.deployed_envs / .deployed_agents
# / .prompt_responses does so as the final synchronous op in execute() — no
# await after the append. Under asyncio, that guarantees `context.<bucket>[-1]`
# at the moment on_step_complete fires is *this* step's item, even under DAG
# parallelism (no yield between the append and the callback in the same
# coroutine). Verifier output is duck-typed via `verifier_id`.
def _step_output_lines(step, context) -> list[str]:
    if step.type == "deploy_env" and context.deployed_envs:
        return [_format_env(context.deployed_envs[-1])]
    if step.type == "deploy_agent" and context.deployed_agents:
        agent = context.deployed_agents[-1]
        lines = [_format_agent(agent)]
        # If this deploy step also loaded a snapshot, surface its provenance.
        loaded = context.metadata.get("agent_loaded_snapshots") or []
        match = next(
            (e for e in reversed(loaded) if e.get("agent_name") == agent.agent_name),
            None,
        )
        if match:
            lines.append(_format_loaded_snapshot(match))
        return lines
    if step.type == "prompt_agent" and context.prompt_responses:
        return [_format_prompt_response(context.prompt_responses[-1])]
    if step.type == "snapshot_agent_state":
        snapshots = context.metadata.get("agent_snapshots") or []
        if snapshots:
            return [_format_snapshot_capture(snapshots[-1])]
    if step.type == "modify_env_tool_access":
        changes = context.metadata.get("tool_access_changes") or []
        match = next((c for c in reversed(changes) if c.get("step_id") == step.id), None)
        if match:
            return [_format_tool_access_change(match)]
    if step.type == "apply_server_config":
        changes = [c for c in (context.metadata.get("server_config_changes") or []) if c.get("step_id") == step.id]
        if changes:
            return [_format_server_config_changes(changes)]
    vid = getattr(step, "verifier_id", None)
    if vid:
        vdata = context.metadata.get("verifications", {}).get(vid)
        if vdata:
            return _format_verifier_output(vid, vdata)
    return []


def _format_verifier_output(vid: str, vdata: dict) -> list[str]:
    lines: list[str] = []
    compact_uri = vdata.get("compact_trajectory_s3_uri")
    if compact_uri:
        lines.append(f"compact_trajectory_s3_uri (verifier_id={vid}): {compact_uri}")
    judge_uri = vdata.get("judge_trajectory_s3_uri")
    if judge_uri:
        lines.append(f"judge_trajectory_s3_uri (verifier_id={vid}): {judge_uri}")
    command = vdata.get("command")
    if command is not None:
        lines.append(f"command (verifier_id={vid}): {command}")
    exit_code = vdata.get("exit_code")
    if exit_code is not None:
        timed_out = vdata.get("timed_out", False)
        lines.append(f"exit_code (verifier_id={vid}): {exit_code} (timed_out={timed_out})")
    stdout_artifact = vdata.get("stdout_artifact")
    if stdout_artifact:
        lines.append(
            f"stdout_artifact (verifier_id={vid}): "
            f"{stdout_artifact.get('id')} v{stdout_artifact.get('version')} -> {stdout_artifact.get('s3_url')}"
        )
    stdout_head = vdata.get("stdout_head")
    if stdout_head:
        lines.append(f"stdout_head (verifier_id={vid}):")
        for line in stdout_head.splitlines():
            lines.append(f"  {line}")
    stderr_artifact = vdata.get("stderr_artifact")
    if stderr_artifact:
        lines.append(
            f"stderr_artifact (verifier_id={vid}): "
            f"{stderr_artifact.get('id')} v{stderr_artifact.get('version')} -> {stderr_artifact.get('s3_url')}"
        )
    stderr_head = vdata.get("stderr_head")
    if stderr_head:
        lines.append(f"stderr_head (verifier_id={vid}):")
        for line in stderr_head.splitlines():
            lines.append(f"  {line}")
    extracted = vdata.get("extracted_files")
    if extracted:
        lines.append(f"extracted_files (verifier_id={vid}):")
        for path, parsed in extracted.items():
            if isinstance(parsed, dict):
                lines.append(f"  {path}: {json.dumps(parsed, default=str)[:600]}")
            elif parsed is None:
                lines.append(f"  {path}: <unavailable>")
            else:
                lines.append(f"  {path}: {str(parsed)[:600]}")
    results = vdata.get("results")
    if results is not None:
        lines.append(f"verification (verifier_id={vid}):")
        lines.append(_format_verification_results(results))
    score = vdata.get("score")
    if score is not None:
        lines.append(f"score (verifier_id={vid}): {score}")
    return lines


# -- Single-run (k=1) callbacks: verbose output, no prefix --

async def _log_step_start(i, total, step, context):
    async with _print_lock:
        if i == 0 and context.instance_id:
            click.echo(click.style(f"Task instance: {context.instance_id}", fg="blue"))
        if i > 0:
            click.echo()
        click.echo(click.style(
            f"Running step [{i+1}/{total}]: {step.type} (id={step.id})...",
            fg="yellow",
        ))


async def _log_step_complete(i, total, step, context, duration):
    async with _print_lock:
        click.echo(click.style(
            f"Completed step [{i+1}/{total}]: {step.type} (id={step.id}) [{duration:.1f}s]",
            fg="green",
        ))
        for line in _step_output_lines(step, context):
            click.echo(click.style(line, fg="green"))


# -- Parallel-run (k>1) callbacks: compact output with [run N] prefix --

def _make_parallel_callbacks(run_index):
    """Return (on_step_start, on_step_complete) callbacks for a parallel run."""
    tag = f"[run {run_index}]"

    async def on_step_start(i, total, step, context):
        async with _print_lock:
            if i == 0 and context.instance_id:
                click.echo(click.style(f"{tag} Task instance: {context.instance_id}", fg="blue"))
            click.echo(click.style(
                f"{tag} Running step [{i+1}/{total}]: {step.type} (id={step.id})...",
                fg="yellow",
            ))

    async def on_step_complete(i, total, step, context, duration):
        async with _print_lock:
            click.echo(click.style(
                f"{tag} Completed step [{i+1}/{total}]: {step.type} (id={step.id}) [{duration:.1f}s]",
                fg="green",
            ))
            vid = getattr(step, "verifier_id", None)
            if vid:
                vdata = context.metadata.get("verifications", {}).get(vid, {})
                score = vdata.get("score")
                if score is not None:
                    click.echo(click.style(f"{tag} score (verifier_id={vid}): {score}", fg="green"))

    return on_step_start, on_step_complete


def _write_context(context, task_id, output_dir, prefix=""):
    """Write context JSON to output_dir and print the path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{fs_safe(task_id)}_{uuid4().hex[:8]}.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(context.to_safe_dict(), f, indent=2)
    click.echo(click.style(f"{prefix}Task context written to: {output_path}", fg="blue"))


def _stamp_agent_env_client_metadata(context_metadata: dict, client_id: str | None) -> None:
    if client_id:
        context_metadata.setdefault("agent_env_hub", {}).setdefault("caller", client_id)


# Step types whose execution path makes LiteLLM calls (directly or via a
# deployed agent that consumes LITELLM_* env vars). Used to decide whether
# the cost-attribution banner is relevant for a given task.
_LITELLM_USING_STEP_TYPES: frozenset[str] = frozenset({
    "deploy_agent",
    "prompt_agent",
    "rubrics_verifier",
})


def _task_uses_litellm(task) -> bool:
    """True if any of the task's steps are known to make LiteLLM calls."""
    return any(getattr(step, "type", None) in _LITELLM_USING_STEP_TYPES for step in task.steps)


def _warn_missing_cost_attribution(
    project_id: str | None,
    task,
) -> None:
    """Yellow CLI banner when `--project-id` is missing on a LiteLLM task.

    Project ID is the primary attribution dimension (it becomes the
    LiteLLM `user` field that spend reporting joins on to derive the
    customer). Skipped when the task has no LiteLLM-using
    steps. Never blocks the run.
    """
    if project_id:
        return
    if not _task_uses_litellm(task):
        return
    click.echo(click.style(
        "⚠  LiteLLM cost attribution is incomplete: --project-id not set.",
        fg="yellow", bold=True,
    ))
    click.echo(click.style(
        "   Runs will still start, but spend won't be tagged to a "
        "project.",
        fg="yellow",
    ))
    click.echo(click.style(
        "   Pass --project-id <project-id> on your next run so the "
        "cost shows up in your team's budget.",
        fg="yellow",
    ))
    click.echo()


async def _run_single(task, run_index, task_id, output_dir, agent_model=None, agent_artifact_id=None, start_step=0, context=None):
    """Execute a single parallel run."""
    tag = f"[run {run_index}] "
    on_start, on_complete = _make_parallel_callbacks(run_index)
    context = await task.run(on_step_start=on_start, on_step_complete=on_complete, agent_model=agent_model, agent_artifact_id=agent_artifact_id, start_step=start_step, context=context)
    if output_dir:
        _write_context(context, task_id, output_dir, prefix=tag)
    return context


@click.command()
@click.option("--id", "task_id", required=True, help="Task id")
@click.option("--version", "task_version", default=None, type=int, help="Task version (defaults to latest)")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False), help="Directory to write context JSON output")
@click.option("--k", "k", default=1, type=int, help="Number of parallel task runs")
@click.option("--agent-model", default=None, type=str, help="Override the model used by prompt_agent steps")
@click.option("--agent-artifact-id", default=None, type=str, help="Override the agent image artifact for deploy_agent steps")
@click.option("--start-step", default=0, type=int, help="0-based step index to start execution from")
@click.option("--context-json", default=None, type=click.Path(exists=True, dir_okay=False), help="Path to a JSON file to seed the TaskStepContext")
@click.option("--litellm-api-key", default=None, type=str, help="Override LITELLM_API_KEY for this run")
@click.option("--judge-litellm-api-key", default=None, type=str, help="Override LITELLM_API_KEY used by rubrics_verifier judge agents (falls back to --litellm-api-key, then to config secret)")
@click.option("--apply-trajectory-filter/--no-trajectory-filter", "apply_trajectory_filter", default=None, help="Force trajectory filtering on/off for rubrics_verifier (overrides the step's trajectory_filter setting)")
@click.option("--a2a-agent-id", default=None, type=str, help="Override A2A agent id for deploy_agent steps")
@click.option("--agent-sandbox", default=None,
              help="Override agent sandbox backend(s) for deploy_agent steps; comma-separated for fallback chain (e.g. 'modal,local')")
@click.option("--project-id", "project_id", default=None, type=str,
              help="Project id for LiteLLM cost attribution. Forwarded as the `user` field on every LLM call and as `projectId:<id>` in `metadata.tags`. Optional, but spend won't be attributed to a project without it.")
@click.option("--env-sandbox", default=None,
              help="Override env sandbox backend(s) for deploy_env steps; comma-separated for fallback chain (e.g. 'modal,local')")
@click.option("--gateway-env-id", default=None, type=str,
              help="Override default_gateway_env_id (required when --env-sandbox=modal until the default gateway image is rebuilt)")
@click.option("--service-db-env-id", default=None, type=str,
              help="Override default_service_db_env_id (on Modal its images must be in the configured image store)")
@click.option("--env-state-type", default=None, type=str,
              help="Override the env state type for deploy_env steps")
@click.option("--env-state-instance-id", default=None, type=str,
              help="Attach env to an existing EnvStateInstance")
def run(task_id: str, task_version: int | None, output_dir: str | None, k: int, agent_model: str | None, agent_artifact_id: str | None, start_step: int, context_json: str | None, litellm_api_key: str | None, judge_litellm_api_key: str | None, apply_trajectory_filter: bool | None, a2a_agent_id: str | None, agent_sandbox: str | None, project_id: str | None, env_sandbox: str | None, gateway_env_id: str | None, service_db_env_id: str | None, env_state_type: str | None, env_state_instance_id: str | None):
    """Run a task by executing its steps sequentially."""
    if k < 1:
        raise click.BadParameter("must be at least 1", param_hint="'--k'")

    from agent_env.config import get_config
    from agent_env.task import Task
    from agent_env.task_step.context import TaskStepContext

    if gateway_env_id:
        get_config().default_gateway_env_id = gateway_env_id
        click.echo(f"Override default_gateway_env_id: {gateway_env_id}")
    if service_db_env_id:
        get_config().default_service_db_env_id = service_db_env_id
        click.echo(f"Override default_service_db_env_id: {service_db_env_id}")
    if output_dir is None:
        output_dir = f"/tmp/taskid={fs_safe(task_id)}-run{uuid4().hex[:6]}"
    if context_json:
        with open(context_json) as f:
            initial_context = TaskStepContext.from_dict(json.load(f))
        click.echo(f"Loaded context from {context_json}")
    else:
        initial_context = TaskStepContext()
    client_id = get_agent_env_client_id()
    _stamp_agent_env_client_metadata(initial_context.metadata, client_id)
    initial_context.metadata.setdefault("run_group_id", uuid4().hex)
    if litellm_api_key:
        initial_context.metadata.setdefault("user_overrides", {})["litellm_api_key"] = litellm_api_key
    if judge_litellm_api_key:
        initial_context.metadata.setdefault("user_overrides", {})["judge_litellm_api_key"] = judge_litellm_api_key
    if apply_trajectory_filter is not None:
        initial_context.metadata.setdefault("user_overrides", {})["apply_trajectory_filter"] = apply_trajectory_filter
    if a2a_agent_id:
        initial_context.metadata.setdefault("user_overrides", {})["a2a_agent_id"] = a2a_agent_id
    if agent_sandbox:
        initial_context.metadata.setdefault("user_overrides", {})["agent_sandbox"] = agent_sandbox
    if env_sandbox:
        initial_context.metadata.setdefault("user_overrides", {})["env_sandbox"] = env_sandbox
    if env_state_type:
        initial_context.metadata.setdefault("user_overrides", {})["env_state_type"] = env_state_type
    if env_state_instance_id:
        initial_context.metadata.setdefault("user_overrides", {})["env_state_instance_id"] = env_state_instance_id
    initial_context.metadata["task_id"] = task_id
    if project_id:
        initial_context.metadata["project_id"] = project_id

    click.echo(f"Fetching task: id={task_id} version={task_version or 'latest'}...")
    task = Task.get(task_id, version=task_version)
    click.echo(f"Found task: id={task.id} version={task.version} steps={len(task.steps)}")

    print_banner()
    _warn_missing_cost_attribution(project_id, task)

    if k > 1:
        # Parallel runs: compact output with [run N] prefixes
        click.echo(click.style(f"Running {k} task runs in parallel...", fg="blue"))
        click.echo()

        async def _run_all():
            coros = [_run_single(task, i, task_id, output_dir, agent_model=agent_model, agent_artifact_id=agent_artifact_id, start_step=start_step, context=copy.deepcopy(initial_context) if initial_context else None) for i in range(1, k + 1)]
            return await asyncio.gather(*coros, return_exceptions=True)

        results = asyncio.run(_run_all())

        click.echo()
        failures = [(i, r) for i, r in enumerate(results, 1) if isinstance(r, BaseException)]
        if failures:
            for run_index, exc in failures:
                click.echo(click.style(f"[run {run_index}] FAILED: {exc}", fg="red"))
            click.echo(click.style(
                f"{len(results) - len(failures)}/{k} runs completed, {len(failures)}/{k} failed.",
                fg="red",
            ))
            raise SystemExit(1)
        else:
            click.echo(click.style(f"All {k} runs completed!", fg="blue"))
    else:
        # Single run: verbose output
        context = asyncio.run(task.run(
            on_step_start=_log_step_start,
            on_step_complete=_log_step_complete,
            agent_model=agent_model,
            agent_artifact_id=agent_artifact_id,
            start_step=start_step,
            context=initial_context,
        ))

        click.echo()
        click.echo(click.style("Task completed!", fg="blue"))

        _write_context(context, task_id, output_dir)


def _seed_universe_id(task, seed: dict) -> str | None:
    """The universe a seed's runs collect into, so runs of one seed share it: derived from the task version and the
    seed's id, else its name. None leaves collect_artifacts to name it after the run."""
    seed_name = seed.get("id") or seed.get("name")
    return derive_id(task.id, f"v{task.version}-{seed_name}") if seed_name else None


@click.command("run-batch")
@click.option("--id", "task_id", required=True, help="Task id")
@click.option("--version", "task_version", default=None, type=int, help="Task version (defaults to latest)")
@click.option("--seeds", required=True, type=click.Path(exists=True, dir_okay=False), help="CSV file with one row per seed")
@click.option("--concurrency", default=5, type=int, help="Max parallel task runs")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False), help="Directory to write context JSON outputs")
@click.option("--agent-model", default=None, type=str, help="Override the model used by prompt_agent steps")
@click.option("--agent-artifact-id", default=None, type=str, help="Override the agent image artifact for deploy_agent steps")
@click.option("--litellm-api-key", default=None, type=str, help="Override LITELLM_API_KEY for all runs")
@click.option("--judge-litellm-api-key", default=None, type=str, help="Override LITELLM_API_KEY used by rubrics_verifier judge agents")
@click.option("--apply-trajectory-filter/--no-trajectory-filter", "apply_trajectory_filter", default=None, help="Force trajectory filtering on/off for rubrics_verifier")
@click.option("--agent-sandbox", default=None,
              help="Override agent sandbox backend(s) for deploy_agent steps; comma-separated for fallback chain (e.g. 'modal,local')")
@click.option("--env-sandbox", default=None,
              help="Override env sandbox backend(s) for deploy_env steps; comma-separated for fallback chain (e.g. 'modal,local')")
@click.option("--project-id", "project_id", default=None, type=str,
              help="Project id for LiteLLM cost attribution. Forwarded as the `user` field on every LLM call and as `projectId:<id>` in `metadata.tags`. Optional, but spend won't be attributed to a project without it.")
@click.option("--env-state-type", default=None, type=str,
              help="Override the env state type for deploy_env steps")
@click.option("--env-state-instance-id", default=None, type=str,
              help="Attach env to an existing EnvStateInstance")
def run_batch(task_id: str, task_version: int | None, seeds: str, concurrency: int, output_dir: str | None, agent_model: str | None, agent_artifact_id: str | None, litellm_api_key: str | None, judge_litellm_api_key: str | None, apply_trajectory_filter: bool | None, agent_sandbox: str | None, env_sandbox: str | None, project_id: str | None, env_state_type: str | None, env_state_instance_id: str | None):
    """Run a task in batch against multiple seeds from a CSV file.

    Each row in the CSV becomes a seed dict passed to the task via
    context.metadata["seed"]. Prompt templates with <placeholder> variables
    are rendered using the seed values before being sent to the agent.
    """
    from agent_env.task import Task
    from agent_env.task_step.context import TaskStepContext

    if output_dir is None:
        output_dir = f"/tmp/taskid={fs_safe(task_id)}-batch{uuid4().hex[:6]}"

    # Read seeds from CSV
    with open(seeds, newline="") as f:
        reader = csv.DictReader(f)
        seed_rows = [row for row in reader if any(row.values())]

    click.echo(f"Loaded {len(seed_rows)} seeds from {seeds}")
    if not seed_rows:
        raise click.ClickException("No seeds found in CSV")

    click.echo(f"Fetching task: id={task_id} version={task_version or 'latest'}...")
    task = Task.get(task_id, version=task_version)
    click.echo(f"Found task: id={task.id} version={task.version} steps={len(task.steps)}")
    if is_local_id(task.id) and any(isinstance(step, CollectArtifactsTaskStep) for step in task.steps):
        for index, seed in enumerate(seed_rows, 1):
            universe_id = _seed_universe_id(task, seed)
            if not universe_id:
                continue
            try:
                validate_local_id(universe_id)
            except ValueError as e:
                raise click.ClickException(f"seed {index} can't name an @local universe: {e}") from e
    click.echo(click.style(f"Running {len(seed_rows)} seeds with concurrency={concurrency}...", fg="blue"))
    click.echo()

    print_banner()
    _warn_missing_cost_attribution(project_id, task)

    sem = asyncio.Semaphore(concurrency)
    batch_run_group_id = uuid4().hex
    client_id = get_agent_env_client_id()

    async def _run_seed(index: int, seed: dict):
        async with sem:
            seed_name = seed.get("name", seed.get("repo_url", f"seed-{index}"))
            tag = f"[seed {index}: {seed_name[:50]}]"
            on_start, on_complete = _make_parallel_callbacks(f"seed {index}")

            ctx = TaskStepContext()
            _stamp_agent_env_client_metadata(ctx.metadata, client_id)
            ctx.metadata["run_group_id"] = batch_run_group_id
            ctx.metadata["seed"] = seed
            universe_id = _seed_universe_id(task, seed)
            if universe_id:
                ctx.metadata["universe_id"] = universe_id
            if litellm_api_key:
                ctx.metadata.setdefault("user_overrides", {})["litellm_api_key"] = litellm_api_key
            if judge_litellm_api_key:
                ctx.metadata.setdefault("user_overrides", {})["judge_litellm_api_key"] = judge_litellm_api_key
            if apply_trajectory_filter is not None:
                ctx.metadata.setdefault("user_overrides", {})["apply_trajectory_filter"] = apply_trajectory_filter
            if agent_sandbox:
                ctx.metadata.setdefault("user_overrides", {})["agent_sandbox"] = agent_sandbox
            if env_sandbox:
                ctx.metadata.setdefault("user_overrides", {})["env_sandbox"] = env_sandbox
            if env_state_type:
                ctx.metadata.setdefault("user_overrides", {})["env_state_type"] = env_state_type
            if env_state_instance_id:
                ctx.metadata.setdefault("user_overrides", {})["env_state_instance_id"] = env_state_instance_id
            ctx.metadata["task_id"] = task_id
            if project_id:
                ctx.metadata["project_id"] = project_id

            click.echo(click.style(f"{tag} Starting...", fg="yellow"))
            context = await task.run(
                on_step_start=on_start,
                on_step_complete=on_complete,
                agent_model=agent_model,
                agent_artifact_id=agent_artifact_id,
                context=ctx,
            )
            if output_dir:
                _write_context(context, f"{task_id}-seed{index}", output_dir, prefix=f"{tag} ")
            return index, seed_name, context

    async def _run_all():
        coros = [_run_seed(i, seed) for i, seed in enumerate(seed_rows, 1)]
        return await asyncio.gather(*coros, return_exceptions=True)

    results = asyncio.run(_run_all())

    click.echo()
    successes = 0
    failures = []
    for r in results:
        if isinstance(r, BaseException):
            failures.append(r)
        else:
            successes += 1

    if failures:
        for exc in failures:
            click.echo(click.style(f"FAILED: {exc}", fg="red"))
        click.echo(click.style(f"{successes}/{len(seed_rows)} succeeded, {len(failures)}/{len(seed_rows)} failed.", fg="red"))
        raise SystemExit(1)
    else:
        click.echo(click.style(f"All {len(seed_rows)} seeds completed!", fg="blue"))
