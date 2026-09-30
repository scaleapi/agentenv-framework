import asyncio
import os
import sys
from pathlib import Path

import click

from agent_env.artifact import DockerImageArtifact, EnvironmentArtifact
from agent_env.cli.utils import (
    deployed_env_from_instance,
    build_platform_option,
    detect_env_metadata,
    env_provider_type_option,
    environment_name_options,
    resolve_environment_name,
    skips_local_validation,
)
from agent_env.utils.card_naming import card_name_from_github, card_name_from_source
from agent_env.utils.docker_build import DEFAULT_BUILD_PLATFORM, build_image
from agent_env.env import Env, MCPServerEnv


@click.group(name="mcp-server")
def mcp_server():
    """MCP server environment commands."""
    pass


def _read_validation_report(search_dirs) -> dict | None:
    """Load ``validation_report.json`` (the env-build → put handoff) from the first of
    ``search_dirs`` that has one; None if absent/unreadable."""
    import json

    for d in search_dirs:
        if not d:
            continue
        p = Path(d) / "validation_report.json"
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                return None
    return None


def _decide_release(validation_gate: dict | None, unit_tests: dict | None, override: bool) -> dict:
    """Decide release from the gate verdict + unit-test verdict. Blocks on any live-server
    gate that failed or did not run; unit tests block only when present and failed (a
    missing verdict is unknown, not a block). ``override`` turns a block into an audited
    publish. Returns ``{promote, blocked, reasons, overridden}``.
    """
    reasons: list[str] = []
    if validation_gate is None:
        reasons.append("live-server validation gate did not run")
    elif not validation_gate.get("passed", False):
        failed = validation_gate.get("failed_gates", [])
        missing = validation_gate.get("missing_gates", [])
        if failed:
            reasons.append(f"failed gates: {', '.join(failed)}")
        if missing:
            reasons.append(f"gates that did not run: {', '.join(missing)}")
        if not failed and not missing:
            reasons.append("validation gate did not pass")
    if unit_tests is not None and not unit_tests.get("passed", False):
        reasons.append("unit tests failed")

    blocked = bool(reasons)
    if blocked and override:
        return {"promote": True, "blocked": True, "reasons": reasons, "overridden": True}
    return {"promote": not blocked, "blocked": blocked, "reasons": reasons, "overridden": False}


def _gate_release(env, report_dirs, override: bool) -> None:
    """Gate a freshly-put env: validate, read the gate + unit-test verdicts, stamp a
    durable ``release_gate`` on the env, and exit non-zero if blocked (unless
    ``--override``). The single publish-enforcement point.
    """
    click.echo("\nValidating environment...")
    instance_id = asyncio.run(env.validate(on_progress=click.echo))
    click.echo(f"Validation task: {instance_id}")

    fresh = Env.get(env.id, env.version)
    validation_gate = fresh.metadata.get("validation_gate")
    report = _read_validation_report(report_dirs)
    unit_tests = report.get("unit_tests") if report else None

    decision = _decide_release(validation_gate, unit_tests, override)
    try:
        fresh.update_metadata({**fresh.metadata, "release_gate": {
            "passed": not decision["blocked"],
            "overridden": decision["overridden"],
            "reasons": decision["reasons"],
        }})
    except Exception as e:  # best-effort audit stamp
        click.echo(f"Warning: could not stamp release_gate on env: {e}", err=True)

    if decision["blocked"] and not decision["overridden"]:
        click.secho(f"\nRelease gate FAILED: {'; '.join(decision['reasons'])}", fg="red", err=True)
        click.echo("Re-run with --override to publish anyway.", err=True)
        sys.exit(1)
    if decision["overridden"]:
        click.secho(f"\nRelease gate OVERRIDDEN ({'; '.join(decision['reasons'])}); publishing anyway.", fg="yellow")
    else:
        click.secho("\nRelease gate PASSED.", fg="green")


@mcp_server.command()
@click.option("--id", "env_id", required=True, help="Env id")
@click.option("--dockerfile", default=None, type=click.Path(exists=True), help="Path to local Dockerfile")
@click.option("--context", "context_path", default=None, type=click.Path(exists=True), help="Docker build context (defaults to Dockerfile's directory)")
@click.option("--dockerfile-github-url", "dockerfile_github_url", default=None, help="GitHub URL to Dockerfile (e.g. https://github.com/owner/repo/tree/main/path/Dockerfile)")
@click.option("--docker-context-github-url", "docker_context_github_url", default=None, help="GitHub URL to build context directory (defaults to Dockerfile's parent)")
@environment_name_options
@env_provider_type_option("What deploys the env: 'gateway' (a gateway and service database in front of the server), 'server' (the "
                          "server on its own), or the type of an installed agent_env.env_providers plugin")
@click.option("--metadata", "metadata_pairs", multiple=True, help="Metadata key=value pair (repeatable)")
@click.option("--validate", "run_validation", is_flag=True, default=False,
              help="Run the release gate after registering: validate the env and exit non-zero if it fails")
@click.option("--override", "override", is_flag=True, default=False,
              help="Run the release gate but publish even if it fails (records an audited override); implies --validate")
@build_platform_option
def put(env_id: str, dockerfile: str | None, context_path: str | None, dockerfile_github_url: str | None, docker_context_github_url: str | None, environment_name: str | None, env_provider_type: str, metadata_pairs: tuple[str, ...], run_validation: bool, override: bool, build_platform: str):
    """Build and upload an MCP server environment."""

    environment_name = resolve_environment_name(environment_name, allow_missing=True)

    if dockerfile and dockerfile_github_url:
        click.echo("Error: --dockerfile and --dockerfile-github-url are mutually exclusive", err=True)
        sys.exit(1)
    if not dockerfile and not dockerfile_github_url:
        click.echo("Error: either --dockerfile or --dockerfile-github-url is required", err=True)
        sys.exit(1)
    if context_path and dockerfile_github_url:
        click.echo("Error: --context cannot be used with --dockerfile-github-url", err=True)
        sys.exit(1)
    if docker_context_github_url and not dockerfile_github_url:
        click.echo("Error: --docker-context-github-url requires --dockerfile-github-url", err=True)
        sys.exit(1)

    user_metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        user_metadata[key] = value

    if dockerfile_github_url:
        if environment_name is None:
            environment_name = card_name_from_github(dockerfile_github_url, docker_context_github_url, github_token=os.environ.get("GITHUB_TOKEN"))
            if not environment_name:
                click.echo("Error: could not read @environment_card(name=...) from the GitHub source; pass --environment-name.", err=True)
                sys.exit(1)
            click.echo(f"Derived environment_name={environment_name!r} from the environment card.")
        if build_platform != DEFAULT_BUILD_PLATFORM:
            click.echo(f"Warning: --platform {build_platform!r} is ignored for --dockerfile-github-url builds", err=True)
        status = [""]
        lines_printed = [0]

        def _on_progress(step: str, message: str, percent: int) -> None:
            if lines_printed[0] > 0:
                click.echo("\033[1A", nl=False)
            status[0] = f"[{percent:3d}%] {message}"
            click.echo(f"\033[2K  {click.style('MCP Server:', fg='cyan')} {status[0]}")
            lines_printed[0] = 1

        env = asyncio.run(MCPServerEnv.put_from_github(
            id=env_id,
            dockerfile_github_url=dockerfile_github_url,
            docker_context_github_url=docker_context_github_url,
            environment_name=environment_name,
            env_provider_type=env_provider_type,
            github_token=os.environ.get("GITHUB_TOKEN"),
            metadata=user_metadata if user_metadata else None,
            on_progress=_on_progress,
        ))
        click.echo(f"Created MCPServerEnv: id={env.id} version={env.version} environment_name={env.environment_name} env_provider_type={env.env_provider_type}")
        if (run_validation or override) and not skips_local_validation(env.id, "env"):
            # GitHub-sourced build: validation_report.json lives in the repo, not
            # locally, so the unit-test verdict is unknown here (doesn't block).
            _gate_release(env, [], override)
        return

    dockerfile_path = Path(dockerfile)
    context = Path(context_path) if context_path else dockerfile_path.parent
    if environment_name is None:
        environment_name = card_name_from_source(str(dockerfile_path), str(context))
        if not environment_name:
            click.echo("Error: no @environment_card(name=...) found in the build source; pass --environment-name.", err=True)
            sys.exit(1)
        click.echo(f"Derived environment_name={environment_name!r} from the environment card.")
    image_tag = f"mcp-server-{env_id}"

    click.echo(f"Building MCP server Docker image...")
    build_image(dockerfile_path, context, image_tag, platform=build_platform)

    click.echo(f"Creating DockerImageArtifact...")
    artifact = DockerImageArtifact.put(
        id=f"mcp-server-{env_id}",
        description="Created from agent-env CLI",
        image_name=image_tag,
        build_context_path=str(context),
        dockerfile_path=str(dockerfile_path),
    )
    click.echo(f"Created artifact: id={artifact.id} version={artifact.version}")

    metadata = detect_env_metadata(dockerfile_path, context)
    metadata.update(user_metadata)

    click.echo(f"Creating MCPServerEnv...")
    env = MCPServerEnv.put(
        id=env_id,
        docker_image_artifact=artifact,
        environment_name=environment_name,
        env_provider_type=env_provider_type,
        metadata=metadata if metadata else None,
    )
    click.echo(f"Created MCPServerEnv: id={env.id} version={env.version} environment_name={env.environment_name} env_provider_type={env.env_provider_type}")
    if (run_validation or override) and not skips_local_validation(env.id, "env"):
        # Local build: look for the env-build handoff report next to the build context.
        _gate_release(env, [str(context), str(dockerfile_path.parent)], override)


_SPEC_FINDING_LIMIT = 20


def _echo_spec_conformance(spec: dict | None) -> None:
    """Print a ``mcp_spec_conformance`` verdict.

    Findings are listed whenever there are any, not only when the verdict fails: the spec
    is a floor, so a conformant server still reports its framework tools as informational
    ``extra_*`` findings, and keying the display off ``passed`` hid that drift entirely.
    Blocking first, so the truncation drops noise rather than signal.
    """
    click.secho("\nSpec conformance:", bold=True)
    if not spec:
        click.echo("  (did not run)")
        return
    if spec.get("skipped"):
        click.secho(f"  SKIPPED: {spec.get('reason', 'nothing to check')}", fg="yellow")
        return

    findings = spec.get("findings", [])
    ops = spec.get("spec_operations", 0)
    blocking = spec.get("blocking_findings", len(findings))
    informational = len(findings) - blocking

    if not findings:
        click.secho(f"  PASSED: {ops} spec operation(s) match the live tool surface", fg="green")
        return
    if blocking:
        click.secho(f"  {blocking} blocking finding(s), {informational} informational", fg="yellow")
    else:
        # Nothing blocks, so the gate passed — but the drift is still worth seeing.
        click.secho(
            f"  PASSED: {ops} spec operation(s) match the live tool surface; "
            f"{informational} informational finding(s)",
            fg="green",
        )

    ordered = sorted(findings, key=lambda f: not f.get("blocking", True))
    for f in ordered[:_SPEC_FINDING_LIMIT]:
        loc = f.get("tool", "")
        if f.get("param"):
            loc = f"{loc}.{f['param']}"
        click.echo(f"    {f['kind']} {loc}: {f.get('detail', '')}")
    if len(ordered) > _SPEC_FINDING_LIMIT:
        click.echo(f"    (+{len(ordered) - _SPEC_FINDING_LIMIT} more)")


@mcp_server.command(name="load-environment-artifact")
@click.option("--id", "env_id", default=None, help="MCPServerEnv id (optional; cross-checked against the instance)")
@click.option("--environment-artifact-id", "environment_artifact_id",
              required=True, help="EnvironmentArtifact id")
@click.option("--instance-id", "instance_id", required=True,
              help="Deployed env instance id, as printed by `env deploy`")
def load_environment_artifact(env_id: str | None, environment_artifact_id: str,
                              instance_id: str):
    """Load an environment artifact into a deployed MCP server environment."""

    env = deployed_env_from_instance(env_id, instance_id)
    click.echo(f"Found env: id={env.id} version={env.version} environment_name={env.environment_name}")

    click.echo(f"Fetching environment artifact: id={environment_artifact_id}...")
    environment_artifact = EnvironmentArtifact.get(environment_artifact_id)
    click.echo(f"Found environment artifact: id={environment_artifact.id} version={environment_artifact.version} environment_name={environment_artifact.environment_name}")

    click.echo("Loading environment artifact...")
    try:
        asyncio.run(env.load_environment_artifact(environment_artifact))
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo("Loaded environment artifact into env")


@mcp_server.command()
@click.argument("env_id")
@click.option("--version", "env_version", type=int, default=None, help="Env version (default: latest)")
def validate(env_id: str, env_version: int | None):
    """Validate an MCP server environment's tool schemas."""
    from agent_env.store.base import NotFoundError
    from agent_env.task import Task
    from agent_env.task_step.task_steps.env_card_validator import VALIDATED_ENVIRONMENT_CARD_KEY

    try:
        env = Env.get(env_id, env_version)
    except NotFoundError:
        click.echo(f"Error: Env '{env_id}' not found", err=True)
        sys.exit(1)

    if not isinstance(env, MCPServerEnv):
        click.echo(f"Error: Env '{env_id}' is type '{env.type}', expected mcp_server", err=True)
        sys.exit(1)

    click.echo(f"Validating env: id={env.id} version={env.version} environment_name={env.environment_name}")
    instance_id = asyncio.run(env.validate(on_progress=click.echo))

    instance = Task.get_instance(instance_id)
    click.echo(f"Task instance: {instance_id} status={instance.status}")

    env = Env.get(env_id, env.version)

    # Schema validation results
    schema = env.metadata.get("mcp_tool_schema_validation", {})
    schema_passed = schema.get("passed")
    total = schema.get("total_tools", 0)
    missing = schema.get("tools_missing_description", [])
    param_issues = schema.get("parameter_issues", {})

    click.echo(click.style("\nSchema validation:", bold=True))
    if schema_passed:
        click.echo(click.style(f"  PASSED: all {total} tool(s) have valid schemas", fg="green"))
    else:
        click.echo(click.style("  FAILED:", fg="red"))
        if missing:
            click.echo(f"    Tools missing descriptions: {missing}")
        for tool_name, issues in param_issues.items():
            click.echo(f"    {tool_name}:")
            for issue in issues:
                click.echo(f"      - {issue}")

    # Spec conformance results (live tool surface vs. the baked /openapi.yaml)
    _echo_spec_conformance(env.metadata.get("mcp_spec_conformance"))

    # Tool correctness results
    correctness = env.metadata.get("mcp_tool_correctness_validation", {})
    if correctness:
        correctness_passed = correctness.get("passed")
        correctness_total = correctness.get("total_tools", 0)
        results = correctness.get("results", [])

        click.echo(click.style("\nTool correctness:", bold=True))
        if correctness_passed:
            click.echo(click.style(f"  PASSED: all {correctness_total} tool(s) work correctly", fg="green"))
        else:
            click.echo(click.style("  FAILED:", fg="red"))
            for r in results:
                if not r.get("passed"):
                    click.echo(f"    {r['tool_name']}: {r.get('error', 'unknown error')}")
                    if r.get("justification"):
                        click.echo(f"      {r['justification']}")

    # Environment card
    card_val = env.metadata.get(VALIDATED_ENVIRONMENT_CARD_KEY)
    click.secho("\nEnvironment card:", bold=True)
    if not card_val:
        click.echo("  (not validated)")
    elif card_val.get("accessible"):
        missing = [f for f, v in card_val.get("required_fields", {}).items() if not v.get("present")]
        click.secho(f"  ACCESSIBLE: {card_val.get('children_count', 0)} child env(s); extensions={card_val.get('extensions', [])}; tools={card_val.get('tools', [])}", fg="green")
        if missing:
            click.secho(f"  missing required fields: {missing}", fg="yellow")
    else:
        click.secho(f"  NOT ACCESSIBLE: {card_val.get('error', 'no card')}", fg="magenta")

    # Consolidated release gate (required live-server gates rolled up)
    gate = env.metadata.get("validation_gate")
    click.secho("\nRelease gate (required):", bold=True)
    if not gate:
        click.echo("  (not aggregated)")
    elif gate.get("passed"):
        click.secho(f"  PASSED: {', '.join(gate.get('required_gates', []))}", fg="green")
    else:
        failed = gate.get("failed_gates", [])
        missing = gate.get("missing_gates", [])
        click.secho(f"  FAILED — failed={failed} did-not-run={missing}", fg="red")
    if gate and gate.get("skipped_gates"):
        click.secho(f"  skipped (nothing to check): {', '.join(gate['skipped_gates'])}", fg="yellow")


@mcp_server.command(name="create-cli")
@click.option("--id", "env_id", required=True, help="MCPServerEnv id")
@click.option("--version", "env_version", type=int, default=None, help="Env version (default: latest)")
@click.option("--command-name", default=None, help="CLI command name (defaults to env's environment_name)")
@click.option("--force", is_flag=True, default=False, help="Rebuild even if a CliArtifact ref exists in env metadata")
def create_cli(env_id: str, env_version: int | None, command_name: str | None, force: bool):
    """Build a Click-based CLI artifact from a deployed MCPServerEnv's tools."""

    env = Env.get(env_id, env_version)
    if not isinstance(env, MCPServerEnv):
        click.echo(f"Error: env '{env_id}' is type '{env.type}', expected mcp_server", err=True)
        sys.exit(1)

    artifact = asyncio.run(env.create_cli(command_name=command_name, on_progress=click.echo, force=force))
    click.echo(
        f"Created CliArtifact: id={artifact.id} version={artifact.version} "
        f"command_name={artifact.command_name} entrypoint={artifact.entrypoint} "
        f"cli_s3_url={artifact.cli_object_url}"
    )
