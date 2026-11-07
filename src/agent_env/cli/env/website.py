import asyncio
import os
import subprocess
import sys
from pathlib import Path

import click

from agent_env.artifact import DockerImageArtifact, EnvironmentArtifact
from agent_env.cli.env import website_dev
from agent_env.cli.env.website_scaffold import ScaffoldError, scaffold, tools_from_json, tools_from_mcp
from agent_env.cli.utils import (
    deployed_env_from_instance,
    build_platform_option,
    detect_env_metadata,
    env_provider_type_option,
    environment_name_options,
    refuse_unwritable_ids,
    resolve_environment_name,
)
from agent_env.utils.card_naming import card_name_from_github, card_name_from_source
from agent_env.utils.docker.build import DEFAULT_BUILD_PLATFORM, build_image
from agent_env.env.envs.website import WebsiteEnv, domains_problem, health_path_problem
from agent_env.providers.env_providers.constants import AGENT_ENV_WEBSITE_HEALTH_PATH
from agent_env.store.ids import derive_id, image_repository



@click.group(name="website")
def website():
    """Website environment commands."""
    pass


@website.command()
@click.argument("name")
@click.option("--dir", "directory", default=None, type=click.Path(file_okay=False, path_type=Path),
              help="Where to write the site (defaults to ./NAME)")
@click.option("--from-mcp", "from_mcp", default=None, type=click.Path(exists=True, path_type=Path),
              help="An MCP server's source, a folder or .py file, whose @tool methods become stub @api routes; read "
                   "without running it")
@click.option("--from-tools-json", "from_tools_json", default=None, type=click.Path(exists=True, dir_okay=False,
              path_type=Path), help="A JSON file of the tools to make stub @api routes of: tools/list's answer, a "
                                    "list of {name, description, inputSchema}, or an environment card")
@click.option("--domain", "domains", multiple=True,
              help="A name a deploy reaches the site by, such as shop.example or *.shop.example (repeatable), written "
                   "to env.toml's domains; the card declares NAME.test")
@click.option("--force", is_flag=True, default=False, help="Write into a directory that isn't empty")
def new(name: str, directory: Path | None, from_mcp: Path | None, from_tools_json: Path | None,
        domains: tuple[str, ...], force: bool):
    """Make a new website env NAME like the Notes example: site.py, data/seed.json, a Vite app in web/ with the SDK's
    browser client, a Dockerfile, env.toml and a README. With --from-mcp or --from-tools-json, each of a server's tools
    is a stub @api route of the same name, and the seed has a collection for each."""
    if from_mcp and from_tools_json:
        click.echo("Error: --from-mcp and --from-tools-json are mutually exclusive", err=True)
        sys.exit(1)
    target = directory or Path(name)
    try:
        tools, source = None, None
        if from_mcp:
            tools, source = tools_from_mcp(from_mcp), str(from_mcp)
        elif from_tools_json:
            tools, source = tools_from_json(from_tools_json), str(from_tools_json)
        written = scaffold(target, name, domains=list(domains), tools=tools, source=source, force=force)
    except ScaffoldError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(f"Created website env {name!r} in {target} ({len(written)} files"
               + (f", {len(tools)} stub route(s) from {source}" if tools is not None else "") + ").")
    click.echo(f"\nNext:\n  cd {target}\n  agent-env env website dev\n"
               f"  agent-env env website put --id {name} --server-dockerfile Dockerfile")


@website.command()
@click.option("--dir", "directory", default=".", show_default=True, type=click.Path(exists=True, file_okay=False,
              path_type=Path), help="The site's folder, with its site.py")
@click.option("--site", "site_file", default="site.py", show_default=True, help="The site's server, in --dir")
@click.option("--host", default="127.0.0.1", show_default=True, help="The address the site listens on")
@click.option("--port", default=8000, show_default=True, type=int, help="The port the site listens on")
@click.option("--clock-port", default=0, type=int, help="The local clock's port (default: any free one)")
@click.option("--clock-time", default=None, help="The universe's time the local clock starts at, RFC3339 (default: now)")
@click.option("--seed", default="data/seed.json", show_default=True,
              help="The seed loaded through the data plane, in --dir; skipped when the file isn't there")
@click.option("--web/--no-web", "build_web", default=True, show_default=True,
              help="Build web/ when it is out of date, if it has a package.json")
@click.option("--watch/--no-watch", default=True, show_default=True, help="Rebuild web/ as its sources change")
@click.option("--check/--no-check", default=True, show_default=True,
              help="Run the website/v1 conformance check once the site is up")
def dev(directory: Path, site_file: str, host: str, port: int, clock_port: int, clock_time: str | None, seed: str,
        build_web: bool, watch: bool, check: bool):
    """Run a site locally for authoring, without Docker: check the folder's env.toml, when it has one, as a bundle
    would read it, build web/ when it is out of date, serve site.py with uvicorn's reload, start a local clock standing in for the gateway's, sync the site to it with clock/v1's
    sync_time and load the seed through the data plane (again after every reload), print the URLs, and run the
    conformance check once. Ctrl-C stops it all."""
    import uvicorn

    from agentenv_protocol.website.clock import parse_time

    site_dir = directory.resolve()
    site_path = site_dir / site_file
    if not site_path.is_file():
        click.echo(f"Error: no {site_file} in {site_dir}", err=True)
        sys.exit(1)
    try:
        start = parse_time(clock_time) if clock_time else None
    except ValueError:
        click.echo(f"Error: --clock-time {clock_time!r} is not an RFC3339 time", err=True)
        sys.exit(1)
    try:
        declared = website_dev.env_toml(site_dir)
    except website_dev.DevError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    try:
        website_dev.site_class(site_path)
    except Exception as e:
        click.echo(f"Error: can't load {site_path}: {e}", err=True)
        sys.exit(1)

    web = site_dir / "web"
    watcher = None
    if build_web and (web / "package.json").is_file():
        if not website_dev.npm_available():
            click.echo("Warning: web/ has a package.json but npm isn't on PATH; serving web/dist as it is", err=True)
        else:
            commands, watch_command = website_dev.web_plan(web, watch=watch)
            for command in commands:
                click.echo(f"[dev] web/: {' '.join(command)}")
                try:
                    website_dev.run_npm(command, web)
                except subprocess.CalledProcessError as e:
                    click.echo(f"Error: {' '.join(command)} failed with exit code {e.returncode}", err=True)
                    sys.exit(1)
            if watch_command:
                click.echo(f"[dev] web/: {' '.join(watch_command)} (in the background)")
                watcher = website_dev.run_npm(watch_command, web, background=True)

    seed_path = site_dir / seed
    clock = website_dev.DevClock(start, port=clock_port).start()
    base_url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    supervisor = website_dev.Supervisor(base_url, clock.url, seed_path if seed_path.is_file() else None, check=check,
                                        echo=click.echo).start()
    clock_base = clock.url.rsplit("/clock/", 1)[0]
    click.echo(f"""
[dev] Site:   {base_url}/
[dev] Card:   {base_url}/.well-known/agent-env.json
[dev] Clock:  {clock.url}
[dev]   curl -X PUT {clock_base}/clock/advance -d '{{"seconds": 3600}}'
[dev]   curl -X PUT {clock_base}/clock/set-time -d '{{"virtual_time": "2030-01-01T09:00:00Z"}}'
[dev] Check:  python -m agentenv_protocol.website.conformance {base_url}
""")
    if declared and declared.get("domains"):
        click.echo(f"[dev] env.toml names it {', '.join(declared['domains'])}, which a deploy reaches it by over the card's")
    excludes = [str(web)] if web.is_dir() else []
    os.environ[website_dev.SITE_ENV] = str(site_path)
    try:
        uvicorn.run("agent_env.cli.env.website_dev:create_app", factory=True, host=host, port=port, reload=True,
                    reload_dirs=[str(site_dir)], reload_includes=website_dev.RELOAD_INCLUDES, reload_excludes=excludes or None)
    finally:
        supervisor.stop()
        clock.close()
        if watcher is not None:
            watcher.terminate()
            try:
                watcher.wait(timeout=website_dev.WATCHER_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                watcher.kill()


@website.command()
@click.option("--id", "env_id", required=True, help="Env id")
@click.option("--server-dockerfile", "--backend-dockerfile", "backend_dockerfile", default=None, type=click.Path(exists=True),
              help="Path to the Dockerfile of the server image, which serves the site's pages and API")
@click.option("--server-docker-context", "--backend-docker-context", "backend_docker_context", default=None,
              type=click.Path(exists=True), help="The server image's build context (defaults to its Dockerfile's directory)")
@click.option("--server-dockerfile-github-url", "--backend-dockerfile-github-url", "backend_dockerfile_github_url",
              default=None, help="GitHub URL to the server image's Dockerfile")
@click.option("--server-docker-context-github-url", "--backend-docker-context-github-url",
              "backend_docker_context_github_url", default=None, help="GitHub URL to the server image's build context directory")
@click.option("--server-image-ref", "server_image_ref", default=None,
              help="A server image already in a registry, e.g. ghcr.io/org/site:v1: registered as it is, with its tag "
                   "pinned to the digest the registry serves now (instead of a Dockerfile)")
@click.option("--frontend-dockerfile", default=None, type=click.Path(exists=True),
              help="Path to a separate frontend Dockerfile (deprecated: build the pages into the server image)")
@click.option("--frontend-docker-context", default=None, type=click.Path(exists=True), help="Frontend Docker build context (defaults to Dockerfile's directory)")
@click.option("--frontend-dockerfile-github-url", default=None, help="GitHub URL to a separate frontend Dockerfile (deprecated)")
@click.option("--frontend-docker-context-github-url", default=None, help="GitHub URL to frontend build context directory")
@click.option("--health-path", default=AGENT_ENV_WEBSITE_HEALTH_PATH, show_default=True,
              help="The path that answers 200 once the site is usable")
@click.option("--domain", "domains", multiple=True,
              help="A name the site is reached by, such as slack.com or *.slack.com (repeatable); without one, the "
                   "names its card declares")
@environment_name_options
@env_provider_type_option("What deploys the env: 'gateway' (a gateway in front of the website), or the type of an installed "
                          "agent_env.env_providers plugin", "website")
@click.option("--metadata", "metadata_pairs", multiple=True, help="Metadata key=value pair (repeatable)")
@click.option("--skip-validation", is_flag=True, default=False, help="Skip environment validation after registration")
@build_platform_option
def put(
    env_id: str,
    backend_dockerfile: str | None,
    backend_docker_context: str | None,
    backend_dockerfile_github_url: str | None,
    backend_docker_context_github_url: str | None,
    server_image_ref: str | None,
    frontend_dockerfile: str | None,
    frontend_docker_context: str | None,
    frontend_dockerfile_github_url: str | None,
    frontend_docker_context_github_url: str | None,
    health_path: str,
    domains: tuple[str, ...],
    environment_name: str | None,
    env_provider_type: str,
    metadata_pairs: tuple[str, ...],
    skip_validation: bool,
    build_platform: str,
):
    """Build and upload a website environment: its server image, which serves the site's pages and API, from a
    Dockerfile, GitHub or a registry, and any deprecated separate frontend image."""
    environment_name = resolve_environment_name(environment_name, allow_missing=True)

    if sum(map(bool, (backend_dockerfile, backend_dockerfile_github_url, server_image_ref))) != 1:
        click.echo("Error: exactly one of --server-dockerfile, --server-dockerfile-github-url and --server-image-ref is "
                   "required", err=True)
        sys.exit(1)
    if frontend_dockerfile and frontend_dockerfile_github_url:
        click.echo("Error: --frontend-dockerfile and --frontend-dockerfile-github-url are mutually exclusive", err=True)
        sys.exit(1)
    use_github = bool(backend_dockerfile_github_url)
    if (frontend_dockerfile and not backend_dockerfile) or (frontend_dockerfile_github_url and not use_github):
        click.echo("Error: a frontend image is built the way the server image is, both from local Dockerfiles or both "
                   "from GitHub URLs", err=True)
        sys.exit(1)
    if server_image_ref and not environment_name:
        click.echo("Error: --server-image-ref needs --environment-name, which a registry image's card can't give here",
                   err=True)
        sys.exit(1)
    if problem := health_path_problem(health_path) or domains_problem(list(domains)):
        click.echo(f"Error: {problem}", err=True)
        sys.exit(1)
    if frontend_dockerfile or frontend_dockerfile_github_url:
        click.echo("Warning: a separate frontend image is deprecated; build the site's pages into its server image",
                   err=True)

    user_metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        user_metadata[key] = value
    site = {"health_path": health_path, "domains": list(domains)}

    if use_github:
        if environment_name is None:
            environment_name = card_name_from_github(backend_dockerfile_github_url, backend_docker_context_github_url, github_token=os.environ.get("GITHUB_TOKEN"))
            if not environment_name:
                click.echo("Error: could not read @environment_card(name=...) from the server image's GitHub source; pass --environment-name.", err=True)
                sys.exit(1)
            click.echo(f"Derived environment_name={environment_name!r} from the environment card.")
        if build_platform != DEFAULT_BUILD_PLATFORM:
            click.echo(f"Warning: --platform {build_platform!r} is ignored for --*-dockerfile-github-url builds", err=True)
        backend_status = [""]
        frontend_status = [""]
        lines_printed = [0]

        def _redraw():
            if lines_printed[0] > 0:
                click.echo(f"\033[{lines_printed[0]}A", nl=False)
            click.echo(f"\033[2K  {click.style('Server:', fg='cyan')}   {backend_status[0]}")
            click.echo(f"\033[2K  {click.style('Frontend:', fg='magenta')} {frontend_status[0]}")
            lines_printed[0] = 2

        def _backend_progress(step: str, message: str, percent: int) -> None:
            backend_status[0] = f"[{percent:3d}%] {message}"
            _redraw()

        def _frontend_progress(step: str, message: str, percent: int) -> None:
            frontend_status[0] = f"[{percent:3d}%] {message}"
            _redraw()

        env = asyncio.run(WebsiteEnv.put_from_github(
            id=env_id,
            backend_dockerfile_github_url=backend_dockerfile_github_url,
            backend_docker_context_github_url=backend_docker_context_github_url,
            frontend_dockerfile_github_url=frontend_dockerfile_github_url or "",
            frontend_docker_context_github_url=frontend_docker_context_github_url,
            environment_name=environment_name,
            metadata=user_metadata if user_metadata else None,
            on_backend_progress=_backend_progress,
            on_frontend_progress=_frontend_progress,
            github_token=os.environ.get("GITHUB_TOKEN"),
            env_provider_type=env_provider_type,
            **site,
        ))
        click.echo(f"Created WebsiteEnv: id={env.id} version={env.version} environment_name={env.environment_name}")
        if not skip_validation:
            click.echo("\nValidating environment (use --skip-validation to skip)...")
            instance_id = asyncio.run(env.validate(on_progress=click.echo))
            click.echo(f"Validation task: {instance_id}")
        return

    backend_id, frontend_id = derive_id(env_id, "backend_image"), derive_id(env_id, "frontend_image")
    refuse_unwritable_ids(env_id, backend_id, *([frontend_id] if frontend_dockerfile else []))
    metadata: dict[str, str] = {}
    if server_image_ref:
        if build_platform != DEFAULT_BUILD_PLATFORM:
            click.echo(f"Warning: --platform {build_platform!r} is ignored for --server-image-ref, which builds nothing", err=True)
        backend_artifact = DockerImageArtifact.put_ref(backend_id, description=f"Website server image for {env_id}",
                                                       image_name=server_image_ref)
        click.echo(f"Registered server image: id={backend_artifact.id} version={backend_artifact.version} image={backend_artifact.image_name}")
    else:
        backend_dockerfile_path = Path(backend_dockerfile)
        backend_ctx = Path(backend_docker_context) if backend_docker_context else backend_dockerfile_path.parent
        if environment_name is None:
            environment_name = card_name_from_source(str(backend_dockerfile_path), str(backend_ctx))
            if not environment_name:
                click.echo("Error: no @environment_card(name=...) found in the server image's source; pass --environment-name.", err=True)
                sys.exit(1)
            click.echo(f"Derived environment_name={environment_name!r} from the environment card.")
        backend_artifact = _build(backend_id, backend_dockerfile_path, backend_ctx, "server", build_platform)
        metadata = detect_env_metadata(backend_dockerfile_path, backend_ctx)

    frontend_artifact = None
    if frontend_dockerfile:
        frontend_dockerfile_path = Path(frontend_dockerfile)
        frontend_ctx = Path(frontend_docker_context) if frontend_docker_context else frontend_dockerfile_path.parent
        frontend_artifact = _build(frontend_id, frontend_dockerfile_path, frontend_ctx, "frontend", build_platform)
    metadata.update(user_metadata)

    click.echo("Creating WebsiteEnv...")
    try:
        env = WebsiteEnv.put(
            id=env_id,
            server_docker_image_artifact=backend_artifact,
            frontend_docker_image_artifact=frontend_artifact,
            environment_name=environment_name,
            metadata=metadata if metadata else None,
            env_provider_type=env_provider_type,
            **site,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Created WebsiteEnv: id={env.id} version={env.version} environment_name={env.environment_name} "
               f"env_provider_type={env.env_provider_type}")
    if not skip_validation:
        click.echo("\nValidating environment (use --skip-validation to skip)...")
        instance_id = asyncio.run(env.validate(on_progress=click.echo))
        click.echo(f"Validation task: {instance_id}")


def _build(image_id: str, dockerfile: Path, context: Path, role: str, platform: str) -> DockerImageArtifact:
    """Build ``dockerfile`` into the image ``image_id`` and store it with its build context, as an a2a agent's image
    is stored."""
    tag = image_repository(image_id)
    click.echo(f"Building website {role} Docker image...")
    build_image(dockerfile, context, tag, platform=platform)
    artifact = DockerImageArtifact.put(id=image_id, description=f"Website {role} image created from agent-env CLI",
                                       image_name=tag, build_context_path=str(context), dockerfile_path=str(dockerfile),
                                       platform=platform)
    click.echo(f"Created {role} artifact: id={artifact.id} version={artifact.version}")
    return artifact


@website.command(name="load-environment-artifact")
@click.option("--id", "env_id", default=None, help="WebsiteEnv id (optional; cross-checked against the instance)")
@click.option("--environment-artifact-id", "environment_artifact_id",
              required=True, help="EnvironmentArtifact id")
@click.option("--instance-id", "instance_id", required=True,
              help="Deployed env instance id, as printed by `env deploy`")
def load_environment_artifact(env_id: str | None, environment_artifact_id: str,
                              instance_id: str):
    """Load an environment artifact into a deployed website environment."""

    env = deployed_env_from_instance(env_id, instance_id)
    click.echo(f"Found env: id={env.id} version={env.version} environment_name={env.environment_name}")

    click.echo(f"Fetching environment artifact: id={environment_artifact_id}...")
    environment_artifact = EnvironmentArtifact.get(environment_artifact_id)
    click.echo(f"Found environment artifact: id={environment_artifact.id} version={environment_artifact.version} environment_name={environment_artifact.environment_name}")

    click.echo("Loading environment artifact...")
    asyncio.run(env.load_environment_artifact(environment_artifact))
    click.echo("Loaded environment artifact into env")
