"""Run SQL in a deployed env's service database, the PostgreSQL every MCP service of a gateway env keeps its state in.

The database runs as the ``servicedb`` compose service on the env's VM host, so SQL reaches it with
``docker compose exec servicedb psql`` from that host, as the env snapshot store does. Envs whose state lives
elsewhere (a container-mode or external store) are refused with a message saying so.
"""
from __future__ import annotations

import logging
import shlex
import uuid
from dataclasses import dataclass
from typing import Any

from agent_env.env.env import DeployedSandboxEnv, require_sandbox
from agent_env.task_step.context import TaskStepContext

logger = logging.getLogger(__name__)

# The directory on the env's VM host that holds the SQL files written for psql to read.
WORK_DIR = "/tmp/agentenv-service-db-sql"


class ServiceDbSqlError(RuntimeError):
    """psql exited non-zero; ``message`` is its error text, e.g. ``psql:<stdin>:3: ERROR: relation "x" does not
    exist``."""

    def __init__(self, what: str, exit_code: int, message: str):
        self.exit_code = exit_code
        self.message = message
        super().__init__(f"{what} failed (psql exit {exit_code}): {message}")


@dataclass
class ServiceDbHost:
    """The env's VM host, and the psql command line that reaches its service database from there."""

    sandbox: Any
    env_id: str

    def psql(self, *flags: str) -> str:
        from agent_env.env.envs.service_db import DATABASE_SERVICE_NAME, DB_NAME, DB_USER
        from agent_env.providers.env_providers.constants import DOCKER_COMPOSE_PATH

        # -X: no psqlrc. -q: no command tags. ON_ERROR_STOP: the first error ends the script with exit 3.
        return " ".join(
            ["docker", "compose", "-f", shlex.quote(DOCKER_COMPOSE_PATH), "exec", "-T", DATABASE_SERVICE_NAME,
             "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "-U", DB_USER, "-d", DB_NAME, *flags]
        )

    async def run(self, script: str) -> tuple[int, str, str]:
        """Run a bash script on the host as root, returning (exit code, stdout, stderr)."""
        return await self.sandbox.exec_with_output("sudo", "bash", "-c", script)

    async def write(self, path: str, text: str) -> None:
        await self.sandbox.write_host_file(text.encode("utf-8"), path)

    async def run_sql_file(self, what: str, sql: str, *flags: str) -> str:
        """Run ``sql`` as one psql script fed on stdin; return its stdout, raise ServiceDbSqlError on a failure."""
        path = f"{WORK_DIR}/{uuid.uuid4().hex}.sql"
        await self.write(path, sql)
        try:
            code, out, err = await self.run(f"{self.psql(*flags, '-f', '-')} < {shlex.quote(path)}")
        finally:
            await self.run(f"rm -f {shlex.quote(path)}")
        if code != 0:
            raise ServiceDbSqlError(what, code, _psql_error(err, out))
        return out


def _psql_error(stderr: str, stdout: str) -> str:
    """psql's error lines, else the tail of what it printed. A missing ``servicedb`` service is named plainly."""
    text = stderr.strip() or stdout.strip()
    if "no such service" in text or "is not running" in text:
        return ("this env has no running 'servicedb' service on its VM host, so its state is not in a local "
                f"service database these steps can reach ({text[-300:]})")
    lines = [line for line in text.splitlines() if "ERROR" in line or "FATAL" in line or line.startswith(("LINE", "DETAIL", "HINT", "CONTEXT"))]
    return ("\n".join(lines) if lines else text)[-2000:]


async def service_db_host(context: TaskStepContext, env_id: str, what: str) -> ServiceDbHost:
    """The VM host of ``env_id``'s deploy in this run."""
    deployed = next((d for d in context.deployed_envs if d.env_id == env_id), None)
    if deployed is None:
        raise RuntimeError(f"{what}: env '{env_id}' is not deployed in this run; put a deploy_env step for it first")
    return await service_db_host_of(deployed, what)


async def service_db_host_of(deployed, what: str) -> ServiceDbHost:
    """The VM host of a deployed env's record, wherever it was deployed from (a task run or a direct deploy)."""
    from agent_env.providers.sandbox_providers.sandbox import VmSandbox
    from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider, get_env_sandbox_provider

    env_id = deployed.env_id
    deployed: DeployedSandboxEnv = require_sandbox(deployed, what)
    if not deployed.sandbox_id:
        raise RuntimeError(f"{what}: env '{env_id}' has no sandbox_id")
    provider = build_sandbox_provider(deployed.sandbox_type) if deployed.sandbox_type else get_env_sandbox_provider()
    sandbox = await provider.get_sandbox(deployed.sandbox_id)
    if not isinstance(sandbox, VmSandbox):
        raise RuntimeError(f"{what}: env '{env_id}' runs in a container sandbox; its service database is only "
                           "reachable on a VM-hosted env")
    return ServiceDbHost(sandbox=sandbox, env_id=env_id)
