"""apply_sql against a real PostgreSQL 16 holding a service schema set up the way a
gateway env's service database is: the shared changelog objects from the state provider's init script and changelog
triggers on the service's tables.

The step reaches the database through ``ServiceDbHost``; here its psql runs through ``docker exec`` into a local
postgres container and its shell commands run on this machine, so everything but the env's VM is real: the scripts,
psql's single transaction and ``\\gexec``, and the triggers.
"""
import asyncio
import os
import shutil
import subprocess
import uuid

import pytest

from agent_env.providers.env_state.env_state_provider import DatabaseStateProvider
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps import apply_sql as apply_sql_module
from agent_env.task_step.task_steps import service_db_sql
from agent_env.task_step.task_steps.apply_sql import ApplySqlTaskStep
from agent_env.task_step.task_steps.service_db_sql import ServiceDbHost, ServiceDbSqlError
from tst.util.capabilities import missing_capability_reason

pytestmark = pytest.mark.integration

_IMAGE = "public.ecr.aws/docker/library/postgres:16-alpine"

_BASE = """\
CREATE TABLE slack.channels (id serial PRIMARY KEY, name text NOT NULL);
CREATE TABLE slack.messages (id serial PRIMARY KEY, channel_id int REFERENCES slack.channels(id), body text,
                             posted_at timestamptz NOT NULL DEFAULT '2026-02-14T09:00:00Z');
CREATE TABLE slack.reactions (message_id int, emoji text);  -- no primary key
INSERT INTO slack.channels (name) VALUES ('general'), ('bugs');
INSERT INTO slack.messages (channel_id, body) VALUES (1, 'hello'), (2, 'login is broken');
INSERT INTO slack.reactions VALUES (2, 'eyes');
SELECT _install_changelog_triggers('slack');
"""


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


class _LocalHost(ServiceDbHost):
    """psql through ``docker exec`` into the test container; shell commands on this machine."""

    def __init__(self, container: str, work_dir: str):
        super().__init__(sandbox=None, env_id="test-env")
        self.container, self.work_dir = container, work_dir

    def psql(self, *flags: str) -> str:
        import shlex

        return " ".join(["docker", "exec", "-i", self.container, "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1",
                         "-U", "agentenv", "-d", "agentenv", *(shlex.quote(f) if " " in f else f for f in flags)])

    async def run(self, script: str):
        script = script.replace(service_db_sql.WORK_DIR, self.work_dir)
        proc = await asyncio.create_subprocess_exec("bash", "-c", script, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        return proc.returncode, out.decode(), err.decode()

    async def write(self, path: str, text: str) -> None:
        path = path.replace(service_db_sql.WORK_DIR, self.work_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


@pytest.fixture(scope="module")
def pg():
    if not _docker_ok():
        pytest.skip(missing_capability_reason("docker_daemon"))
    name = f"agentenv-sql-test-{uuid.uuid4().hex[:8]}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "-e", "POSTGRES_USER=agentenv",
                    "-e", "POSTGRES_PASSWORD=agentenv", "-e", "POSTGRES_DB=agentenv", _IMAGE],
                   check=True, capture_output=True)
    try:
        for _ in range(60):
            if subprocess.run(["docker", "exec", name, "pg_isready", "-U", "agentenv", "-d", "agentenv"],
                              capture_output=True).returncode == 0:
                # pg_isready answers during the image's init-time restart; a query proves the final server.
                if subprocess.run(["docker", "exec", name, "psql", "-U", "agentenv", "-d", "agentenv", "-c",
                                   "SELECT 1"], capture_output=True).returncode == 0:
                    break
            subprocess.run(["sleep", "1"])
        else:
            pytest.fail("postgres did not start")
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _sql(container: str, sql: str) -> str:
    res = subprocess.run(["docker", "exec", "-i", container, "psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1",
                          "-U", "agentenv", "-d", "agentenv", "-f", "-"], input=sql, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


@pytest.fixture
def base(pg, tmp_path, monkeypatch):
    """A fresh service database holding the base universe, with an empty changelog, as a loaded universe leaves it."""
    _sql(pg, "DROP SCHEMA IF EXISTS slack CASCADE; DROP TABLE IF EXISTS public._changelog CASCADE;")
    _sql(pg, DatabaseStateProvider.get_init_script(["slack"]))
    _sql(pg, _BASE)
    _sql(pg, "DELETE FROM public._changelog;")
    host = _LocalHost(pg, str(tmp_path))

    async def fake(context, env_id, what):
        return host

    monkeypatch.setattr(apply_sql_module, "service_db_host", fake)
    return pg


_INJECTION = """\
-- a CB's injection, as pasted: explicit ids, a quote, a tab and a newline in a value, a ':' that psql must not read
INSERT INTO slack.channels (id, name) VALUES (7, 'zombie-bugs');
INSERT INTO slack.messages (id, channel_id, body) VALUES
  (10, 7, 'Login 500s again: see ENG-1'),
  (11, 7, E'it''s back\tafter the\nhotfix :ticket');
UPDATE slack.messages SET body = 'login is broken (prod)' WHERE id = 2
"""


@pytest.mark.asyncio
async def test_apply_sql_applies_clears_its_changelog_and_syncs_sequences(base):
    ctx = TaskStepContext()
    await ApplySqlTaskStep(id="inject", version=1, env_id="test-env", sql=_INJECTION).execute(ctx)

    assert _sql(base, "SELECT count(*) FROM slack.messages") == "4"
    assert _sql(base, "SELECT body FROM slack.messages WHERE id = 11") == "it's back\tafter the\nhotfix :ticket"
    # Nothing the injection wrote is left in the changelog for the run to attribute to an agent ...
    assert _sql(base, "SELECT count(*) FROM public._changelog") == "0"
    # ... but the step recorded what it wrote, per table and operation.
    assert {(c["table_name"], c["operation"], c["rows"]) for c in ctx.metadata["applied_sql"]["inject"]["changes"]} \
        == {("channels", "INSERT", 1), ("messages", "INSERT", 2), ("messages", "UPDATE", 1)}
    # A service's next insert gets a fresh id instead of colliding with the injection's explicit ones.
    assert _sql(base, "INSERT INTO slack.messages (channel_id, body) VALUES (1, 'agent') RETURNING id") == "12"
    assert _sql(base, "INSERT INTO slack.channels (name) VALUES ('new') RETURNING id") == "8"
    # And that write, the agent's, is the changelog's only entry.
    assert _sql(base, "SELECT count(*) FROM public._changelog WHERE operation = 'INSERT'") == "2"


@pytest.mark.asyncio
async def test_failing_sql_rolls_back_everything_and_names_the_error(base):
    ctx = TaskStepContext()
    sql = "INSERT INTO slack.channels (id, name) VALUES (7, 'zombie-bugs');\nINSERT INTO slack.nope VALUES (1);\n"
    with pytest.raises(ServiceDbSqlError) as e:
        await ApplySqlTaskStep(id="inject", version=1, env_id="test-env", sql=sql).execute(ctx)
    assert 'relation "slack.nope" does not exist' in e.value.message
    assert "line 2 of the SQL:" in e.value.message  # the failing statement's line, counted in the SQL as pasted
    assert _sql(base, "SELECT count(*) FROM slack.channels") == "2"  # the first insert was rolled back too
    assert ctx.metadata["applied_sql"]["inject"]["applied"] is False

