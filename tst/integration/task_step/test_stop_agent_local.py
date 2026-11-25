"""stop_agent in a real container on a local sandbox: it ends what an agent left running there and spares the server,
what ran before it and another user's processes, on a busybox and a Debian userland; and a verifier reading the
container after it grades what the tests wrote, where before it the agent's leftover overwrote the reward. The
verifier reads a result file in the container, refusing a link. Needs Docker and a pull of two small public images."""

import asyncio
import shutil
import subprocess
import uuid

import pytest
import pytest_asyncio

from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.stop_agent import StopAgentTaskStep
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)
from tst.util.capabilities import missing_capability_reason

pytestmark = pytest.mark.integration

PORT = 8000
WRITER = "while :; do echo 0 > /tmp/reward; sleep 0.2; done"
DEAF = "trap '' TERM; while :; do sleep 1; done"
TREE = "sleep 1001 & sleep 1002 & wait"
SERVICE = "while :; do sleep 0.3; done"
# Per image: the agent's server, listening on PORT, and a process of another user.
USERLANDS = {
    "busybox": ("mirror.gcr.io/library/alpine:3.20", f"nc -lk -p {PORT}", "su nobody -s /bin/sh -c 'sleep 1003'"),
    "debian": ("mirror.gcr.io/library/debian:bookworm-slim",
               f"perl -MIO::Socket::INET -e 'my $s = IO::Socket::INET->new(LocalPort => {PORT}, Listen => 5, "
               "ReuseAddr => 1) or die; sleep 100000'",
               "setpriv --reuid=65534 --regid=65534 --clear-groups sleep 1003"),
}


@pytest.fixture
def local_backends(monkeypatch, tmp_path):
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    configure()
    reset_artifact_store()
    try:
        yield tmp_path
    finally:
        reset_artifact_store()
        reset_config()


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout.strip()


def _running(container: str) -> list[str]:
    """The command line of each process in ``container`` that hasn't exited."""
    listing = _docker("exec", container, "sh", "-c", 'for d in /proc/[0-9]*; do read -r s < "$d/stat"; '
                      'set -- ${s##*") "}; [ "$1" = Z ] || tr "\\000" " " < "$d/cmdline"; echo; done')
    return [line.strip() for line in listing.splitlines() if line.strip()]


async def _until(check, what: str):
    for _ in range(100):
        if check():
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


@pytest_asyncio.fixture
async def agent_left_running(local_backends, request):
    """A container on a local sandbox, as an agent left it: a service of its own started before the agent's server,
    then the server, then the agent's leftovers (a writer that daemonized, one deaf to SIGTERM, a tree), orphaned as a
    shell's background jobs are when it exits, and another user's process; a reward the agent wrote and a link it
    planted. Yields the sandbox, the container and the run's context, the agent recorded as installed in the
    container."""
    image, server, other = USERLANDS[request.param]
    sandbox = await LocalSandboxProvider().create_vm()
    container = sandbox.scoped_name(f"repo-{uuid.uuid4().hex[:8]}")
    try:
        _docker("run", "-d", "--name", container, image, "sleep", "infinity")
        _docker("exec", "-d", container, "sh", "-c", f"({SERVICE}) & sleep 1.1; {server} & sleep 1.1; "
                f"setsid sh -c '{WRITER}' > /dev/null 2>&1 & sh -c \"{DEAF}\" & sh -c '{TREE}' & {other} & "
                "mkdir -p /logs; echo 1 > /logs/reward.txt; ln -s /etc/hostname /logs/link.txt; touch /logs/up")
        await _until(lambda: "sleep 1003" in _running(container) and "up" in _docker("exec", container, "ls", "/logs"),
                     "the agent's leftovers")
        yield sandbox, container, TaskStepContext(
            deployed_agents=[DeployedAgent(agent_name="coder", api_url="http://127.0.0.1:9",
                                           sandbox_id=sandbox.sandbox_id, sandbox_type="local",
                                           installed_container=container, a2a_port=PORT)],
            deployed_sandboxes=[DeployedSandbox(sandbox_name="box", sandbox_id=sandbox.sandbox_id, sandbox_mode="vm",
                                                sandbox_type="local")],
            metadata={"deployed_docker_containers": [{"container_name": container.removesuffix(
                f"-{sandbox.sandbox_id}"), "sandbox_name": "box"}]})
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        await sandbox.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_left_running", sorted(USERLANDS), indirect=True)
async def test_stop_agent_ends_what_the_agent_left_and_spares_the_rest(agent_left_running):
    sandbox, container, context = agent_left_running
    step = StopAgentTaskStep(id="stop", version=None, agent_name="coder", grace_s=1,
                             clear_paths=["/logs/reward.txt", "/logs/link.txt", "/logs/none"])

    await step.execute(context)

    report = context.metadata["stop_agent"]["stop"]
    signalled = {entry["command"]: entry["signals"] for entry in report["processes"]}
    assert signalled[f"sh -c {WRITER}"] == ["TERM"] and signalled[f"sh -c {DEAF}"] == ["TERM", "KILL"]
    assert signalled[f"sh -c {TREE}"] == ["TERM"] and {"sleep 1001", "sleep 1002"} <= set(signalled)
    assert str(PORT) in report["server"]["command"]
    assert report["processes_left"] == []
    left = _running(container)
    assert [c for c in left if c.startswith(("sh -c while", "sh -c trap", "sleep 100", "sleep 0.2"))] == ["sleep 1003"]
    assert any(SERVICE in c for c in left) and any(str(PORT) in c for c in left)
    assert [(e["path"], e["kind"], e["removed"]) for e in report["cleared"]] == [
        ("/logs/reward.txt", "file", True), ("/logs/link.txt", "link", True), ("/logs/none", "missing", False)]
    assert report["cleared"][0]["size"] == 2 and report["cleared"][1]["target"] == "/etc/hostname"
    _docker("exec", container, "rm", "-f", "/tmp/reward")
    await asyncio.sleep(0.6)
    assert "reward" not in _docker("exec", container, "ls", "/tmp")


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_left_running", ["debian"], indirect=True)
async def test_a_verifier_after_stop_agent_grades_the_tests_reward_not_the_agents_leftover(agent_left_running):
    sandbox, container, context = agent_left_running
    name = container.removesuffix(f"-{sandbox.sandbox_id}")

    def tests(step_id):
        return RunContainerUnitTestsVerifierTaskStep(
            id=step_id, version=None, sandbox_name="box", container_name=name, verifier_id=step_id,
            command="echo 1 > /tmp/reward; sleep 1", reward_path="/tmp/reward", result_paths=["/logs/link.txt"])

    await tests("before").execute(context)
    await StopAgentTaskStep(id="stop", version=None, agent_name="coder", grace_s=1).execute(context)
    await tests("after").execute(context)

    graded = context.metadata["verifications"]
    assert (graded["before"]["score"], graded["after"]["score"]) == (0.0, 1.0)
    # A link, which a read of a copy on the sandbox's host would follow there: refused.
    assert graded["after"]["extracted_files"]["/logs/link.txt"] is None


@pytest.mark.asyncio
async def test_a_server_of_another_user_is_found_though_root_cant_read_its_descriptors(local_backends):
    """Root in a container has no CAP_SYS_PTRACE, so it reads only its own user's descriptors: the server, running as
    nobody as an image's agent user would, is the oldest process of its socket's user. The writer that user left ends;
    root's own process, started after, stays."""
    image, server, _ = USERLANDS["busybox"]
    sandbox = await LocalSandboxProvider().create_vm()
    container = sandbox.scoped_name(f"user-{uuid.uuid4().hex[:8]}")
    try:
        _docker("run", "-d", "--name", container, image, "sleep", "infinity")
        _docker("exec", "-d", container, "sh", "-c", f"su nobody -s /bin/sh -c '{server}' & sleep 1.1; "
                f"su nobody -s /bin/sh -c \"setsid sh -c '{WRITER}' > /dev/null 2>&1 &\"; (sleep 1004 &); "
                "touch /tmp/up")
        await _until(lambda: "up" in _docker("exec", container, "ls", "/tmp"), "the agent's leftovers")
        context = TaskStepContext(deployed_agents=[DeployedAgent(
            agent_name="coder", api_url="http://127.0.0.1:9", sandbox_id=sandbox.sandbox_id, sandbox_type="local",
            installed_container=container, a2a_port=PORT)])

        await StopAgentTaskStep(id="stop", version=None, agent_name="coder", grace_s=1).execute(context)

        report = context.metadata["stop_agent"]["stop"]
        assert report["server"]["command"] == server
        assert f"sh -c {WRITER}" in {entry["command"] for entry in report["processes"]}
        left = _running(container)
        assert "sleep 1004" in left and server in left and f"sh -c {WRITER}" not in left
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        await sandbox.terminate()


@pytest.mark.asyncio
@pytest.mark.parametrize("second_user", ["root", "nobody"])
async def test_stopping_one_agent_spares_the_server_of_another_in_its_container(local_backends, second_user):
    """Two agents installed into one container, the second's server started after the first's, as root too or as
    another user: stopping the first ends its leftover, and spares the second's server and what that server
    started."""
    image, server, _ = USERLANDS["busybox"]
    sandbox = await LocalSandboxProvider().create_vm()
    container = sandbox.scoped_name(f"two-{uuid.uuid4().hex[:8]}")
    second = f"nc -lk -p {PORT + 1}"
    start_second = f"sh -c 'sleep 1005 & exec {second}'"
    if second_user != "root":
        start_second = f"su {second_user} -s /bin/sh -c \"{start_second}\""
    try:
        _docker("run", "-d", "--name", container, image, "sleep", "infinity")
        _docker("exec", "-d", container, "sh", "-c", f"{server} & sleep 1.1; {start_second} & "
                f"sleep 1.1; setsid sh -c '{WRITER}' > /dev/null 2>&1 & touch /tmp/up")
        await _until(lambda: "up" in _docker("exec", container, "ls", "/tmp"), "both servers")
        agents = [DeployedAgent(agent_name=name, api_url="http://127.0.0.1:9", sandbox_id=sandbox.sandbox_id,
                                sandbox_type="local", installed_container=container, a2a_port=port)
                  for name, port in (("coder", PORT), ("helper", PORT + 1))]

        await StopAgentTaskStep(id="stop", version=None, agent_name="coder", grace_s=1).execute(
            TaskStepContext(deployed_agents=agents))

        left = _running(container)
        assert server in left and second in left and "sleep 1005" in left and f"sh -c {WRITER}" not in left
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        await sandbox.terminate()
