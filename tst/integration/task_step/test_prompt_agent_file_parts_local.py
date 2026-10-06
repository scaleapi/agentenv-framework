"""prompt_agent hands store files to the agents it talks to, on the local defaults. A deployed agent is sent a URL it
can fetch for each file in its prompt, and a user-sim one for each file the solver passes back of those it was sent;
a file the solver names but was never sent stays as it is, so the user-sim can't read it. A human peer is sent the
store's own URL and reads the store itself. The run records the store's own URLs throughout. The agents are the echo
agent (tst/data/a2a_agent), which reports what each file part held. Needs Docker and a throwaway local registry."""

import importlib.util
import json
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid

import httpx
import pytest
import uvicorn

from agent_env.a2a_agent import conversation_store
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, get_config, reset_config, set_image_store
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.task import Task
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_human_agent import DeployHumanAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.util.a2a_test_agent import AGENT_DIR, put_test_agent

pytestmark = pytest.mark.integration


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def local_registry(monkeypatch, tmp_path):
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    port = _free_port()
    name = f"file-parts-registry-{uuid.uuid4().hex[:8]}"
    started = _docker("run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5000", "registry:2")
    assert started.returncode == 0, started.stderr
    host = f"localhost:{port}"
    deadline = time.time() + 45
    while time.time() < deadline:
        try:
            if httpx.get(f"http://{host}/v2/", timeout=2).status_code in (200, 401):
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    configure()
    set_image_store(LocalRegistryImageStore(host))
    reset_artifact_store()
    try:
        yield tmp_path
    finally:
        # Only this test's containers: every local sandbox it deployed has a work dir in its own sandbox root, and its
        # container is named after that sandbox. Images are shared, so matching by image would reach other runs.
        for work_dir in sandboxes.glob("agent-env-*"):
            if m := re.match(r"agent-env-(local-[0-9a-f]+)-", work_dir.name):
                _docker("rm", "-f", f"agent-{m.group(1)}")
        _docker("rm", "-f", name)
        shutil.rmtree(sandboxes, ignore_errors=True)
        # The agents fetched through this process's grant server, whose certificate is from this test's state root;
        # a later test's agents trust another root's CA, so it must start afresh.
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


@pytest.fixture
def human_peer():
    """The echo agent served in this process, beside the store, as a human's hub would be: its URL."""
    spec = importlib.util.spec_from_file_location("echo_agent_here", AGENT_DIR / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(module.EchoAgent().create_app(), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _files(suffix: str) -> dict[str, str]:
    """Two text files in the store, by name: the object URL of each, whose content is ``<name>-<suffix>``."""
    store = get_config().get_object_store()
    return {
        name: store.put(f"file-parts-{suffix}/{name}.txt", f"{name}-{suffix}".encode(), content_type="text/plain")
        for name in ("brief", "report")
    }


def _deploy(agent, name: str) -> DeployAgentTaskStep:
    return DeployAgentTaskStep(
        id=f"deploy-{name}", version=None, env_ids=[], a2a_agent_id=agent.id, a2a_agent_version=agent.version,
        agent_name=name, sandbox_type="local",
    )


def _conversation(ctx) -> list[dict]:
    """The conversation's messages: the solver's prompt, its reply, the peer's reply, the solver's second reply."""
    assert not ctx.metadata.get("failed_steps"), ctx.metadata.get("failed_steps")
    return conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["ask"])["messages"]


def _text(message: dict) -> str:
    return "\n".join(part["text"] for part in message["parts"] if part["kind"] == "text")


def _uris(message: dict) -> list[str]:
    return [part["file"]["uri"] for part in message["parts"] if part["kind"] == "file"]


@pytest.mark.asyncio
async def test_the_solver_and_a_user_sim_fetch_the_files_the_conversation_shares_and_no_other(local_registry):
    suffix = uuid.uuid4().hex[:8]
    urls = _files(suffix)
    agent = put_test_agent(f"file-parts-agent-{suffix}")
    task = Task.put(id=f"file-parts-{suffix}", steps=[
        _deploy(agent, "solver"),
        _deploy(agent, "user"),
        PromptAgentTaskStep(
            id="ask", version=None, prompt_id="ask", agent_name="solver", user_agent_name="user",
            max_conversation_turns=2, poll_interval_seconds=1,
            parts=[
                {"kind": "text", "text": f"send-file {urls['brief']}\nsend-file {urls['report']}"},
                {"kind": "file", "file": {"uri": urls["brief"], "name": "brief.txt", "mimeType": "text/plain"}},
            ],
        ),
    ])

    prompt, reply, user_reply, _ = _conversation(await task.run())

    assert f"read brief.txt over https: brief-{suffix}" in _text(reply)
    # The user-sim's own lines follow its echo of the reply, one per file it was sent.
    shared, never_sent = _text(user_reply).splitlines()[-2:]
    assert shared == f"read brief.txt over https: brief-{suffix}"
    assert never_sent.startswith("could not read report.txt over file:")
    assert (_uris(prompt), _uris(reply)) == ([urls["brief"]], [urls["brief"], urls["report"]])
    assert "https://" not in json.dumps([prompt, reply, user_reply])


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [False, True], ids=["user_a2a_url", "registered"])
async def test_a_human_peer_is_sent_the_stores_own_url_and_reads_the_store_itself(local_registry, human_peer, registered):
    suffix = uuid.uuid4().hex[:8]
    urls = _files(suffix)
    agent = put_test_agent(f"file-parts-agent-{suffix}")
    peer = {"user_agent_name": "human"} if registered else {"user_a2a_url": human_peer}
    task = Task.put(id=f"file-parts-human-{suffix}", steps=[
        _deploy(agent, "solver"),
        *([DeployHumanAgentTaskStep(id="register-human", version=None, agent_name="human", a2a_url=human_peer)]
          if registered else []),
        PromptAgentTaskStep(
            id="ask", version=None, prompt_id="ask", agent_name="solver", max_conversation_turns=2,
            poll_interval_seconds=1, prompt=f"send-file {urls['report']}", **peer,
        ),
    ])

    _, reply, peer_reply, _ = _conversation(await task.run())

    assert _uris(reply) == [urls["report"]]
    assert f"read report.txt over file: report-{suffix}" in _text(peer_reply)
