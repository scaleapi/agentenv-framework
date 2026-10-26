"""The proxy's folder installed into a running container by its own install.sh, as install_agent installs it there. It
brings its own Python and serves; its bash runs commands with bash, or with sh in a container that has only sh (Alpine,
musl), and says which. The container installs this checkout's protocol, so the test covers the code beside it."""

import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

import agent_env

pytestmark = [pytest.mark.integration]

AGENT_DIR = Path(agent_env.__file__).parent / "agents" / "agents" / "proxy"
PROTOCOL_DIR = Path(agent_env.__file__).parents[2] / "packages" / "agentenv-protocol"
PORT = 8000
TOKEN = "t" * 32


@pytest.fixture(params=[("alpine:3.20", "sh"), ("debian:12-slim", "bash")], ids=["sh-only", "with-bash"])
def container(request):
    image, shell = request.param
    name = f"proxy-{shell}-{uuid.uuid4().hex[:8]}"
    started = subprocess.run(["docker", "run", "-d", "--name", name, "-p", f"127.0.0.1::{PORT}",
                              f"mirror.gcr.io/library/{image}", "sleep", "infinity"], capture_output=True, text=True)
    assert started.returncode == 0, started.stderr
    try:
        yield name, shell
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_installed_into_a_container_the_proxy_serves_and_its_bash_runs_the_shell_there(container, tmp_path):
    name, shell = container
    has_bash = subprocess.run(["docker", "exec", name, "sh", "-c", "command -v bash"], capture_output=True).returncode == 0
    assert has_bash == (shell == "bash")
    folder = tmp_path / "agent"
    folder.mkdir()
    for file in ("install.sh", "a2a_server.py"):
        shutil.copy(AGENT_DIR / file, folder / file)
    (folder / "requirements.txt").write_text("agentenv-framework-protocol[agent] @ file:///tmp/protocol\n")
    subprocess.run(["docker", "cp", f"{PROTOCOL_DIR}/.", f"{name}:/tmp/protocol"], check=True, capture_output=True)

    subprocess.run(["bash", str(folder / "install.sh"), "container", name, str(PORT)], check=True)

    with httpx.Client(base_url=f"http://{_published(name)}", timeout=30) as client:
        _serving(client)
        assert client.post("/ext/proxy/claim", json={"token": TOKEN}).status_code == 200
        task_id = _send(client, "go")
        tools = _proxy(client, "tools", task_id=task_id).json()["tools"]
        result = _done(client, task_id, {"kind": "bash", "command": "echo $0; echo out"})

    assert [tool["name"] for tool in tools] == ["bash"]
    assert tools[0]["description"].startswith(
        "Run a bash command where" if shell == "bash" else "Run a command where the agent works, as its user, in its "
                                                           "folder, with POSIX sh: this place has no bash")
    assert (result["exit_code"], result["stdout"]) == (0, f"{shell}\nout\n")


def _published(container: str) -> str:
    return subprocess.run(["docker", "port", container, str(PORT)], check=True, capture_output=True,
                          text=True).stdout.splitlines()[0]


def _serving(client: httpx.Client) -> None:
    deadline = time.monotonic() + 120
    while True:
        try:
            if client.get("/.well-known/agent.json").status_code == 200:
                return
        except httpx.TransportError:
            pass
        assert time.monotonic() < deadline, "the proxy never served its card"
        time.sleep(1)


def _send(client: httpx.Client, text: str) -> str:
    response = client.post("/a2a", json={
        "jsonrpc": "2.0", "id": "1", "method": "message/send",
        "params": {"message": {"kind": "message", "messageId": f"message-{text}", "role": "user",
                               "contextId": "context-1", "parts": [{"kind": "text", "text": text}]},
                   "configuration": {"blocking": False}},
    })
    return response.json()["result"]["id"]


def _proxy(client: httpx.Client, operation: str, **body: Any) -> httpx.Response:
    return client.post(f"/ext/proxy/{operation}", json={"token": TOKEN, **body})


def _done(client: httpx.Client, task_id: str, action: dict[str, Any]) -> dict[str, Any]:
    state = _proxy(client, "act", task_id=task_id, action_id="a1", action=action).json()
    while not state["done"]:
        state = _proxy(client, "result", action_id="a1").json()
    return state["result"]
