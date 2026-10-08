"""Sail Sailbox smoke test.

Runs when the resolved config can build the ``sail_vm`` sandbox provider (``[sandbox.providers.sail_vm.config]``
with ``api_key = "secret:sail_api_key"``), the ``remote_sandbox`` capability, and skips with the declared
reason otherwise. The test never reads or prints the API key.
"""

from __future__ import annotations

import hashlib
import json
import secrets

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.sail_vm.model_key import PLACEHOLDER, secret_name
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

_PORT = 8080

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("sail_vm"),
]


@pytest.fixture(scope="module")
def sail_provider():
    return build_sandbox_provider("sail_vm")


@pytest_asyncio.fixture(scope="module")
async def sail_sandbox(sail_provider):
    sandbox = await sail_provider.create_vm(cpu=1.0, memory=2048, exposed_ports=[_PORT], timeout=900)
    try:
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_sailbox_runs_docker_serves_its_port_and_reconnects(sail_provider, sail_sandbox):
    assert (sail_sandbox.type, sail_sandbox.mode) == ("sail_vm", "vm")

    await sail_sandbox.exec_script(
        f"docker run -d --name web -p {_PORT}:80 public.ecr.aws/nginx/nginx:alpine > /dev/null"
    )
    url = sail_sandbox.tunnel_urls[_PORT]
    assert url.startswith("https://") and url.endswith(".sail.box")
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(30):
            response = await client.get(url)
            if response.status_code == 200:
                break
    assert response.status_code == 200
    assert "nginx" in response.text

    data = bytes(range(256)) * 4096
    await sail_sandbox.write_host_file(data, "/tmp/agent-env-smoke/blob.bin")
    exit_code, digest, stderr = await sail_sandbox.exec_with_output("sha256sum", "/tmp/agent-env-smoke/blob.bin")
    assert exit_code == 0, stderr
    assert digest.split()[0] == hashlib.sha256(data).hexdigest()

    reconnected = await sail_provider.get_sandbox(sail_sandbox.sandbox_id)
    assert reconnected.sandbox_id == sail_sandbox.sandbox_id
    assert reconnected.tunnel_urls == sail_sandbox.tunnel_urls
    assert reconnected.network_policy == NetworkPolicy()


async def test_an_allowlist_is_enforced_for_containers_in_the_sailbox(sail_provider):
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",))
    sandbox = await sail_provider.create_vm(cpu=1.0, memory=2048, exposed_ports=[], timeout=600, network_policy=policy)
    try:
        assert "*.sail.box" in sandbox.network_policy.allow_hosts
        exit_code, out, stderr = await sandbox.exec_with_output(
            "bash", "-c",
            "curl -s -m 10 -o /dev/null -w '%{http_code}' https://pypi.org/simple/; echo; "
            "curl -s -m 10 -o /dev/null https://example.com && echo example-reachable || echo example-blocked",
        )
        assert exit_code == 0, stderr
        assert out.split() == ["200", "example-blocked"]
        reconnected = await sail_provider.get_sandbox(sandbox.sandbox_id)
        assert reconnected.network_policy == sandbox.network_policy
    finally:
        await sandbox.terminate()


_ECHO_HOST = "httpbin.org"
_CLIENT = (
    "import json, os, urllib.request; "
    f"r = urllib.request.Request('https://{_ECHO_HOST}/headers', headers={{'Authorization': 'Bearer ' + os.environ['LITELLM_API_KEY']}}); "
    "print(json.loads(urllib.request.urlopen(r, timeout=20).read())['headers']['Authorization'])"
)


async def test_an_agents_model_key_is_injected_by_sail_and_never_enters_the_sailbox(sail_provider):
    """A throwaway key against a header-echo host stands in for the model endpoint. The provider leaves the
    key's secret in Sail; this test deletes its own."""
    key = f"sk-agentenv-smoke-{secrets.token_hex(16)}"
    env = {"LITELLM_API_KEY": key, "LITELLM_BASE_URL": f"https://{_ECHO_HOST}/v1"}
    sandbox = await sail_provider.create_sandbox(image_name="unused", port=_PORT, env=env, cpu=1.0, memory=2048, timeout=900)
    try:
        flags = " ".join(f"-e {name}='{value}'" for name, value in env.items())
        await sandbox.exec_script(f"docker run -d --name agent-api {flags} public.ecr.aws/docker/library/python:3.12-slim sleep 600 > /dev/null")
        exit_code, sent, stderr = await sandbox.exec_with_output("docker", "exec", "agent-api", "python", "-c", _CLIENT)
        assert exit_code == 0, stderr
        assert sent.strip() == f"Bearer {key}"

        _, inside, _ = await sandbox.exec_with_output("docker", "exec", "agent-api", "printenv", "LITELLM_API_KEY")
        assert inside.strip() == PLACEHOLDER
        _, config, _ = await sandbox.exec_with_output("docker", "inspect", "agent-api")
        assert key not in config and PLACEHOLDER in json.dumps(json.loads(config)[0]["Config"]["Env"])
    finally:
        await sandbox.terminate()
    secret = await sandbox._sdk.Secret.get.aio(secret_name(key))
    await secret.delete.aio()
