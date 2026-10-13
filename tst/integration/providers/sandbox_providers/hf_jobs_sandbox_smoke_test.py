"""HF Jobs smoke test.

Runs when the resolved config can build the ``hf_jobs`` sandbox provider (a Hugging Face token, configured in
``[sandbox.providers.hf_jobs.config]`` or from ``hf auth login``) — the ``remote_sandbox`` capability — and skips with
the declared reason otherwise. Two cpu-basic jobs run for a minute or two, a cent's worth of credits.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

_PORT = 8080
_IMAGE = "python:3.12-slim"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("hf_jobs"),
]


@pytest.fixture(scope="module")
def hf_jobs_provider():
    return build_sandbox_provider("hf_jobs")


@pytest_asyncio.fixture(scope="module")
async def server(hf_jobs_provider):
    """A web server on the provider's network group, its port public; always cancelled after the test."""
    sandbox = await hf_jobs_provider.create_container(
        image_name=_IMAGE, port=_PORT, env={"GREETING": "hello"}, command=["python", "-m", "http.server", str(_PORT)],
        private_network=True, timeout=900)
    try:
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_a_job_serves_its_port_publicly_runs_commands_and_is_reached_by_another_on_the_group(hf_jobs_provider,
                                                                                                   server):
    assert server.type == "hf_jobs" and server.private_host
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(60):
            try:
                if (await client.get(server.tunnel_urls[_PORT])).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        else:
            pytest.fail(f"{server.tunnel_urls[_PORT]} never answered 200")

    exit_code, greeting, stderr = await server.exec_with_output("sh", "-c", 'printf %s "$GREETING"')
    assert (exit_code, greeting) == (0, "hello"), stderr

    client_job = await hf_jobs_provider.create_container(
        image_name=_IMAGE, port=_PORT, env={}, command=["sleep", "600"], expose_externally=False,
        private_network=True, timeout=900)
    try:
        exit_code, status, stderr = await client_job.exec_with_output(
            "python", "-c", f"import urllib.request; print(urllib.request.urlopen('http://{server.private_host}:{_PORT}/',"
                            " timeout=30).status)")
        assert (exit_code, status.strip()) == (0, "200"), stderr
    finally:
        await client_job.terminate()

    found = await hf_jobs_provider.get_sandbox(server.sandbox_id)
    assert (found.private_host, found.tunnel_urls) == (server.private_host, server.tunnel_urls)
