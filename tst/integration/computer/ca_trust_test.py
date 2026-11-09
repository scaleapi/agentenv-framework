"""A computer routing a site trusts the deploy's CA in every runtime it has, however a program started: an app launched
at boot loads the site through Node, an apt JDK, a JDK copied under /opt with a store of its own, and curl; a command
the computer runs sees the trust variables and loads it through Python with SSL_CERT_FILE alone; the CA is at
/etc/agentenv/ca.pem, among the CAs agentenv adds, and in /etc/agentenv/ca-bundle.pem beside the public roots; and the
deploy's record keeps each runtime's check and each JDK store it trusted. Needs Docker, the local registry on
localhost:5000, and Ubuntu's archive for the app's packages."""

import json

import pytest

from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from tst.integration.computer.start_tabs import host_pages

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

URL = "https://shop.test/landed"
# Loads the URL it's given through SSL_CERT_FILE alone, printing the status it's answered.
THROUGH_CERT_FILE = ("import os, ssl, sys, urllib.request\n"
                     "context = ssl.create_default_context(cafile=os.environ['SSL_CERT_FILE'])\n"
                     "print(urllib.request.urlopen(sys.argv[1], context=context).status)")


@pytest.mark.asyncio
async def test_every_runtime_of_a_computer_routing_a_site_trusts_its_ca_launched_at_boot_or_run_by_the_computer(
        tls_desktop):
    with host_pages() as port, namespace_routing():
        params = {"site_routing": True, "sites": [{"name": "shop", "url": f"http://127.0.0.1:{port}/website/shop/",
                                                   "domains": ["shop.test"]}],
                  "app_settings": {"clients": {"url": URL}}}
        deployed = await Env.get(tls_desktop.id).deploy(env_params=params)
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            at_boot = json.loads((await inside.run(deployed, ["cat", ".tls.json"], timeout_s=30))["stdout"])
            variables = (await inside.run(deployed, ["sh", "-c", "echo $SSL_CERT_FILE $REQUESTS_CA_BUNDLE "
                                                     "$NODE_EXTRA_CA_CERTS"], timeout_s=30))["stdout"]
            through_cert_file = await inside.run(deployed, ["python3", "-c", THROUGH_CERT_FILE, URL], timeout_s=60)
            added = (await inside.run(deployed, ["cat", "/etc/agentenv/ca.pem"], timeout_s=30))["stdout"]
            mode = (await inside.run(deployed, ["stat", "-c", "%a", "/etc/agentenv/ca.pem"], timeout_s=30))["stdout"]
            bundle = (await inside.run(deployed, ["cat", "/etc/agentenv/ca-bundle.pem"], timeout_s=30))["stdout"]
        finally:
            report = await teardown_run(context)

    routed = deployed.metadata["computer"]["sites"]
    assert at_boot == {"node": "200", "java": "200", "copied java": "200", "curl": "200"}, at_boot
    assert variables.split() == ["/etc/agentenv/ca-bundle.pem", "/etc/agentenv/ca-bundle.pem", "/etc/agentenv/ca.pem"]
    assert through_cert_file["exit_code"] == 0 and through_cert_file["stdout"].strip() == "200", through_cert_file
    assert "forward_port" not in routed
    assert routed["ca_path"] == "/etc/agentenv/ca.pem" and routed["ca"].strip() in added and mode.strip() == "644"
    assert routed["ca"].strip() in bundle and bundle.count("BEGIN CERTIFICATE") > 100
    assert {runtime: "status" in result for runtime, result in routed["runtimes"].items()} == {
        "python SSL_CERT_FILE": True, "node": True, "curl": True}, routed["runtimes"]
    stores = routed["trust"]["java"]
    assert len(stores) == 2 and all(result == {"trusted": True} for result in stores.values()), stores
    assert any(store.startswith("/opt/copied-jdk/") for store in stores) and not report.failed
