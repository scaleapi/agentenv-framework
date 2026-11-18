"""The conformance probe at full against the deployed Omarchy computer: one result per check, passing when it passed or
didn't apply. cua-driver declares no known failures on Omarchy, so none is waived."""
from agentenv_protocol.computer.conformance import run_probe


async def verify(mcp_url, env_url):
    report = await run_probe(env_url, "full")
    return [{"id": f"{check.check} {check.name}", "result": check.status in ("passed", "not_applicable"),
             "description": f"{check.status}: {check.detail}" if check.detail else check.status}
            for check in report.checks]
