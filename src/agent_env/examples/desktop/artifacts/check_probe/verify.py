"""The conformance probe at full against the deployed computer: one result per check, passing when it passed or
didn't apply."""
from agentenv_protocol.computer.conformance import run_probe


async def verify(mcp_url):
    report = await run_probe(mcp_url.rsplit("/mcp", 1)[0], "full")
    return [{"id": f"{check.check} {check.name}", "result": check.status in ("passed", "not_applicable"),
             "description": check.detail or check.status} for check in report.checks]
