"""The conformance probe at full against the deployed Windows computer: one result per check, passing when it passed,
didn't apply, or couldn't be verified, as checks that need computer-exec can't be on Windows yet."""
from agentenv_protocol.computer.conformance import run_probe


async def verify(mcp_url, env_url):
    report = await run_probe(env_url, "full")
    return [{"id": f"{check.check} {check.name}", "result": check.status in ("passed", "not_applicable", "unverified"),
             "description": f"{check.status}: {check.detail}" if check.detail else check.status}
            for check in report.checks]
