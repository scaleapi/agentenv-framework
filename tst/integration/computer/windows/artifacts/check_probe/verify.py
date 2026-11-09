"""The conformance probe at full against the deployed Windows computer: one result per check, passing when it passed,
didn't apply, or is the failure cua-driver declares it's known for on Windows (env.toml's known_failures.windows), as
gate 3 waives it."""
from agentenv_protocol.computer.conformance import run_probe

KNOWN = {("P4", "16 clicks sent at once all land")}


async def verify(mcp_url, env_url):
    report = await run_probe(env_url, "full")
    return [{"id": f"{check.check} {check.name}",
             "result": check.status in ("passed", "not_applicable") or (check.check, check.name) in KNOWN,
             "description": f"{check.status}: {check.detail}" if check.detail else check.status}
            for check in report.checks]
