"""The conformance probe at full against the desktop, RUNS times: a check passes only when it passed, or didn't
apply, in every run."""
from agentenv_protocol.computer.conformance import run_probe

RUNS = 3


async def verify(mcp_url):
    base, outcomes = mcp_url.rsplit("/mcp", 1)[0], {}
    for _ in range(RUNS):
        for check in (await run_probe(base, "full")).checks:
            seen = outcomes.setdefault(f"{check.check} {check.name}", [])
            seen.append(check.status if check.status in ("passed", "not_applicable") else f"{check.status}: {check.detail}")
    return [{"id": name, "result": all(s in ("passed", "not_applicable") for s in seen), "description": "; ".join(seen)}
            for name, seen in outcomes.items()]
