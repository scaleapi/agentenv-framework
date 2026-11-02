"""The conformance probe: checks a running computer driver against the contract through its card, tools and
extensions, and reports each check and each declared extension. ``basic`` needs nothing on the computer; ``full``
stages the Linux fixture with data/add and reads it back with computer-exec. Run it as
``python -m agentenv_protocol.computer.conformance <driver url> [--level basic|full] [--json report.json]``."""
from __future__ import annotations

from .checks import Probe
from .client import Driver
from .report import CheckResult, Level, ProbeReport, Status, Verdict

__all__ = ["run_probe", "ProbeReport", "CheckResult", "Level", "Status", "Verdict"]


async def run_probe(url: str, level: Level = "full", *, timeout_s: float = 60, ready_timeout_s: float = 60,
                    token: str | None = None) -> ProbeReport:
    """Probe the driver serving its card at ``url``, sending ``token``, its control token, to its framework-only routes.
    ``ready_timeout_s`` is how long it may take to report ready."""
    driver = await Driver.connect(url, timeout_s, token)
    async with driver.session() as session:
        probe = Probe(driver, session, ready_timeout_s)
        await probe.basic()
        if level == "full":
            await probe.full()
    return ProbeReport.of(level, driver.name, probe.results, driver.extensions)
