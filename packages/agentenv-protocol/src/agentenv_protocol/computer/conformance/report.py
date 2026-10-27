"""What a probe found: each check's status, and each declared extension verified, unverified or failed. The report is
the value M0.7 stores in an env's ``metadata.validated_boot``."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["basic", "full"]
Status = Literal["passed", "failed", "unverified", "not_applicable"]
Verdict = Literal["verified", "unverified", "failed"]


class CheckResult(BaseModel):
    #: The design's check id, P0 to P9, or X for a check of one extension's own methods.
    check: str
    name: str
    status: Status
    detail: str = ""
    #: The extension URNs this result counts toward.
    extensions: list[str] = Field(default_factory=list)


class ProbeReport(BaseModel):
    level: Level
    driver: str
    passed: bool
    checks: list[CheckResult]
    extensions: dict[str, Verdict]

    @classmethod
    def of(cls, level: Level, driver: str, checks: list[CheckResult], declared: list[str]) -> ProbeReport:
        """The report for ``checks``: it passes when none failed, and each declared extension is failed when one of
        its checks failed, verified when one passed, and unverified otherwise."""
        extensions: dict[str, Verdict] = {}
        for uri in declared:
            mine = [c.status for c in checks if uri in c.extensions]
            extensions[uri] = "failed" if "failed" in mine else "verified" if "passed" in mine else "unverified"
        return cls(level=level, driver=driver, passed=all(c.status != "failed" for c in checks), checks=checks,
                   extensions=extensions)

    def counts(self) -> dict[str, int]:
        return {status: sum(c.status == status for c in self.checks) for status in Status.__args__}
