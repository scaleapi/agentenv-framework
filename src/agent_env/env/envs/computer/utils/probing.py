"""Gate 3: a computer env's build booted on a provider that runs it natively and probed at ``full``, by its compose or,
for a build its compose couldn't boot, on its first deploy."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import httpx
from agentenv_protocol.computer.conformance import run_probe

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.config.errors import ConfigError
from agent_env.env.env import Env
from agent_env.env.envs.computer.utils import boot
from agent_env.env.envs.computer.utils.composed import Record, failed, now
from agent_env.env.envs.computer.utils.install import for_os, platform_of
from agent_env.store.ids import derive_id
from agent_env.store.local_state import awaiting_lock

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agentenv_protocol.computer.conformance import ProbeReport

    from agent_env.env.envs.computer.driver_env import ComputerDriverEnv
    from agent_env.env.envs.computer.env import ComputerEnv
    from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

#: How long a probe's computer may run before its sandbox provider stops it.
_PROBE_TTL_S = 1800
#: How long the probe at full may take on a computer that booted: about ten seconds on a healthy one.
_PROBE_TIMEOUT_S = 120
#: How many probes, each on a fresh boot, a build gets when each ends with the connection to its driver dropped, refused
#: or timed out, which says nothing of the build, before the last one's result is recorded.
_PROBE_ATTEMPTS = 3


async def probe_first_boot(env: ComputerEnv, form: str, arch: str, sandbox_provider: SandboxProvider) -> dict:
    """Gate 3 for a build its compose couldn't probe, on its first boot on ``sandbox_provider``, which boots it
    natively: the result, recorded. Deploys of one build at once, in this process or another, probe it once: each
    waits for the one probing, then takes the result it recorded, a failure's too, which keeps the build's key. A record
    another compose has taken over since, as an agent-env that composes differently does, stays that compose's: the
    build this deploy picked is probed for it alone."""
    build = env.metadata["composed"][form][arch]
    async with awaiting_lock(f"computer-first-boot:{env.stored_id}:{form}:{arch}:{build['image']}"):
        recorded = (await asyncio.to_thread(Env.get, env.stored_id, env.version)).metadata["composed"][form][arch]
        if (recorded.get("key"), recorded.get("composer")) != (build["key"], build.get("composer")):
            return await checked([sandbox_provider], env, env.driver, form, arch, build)
        if recorded["status"] != "unverified":
            return recorded
        result = await checked([sandbox_provider], env, env.driver, form, arch, recorded)
        await asyncio.to_thread(Record(env).finished, form, arch, result)
        return result


class Unprobed(Exception):
    """The probe couldn't boot the build, for a setting it boots with that doesn't resolve here: which says nothing of
    the build."""


async def checked(links: list[SandboxProvider], env: ComputerEnv, driver: ComputerDriverEnv, form: str, arch: str,
                  result: dict) -> dict:
    """The build of ``result`` booted on the first of ``links`` that boots it and probed at ``full``: ready, with the
    driver's card, or failed at the probe; each with a summary of the probe and its report stored. A probe that ended
    on a transport error talking to the driver runs again on a fresh boot first, up to ``_PROBE_ATTEMPTS`` in all, the
    record then keeping each attempt; one with a failed check is recorded at once. A build the probe couldn't boot for
    a setting that doesn't resolve here stays as it was, unverified, saying why, so a compose once it resolves probes
    it."""
    attempts: list[dict] = []
    while True:
        try:
            report, card, provider = await _probe(links, env, driver, result["image"], form)
        except Unprobed as e:
            return {**result, "reason": f"not probed: {e}"}
        except TimeoutError:
            outcome = failed(env, form, arch, "probe", result["key"],
                             f"the probe didn't finish within {_PROBE_TIMEOUT_S} s")
        except Exception as e:
            if not _dropped(e):
                outcome = failed(env, form, arch, "probe", result["key"], f"it didn't boot: {e}")
            else:
                attempts.append({"status": "dropped", "reason": _transport_errors(e), "dropped_at": now()})
                if len(attempts) < _PROBE_ATTEMPTS:
                    continue
                return {**failed(env, form, arch, "probe", result["key"],
                                 f"each of its {_PROBE_ATTEMPTS} probes, each on a fresh boot, ended on a transport "
                                 f"error talking to its driver, the last {attempts[-1]['reason']}"),
                        "attempts": attempts}
        else:
            outcome = _reported(env, driver, form, arch, result, report, card, provider)
        return _attempted(attempts, outcome)


def _reported(env: ComputerEnv, driver: ComputerDriverEnv, form: str, arch: str, result: dict, report: ProbeReport,
              card: dict, provider: str) -> dict:
    """``result`` probed: ready with the driver's card, or failed at the probe, with a summary of the probe and its
    report stored. A failed check the driver declares it's known to fail on the base's OS is recorded in the summary,
    and doesn't fail the build; one it declares that didn't fail is recorded and warned of, so a waiver doesn't outlive
    its bug."""
    stored = FileArtifact.put_bytes(derive_id(env.stored_id, f"probe_report_{form}_{arch}"), description=(
        f"the {form}/{arch} probe of {env.stored_id} v{env.version}"), filename="probe.json",
        content=report.model_dump_json(indent=1).encode(), content_type="application/json")
    probe = {"level": report.level, "provider": provider, "passed": report.passed, "checks": report.counts(),
             "extensions": report.extensions, "report": {"id": stored.id, "version": stored.version},
             "probed_at": now()}
    known = {(failure.check, failure.name): failure
             for failure in for_os(driver.known_failures, env.computer.os) or []}
    failed_checks = [c for c in report.checks if c.status == "failed"]
    if waived := [c for c in failed_checks if (c.check, c.name) in known]:
        probe["known_failures"] = [{"check": c.check, "name": c.name, "detail": c.detail,
                                    "reason": known[c.check, c.name].reason} for c in waived]
    seen = {(c.check, c.name): c.status for c in report.checks}
    if unused := [{"check": check, "name": name, "status": seen.get((check, name), "absent")}
                  for check, name in known if seen.get((check, name)) != "failed"]:
        probe["unused_known_failures"] = unused
        for failure in unused:
            saw = ("has no such check: correct its name" if failure["status"] == "absent" else
                   f"found it {failure['status'].replace('_', ' ')}: once what it declares is fixed")
            logger.warning(f"{env.stored_id}: {driver.id} declares {failure['check']} {failure['name']!r} a known "
                           f"failure on {env.computer.os.type}, and its {form}/{arch} probe {saw}, take it out of the "
                           "driver's known_failures")
    if failing := [c for c in failed_checks if (c.check, c.name) not in known]:
        return {**result, "status": "failed", "stage": "probe", "failed_at": now(), "probe": probe,
                "reason": "the probe failed: " + "; ".join(f"{c.check} {c.name}: {c.detail}" for c in failing)}
    return {**result, "status": "ready", "card": card, "probe": probe}


def _dropped(error: BaseException) -> bool:
    """Whether ``error`` is only the connection to the driver dropping, refused or timing out, as httpx raises it, alone
    or from a task group."""
    if isinstance(error, BaseExceptionGroup):
        return all(_dropped(e) for e in error.exceptions)
    return isinstance(error, httpx.TransportError)


def _transport_errors(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(dict.fromkeys(_transport_errors(e) for e in error.exceptions))
    return f"{type(error).__name__}: {error}"


def _attempted(attempts: list[dict], result: dict) -> dict:
    """``result``, with the attempts before it and itself when it was rerun."""
    if not attempts:
        return result
    last = {key: result[key] for key in ("status", "reason", "failed_at", "probe") if key in result}
    return {**result, "attempts": [*attempts, last]}


async def _probe(links: list[SandboxProvider], env: ComputerEnv, driver: ComputerDriverEnv, image: str,
                 form: str = "container") -> tuple[ProbeReport, dict, str]:
    """``image``, of ``form``, booted at the env's settings and its apps' defaults with a control token of its own, as
    a deploy with no ``env_params`` boots it, on the first of ``links`` that boots it, probed at ``full`` with that
    token within ``_PROBE_TIMEOUT_S`` and torn down; the report, the driver's card and the provider's name. Raises
    Unprobed, booting nothing, when a setting it boots with doesn't resolve here."""
    token = boot.control_token()
    try:
        environment = {
            **boot.settings_environment(env.computer, env.computer_values(), environment_name=env.computer_name),
            **boot.app_settings_environment(env.composed_apps(), env.app_defaults(), platform_of(env.computer.os))}
    except ConfigError as e:  # an env: or secret: reference that doesn't resolve here, or no secret store to resolve it
        raise Unprobed(f"a setting it boots with didn't resolve here: {e}") from e
    problems = []
    for link in links:
        try:
            booted = await boot.boot(link, image=image, port=driver.port, environment=environment, token=token,
                                     network_policy=None, timeout=_PROBE_TTL_S, attribution=None,
                                     ready_s=boot.ready_timeout(env.composed_apps(), platform_of(env.computer.os)),
                                     form=form, platform=platform_of(env.computer.os), shm_mb=env.shm_mb())
        except Exception as e:
            problems.append(f"{type(link).__name__}: {e}")
            continue
        try:
            async with asyncio.timeout(_PROBE_TIMEOUT_S):
                return await run_probe(booted.url, "full", token=token), booted.card, type(link).__name__
        finally:
            await boot.discard(booted.sandbox)
    raise RuntimeError("; ".join(problems))
