"""What always_on and prompt_agent share when they read a user deployed by deploy_user_sim_agent."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def person_of(us: dict) -> dict:
    """The user a deploy_user_sim_agent record describes, as the user sim's ``answer`` takes them; the caller adds the
    run's ``time_zone``, ``opening`` and ``done_ends_run``."""
    name = us.get("display_name") or us["name"]
    return {"key": us["name"], "name": name, "first": us.get("first_name") or name.split()[0],
            "email": us.get("email"), "time_zone": us.get("time_zone"),
            "description": us.get("description"), "versions": us.get("versions") or {},
            "output_format": us.get("output_format"),
            "permissions": us.get("permissions"), "keep_private": us.get("keep_private")}


def task_zone(us: dict, clocks: list[dict], where: str) -> str:
    """The user's time zone, else the one sync_env_clock gave the world; refused when the two disagree. ``where`` names
    the step in the error."""
    zones = sorted({c["timezone"] for c in clocks if c.get("timezone")})
    zone = us.get("time_zone")
    if zone and zones and zones != [zone]:
        raise ValueError(f"{where}: user sim '{us['name']}' is in {zone} but sync_env_clock set {zones}")
    if not zone and len(zones) > 1:
        raise ValueError(f"{where}: sync_env_clock set several time zones {zones}; the user sim needs one")
    return zone or (zones[0] if zones else "UTC")


def utc(text: str) -> datetime:
    at = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError(f"{text!r} has no UTC offset")
    return at.astimezone(timezone.utc)


def iso(value: Optional[datetime]) -> Optional[str]:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if value else None
