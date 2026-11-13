"""What always_on and prompt_agent share when they read a user deployed by deploy_user_sim_agent."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Optional

# Where a run's rating by its user sim (user_sim_versions.rate) is recorded: metadata[SATISFACTION_KEY][step_id].
SATISFACTION_KEY = "user_sim_satisfaction"


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


# ---- the opening: the user's first message ---------------------------------------------------------------------------
# ``opening_prompt`` is text, or a list of A2A text and file parts (prompt_agent's ``parts``). A file part may carry a
# one-line ``metadata.description`` for the user sim, which can't open files; the agent is sent the file without it.
OPENING_KINDS = ("text", "file")
DESCRIPTION = "description"


def opening_parts(opening: Any) -> list[dict]:
    """``opening_prompt`` as parts, a copy: text is one text part."""
    if isinstance(opening, str):
        return [{"kind": "text", "text": opening}]
    return copy.deepcopy(list(opening or []))


def recorded_opening(us: dict) -> list[dict]:
    """The opening a deploy_user_sim_agent record holds, as parts, its seed not yet filled in."""
    return opening_parts(us["opening_parts"] if us.get("opening_parts") is not None else us.get("opening") or "")


def opening_text(parts: list[dict]) -> str:
    """The opening's text parts joined, as the user's one line of text."""
    return "\n".join(p.get("text", "") for p in parts if p.get("kind") == "text")


def file_name(part: dict) -> str:
    """What the user calls a file part: its name, else the last segment of its address."""
    file = part.get("file") or {}
    return file.get("name") or (file.get("uri") or "").rstrip("/").rsplit("/", 1)[-1] or "a file"


def attached(parts: list[dict]) -> list[dict]:
    """The files the opening attaches, as the user sim is told of them: ``{"name", "description"}``, the description
    only when the part has one."""
    out = []
    for part in parts:
        if part.get("kind") == "file":
            description = (part.get("metadata") or {}).get(DESCRIPTION)
            out.append({"name": file_name(part), **({"description": description} if description else {})})
    return out


def for_agent(part: dict) -> dict:
    """A part as the agent is sent it: without the description, which is for the user sim."""
    metadata = part.get("metadata")
    if not isinstance(metadata, dict) or DESCRIPTION not in metadata:
        return part
    rest = {k: v for k, v in metadata.items() if k != DESCRIPTION}
    return {**{k: v for k, v in part.items() if k != "metadata"}, **({"metadata": rest} if rest else {})}


def opening_problems(opening: Any) -> list[str]:
    """What's wrong with ``opening_prompt``, one message each; empty when it's fine."""
    required = "opening_prompt is required: the user's first message, which always_on starts the run with"
    if opening is None or isinstance(opening, str):
        return [] if isinstance(opening, str) and opening.strip() else [required]
    if not isinstance(opening, list):
        return ["opening_prompt must be text, or a list of text and file parts"]
    from a2a.types import FilePart, TextPart
    from pydantic import ValidationError

    out = []
    for i, part in enumerate(opening):
        where = f"opening_prompt[{i}]"
        if not isinstance(part, dict):
            out.append(f"{where} must be a part, an object")
            continue
        if part.get("kind") not in OPENING_KINDS:
            out.append(f"{where} is a {part.get('kind')!r} part; the user's first message takes text and file parts")
            continue
        try:
            (TextPart if part["kind"] == "text" else FilePart).model_validate(part)
        except ValidationError as e:
            why = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            out.append(f"{where} is not a valid A2A {part['kind']} part ({why})")
            continue
        metadata = part.get("metadata")
        if part["kind"] == "file" and isinstance(metadata, dict) and DESCRIPTION in metadata:
            description = metadata[DESCRIPTION]
            if not isinstance(description, str) or not description.strip() or "\n" in description.strip():
                out.append(f"{where}'s metadata.description must be one line of text: what the user sim is told "
                           "the file is")
    if not opening_text([p for p in opening if isinstance(p, dict)]).strip():
        out.append("opening_prompt needs a text part with text: the user's first message, which always_on starts "
                   "the run with")
    return out
