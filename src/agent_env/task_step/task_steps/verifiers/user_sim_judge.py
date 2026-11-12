"""The user_sim_judge step: after an always_on run, grade the user sim (the AI playing the user), not the assistant.

It reads the always_on step's record of the run: the user sim's instructions (rebuilt from its ``user_sim`` settings
the way the built-in plug-in wrote them), and every ``user_sim_read`` in the stored timeline at the record's
``object_url``: the exact check-in it got (``input``) and its answer. When the timeline wasn't stored it reads the
record's ``timeline_tail``, which keeps answers but not inputs, and says so to the judge. ``used`` is never shown to
the judge: it is the user sim's own pulse check, not evidence.

Each check (``user_sim_judge_core.CHECKS``) is one narrow question, asked of every model in ``judge_models`` in its
own call; a task's ``rubric_items`` ride along with the check that owns their family. Where the judges differ the
result is ``split``. A check in ``error_checks`` that fails in a way that could change the assistant's grade (every
judge agreeing) marks the run a user sim error; a split marks it only with ``error_on_split``.

What it writes:
- ``metadata["user_sim_judge"][verifier_id]``: every verdict with each judge's reason and cited turns, the task items,
  ``user_sim_error`` (``value`` yes / no / split, ``marked``, ``because``), token use and cost per model.
- ``metadata["always_on"][always_on_step_id]["user_sim_judge"]``: a short pointer to it with ``user_sim_error``, so a
  reader of the run's record finds it without knowing the verifier id.

It writes no ``verifications`` entry, so its verdicts never mix into the assistant's scores. Graders that should skip a
run the user sim spoiled must run after this step and read ``user_sim_error``.

A run that ended in an error (the always_on record's ``error_type``) is not judged: its prompt is skipped by every
verifier anyway.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers import user_sim_judge_core as core

logger = logging.getLogger(__name__)

JUDGE_KEY = "user_sim_judge"
ALWAYS_ON_KEY = "always_on"  # always_on.ALWAYS_ON_KEY; not imported, as that module imports the verifiers' peers
USER_SIMS_KEY = "user_sims"  # user_sim.USER_SIMS_KEY
DEFAULT_JUDGE_MODELS = ("anthropic/claude-opus-4-8", "openai/gpt-5.5")
FLAG_ASSISTANT = "the always_on step flagged this answer as looking like the assistant's (very long, or markdown)"
FLAG_LAST = "after the run's last wake: this answer never reached the assistant"


class UserSimJudgeTaskStep(TaskStep):
    type: ClassVar[str] = "user_sim_judge"
    entity_refs = ()
    DEFAULT_TIMEOUT_SECONDS: ClassVar[int] = 600
    DEFAULT_MAX_RETRIES: ClassVar[int] = 3

    def __init__(
        self,
        id: str,
        version: Optional[int],
        always_on_step_id: Optional[str] = None,
        judge_models: Optional[list[str]] = None,
        checks: Optional[list[str]] = None,
        error_checks: Optional[list[str]] = None,
        error_on_split: bool = False,
        rubric_items: Optional[list[dict]] = None,
        verifier_id: Optional[str] = None,
        judge_timeout_seconds: Optional[int] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.always_on_step_id = always_on_step_id
        self.judge_models = list(judge_models) if judge_models is not None else list(DEFAULT_JUDGE_MODELS)
        self.checks = list(checks) if checks is not None else list(core.CHECK_NAMES)
        self.error_checks = (list(error_checks) if error_checks is not None
                             else [c for c in core.DEFAULT_ERROR_CHECKS if c in self.checks])
        self.error_on_split = bool(error_on_split)
        self.rubric_items = list(rubric_items or [])
        self.verifier_id = verifier_id or uuid.uuid4().hex
        self.judge_timeout_seconds = judge_timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS

    # ---- settings ----------------------------------------------------------------------------------------------
    def config_problems(self) -> list[str]:
        return _problems(self.judge_models, self.checks, self.error_checks, self.rubric_items)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update(always_on_step_id=self.always_on_step_id, judge_models=self.judge_models, checks=self.checks,
                    error_checks=self.error_checks, error_on_split=self.error_on_split,
                    rubric_items=self.rubric_items, verifier_id=self.verifier_id,
                    judge_timeout_seconds=self.judge_timeout_seconds)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "UserSimJudgeTaskStep":
        return cls(**cls._base_from_dict(data), always_on_step_id=data.get("always_on_step_id"),
                   judge_models=data.get("judge_models"), checks=data.get("checks"),
                   error_checks=data.get("error_checks"), error_on_split=data.get("error_on_split", False),
                   rubric_items=data.get("rubric_items"), verifier_id=data.get("verifier_id"),
                   judge_timeout_seconds=data.get("judge_timeout_seconds"))

    def _settings(self, context: TaskStepContext) -> dict:
        """This run's settings: a run's ``step_params`` for this step override the stored ones."""
        over = self.step_param_overrides(context)
        models = list(over.get("judge_models", self.judge_models))
        checks = list(over.get("checks", self.checks))
        error_checks = list(over["error_checks"]) if "error_checks" in over else (
            [c for c in self.error_checks if c in checks] if "checks" in over else list(self.error_checks))
        items = list(over.get("rubric_items", self.rubric_items))
        problems = _problems(models, checks, error_checks, items)
        if problems:
            raise ValueError(f"user_sim_judge '{self.id}': " + "; ".join(problems))
        return {"models": models, "checks": checks, "error_checks": error_checks,
                "error_on_split": bool(over.get("error_on_split", self.error_on_split)),
                "items": [core.RubricItem.from_dict(i) for i in items]}

    # ---- the run -----------------------------------------------------------------------------------------------
    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        settings = self._settings(context)
        step_id, record = self._record(context)
        entry: dict[str, Any] = {"always_on_step_id": step_id, "user_sim": (record.get("user_sim") or {}).get("name"),
                                 "checks_run": settings["checks"], "error_checks": settings["error_checks"],
                                 "rubric_items": [i.to_dict() for i in settings["items"]]}
        if record.get("error_type"):
            entry.update(skipped=f"the run ended with error_type={record['error_type']}",
                         user_sim_error={"value": "no", "marked": False, "because": [], "split_on": []})
            logger.warning(f"user_sim_judge '{self.id}': skipped, {entry['skipped']}")
        else:
            run = await asyncio.to_thread(run_from_record, record, context.metadata)
            entry.update(await core.judge_run(
                run, models=settings["models"], checks=settings["checks"], items=settings["items"],
                error_checks=settings["error_checks"], error_on_split=settings["error_on_split"],
                complete=self._complete(context)))
        context.metadata.setdefault(JUDGE_KEY, {})[self.verifier_id] = entry
        error = entry["user_sim_error"]
        record[JUDGE_KEY] = {"verifier_id": self.verifier_id, "user_sim_error": error["value"],
                             "marked": error["marked"], "because": error["because"]}
        logger.info(f"user_sim_judge '{self.id}': user_sim_error={error['value']} "
                    f"({', '.join(error['because'] + error['split_on']) or 'no check'}); "
                    + ", ".join(f"{k}={v['verdict']}" for k, v in (entry.get("checks") or {}).items()))
        return context

    def _record(self, context: TaskStepContext) -> tuple[str, dict]:
        records = context.metadata.get(ALWAYS_ON_KEY) or {}
        if self.always_on_step_id:
            if self.always_on_step_id not in records:
                raise RuntimeError(f"user_sim_judge '{self.id}': no always_on record for step "
                                   f"'{self.always_on_step_id}' (have {sorted(records)})")
            return self.always_on_step_id, records[self.always_on_step_id]
        if len(records) != 1:
            raise RuntimeError(f"user_sim_judge '{self.id}': set always_on_step_id; this run has "
                               f"{len(records)} always_on records ({sorted(records)})")
        return next(iter(records.items()))

    def _complete(self, context: TaskStepContext) -> core.Complete:
        """One judge call through LiteLLM, with the key and base the run's overrides name (as rubrics_verifier's
        judge), the run's two shared parts marked for Anthropic's prompt cache."""
        overrides = context.metadata.get("user_overrides", {})
        timeout, retries = self.judge_timeout_seconds, self.DEFAULT_MAX_RETRIES

        async def complete(model: str, system: str, parts: list[str], schema: dict) -> dict:
            import litellm

            from agent_env.config import get_config
            from agent_env.config.model import ModelParam

            config = get_config()
            call = config.resolve_model_call(
                model,
                default_api_key=config.resolve_reference(overrides.get("judge_litellm_api_key")
                                                         or overrides.get("litellm_api_key")),
                base_override=overrides.get("judge_litellm_base_url") or overrides.get("litellm_base_url"))
            cache = _anthropic(model)
            content = [{"type": "text", "text": p, **({"cache_control": {"type": "ephemeral"}} if cache and i < 2
                                                      else {})}
                       for i, p in enumerate(parts) if p]
            kwargs = dict(model=model, messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": content}],
                          timeout=timeout, **call.client_kwargs())
            kwargs[ModelParam.RESPONSE_FORMAT] = {"type": "json_schema", "json_schema": {
                "name": "user_sim_check", "schema": schema, "strict": True}}
            last: Optional[Exception] = None
            for attempt in range(1, retries + 1):
                try:
                    response = await litellm.acompletion(**kwargs)
                    usage = getattr(response, "usage", None)
                    hidden = getattr(response, "_hidden_params", None) or {}
                    return {"text": response.choices[0].message.content or "",
                            "usage": usage.model_dump() if hasattr(usage, "model_dump") else dict(usage or {}),
                            "cost": hidden.get("response_cost")}
                except Exception as e:
                    last = e
                    logger.warning(f"user_sim_judge: {model} call {attempt}/{retries} failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"{model} failed after {retries} attempts") from last

        return complete


def _anthropic(model: str) -> bool:
    m = model.lower()
    return m.startswith(("anthropic/", "bedrock/", "vertex_ai/")) and "claude" in m or m.startswith("claude")


def _problems(models: Any, checks: Any, error_checks: Any, items: Any) -> list[str]:
    out = []
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m.strip() for m in models):
        out.append("judge_models must be a non-empty list of model names")
    elif len(set(models)) != len(models):
        out.append("judge_models names a model twice")
    if not isinstance(checks, list) or not all(isinstance(c, str) for c in checks):
        out.append(f"checks must be a list from {list(core.CHECK_NAMES)}")
    else:
        unknown = sorted(set(checks) - set(core.CHECK_NAMES))
        if unknown:
            out.append(f"checks has unknown checks {unknown}; the checks are {list(core.CHECK_NAMES)}")
        if not checks and not items:
            out.append("checks is empty and there are no rubric_items: nothing to judge")
    if not isinstance(error_checks, list) or not all(isinstance(c, str) for c in error_checks):
        out.append("error_checks must be a list of check names")
    elif isinstance(checks, list):
        not_run = sorted(set(error_checks) - set(checks))
        if not_run:
            out.append(f"error_checks names checks that don't run: {not_run}")
    if not isinstance(items, list):
        out.append("rubric_items must be a list of {id, family, text}")
    else:
        out += core.item_problems(items)
    return out


# ---- the run's record -----------------------------------------------------------------------------------------------
def run_from_record(record: dict, metadata: dict) -> core.UserSimRun:
    """The user sim's run, from an always_on record and the run's metadata (for its ``user_sim`` settings)."""
    notes: list[str] = []
    timeline = None
    if record.get("object_url"):
        try:
            from agent_env.config import get_config

            url = record["object_url"]
            timeline = json.loads(get_config().get_object_store_at(url).get(url))["timeline"]
        except Exception as e:
            notes.append(f"the run's stored record could not be read ({type(e).__name__}); only the last events, "
                         "without what the user sim was sent, are shown")
    if timeline is None:
        timeline = record.get("timeline_tail") or []
        if not record.get("object_url"):
            notes.append("the run's full record wasn't stored; only its last events, without what the user sim was "
                         "sent, are shown")
    reads = [e for e in timeline if e.get("event") == "user_sim_read"]
    name = (record.get("user_sim") or {}).get("name")
    settings = (metadata.get(USER_SIMS_KEY) or {}).get(name) or {}
    instructions, more = _instructions(settings, record, reads)
    notes += more
    turns = []
    for n, e in enumerate(reads, 1):
        flags = []
        if e.get("looks_like_assistant"):
            flags.append(FLAG_ASSISTANT)
        if e.get("schema_errors"):
            flags.append("its answer was malformed, so nothing was sent: " + "; ".join(e["schema_errors"])[:300])
        elif e.get("retried"):
            flags.append("its first answer was malformed and it answered again")
        if e.get("user_sim_error"):
            flags.append(f"the user sim agent failed at this check-in: {str(e['user_sim_error'])[:200]}")
        if e.get("last_read"):
            flags.append(FLAG_LAST)
        turns.append(core.Turn(n=n, outcome=e.get("outcome") or "no_reply", message=e.get("message") or "",
                               input=e.get("input"), flags=flags,
                               time=None if e.get("input") else f"{e.get('virtual_time')} UTC"))
    return core.UserSimRun(name=name or "the user", instructions=instructions, turns=turns, notes=notes)


def _instructions(settings: dict, record: dict, reads: list[dict]) -> tuple[Optional[str], list[str]]:
    """The user sim's system prompt, as the built-in plug-in wrote it for this run, and notes on what may differ; for a
    plug-in of the task's own, its settings, which is all the run keeps."""
    if not settings:
        return None, ["the user sim's settings were not found, so its instructions aren't shown"]
    version = (record.get("user_sim") or {}).get("version") or settings.get("user_version")
    if settings.get("plugin") or version not in (None, "ours"):
        shown = {k: settings.get(k) for k in ("description", "permissions", "keep_private") if settings.get(k)}
        return (json.dumps(shown, indent=1, ensure_ascii=False),
                ["the user sim is a plug-in of the task's own; its instructions are shown as its settings"])
    from agent_env.task_step.task_steps.agent import user_sim_versions

    name = settings.get("display_name") or settings.get("name") or "the user"
    first_read = (reads[0].get("read") or {}) if reads else {}
    opening = next(iter(first_read.get("you_sent") or []), None)
    person = {"key": settings.get("name"), "name": name, "first": settings.get("first_name") or name.split()[0],
              "email": settings.get("email"), "time_zone": record.get("time_zone") or settings.get("time_zone"),
              "description": settings.get("description"), "versions": settings.get("versions") or {},
              "output_format": settings.get("output_format"), "permissions": settings.get("permissions"),
              "keep_private": settings.get("keep_private"), "opening": opening,
              "done_ends_run": not record.get("multi_day")}
    try:
        prompt = user_sim_versions.system_prompt(person)
    except Exception as e:
        return None, [f"the user sim's instructions could not be rebuilt ({type(e).__name__}: {e})"]
    notes = []
    if not opening:
        notes.append("the opening request wasn't recorded, so it is missing from these instructions")
    chars = (record.get("user_sim") or {}).get("system_prompt_chars")
    if chars and chars != len(prompt):
        notes.append(f"these instructions were rebuilt from the user sim's settings ({len(prompt)} characters; the "
                     f"run's were {chars}), so they may differ slightly from what it got")
    return prompt, notes
