"""The user_sim_judge step: after a run with the built-in user sim, grade the user sim (the AI playing the user), not
the assistant. A task writer adds it after the run's step; nothing turns it on by default.

It judges either kind of run the built-in user sim plays (``deploy_user_sim_agent``):

- an ``always_on`` run: every ``user_sim_read`` in the timeline stored at ``metadata["always_on"][step]["object_url"]``
  (else the record's ``timeline_tail``, which keeps answers but not what the user sim was sent);
- a ``prompt_agent`` chat run: each short entry in ``metadata["user_sim_reads"][step]``, read whole from its
  ``object_url`` (else the entry, which keeps the answer only).

``run_step_id`` names that step; unset, the run must have exactly one of them. The user sim's instructions are rebuilt
from its ``metadata["user_sims"]`` record with ``user_sim_versions.setup``, the way ``deploy_user_sim_agent`` wrote
them, and checked against the record's ``prompt_sha256``. ``used`` is never shown to the judge: it is the user sim's
own pulse check, not evidence.

Each check (``user_sim_judge_core.CHECKS``) is one narrow question to ``judge_model``, in its own call; a task's
``rubric_items`` ride along with the check that owns their family.

What it writes: ``metadata["user_sim_judge"][verifier_id]``, with every verdict, its reason and cited turns, the task
items, ``user_sim_error`` (``value`` yes / no / unknown, ``because``, ``marked``), the instructions check, token use
and cost. It writes no ``verifications`` entry, so its verdicts never mix into the assistant's scores.

Record only, by default: with ``mark_user_sim_error`` off it changes nothing a grader reads. Turned on, a run whose
``user_sim_error`` is yes, judged against instructions that match the recorded hash, has its prompt response marked
``error_type = "user_sim_error"``, as a user sim that fails mid-run marks it, so graders that run after this step skip
it. A failure of this step doesn't fail the task unless ``fail_task_on_error`` says so.

A run that ended in an error (the always_on record's or the prompt response's ``error_type``) is not judged: every
verifier skips its prompt anyway.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from typing import Any, ClassVar, Optional

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_step import TaskStep, TaskStepDependency
from agent_env.task_step.task_steps.verifiers import user_sim_judge_core as core

logger = logging.getLogger(__name__)

JUDGE_KEY = "user_sim_judge"
# Not imported: always_on and prompt_agent import the verifiers' peers. Each is the key its module names.
ALWAYS_ON_KEY = "always_on"  # always_on.ALWAYS_ON_KEY
READS_KEY = "user_sim_reads"  # prompt_agent's _BuiltInUserSim
USER_SIMS_KEY = "user_sims"  # deploy_user_sim_agent.USER_SIMS_KEY
ALWAYS_ON, CHAT = "always_on", "prompt_agent"
RUN_STEP_TYPES = (ALWAYS_ON, CHAT)
FALLBACK_JUDGE_MODEL = "claude-sonnet-4-6"  # rubrics_verifier's, when the config names no judge
USER_SIM_ERROR = "user_sim_error"
FLAG_ASSISTANT = "this answer looks like the assistant's (very long, or markdown)"
FLAG_LAST = "after the run's last wake: this answer never reached the assistant"
# always_on.ASSISTANT_LIKE_MESSAGE_CHARS: a chat run's reads carry no flag of their own, so it's computed the same way.
ASSISTANT_LIKE_MESSAGE_CHARS = 1200


def default_judge_model() -> str:
    """The model rubrics_verifier defaults to: the config's ``judge`` role, else claude-sonnet-4-6."""
    from agent_env.config import get_config

    return get_config().get_model_for_role("judge") or FALLBACK_JUDGE_MODEL


class UserSimJudgeTaskStep(TaskStep):
    type: ClassVar[str] = "user_sim_judge"
    entity_refs = ()
    DEFAULT_TIMEOUT_SECONDS: ClassVar[int] = 600
    DEFAULT_MAX_RETRIES: ClassVar[int] = 3

    def __init__(
        self,
        id: str,
        version: Optional[int],
        run_step_id: Optional[str] = None,
        judge_model: Optional[str] = None,
        checks: Optional[list[str]] = None,
        error_checks: Optional[list[str]] = None,
        rubric_items: Optional[list[dict]] = None,
        mark_user_sim_error: bool = False,
        verifier_id: Optional[str] = None,
        judge_timeout_seconds: Optional[int] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = False,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.run_step_id = run_step_id
        self.judge_model = judge_model  # None: default_judge_model() when the step runs
        self.checks = list(checks) if checks is not None else list(core.CHECK_NAMES)
        self.error_checks = (list(error_checks) if error_checks is not None
                             else [c for c in core.DEFAULT_ERROR_CHECKS if c in self.checks])
        self.rubric_items = list(rubric_items or [])
        self.mark_user_sim_error = mark_user_sim_error
        self.verifier_id = verifier_id or uuid.uuid4().hex
        self.judge_timeout_seconds = judge_timeout_seconds or self.DEFAULT_TIMEOUT_SECONDS

    # ---- settings ----------------------------------------------------------------------------------------------
    def config_problems(self) -> list[str]:
        return _problems(self.judge_model, self.checks, self.error_checks, self.rubric_items, self.mark_user_sim_error)

    def task_problems(self, steps: list[TaskStep]) -> list[str]:
        """The run it judges must be an always_on or prompt_agent step's; unset, the task must have exactly one."""
        runs = [s for s in steps if s.type in RUN_STEP_TYPES]
        if self.run_step_id is not None:
            named = next((s for s in steps if s.id == self.run_step_id), None)
            if named is not None and named.type not in RUN_STEP_TYPES:
                return [f"step '{self.id}' ({self.type}) names the user sim of step '{named.id}' ({named.type}); "
                        "it judges an always_on or prompt_agent step's"]
            return []
        if len(runs) > 1:
            return [f"step '{self.id}' ({self.type}): set run_step_id; this task has {len(runs)} steps it could judge "
                    f"({sorted(s.id for s in runs)})"]
        return []

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update(run_step_id=self.run_step_id, judge_model=self.judge_model, checks=self.checks,
                    error_checks=self.error_checks, rubric_items=self.rubric_items,
                    mark_user_sim_error=self.mark_user_sim_error, verifier_id=self.verifier_id,
                    judge_timeout_seconds=self.judge_timeout_seconds)
        return base

    @classmethod
    def from_dict(cls, data: dict) -> "UserSimJudgeTaskStep":
        base = {**cls._base_from_dict(data), "fail_task_on_error": data.get("fail_task_on_error", False)}
        return cls(**base, run_step_id=data.get("run_step_id"), judge_model=data.get("judge_model"),
                   checks=data.get("checks"), error_checks=data.get("error_checks"),
                   rubric_items=data.get("rubric_items"), mark_user_sim_error=data.get("mark_user_sim_error", False),
                   verifier_id=data.get("verifier_id"), judge_timeout_seconds=data.get("judge_timeout_seconds"))

    def _settings(self, context: TaskStepContext) -> dict:
        """This run's settings: a run's ``step_params`` for this step override the stored ones."""
        over = self.step_param_overrides(context)
        model = over.get("judge_model", self.judge_model)
        checks = list(over.get("checks", self.checks))
        error_checks = list(over["error_checks"]) if "error_checks" in over else (
            [c for c in self.error_checks if c in checks] if "checks" in over else list(self.error_checks))
        items = list(over.get("rubric_items", self.rubric_items))
        mark = over.get("mark_user_sim_error", self.mark_user_sim_error)
        problems = _problems(model, checks, error_checks, items, mark)
        if problems:
            raise ValueError(f"user_sim_judge '{self.id}': " + "; ".join(problems))
        return {"model": model or default_judge_model(), "checks": checks, "error_checks": error_checks,
                "mark": mark, "items": [core.RubricItem.from_dict(i) for i in items]}

    # ---- the run -----------------------------------------------------------------------------------------------
    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        settings = self._settings(context)
        kind, step_id = self._find_run(context)
        response = _prompt_response(context, step_id)
        entry: dict[str, Any] = {"run_step_id": step_id, "run_kind": kind, "judge_model": settings["model"],
                                 "checks_run": settings["checks"], "error_checks": settings["error_checks"],
                                 "rubric_items": [i.to_dict() for i in settings["items"]],
                                 "mark_user_sim_error": settings["mark"]}
        record = (context.metadata.get(ALWAYS_ON_KEY) or {}).get(step_id) if kind == ALWAYS_ON else None
        error_type = (record or {}).get("error_type") or (response.error_type if response else None)
        if error_type:
            entry.update(skipped=f"the run ended with error_type={error_type}")
        elif kind == CHAT and not (context.metadata.get(READS_KEY) or {}).get(step_id):
            entry.update(skipped="the user sim never read: the run has no user_sim_reads")
        else:
            run, about = await asyncio.to_thread(_read_run, kind, step_id, record, context)
            entry["user_sim"], entry["instructions"] = about["user_sim"], about["instructions"]
            context.record_model("user sim judge", self.id, settings["model"])
            result = await core.judge_run(
                run, model=settings["model"], checks=settings["checks"], items=settings["items"],
                error_checks=settings["error_checks"], complete=self._complete(context))
            result.pop("model", None)  # the entry's judge_model
            entry.update(result)
        if "skipped" in entry:
            logger.warning(f"user_sim_judge '{self.id}': skipped, {entry['skipped']}")
            entry["user_sim_error"] = dict(core.NOT_JUDGED)
        entry["user_sim_error"].update(self._mark(entry, settings["mark"], response))
        context.metadata.setdefault(JUDGE_KEY, {})[self.verifier_id] = entry
        error = entry["user_sim_error"]
        logger.info(f"user_sim_judge '{self.id}': {kind} '{step_id}' user_sim_error={error['value']} "
                    f"({', '.join(error['because'] + error['unknown']) or 'no check'}), marked={error['marked']}; "
                    + ", ".join(f"{k}={v['verdict']}" for k, v in (entry.get("checks") or {}).items()))
        return context

    def _find_run(self, context: TaskStepContext) -> tuple[str, str]:
        """The kind of run and its step id: ``run_step_id``'s, else the run's only always_on or chat run."""
        records = context.metadata.get(ALWAYS_ON_KEY) or {}
        reads = context.metadata.get(READS_KEY) or {}
        if self.run_step_id:
            if self.run_step_id in records:
                return ALWAYS_ON, self.run_step_id
            if self.run_step_id in reads or _prompt_response(context, self.run_step_id) is not None:
                return CHAT, self.run_step_id  # a chat run whose user sim never read leaves a response, no reads
            raise RuntimeError(f"user_sim_judge '{self.id}': step '{self.run_step_id}' left no run to judge (always_on "
                               f"records: {sorted(records)}; prompt_agent user sim reads: {sorted(reads)})")
        runs = [(ALWAYS_ON, s) for s in records] + [(CHAT, s) for s in reads if s not in records]
        if len(runs) != 1:
            raise RuntimeError(f"user_sim_judge '{self.id}': set run_step_id; this run has {len(runs)} runs with a "
                               f"built-in user sim ({sorted(s for _, s in runs)})")
        return runs[0]

    def _mark(self, entry: dict, mark: bool, response: Optional[PromptResponse]) -> dict:
        """With ``mark`` on, a user sim error marks the run's prompt response, so graders after this step skip it."""
        error = entry["user_sim_error"]
        if error["value"] != "yes":
            return {"marked": False}
        if not mark:
            return {"marked": False, "not_marked": "mark_user_sim_error is off: verdicts are recorded only"}
        if not (entry.get("instructions") or {}).get("matches"):
            return {"marked": False, "not_marked": "the judge's copy of the user sim's instructions isn't confirmed "
                                                   "to be the one it ran with"}
        if response is None:
            return {"marked": False, "not_marked": f"step '{entry['run_step_id']}' left no prompt response"}
        response.error_type = USER_SIM_ERROR
        response.error_message = response.error_message or (
            "The user sim judge found a user sim error: " + ", ".join(error["because"]))
        return {"marked": True}

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


def _problems(model: Any, checks: Any, error_checks: Any, items: Any, mark: Any) -> list[str]:
    out = []
    if model is not None and (not isinstance(model, str) or not model.strip()):
        out.append("judge_model must be a model name")
    if not isinstance(mark, bool):
        out.append("mark_user_sim_error must be true or false")
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


def _prompt_response(context: TaskStepContext, step_id: str) -> Optional[PromptResponse]:
    return next((p for p in reversed(context.prompt_responses) if p.step_id == step_id), None)


# ---- the run as the judge reads it ----------------------------------------------------------------------------------
def _read_run(kind: str, step_id: str, record: Optional[dict], context: TaskStepContext) -> tuple[core.UserSimRun, dict]:
    """The user sim's run, and what the entry records about it: ``user_sim`` (its name) and ``instructions``."""
    metadata = context.metadata
    if kind == ALWAYS_ON:
        reads, notes = _always_on_reads(record)
        name = (record.get("user_sim") or {}).get("name")
        multi_day = record.get("multi_day")
    else:
        reads, notes = _chat_reads(metadata.get(READS_KEY, {}).get(step_id) or [])
        name = _chat_user_sim(metadata, step_id)
        multi_day = False
    us = (metadata.get(USER_SIMS_KEY) or {}).get(name) if name else None
    instructions, check, more = _instructions(us, metadata.get("seed") or {}, multi_day)
    turns = [_turn(n, e) for n, e in enumerate(reads, 1)]
    run = core.UserSimRun(name=name or "the user", instructions=instructions, turns=turns, notes=notes + more)
    return run, {"user_sim": name, "instructions": check}


def _load(url: str) -> dict:
    from agent_env.config import get_config

    return json.loads(get_config().get_object_store_at(url).get(url))


def _always_on_reads(record: dict) -> tuple[list[dict], list[str]]:
    """Every ``user_sim_read`` in the always_on step's stored timeline, else its tail."""
    if record.get("object_url"):
        try:
            timeline = _load(record["object_url"])["timeline"]
            return [e for e in timeline if e.get("event") == "user_sim_read"], []
        except Exception as e:
            note = (f"the run's stored record could not be read ({type(e).__name__}); only its last events, without "
                    "what the user sim was sent, are shown")
    else:
        note = "the run's full record wasn't stored; only its last events, without what the user sim was sent, are shown"
    reads = [e for e in record.get("timeline_tail") or [] if e.get("event") == "user_sim_read"]
    return reads, [note]


def _chat_reads(summaries: list[dict]) -> tuple[list[dict], list[str]]:
    """Each read of a chat run, whole from its ``object_url``, else its summary, which keeps only the answer."""
    reads, missing = [], 0
    for summary in summaries:
        full = None
        if summary.get("object_url"):
            try:
                full = _load(summary["object_url"])
            except Exception as e:
                logger.warning(f"user_sim_judge: could not read {summary['object_url']}: {type(e).__name__}: {e}")
        if full is None:
            missing += 1
        read = {**summary, **(full or {})}
        message = read.get("message") or ""
        read["looks_like_assistant"] = (len(message) > ASSISTANT_LIKE_MESSAGE_CHARS
                                        or bool(re.search(r"^\s*(#|\|)", message, re.M)))
        reads.append(read)
    notes = [f"{missing} of the run's {len(summaries)} check-ins weren't stored whole, so what the user sim was sent "
             "at them isn't shown"] if missing else []
    return reads, notes


def _chat_user_sim(metadata: dict, step_id: str) -> Optional[str]:
    """Which user sim played a chat run: its conversation's user, else the run's only user sim."""
    users = metadata.get(USER_SIMS_KEY) or {}
    conversation_id = (metadata.get("a2a_conversations") or {}).get(step_id)
    if conversation_id:
        try:
            from agent_env.a2a_agent import conversation_store

            source = (conversation_store.get_conversation(conversation_id) or {}).get("source_agent_name")
            name = next((k for k, u in users.items() if source in (k, u.get("agent_name"))), None)
            if name:
                return name
        except Exception as e:
            logger.warning(f"user_sim_judge: could not read conversation {conversation_id}: {type(e).__name__}: {e}")
    return next(iter(users)) if len(users) == 1 else None


def _turn(n: int, e: dict) -> core.Turn:
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
    when = e.get("virtual_time") or (e.get("read") or {}).get("time")
    return core.Turn(n=n, outcome=e.get("outcome") or "no_reply", message=e.get("message") or "",
                     input=e.get("input"), flags=flags, time=None if e.get("input") or not when else f"{when} UTC")


def _instructions(us: Optional[dict], seed: dict, multi_day: Optional[bool]) -> tuple[Optional[str], dict, list[str]]:
    """The user sim's system prompt, rebuilt from its ``user_sims`` record as deploy_user_sim_agent wrote it, what the
    entry records of the check against the record's ``prompt_sha256``, and notes for the judge."""
    if not us:
        return None, {"source": "missing", "matches": False}, [
            "the user sim's settings were not found, so its instructions aren't shown"]
    from agent_env.task_step.task_steps.agent import user_sim_versions

    name = us.get("display_name") or us.get("name") or "the user"
    opening = us.get("opening")
    if opening:
        for key, value in seed.items():
            opening = opening.replace(f"<{key}>", str(value))
    if us.get("multi_day") is not None:
        multi_day = us["multi_day"]
    person = {"key": us.get("name"), "name": name, "first": us.get("first_name") or name.split()[0],
              "description": us.get("description"), "permissions": us.get("permissions"),
              "keep_private": us.get("keep_private"), "opening": {"text": opening} if opening else None,
              "done_ends_run": not multi_day}
    try:
        prompt = user_sim_versions.setup(us.get("user_version") or user_sim_versions.VERSION, person)["system_prompt"]
    except Exception as e:
        return None, {"source": "missing", "matches": False, "error": f"{type(e).__name__}: {e}"}, [
            f"the user sim's instructions could not be rebuilt ({type(e).__name__}), so they aren't shown"]
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    recorded = us.get("prompt_sha256")
    check = {"source": "rebuilt", "prompt_sha256": sha, "recorded_prompt_sha256": recorded,
             "matches": recorded == sha}
    if recorded is None:
        return prompt, check, ["these instructions were rebuilt from the user sim's settings, and the run recorded no "
                               "hash of the ones it got, so they may differ from what it got"]
    if recorded != sha:
        return prompt, check, ["these instructions were rebuilt from the user sim's settings and differ from the ones "
                               "it got, so a line may be missing or changed"]
    if not opening:
        return prompt, check, ["the opening request wasn't recorded, so it is missing from these instructions"]
    return prompt, check, []
