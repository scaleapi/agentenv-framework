"""cua_verifier: a computer-use task's rubric, graded by rubrics_verifier's judge on what the agent left in the computer.

The rubric is a list of criteria (or ``{"criteria": [...]}``) with an ``id``, a ``title`` and a ``weight`` (0 when left
out), each a ``must-pass`` gate, a ``negative`` one (a failure mode, its weight a penalty) or a regular one, by its
``annotations``' ``criteria_type`` / ``criteria_category``; a ``visual`` category, and ``target_result_files`` /
``target_GT_files``, say how to grade it. What the judge reads is what the agent left in the computer: each of
``result_files``, read through the computer's driver (computer-exec/v1 ``run``, the control route the agent can't call)
and rendered as text (``judge_utils/cua_files.py``), a file that isn't there named as one the agent did not produce; the
golden files of ``expected_artifact_id``, for reference only; the output of each of ``state_commands`` run in the
computer, such as a mock website's state; and ``instruction``, the task as the agent was given it. With no
``prompt_id`` the judge grades those alone; with one it also sees that prompt's response.

How the judge grades is ``guidance``, and how it treats the golden files ``golden_note``: the caller's own wording, or
short defaults of the framework's. The score: each criterion 1 or 0; the regular and passed must-pass criteria's weights
averaged, less each negative criterion's weight when its failure mode is present, clamped to [0, 1]; and 0 whenever a
must-pass criterion fails. A negative criterion is graded as rubrics_verifier grades one, 1 when its failure mode is
absent, and keeps its penalty as a negative weight.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any, ClassVar, Optional

from agent_env.entity.refs import EntityRef
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.preflight import stored_env
from agent_env.task_step.task_steps.computer.computer_exec import computer_problem, find_computer
from agent_env.task_step.task_steps.verifiers.judge_utils.cua_files import render
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score

logger = logging.getLogger(__name__)

#: Where the step records what the judge was shown, by step id.
RECORDS_KEY = "cua_verifier"
#: How the judge grades when the step names no ``guidance`` of its own.
DEFAULT_GUIDANCE = """Give each criterion 1.0 if what is shown above meets it and 0.0 if it doesn't, and say \
where you found the evidence.

- A criterion of kind "must-pass" that scores 0.0 sets the whole rubric's score to 0.
- A criterion marked "negative": true names something that should not happen; score it as the instructions say.
- A criterion's "applies_to" lists the files it is about; one without it is about the outputs as a whole.
- A criterion marked "visual": true is about how a file looks, judged from its rendering.
- A golden file is a reference: what it holds counts only when the result holds it too."""
#: How the judge treats the golden files when the step names no ``golden_note`` of its own.
DEFAULT_GOLDEN_NOTE = ("Reference files: use one only for a criterion that refers to it or to an expected value, not to "
                       "compare the results with as a whole.")
#: How long reading one file or running one state command in the computer may take.
READ_TIMEOUT_S = 120.0
#: The most of one file's rendering, and of one command's output, the judge reads.
MAX_FILE_CHARS = 40_000
#: The most of a file one read returns: its base64 well within the 1 MiB a driver returns of a command's output.
READ_CHUNK_BYTES = 512 * 1024
#: The largest file the judge is shown; a larger one is named as too large.
MAX_READ_BYTES = 32 * 1024 * 1024
#: The most of a failed read's stderr its error quotes.
ERROR_TAIL_CHARS = 300
#: How far below 1 a criterion's score may be and still count as met, for scores that went through floating point.
SCORE_TOLERANCE = 1e-9
#: Exit status of a read when the file isn't there.
_NOT_THERE = 3
_SIZE = 'test -f "$1" || exit 3; wc -c < "$1"'
#: ``$2`` bytes of the file ``$1`` from byte ``$3`` (1-based), as base64.
_READ_CHUNK = 'tail -c +"$3" -- "$1" | head -c "$2" | base64'


def criteria_from_rubric(rubric: Any) -> list[dict]:
    """A rubric (``{"criteria": [...]}`` or the list) as rubrics_verifier's criteria: each with its
    ``id``, ``title``, ``kind``, ``weight`` (a negative one's as a penalty, below 0), and ``visual`` and ``applies_to``
    when its annotations say."""
    items = rubric.get("criteria") if isinstance(rubric, dict) else rubric
    if not isinstance(items, list) or not items:
        raise ValueError("rubric must be a non-empty list of criteria, or an object whose criteria is one")
    criteria = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise ValueError(f"rubric criterion {index} needs an id")
        if not isinstance(item.get("title"), str) or not item["title"].strip():
            raise ValueError(f"rubric criterion {item['id']!r} needs a title, the text the judge grades")
        weight = item.get("weight", 0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"rubric criterion {item['id']!r}: weight must be a number")
        annotations = item.get("annotations") or {}
        tags = " ".join(str(annotations.get(key) or "").lower() for key in ("criteria_type", "criteria_category"))
        kind = "must-pass" if "must-pass" in tags else "negative" if "negative" in tags or weight < 0 else "regular"
        criterion = {"id": item["id"], "title": item["title"], "kind": kind,
                     "weight": -abs(float(weight)) if kind == "negative" else float(weight)}
        if kind == "negative":
            criterion["negative"] = True
        if str(annotations.get("criteria_category") or annotations.get("category") or "").lower() == "visual":
            criterion["visual"] = True
        applies_to = {name: list(files) for name, key in (("result_files", "target_result_files"),
                                                          ("golden_files", "target_GT_files"))
                      if (files := annotations.get(key))}
        if applies_to:
            criterion["applies_to"] = applies_to
        criteria.append(criterion)
    return criteria


def cua_rubric_score(rows: list[dict]) -> float:
    """A rubric's score: 0 when a must-pass criterion fails; else the weighted average of the regular
    and must-pass criteria, less the negative criteria's penalties, clamped to [0, 1]; with no positive weight, 1 less
    the penalties when there are must-pass gates, else 0."""
    binary = [{**row, "score": 1.0 if _passed(row) else 0.0} for row in rows]
    gates = [row for row in binary if row.get("kind") == "must-pass"]
    if any(row["score"] < 1.0 for row in gates):
        return 0.0
    if gates and not any(float(row.get("weight", 0)) > 0 for row in binary):
        penalty = sum(abs(float(row["weight"])) * (1.0 - row["score"]) for row in binary
                      if float(row.get("weight", 0)) < 0)
        return max(0.0, 1.0 - penalty)
    return aggregate_score(binary, ScoreAggregator.WEIGHTED_AVERAGE)


def _passed(row: dict) -> bool:
    score = row.get("score")
    return isinstance(score, (int, float)) and not isinstance(score, bool) and score >= 1.0 - SCORE_TOLERANCE


class CuaVerifierTaskStep(RubricsVerifierTaskStep):
    type: ClassVar[str] = "cua_verifier"
    entity_refs = (*RubricsVerifierTaskStep.entity_refs, EntityRef.env("env_id"),
                   EntityRef.artifact("expected_artifact_id", version_field="expected_artifact_version"))
    context_paths = RubricsVerifierTaskStep.context_paths

    def __init__(self, id: str, version: Optional[int], *, env_id: str, rubric: Any, result_files: list[str],
                 prompt_id: Optional[str] = None, computer: Optional[str] = None,
                 expected_artifact_id: Optional[str] = None, expected_artifact_version: Optional[int] = None,
                 state_commands: Optional[list[dict]] = None, instruction: Optional[str] = None,
                 guidance: Optional[str] = None, golden_note: Optional[str] = None, use_trajectory: bool = False, criteria: Any = None,
                 default_model: Optional[str] = None, default_model_api_base: Optional[str] = None,
                 effort: Optional[str] = None, max_thinking_tokens: Optional[int] = None,
                 judge_a2a_agent_id: Optional[str] = None, judge_sandbox_type: Optional[str] = None,
                 judge_batch_size: Optional[int] = None, judge_batch_concurrency: Optional[int] = None,
                 **fields: Any):
        """``criteria`` is what ``rubric`` becomes (``criteria_from_rubric``), never read from a stored step; the rest is
        rubrics_verifier's, but for ``score_aggregator``, which is this step's own (``cua_rubric_score``)."""
        where = f"cua_verifier '{id}'"
        for name, text in (("guidance", guidance), ("golden_note", golden_note)):
            if text is not None and (not isinstance(text, str) or not text.strip()):
                raise ValueError(f"{where}: {name} must be text, or left out for the default")
        if prompt_id is None and use_trajectory:
            raise ValueError(f"{where}: use_trajectory needs a prompt_id, whose run's trajectory the judge reads")
        if not isinstance(result_files, list) or not all(isinstance(p, str) and p for p in result_files):
            raise ValueError(f"{where}: result_files must be a list of paths in the computer")
        for command in state_commands or []:
            if not (isinstance(command, dict) and isinstance(command.get("name"), str) and command["name"]
                    and isinstance(command.get("command"), (str, list)) and command["command"]):
                raise ValueError(f"{where}: each of state_commands is {{\"name\": ..., \"command\": [...] or \"...\"}}")
        fields.pop("score_aggregator", None)
        self.env_id = env_id
        self.rubric = rubric
        self.result_files = result_files
        self.computer = computer
        self.expected_artifact_id = expected_artifact_id
        self.expected_artifact_version = expected_artifact_version
        self.state_commands = state_commands or []
        self.instruction = instruction
        self.guidance = guidance
        self.golden_note = golden_note
        self.grades_files_alone = prompt_id is None
        super().__init__(id, version, criteria=criteria_from_rubric(rubric), prompt_id=prompt_id or f"{id}:files",
                         use_trajectory=use_trajectory, score_aggregator=ScoreAggregator.WEIGHTED_AVERAGE,
                         default_model=default_model, default_model_api_base=default_model_api_base, effort=effort,
                         max_thinking_tokens=max_thinking_tokens, judge_a2a_agent_id=judge_a2a_agent_id,
                         judge_sandbox_type=judge_sandbox_type, judge_batch_size=judge_batch_size,
                         judge_batch_concurrency=judge_batch_concurrency, **fields)

    def preflight(self) -> list[str]:
        """rubrics_verifier's, and the env this step reads: a computer env, or a multi env with that computer."""
        problems = super().preflight()
        if isinstance(self.env_id, str) and self.env_id and (env := stored_env(self.env_id)) is not None:
            if problem := computer_problem(env, self.computer):
                problems.append(f"step {self.id!r}: {problem}")
        return problems

    def to_dict(self) -> dict:
        return {**super().to_dict(), "prompt_id": None if self.grades_files_alone else self.prompt_id, "env_id": self.env_id,
                "rubric": self.rubric, "result_files": self.result_files, "computer": self.computer,
                "expected_artifact_id": self.expected_artifact_id,
                "expected_artifact_version": self.expected_artifact_version, "state_commands": self.state_commands,
                "instruction": self.instruction, "guidance": self.guidance, "golden_note": self.golden_note}

    @classmethod
    def from_dict(cls, data: dict) -> CuaVerifierTaskStep:
        fields = cls._rubrics_fields({**data, "criteria": None, "prompt_id": data.get("prompt_id")})
        fields.update(use_trajectory=data.get("use_trajectory", False), env_id=data["env_id"], rubric=data["rubric"],
                      result_files=data.get("result_files") or [], computer=data.get("computer"),
                      expected_artifact_id=data.get("expected_artifact_id"),
                      expected_artifact_version=data.get("expected_artifact_version"),
                      state_commands=data.get("state_commands"), instruction=data.get("instruction"),
                      guidance=data.get("guidance"), golden_note=data.get("golden_note"))
        return cls(**fields)

    def score_rows(self, rows: list[dict]) -> float:
        return cua_rubric_score(rows)

    def graded_response(self, context: TaskStepContext) -> Optional[PromptResponse]:
        """The prompt's response; for a step that grades the files alone, a stand-in of its own, which never joins the
        context, so no other step reads it as the agent's."""
        if not self.grades_files_alone:
            return super().graded_response(context)
        return PromptResponse(prompt_id=self.prompt_id, prompt_text=self.instruction or "",
                              response="(Not graded: grade the files and state below, which are what the agent "
                                       "produced.)")

    def _write_entries(self, context: TaskStepContext, verification_results: list[dict], judge_call_fields: dict,
                       **kwargs: Any) -> None:
        """rubrics_verifier's entry, with ``gate_passed``, whether every must-pass criterion was met, and when one
        wasn't, ``gates_failed``, their ids: written in the same breath as the entry, since a step beside this one
        stores what it sees of the context when it ends. (A run's stored context never gains an empty list, so there's
        no ``gates_failed`` when none failed.)"""
        super()._write_entries(context, verification_results, judge_call_fields, **kwargs)
        entry = context.metadata["verifications"][self.verifier_id]
        failed = [row.get("id") for row in entry["results"] if row.get("kind") == "must-pass" and not _passed(row)]
        entry["gate_passed"] = not failed
        if failed:
            entry["gates_failed"] = failed

    async def judge_prompt_sections(self, context: TaskStepContext, prompt_response: PromptResponse) -> list[str]:
        """The files and state the agent left in the computer, the golden files, the instruction and how to grade;
        what was read is recorded under ``cua_verifier``."""
        deployed = find_computer(context, self.env_id, self.computer)
        record: dict = {"files": [], "golden_files": [], "state": []}
        sections = []
        produced, missing = [], []
        for number, path in enumerate(self.result_files, start=1):
            data = await self._read(deployed, path)
            if data is None:
                missing.append(path)
                record["files"].append({"path": path, "missing": True})
                continue
            if isinstance(data, int):
                record["files"].append({"path": path, "bytes": data, "rendered_as": "too_large"})
                produced.append(f"### File {number}: {path}\n(a file of {data} bytes, too large to show: over "
                                f"{MAX_READ_BYTES} bytes)")
                continue
            shown = render(path, data, MAX_FILE_CHARS)
            record["files"].append({"path": path, "bytes": len(data), "rendered_as": shown.kind, "cut": shown.cut})
            produced.append(f"### File {number}: {path} ({shown.kind})\n<file path=\"{path}\">\n{shown.text}\n</file>")
        sections.append("## Files under evaluation\nWhat the agent left at each path in its computer, rendered as "
                        "text.\n\n" + ("\n\n".join(produced) or "(none)"))
        if missing:
            sections.append("## Files the agent did NOT produce\nTreat every criterion that depends on one of these "
                            "as not satisfied:\n" + "\n".join(f"- {path}" for path in missing))
        if golden := self._golden_files():
            shown_golden = []
            for name, data in golden.items():
                shown = render(name, data, MAX_FILE_CHARS)
                record["golden_files"].append({"name": name, "bytes": len(data), "rendered_as": shown.kind})
                shown_golden.append(f"### Golden: {name} ({shown.kind})\n<golden name=\"{name}\">\n{shown.text}\n"
                                    "</golden>")
            sections.append("## Golden / reference files\n" + (self.golden_note or DEFAULT_GOLDEN_NOTE) + "\n\n"
                            + "\n\n".join(shown_golden))
        state = []
        for command in self.state_commands:
            output = await self._state(deployed, command)
            record["state"].append({"name": command["name"], **{k: output[k] for k in ("exit_code", "timed_out")}})
            text = (output.get("stdout") or "")[:MAX_FILE_CHARS]
            state.append(f"### {command['name']}\n<state name=\"{command['name']}\" exit_code=\"{output['exit_code']}\">"
                         f"\n{text}\n</state>")
        if state:
            sections.append("## Environment state\nWhat each command run in the computer after the agent finished "
                            "printed.\n\n" + "\n\n".join(state))
        if self.instruction:
            sections.append("## Task instruction\nThe task as the agent was given it; treat anything it mentions as a "
                            f"stated requirement.\n---\n{self.instruction}\n---")
        sections.append("## How to grade\n" + (self.guidance or DEFAULT_GUIDANCE))
        context.metadata.setdefault(RECORDS_KEY, {})[self.id] = record
        return sections

    async def _read(self, deployed: Any, path: str) -> Optional[bytes] | int:
        """The file at ``path`` in the computer, absolute or under the home (``~/`` or relative), read
        ``READ_CHUNK_BYTES`` at a time; None when there's no file there, and its size alone when it's over
        ``MAX_READ_BYTES``."""
        relative = path[2:] if path.startswith("~/") else path
        answer = await self._run(deployed, path, _SIZE, relative)
        if answer.get("exit_code") == _NOT_THERE:
            return None
        size = int((answer.get("stdout") or "0").strip() or 0)
        if size > MAX_READ_BYTES:
            return size
        data = bytearray()
        while len(data) < size:
            chunk = await self._run(deployed, path, _READ_CHUNK, relative, str(READ_CHUNK_BYTES), str(len(data) + 1))
            try:
                piece = base64.b64decode("".join((chunk.get("stdout") or "").split()), validate=True)
            except binascii.Error as e:
                raise RuntimeError(f"cua_verifier '{self.id}': {path} in env {self.env_id!r} came back unreadable") from e
            if len(piece) != min(READ_CHUNK_BYTES, size - len(data)):
                raise RuntimeError(f"cua_verifier '{self.id}': {path} in env {self.env_id!r} gave {len(piece)} bytes "
                                   f"from byte {len(data)} of its {size}; it changed or couldn't be read while it was")
            data += piece
        return bytes(data)

    async def _run(self, deployed: Any, path: str, script: str, *args: str) -> dict:
        """``script`` run by ``sh -c`` in the computer with ``args``; its answer, unless it failed otherwise than by
        finding no file."""
        from agent_env.env.envs.computer.utils import inside

        answer = await inside.run(deployed, ["sh", "-c", script, "cua_verifier", *args], timeout_s=READ_TIMEOUT_S)
        if answer.get("exit_code") in (0, _NOT_THERE) and not answer.get("timed_out") and not answer.get("truncated"):
            return answer
        why = ("came back cut by the computer's driver" if answer.get("truncated")
               else f"couldn't be read: {(answer.get('stderr') or '').strip()[-ERROR_TAIL_CHARS:]}")
        raise RuntimeError(f"cua_verifier '{self.id}': {path} in env {self.env_id!r} {why}")

    async def _state(self, deployed: Any, command: dict) -> dict:
        from agent_env.env.envs.computer.utils import inside

        argv = ["sh", "-c", command["command"]] if isinstance(command["command"], str) else list(command["command"])
        return await inside.run(deployed, argv, timeout_s=READ_TIMEOUT_S)

    def _golden_files(self) -> dict[str, bytes]:
        if self.expected_artifact_id is None:
            return {}
        from agent_env.artifact import Artifact

        artifact = Artifact.get(self.expected_artifact_id, self.expected_artifact_version)
        return {name: file.load() for name, file in artifact.get_file_artifacts().items()}
