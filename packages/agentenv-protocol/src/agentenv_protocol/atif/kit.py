"""The adapter kit, for an agent's CI: run an adapter over recorded tasks, as the SDK runs it live, and check what it
writes. The live rule holds after every record, the final document passes the validator as a producer's, it equals
the checked-in document when one is given, and ``from_atif`` turns ATIF without the profile back into a native log."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapter import Adapter, Native
from .profile import profile_of
from .reader import strip_profile
from .runner import AdapterRun, RecordedTask
from .validator import ERROR, WARNING, Finding, Rule, _difference, errors_in, validate_live

KIT_RULES: dict[str, Rule] = {rule.id: rule for rule in (
    Rule("kit-run", ERROR, None, "The adapter takes every recorded record without raising"),
    Rule("kit-expected", ERROR, None, "The document equals the checked-in expected document"),
    Rule("kit-unmatched", WARNING, None, "Every native record has a handler; one that none takes is only counted"),
    Rule("kit-from-atif", ERROR, None, "`from_atif` turns a document without the profile into a native log"),
)}


@dataclass
class KitResult:
    """What the kit found, with the document and every live page the adapter's run served."""

    document: dict[str, Any] | None
    pages: list[dict[str, Any] | None] = field(default_factory=list)
    media: dict[str, bytes] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return errors_in(self.findings)

    def raise_for_errors(self) -> None:
        """Raise AssertionError listing every finding when any is an error."""
        if self.errors:
            raise AssertionError("the adapter kit found:\n" + "\n".join(str(f) for f in self.findings))


def check_adapter(adapter: type[Adapter], recorded: Sequence[RecordedTask], *,
                  expected: Mapping[str, Any] | Path | None = None,
                  from_atif: Sequence[Mapping[str, Any]] | None = None,
                  media: Mapping[str, bytes] | None = None) -> KitResult:
    """Run ``adapter`` over ``recorded``, one A2A context's tasks in order, and check it. ``expected`` is the document
    the run should write, or the file holding it. ``from_atif`` are the documents its ``from_atif`` is tried on; by
    default, the run's own document with the profile stripped. ``media`` holds, by path, the files of any Media in the
    recorded tasks' inbound or final content; the adapter's own media come from the run."""
    run = AdapterRun(adapter)
    result = KitResult(document=None)
    where = ""
    try:
        for t, task in enumerate(recorded):
            where = f"task {t}, its start"
            run.start_task(task.context)
            result.pages.append(run.page())
            for r, fed in enumerate(task.records):
                where = f"task {t}, record {r}" + (f" of aux {fed.aux!r}" if fed.aux is not None else "")
                if fed.aux is None:
                    run.native(fed.record)
                else:
                    run.aux(fed.aux, fed.record)
                result.pages.append(run.page())
            where = f"task {t}, its end"
            run.end_task(state=task.state, final=task.final, ended_at=task.ended_at)
            result.pages.append(run.page())
    except Exception as exc:
        result.findings.append(Finding("kit-run", ERROR, where, f"{type(exc).__name__}: {exc}"))
        return result
    result.document = run.document()
    result.media = {**(media or {}), **run.media_files}
    if result.document is None:
        result.findings.append(Finding("kit-run", ERROR, "", "the run wrote no step"))
        return result
    result.findings.extend(validate_live(result.pages, result.document, media=result.media))
    unmatched = profile_of(result.document).get("unmatched")
    if unmatched:
        counts = ", ".join(f"{kind} ({count})" for kind, count in sorted(unmatched.items()))
        result.findings.append(Finding("kit-unmatched", WARNING, "extra.agentenv.unmatched",
                                       f"no handler takes these native records: {counts}"))
    if expected is not None:
        want = json.loads(expected.read_text(encoding="utf-8")) if isinstance(expected, Path) else expected
        if result.document != want:
            result.findings.append(Finding("kit-expected", ERROR, _difference(want, result.document),
                                           "the document differs from the expected one here"))
    for index, document in enumerate(from_atif if from_atif is not None else [strip_profile(result.document)]):
        _from_atif(adapter, document, index, result)
    return result


def _from_atif(adapter: type[Adapter], document: Mapping[str, Any], index: int, result: KitResult) -> None:
    where = f"from_atif document {index}"
    try:
        native = adapter.from_atif(document)
    except NotImplementedError as exc:
        result.findings.append(Finding("kit-from-atif", WARNING, where, str(exc)))
        return
    except Exception as exc:
        result.findings.append(Finding("kit-from-atif", ERROR, where, f"raised {type(exc).__name__}: {exc}"))
        return
    if not isinstance(native, Native) or not isinstance(native.log, list):
        result.findings.append(Finding("kit-from-atif", ERROR, where,
                                       f"returned {type(native).__name__}, not a Native holding a list log"))
