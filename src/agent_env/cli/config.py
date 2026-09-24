"""`agent-env config` — inspect the config surface without touching it."""

import datetime
import json

import click

from agent_env.config import loader as config_loader
from agent_env.config.describe import (
    SOURCE_ENV,
    SOURCE_NONE,
    SOURCE_WALK_UP,
    ConfigReport,
    Explained,
    SectionsReport,
    as_dict,
    describe_config,
    explain_path,
    search_path,
    sources,
)
from agent_env.config.provenance import KIND_DEFAULT, headline, lines as section_lines



def _json(payload: dict) -> str:
    """The group's one serializer. TOML has first-class date/time types, so a perfectly
    valid config file reaches here holding values `json` refuses; anything else unexpected
    still raises, because silently stringifying it would hide a real bug."""
    def encode(value):
        if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
            return value.isoformat()
        raise TypeError(f"cannot serialize {type(value).__name__} in a config report")
    return json.dumps(payload, indent=2, default=encode)


def _source_prose(report: ConfigReport) -> str:
    """The report says *how* the file was found; the sentence saying so belongs here."""
    if report.config_source == SOURCE_ENV:
        return f"via ${config_loader.ENV_CONFIG_PATH}"
    if report.config_source == SOURCE_WALK_UP:
        return "discovered by walking up from the working directory"
    if report.config_source == SOURCE_NONE:
        root = report.search_root
        return f"none found — walked up from {root} to {root.anchor or '/'}"
    return "discovery failed"


def render(report: ConfigReport) -> str:
    """Render a `ConfigReport` as the operator-facing block. Kept separate from the command
    so the format is assertable without a CliRunner."""
    labels = ["config:"] + [f"{s.name}:" for s in report.sections]
    width = max(len(label) for label in labels) + 1
    lines = []
    where = report.config_path if report.config_path is not None else "(none)"
    lines.append(f"{'config:':<{width}} {where}")
    lines.append(f"{'':<{width}} {_source_prose(report)}")
    if report.error:
        lines.append(f"{'':<{width}} (error) {report.error}")
    for warning in report.warnings:
        lines.append(f"{'':<{width}} (warning) {warning}")
    lines.append("")
    for section in report.sections:
        label = f"{section.name}:"
        body = [f"(unresolved) {section.error}"] if section.error else section_lines(section.value)
        head, rest = (body[0] or ("(unset)" if len(body) == 1 else "")), body[1:]
        lines.append(f"{label:<{width}} {head}".rstrip())
        lines += [f"{'':<{width}} {extra}" for extra in rest]
        lines.append(f"{'':<{width}} from {section.winner.where}")
        for beaten in section.shadowed:
            if beaten.error is not None:
                lines.append(f"{'':<{width}} ; {beaten.where} unreadable: {beaten.error}")
            elif beaten.kind != KIND_DEFAULT:
                lines.append(f"{'':<{width}} ; {beaten.summary()} in {beaten.where}, shadowed")
    return "\n".join(lines)


@click.group()
def config():
    """Inspect the resolved agent-env configuration."""


@config.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable, provenance included.")
def show(as_json: bool):
    """Print the config file that won and what each section resolves to.

    Read-only: no store is constructed, no network call is made, and no directory is
    created. Secret values are never printed — `env:` / `secret:` references are shown as
    references, and a literal under a secret-shaped key is masked.
    """
    report = describe_config()
    click.echo(_json(as_dict(report)) if as_json else render(report))


@config.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable.")
def debug(as_json: bool):
    """Print every path discovery considers, in order, and whether it is there."""
    candidates = [{"path": str(c.path), "why": c.why, "exists": c.exists} for c in search_path()]
    if as_json:
        click.echo(_json({"search_path": candidates}))
        return
    for c in candidates:
        mark = "->" if c["exists"] else "  "
        click.echo(f"{mark} {c['path']}  ({c['why']}, exists: {'yes' if c['exists'] else 'no'})")


def _render_children(report: SectionsReport) -> list[str]:
    out = [f"  covers {len(report.children)} sections, each resolving on its own:"]
    width = max(len(c.path) for c in report.children) + 2
    for child in report.children:
        where = child.winner.where if child.winner is not None else "(none)"
        summary = f"(unresolved) {child.error}" if child.error else headline(child.value)
        out.append(f"    {child.path:<{width}}{summary}  from {where}")
    return out


def render_explain(report: Explained) -> str:
    """One path's resolution, in the same vocabulary `show` uses for a whole section."""
    lines = [f"{report.path}"]
    if isinstance(report, SectionsReport):
        return "\n".join(lines + _render_children(report))
    if report.error:
        lines.append(f"  (unresolved) {report.error}")
    if report.winner is None and not report.error:
        lines.append("  (unset) — no layer supplies this path")
        return "\n".join(lines)
    if not report.error:
        body = section_lines(report.value)
        # `(unset)` only when there is nothing else to say: substituting it whenever the
        # first line is blank prints it directly above the table it is denying.
        head = body[0] or ("(unset)" if len(body) == 1 else "")
        lines.append(f"  {head}".rstrip())
        lines += [f"  {extra}" for extra in body[1:]]
    if report.winner is not None:
        lines.append(f"  from {report.winner.where}")
    for beaten in report.shadowed:
        if beaten.error is not None:
            lines.append(f"  ; {beaten.where} unreadable: {beaten.error}")
        elif beaten.kind != KIND_DEFAULT:
            lines.append(f"  ; {beaten.summary()} in {beaten.where}, shadowed")
    if report.file_only and report.winner is not None:
        lines.append("  ; read from the file only — no resolver owns this path")
    return "\n".join(lines)


@config.command()
@click.argument("path")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable, provenance included.")
def explain(path: str, as_json: bool):
    """Print where one dotted PATH's value came from.

    PATH is a section name or a TOML path — `document` and `stores.document` both work.
    Read-only, on the same terms as `show`.
    """
    report = explain_path(path)
    if as_json and isinstance(report, SectionsReport):
        # The wire shape stays flat even though the types split: an ancestor answered with
        # a different set of keys would break a reader that just indexes them, and the type
        # split was for this module's clarity, not the caller's.
        click.echo(_json({
            "path": report.path,
            "section": None,
            "value": None,
            "winner": None,
            "shadowed": [],
            "error": None,
            "file_only": False,
            "children": [
                {"path": c.path, "section": c.section,
                 "winner": None if c.winner is None else c.winner.where,
                 "value": c.value, "error": c.error}
                for c in report.children
            ],
        }))
        return
    if as_json:
        click.echo(_json({
            "path": report.path,
            "section": report.section,
            "value": report.value,
            "winner": None if report.winner is None else {
                "kind": report.winner.kind, "where": report.winner.where,
                "value": report.winner.summary(),
            },
            "shadowed": [{"kind": b.kind, "where": b.where, "value": b.summary(), "error": b.error}
                         for b in report.shadowed],
            "error": report.error,
            "file_only": report.file_only,
            "children": [],
        }))
        return
    click.echo(render_explain(report))


@config.command(name="sources")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable.")
def sources_command(as_json: bool):
    """Print every layer the resolver consults, lowest precedence first.

    `debug` answers which *file*; this answers which *layers* — including the ones
    contributing nothing, because "the file I edited is not being read" is the thing worth
    seeing.
    """
    layers = sources()
    if as_json:
        # Array order *is* the precedence order; no numeric layer id is published,
        # because the target chain's numbering is not settled and would renumber.
        click.echo(_json({"sources": [
            {"kind": s.kind, "where": s.where, "present": s.present,
             "detail": s.detail, "shadows": s.shadows}
            for s in layers
        ]}))
        return
    # Deliberately not column-aligned on `where`: one entry is a filesystem path, and
    # padding every other row to its width pushes the detail off the edge of a terminal.
    for s in layers:
        mark = "->" if s.present else "  "
        shadows = f" -> [{s.shadows}]" if s.shadows else ""
        detail = f"  ({s.detail})" if s.detail else ""
        click.echo(f"{mark} {s.kind:<8} {s.where}{shadows}{detail}")
