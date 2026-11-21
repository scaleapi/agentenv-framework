"""Files a CUA task's agent produced, and their golden counterparts, as text a rubric judge reads in its prompt
(``computer_verifier``): text formats as they are; spreadsheets, documents, slides and PDFs dumped by the libraries the
``cua`` extra installs (openpyxl, python-docx, python-pptx, pypdf), or named as not rendered without them; anything else
by its size and digest. Each rendering is cut at ``max_chars``. ``render_isolated`` renders in a child process held to
``RENDER_MEMORY_BYTES`` and killed after ``RENDER_TIMEOUT_S``, since a crafted file can make a library's work large."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import posixpath
import resource
import sys
import zipfile
from dataclasses import dataclass
from typing import Callable

#: Read as text, as they are.
TEXT_EXTENSIONS = frozenset({
    ".txt", ".csv", ".tsv", ".json", ".jsonl", ".md", ".html", ".htm", ".xml", ".log", ".py", ".js", ".ts", ".css",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sql", ".ics", ".eml", ".srt", ".vtt", ".tex", ".sh", ".rtf"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".tif", ".tiff"})
#: The most cells of a workbook a rendering lists.
MAX_CELLS = 6000
#: The most cells of a workbook a rendering visits, empty ones included: a sparse sheet with a far corner cell spans
#: billions of them.
MAX_VISITED_CELLS = 200_000
#: How much of a file without an extension is sampled to tell whether it's text.
TEXT_SAMPLE_BYTES = 4096
#: The most an Office file (a zip package) may expand to before it's opened: a small file can hold members that
#: decompress to far more, which its library would read into memory.
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
#: How much of a package member is decompressed at a time while it's measured.
EXPAND_CHUNK_BYTES = 1024 * 1024
#: The most pages of a PDF a rendering reads.
MAX_PDF_PAGES = 200
#: The most memory (address space) a rendering's child process may take.
RENDER_MEMORY_BYTES = 2 * 1024 * 1024 * 1024
#: How long a rendering's child process may run before it's killed.
RENDER_TIMEOUT_S = 60.0
#: The child process ``render_isolated`` runs: this module's ``main``.
RENDER_COMMAND = (sys.executable, "-m", __name__)


@dataclass(frozen=True)
class Rendered:
    """A file as the judge reads it: ``text``, and how it was read (``text``, ``xlsx``, ``docx``, ``pptx``, ``pdf``,
    ``image``, ``binary``, or ``unrendered`` when the library its format needs isn't installed); ``cut`` when
    ``text`` was cut at the limit."""

    kind: str
    text: str
    cut: bool = False


def render(name: str, data: bytes, max_chars: int) -> Rendered:
    """``data``, the bytes of the file ``name``, as the judge reads it."""
    extension = posixpath.splitext(name.lower())[1]
    renderer = _RENDERERS.get(extension)
    if renderer is not None:
        kind, library, read = renderer
        try:
            text = read(data)
        except _TooLarge:
            return Rendered("unrendered", f"({extension} file of {len(data)} bytes, not rendered: it expands past "
                                          f"{MAX_EXPANDED_BYTES} bytes, more than a rendering opens)")
        except ImportError:
            return Rendered("unrendered", f"({extension} file of {len(data)} bytes, not rendered: {library} isn't "
                                          "installed; install agentenv-framework[cua])")
        except Exception as e:  # a file the agent broke is evidence too
            return Rendered("unrendered", f"({extension} file of {len(data)} bytes that {library} couldn't read: "
                                          f"{type(e).__name__}: {e})")
        return _cut(kind, text, max_chars)
    if extension in TEXT_EXTENSIONS or (not extension and _looks_like_text(data)):
        return _cut("text", data.decode("utf-8", errors="replace"), max_chars)
    kind = "image" if extension in IMAGE_EXTENSIONS else "binary"
    return Rendered(kind, f"({kind} file of {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}; its content "
                          "isn't shown)")


class _TooLarge(Exception):
    """An Office package that decompresses past ``MAX_EXPANDED_BYTES``."""


def _check_expansion(data: bytes) -> None:
    """Raises ``_TooLarge`` when the zip package ``data`` decompresses past ``MAX_EXPANDED_BYTES``, counted by
    decompressing it a chunk at a time rather than by its headers, which can lie. A package that can't be read raises
    as its library would, for ``render`` to name it unreadable."""
    total = 0
    with zipfile.ZipFile(io.BytesIO(data)) as package:
        for member in package.infolist():
            with package.open(member) as stream:
                while chunk := stream.read(EXPAND_CHUNK_BYTES):
                    total += len(chunk)
                    if total > MAX_EXPANDED_BYTES:
                        raise _TooLarge


async def render_isolated(name: str, data: bytes, max_chars: int) -> Rendered:
    """``render`` in a child process (``RENDER_COMMAND``) held to ``RENDER_MEMORY_BYTES`` and killed after
    ``RENDER_TIMEOUT_S``; a file whose rendering ran out of either is named as not rendered. The child is killed
    whenever the wait ends early, a cancelled grade included, so none outlives its rendering."""
    child = await asyncio.create_subprocess_exec(
        *RENDER_COMMAND, name, str(max_chars), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, preexec_fn=_hold_memory)
    try:
        out, _ = await asyncio.wait_for(child.communicate(data), RENDER_TIMEOUT_S)
    except TimeoutError:
        return Rendered("unrendered", f"(file of {len(data)} bytes, not rendered: rendering it took over "
                                      f"{RENDER_TIMEOUT_S:g} s)")
    finally:
        if child.returncode is None:
            child.kill()
            await asyncio.shield(child.wait())
    if child.returncode != 0:
        return Rendered("unrendered", f"(file of {len(data)} bytes, not rendered: rendering it ran out of memory or "
                                      f"failed, exit {child.returncode})")
    shown = json.loads(out)
    return Rendered(shown["kind"], shown["text"], shown["cut"])


def _hold_memory() -> None:
    """In the child, before it runs: its address space capped at ``RENDER_MEMORY_BYTES`` where the OS enforces it."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (RENDER_MEMORY_BYTES, RENDER_MEMORY_BYTES))
    except (ValueError, OSError):  # macOS doesn't enforce RLIMIT_AS; the time limit still holds
        pass


def main() -> None:
    """The child ``render_isolated`` runs: the file's bytes on stdin, its name and ``max_chars`` as arguments, its
    rendering as JSON on stdout."""
    name, max_chars = sys.argv[1], int(sys.argv[2])
    shown = render(name, sys.stdin.buffer.read(), max_chars)
    json.dump({"kind": shown.kind, "text": shown.text, "cut": shown.cut}, sys.stdout)


def _cut(kind: str, text: str, max_chars: int) -> Rendered:
    if len(text) <= max_chars:
        return Rendered(kind, text)
    return Rendered(kind, text[:max_chars] + f"\n… (cut: {len(text) - max_chars} more characters)", cut=True)


def _looks_like_text(data: bytes) -> bool:
    sample = data[:TEXT_SAMPLE_BYTES]
    if b"\0" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as e:
        return e.start > len(sample) - 4  # a character cut by the sample's end
    return True


def _xlsx(data: bytes) -> str:
    """Each sheet's non-empty cells, ``A1 = value``, with ``[formula: …]`` for a formula cell: the formulas from the
    workbook, the values as last calculated (none for a workbook no application has calculated). Both are streamed
    (``read_only``), and the walk stops at ``MAX_CELLS`` listed or ``MAX_VISITED_CELLS`` visited."""
    import openpyxl
    from openpyxl.utils import get_column_letter

    _check_expansion(data)
    formulas = openpyxl.load_workbook(io.BytesIO(data), data_only=False, read_only=True)
    values = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    lines, listed, visited = [], 0, 0
    try:
        for sheet in formulas.worksheets:
            lines.append(f"## Sheet {sheet.title!r}")
            rows = zip(sheet.iter_rows(values_only=True), values[sheet.title].iter_rows(values_only=True))
            for row_number, (formula_row, value_row) in enumerate(rows, start=1):
                for column, (formula, value) in enumerate(zip(formula_row, value_row), start=1):
                    visited += 1
                    if visited > MAX_VISITED_CELLS:
                        lines.append(f"… (stopped after visiting {MAX_VISITED_CELLS} cells)")
                        return "\n".join(lines)
                    if formula is None:
                        continue
                    if listed == MAX_CELLS:
                        lines.append(f"… (stopped at {MAX_CELLS} cells)")
                        return "\n".join(lines)
                    listed += 1
                    cell = f"{get_column_letter(column)}{row_number}"
                    if isinstance(formula, str) and formula.startswith("="):
                        lines.append(f"{cell} = {value!r} [formula: {formula}]")
                    else:
                        lines.append(f"{cell} = {formula!r}")
    finally:
        formulas.close()
        values.close()
    return "\n".join(lines)


def _docx(data: bytes) -> str:
    """Each paragraph with its style, then each table's rows, cells split by `` | ``."""
    import docx

    _check_expansion(data)
    document = docx.Document(io.BytesIO(data))
    lines = [f"[{paragraph.style.name}] {paragraph.text}" for paragraph in document.paragraphs if paragraph.text.strip()]
    for number, table in enumerate(document.tables, start=1):
        lines.append(f"## Table {number}")
        lines.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows)
    return "\n".join(lines)


def _pptx(data: bytes) -> str:
    """Each slide's shapes' text and tables, a group's shapes within it included, and its notes."""
    import pptx

    _check_expansion(data)

    def shape_lines(shapes) -> list[str]:
        lines = []
        for shape in shapes:
            if hasattr(shape, "shapes"):  # a group: its shapes hold the text
                lines.extend(shape_lines(shape.shapes))
                continue
            if shape.has_text_frame and shape.text_frame.text.strip():
                lines.append(f"[{shape.name}] {shape.text_frame.text}")
            if getattr(shape, "has_table", False) and shape.has_table:
                lines.extend(" | ".join(cell.text.strip() for cell in row.cells) for row in shape.table.rows)
        return lines

    lines = []
    for number, slide in enumerate(pptx.Presentation(io.BytesIO(data)).slides, start=1):
        lines.append(f"## Slide {number}")
        lines.extend(shape_lines(slide.shapes))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            lines.append(f"[notes] {slide.notes_slide.notes_text_frame.text}")
    return "\n".join(lines)


def _pdf(data: bytes) -> str:
    """Each page's text, up to ``MAX_PDF_PAGES`` pages; pypdf bounds each stream it decompresses."""
    import pypdf

    pages = pypdf.PdfReader(io.BytesIO(data)).pages
    lines = [f"## Page {number}\n{page.extract_text() or ''}"
             for number, page in enumerate(pages[:MAX_PDF_PAGES], start=1)]
    if len(pages) > MAX_PDF_PAGES:
        lines.append(f"… (stopped at {MAX_PDF_PAGES} of {len(pages)} pages)")
    return "\n".join(lines)


_RENDERERS: dict[str, tuple[str, str, Callable[[bytes], str]]] = {
    ".xlsx": ("xlsx", "openpyxl", _xlsx), ".xlsm": ("xlsx", "openpyxl", _xlsx),
    ".docx": ("docx", "python-docx", _docx), ".pptx": ("pptx", "python-pptx", _pptx), ".pdf": ("pdf", "pypdf", _pdf)}


if __name__ == "__main__":
    main()
