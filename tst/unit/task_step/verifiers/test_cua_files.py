"""cua_files: each format a CUA task's agent leaves as the judge reads it: documents, slides, spreadsheets and PDFs
dumped as text, text as it is and cut at the limit, anything else by its size and digest, and a format whose library
isn't installed, a file its library can't read, or an Office file that expands past the limit, named as not rendered;
a long PDF read up to its page limit; a damaged one named as unreadable; and a rendering in its own process, killed when
it runs too long and named as not rendered when it fails."""

from __future__ import annotations

import builtins
import io
import zipfile

import docx
import pptx
import pypdf
import pytest

from agent_env.task_step.task_steps.verifiers.judge_utils import cua_files
from agent_env.task_step.task_steps.verifiers.judge_utils.cua_files import render, render_isolated


def _docx() -> bytes:
    document = docx.Document()
    document.add_heading("Dear Editor", level=1)
    document.add_paragraph("The figures are corrected.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "page", "12"
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()


def _pptx() -> bytes:
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[1])
    slide.shapes.title.text = "Q3 results"
    slide.notes_slide.notes_text_frame.text = "Say the margin."
    out = io.BytesIO()
    deck.save(out)
    return out.getvalue()


def _pdf() -> bytes:
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_documents_slides_and_pdfs_are_dumped_as_text():
    shown = render("letter.docx", _docx(), 10_000)
    assert shown.kind == "docx" and "[Heading 1] Dear Editor" in shown.text and "page | 12" in shown.text

    shown = render("deck.pptx", _pptx(), 10_000)
    assert shown.kind == "pptx" and "## Slide 1" in shown.text and "Q3 results" in shown.text
    assert "[notes] Say the margin." in shown.text

    assert render("memo.pdf", _pdf(), 10_000) == render("memo.pdf", _pdf(), 10_000)
    assert render("memo.pdf", _pdf(), 10_000).text.startswith("## Page 1")


def test_text_is_shown_as_it_is_and_cut_at_the_limit():
    assert render("data.csv", b"a,b\n1,2\n", 100).text == "a,b\n1,2\n"
    cut = render("log", b"x" * 50, 10)
    assert cut.kind == "text" and cut.cut and cut.text.startswith("x" * 10) and "40 more characters" in cut.text


def test_images_and_other_binaries_are_named_by_size_and_digest():
    shown = render("chart.png", b"\x89PNG\r\n\x1a\n\0\0", 100)
    assert shown.kind == "image" and "image file of 10 bytes, sha256" in shown.text
    assert render("blob", b"\0\1\2", 100).kind == "binary"


def test_a_format_whose_library_isnt_installed_or_cant_read_it_is_named(monkeypatch):
    assert "that openpyxl couldn't read" in render("broken.xlsx", b"not a workbook", 100).text

    real_import = builtins.__import__

    def without_docx(name, *args, **kwargs):
        if name == "docx":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_docx)
    shown = render("letter.docx", b"PK", 100)
    assert shown.kind == "unrendered" and "python-docx isn't installed; install agentenv-framework[cua]" in shown.text


@pytest.mark.parametrize("name", ["a.xlsx", "a.docx", "a.pptx", "a.pdf"])
def test_an_empty_file_of_a_rendered_format_is_named_not_raised(name):
    assert render(name, b"", 100).kind == "unrendered"


def test_a_sparse_workbook_stops_after_the_cells_it_may_visit():
    import time

    import openpyxl

    book = openpyxl.Workbook()
    book.active["A1"], book.active["XFD1048576"] = "start", "far corner"
    out = io.BytesIO()
    book.save(out)
    started = time.monotonic()

    shown = render("sparse.xlsx", out.getvalue(), 10_000)

    assert time.monotonic() - started < 10
    assert "A1 = 'start'" in shown.text and "stopped after visiting 200000 cells" in shown.text


def test_grouped_slide_text_is_shown():
    from pptx.util import Inches

    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    group = slide.shapes.add_group_shape()
    group.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1)).text_frame.text = "Revenue up 12%"
    out = io.BytesIO()
    deck.save(out)

    assert "Revenue up 12%" in render("deck.pptx", out.getvalue(), 10_000).text


def test_an_office_file_that_expands_past_the_limit_is_named_not_opened(monkeypatch):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("word/document.xml", b"\0" * (4 * 1024 * 1024))
    bomb, memo = out.getvalue(), _docx()
    monkeypatch.setattr(cua_files, "MAX_EXPANDED_BYTES", 1024 * 1024)
    assert render("memo.docx", memo, 1000).kind == "docx"
    opened = []
    monkeypatch.setattr(docx, "Document", lambda *a: opened.append(a))

    shown = render("memo.docx", bomb, 1000)

    assert shown.kind == "unrendered" and "expands past 1048576 bytes" in shown.text and len(bomb) < 64 * 1024
    assert opened == []


def test_a_long_pdf_is_read_up_to_its_page_limit(monkeypatch):
    writer = pypdf.PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=200, height=200)
    out = io.BytesIO()
    writer.write(out)
    monkeypatch.setattr(cua_files, "MAX_PDF_PAGES", 2)

    shown = render("long.pdf", out.getvalue(), 10_000)

    assert "## Page 2" in shown.text and "## Page 3" not in shown.text and "stopped at 2 of 3 pages" in shown.text


def test_a_document_with_damaged_compressed_data_is_named_as_unreadable():
    memo = bytearray(_docx())
    start = memo.find(b"word/document.xml") + len("word/document.xml")
    memo[start + 20:start + 60] = b"\xff" * 40  # inside the member's deflate stream

    shown = render("memo.docx", bytes(memo), 1000)

    assert shown.kind == "unrendered" and "couldn't read" in shown.text


@pytest.mark.asyncio
async def test_a_rendering_runs_in_its_own_process():
    shown = await render_isolated("memo.docx", _docx(), 1000)
    assert shown.kind == "docx" and "The figures are corrected." in shown.text


@pytest.mark.asyncio
@pytest.mark.parametrize("command, problem", [
    (("sh", "-c", "sleep 5"), "rendering it took over 0.2 s"),
    (("sh", "-c", "exit 9"), "ran out of memory or failed, exit 9"),
])
async def test_a_rendering_that_runs_too_long_or_fails_is_named_not_rendered(monkeypatch, command, problem):
    monkeypatch.setattr(cua_files, "RENDER_COMMAND", command)
    monkeypatch.setattr(cua_files, "RENDER_TIMEOUT_S", 0.2)

    shown = await render_isolated("memo.docx", _docx(), 1000)

    assert shown.kind == "unrendered" and problem in shown.text
