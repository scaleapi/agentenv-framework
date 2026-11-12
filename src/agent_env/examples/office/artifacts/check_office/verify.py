"""Whether the agent saved the memo and the presentation the task asks for, in Office formats, read inside the computer
through the driver's computer-exec: the memo a .docx with its title and at least two paragraphs under it; the
presentation a .pptx of two slides titled as asked, the second with a list of three items, that list a shape an entrance
animation targets."""
import json
import uuid

from agentenv_protocol import client as protocol

EXEC = "urn:agentenv:computer-exec/v1"
#: Run in the computer: what each file holds, as JSON; each slide's shapes with their paragraphs, and whether an
#: entrance animation targets them.
READ = r'''
import json, re, sys, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

def paragraphs(element, paragraph, run):
    found = []
    for p in element.iter(paragraph):
        text = "".join(t.text or "" for t in p.iter(run)).strip()
        if text:
            found.append(text)
    return found

def shapes(slide):
    entering = {target.get("spid") for ctn in slide.iter(P + "cTn") if ctn.get("presetClass") == "entr"
                for target in ctn.iter(P + "spTgt")}
    found = []
    for sp in slide.iter(P + "sp"):
        named = sp.find(f"{P}nvSpPr/{P}cNvPr")
        shape_id = named.get("id") if named is not None else None
        found.append({"paragraphs": paragraphs(sp, A + "p", A + "t"), "enters": shape_id in entering})
    return found

def opened(name):
    path = Path.home() / "Documents" / name
    try:
        return zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        return f"{path} isn't there or isn't an Office file: {e}"

out = {}
memo = opened("quarterly-plan.docx")
out["memo"] = memo if isinstance(memo, str) else {
    "paragraphs": paragraphs(ET.fromstring(memo.read("word/document.xml")), W + "p", W + "t")}
deck = opened("quarterly-plan.pptx")
if isinstance(deck, str):
    out["deck"] = deck
else:
    slides = sorted((n for n in deck.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                    key=lambda n: int(re.search(r"\d+", n).group()))
    out["deck"] = {"slides": [shapes(ET.fromstring(deck.read(n))) for n in slides]}
json.dump(out, sys.stdout)
'''


async def verify(mcp_url, env_url):
    card = await protocol.get_card(env_url)
    ran = await protocol.invoke_extension(env_url, card, EXEC, {"argv": ["python3", "-c", READ],
                                                                "request_id": uuid.uuid4().hex}, method="run")
    found = json.loads(ran["stdout"]) if ran["exit_code"] == 0 else {}
    memo, deck = found.get("memo", ran.get("stderr", "")[-300:]), found.get("deck", "")
    paragraphs = memo.get("paragraphs", []) if isinstance(memo, dict) else []
    titled = "Quarterly Plan" in paragraphs
    under = paragraphs[paragraphs.index("Quarterly Plan") + 1:] if titled else []
    slides = deck.get("slides", []) if isinstance(deck, dict) else []
    texts = [[shape["paragraphs"] for shape in slide] for slide in slides]
    second = slides[1] if len(slides) == 2 else []
    lists = [shape for shape in second if len(shape["paragraphs"]) == 3 and "Next Steps" not in shape["paragraphs"]]
    titled_slides = len(slides) == 2 and any("Quarterly Plan" in shape["paragraphs"] for shape in slides[0]) and any(
        shape["paragraphs"] == ["Next Steps"] for shape in second)
    animated = any(shape["enters"] for shape in lists)
    return [
        {"id": "memo", "result": titled and len(under) >= 2,
         "description": f"the memo's paragraphs: {paragraphs}" if isinstance(memo, dict) else memo},
        {"id": "slides", "result": titled_slides and bool(lists),
         "description": f"the slides' shapes' text: {texts}" if isinstance(deck, dict) else deck},
        {"id": "animation", "result": animated,
         "description": "an entrance animation targets the second slide's list of three" if animated else
         "no entrance animation targets a list of three on the second slide"},
    ]
