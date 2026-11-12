"""Whether the agent saved the memo and the presentation the task asks for, in Office formats, read inside the computer
through the driver's computer-exec: the memo a .docx with its title and at least two paragraphs under it; the
presentation a .pptx of two slides titled as asked, the second with a list of three items and an entrance animation."""
import json
import uuid

from agentenv_protocol import client as protocol

EXEC = "urn:agentenv:computer-exec/v1"
#: Run in the computer: what each file holds, as JSON.
READ = r'''
import json, re, sys, zipfile
from pathlib import Path

def paragraphs(xml, paragraph, run):
    found = []
    for p in re.findall(rf"<{paragraph}\b.*?</{paragraph}>", xml, re.S):
        text = "".join(re.findall(rf"<{run}(?:\s[^>]*)?>([^<]*)</{run}>", p)).strip()
        if text:
            found.append(text)
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
    "paragraphs": paragraphs(memo.read("word/document.xml").decode(), "w:p", "w:t")}
deck = opened("quarterly-plan.pptx")
if isinstance(deck, str):
    out["deck"] = deck
else:
    slides = sorted((n for n in deck.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                    key=lambda n: int(re.search(r"\d+", n).group()))
    xml = [deck.read(n).decode() for n in slides]
    out["deck"] = {"slides": [paragraphs(s, "a:p", "a:t") for s in xml],
                   "entrance": ['presetClass="entr"' in s for s in xml]}
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
    second = slides[1] if len(slides) == 2 else []
    animated = isinstance(deck, dict) and len(slides) == 2 and deck["entrance"][1]
    return [
        {"id": "memo", "result": titled and len(under) >= 2,
         "description": f"the memo's paragraphs: {paragraphs}" if isinstance(memo, dict) else memo},
        {"id": "slides", "result": len(slides) == 2 and "Quarterly Plan" in slides[0] and second[:1] == ["Next Steps"]
         and len(second) >= 4, "description": f"the slides' text: {slides}" if isinstance(deck, dict) else deck},
        {"id": "animation", "result": animated,
         "description": "the second slide's list has an entrance animation" if animated else
         "the second slide has no entrance animation"},
    ]
