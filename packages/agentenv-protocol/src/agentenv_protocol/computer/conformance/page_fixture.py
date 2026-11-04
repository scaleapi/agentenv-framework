"""The page fixture the conformance probe stages with data/add and starts with computer-exec on a computer whose screen
is a page: a loopback server for the HTML Fixture Desk, the Tk desk's widgets at the same places, which reports their
boxes and every effect that reaches it here, written to <state-dir>/layout.json and <state-dir>/state.json as the Tk
desk writes them, and reads <state-dir>/inbox.txt back to it. Its port goes to <state-dir>/port. Run as
``python3 page_fixture.py <state-dir>``."""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

TITLE = "Fixture Desk"
TWO = "/two"
TWO_TITLE = "Fixture Desk two"
QUESTION = "The Fixture Desk asks for a word"

DESK = """<!doctype html>
<html><head><meta charset="utf-8"><title>Fixture Desk</title>
<style>
html, body { margin: 0; width: 100%; height: 100%; overflow: hidden; background: #f0f0f0;
             font: 16px "DejaVu Sans", sans-serif; }
.at { position: absolute; box-sizing: border-box; margin: 0; }
#name { left: 140px; top: 40px; width: 420px; height: 34px; font: inherit; }
#save, #ask { width: 120px; height: 44px; font: inherit; }
#save { left: 560px; top: 34px; }
#subscribe { left: 40px; top: 100px; width: 160px; height: 34px; display: flex; align-items: center; gap: 8px; }
#subscribe input { margin: 0 0 0 4px; width: 18px; height: 18px; }
#notes { left: 760px; top: 40px; width: 380px; height: 90px; font: 14px "DejaVu Sans", sans-serif; resize: none; }
#list { left: 40px; top: 160px; width: 300px; height: 400px; overflow-y: scroll; background: #fff;
        border: 1px solid #999; font: 14px "DejaVu Sans", sans-serif; }
#list div { height: 24px; line-height: 24px; padding: 0 6px; }
#list div.chosen { background: #3874d8; color: #fff; }
#grid { left: 400px; top: 160px; }
#pad { left: 840px; top: 160px; background: #fffbe6; }
#wheel { left: 840px; top: 500px; width: 200px; height: 120px; background: #e6f0ff; }
#field { left: 1080px; top: 500px; }
#ask { left: 40px; top: 600px; }
#file { left: 40px; top: 670px; width: 300px; height: 34px; }
</style></head>
<body>
<label class="at" style="left: 40px; top: 40px; line-height: 34px">Name:</label>
<input id="name" class="at" autocomplete="off" spellcheck="false">
<button id="save" class="at">Save</button>
<label id="subscribe" class="at"><input type="checkbox" id="subscribed">Subscribe</label>
<textarea id="notes" class="at" spellcheck="false"></textarea>
<div id="list" class="at"></div>
<canvas id="grid" class="at" width="400" height="400"></canvas>
<canvas id="pad" class="at" width="400" height="300"></canvas>
<div id="wheel" class="at"></div>
<canvas id="field" class="at" width="48" height="48"></canvas>
<button id="ask" class="at">Ask</button>
<input id="file" class="at" type="file">
<script>
const COLOURS = ["#ffffff", "#ff0000", "#00aa00", "#0000ff"], CELL = 50, GRID = 8;
const MARKERS = [[11, 11, "#ff0000"], [12, 11, "#00ff00"], [11, 12, "#0000ff"], [12, 12, "#ffff00"],
                 [37, 20, "#ff00ff"], [20, 37, "#00ffff"]];
const CLIPBOARD = "agentenv-probe clipboard", MIN = [1280, 800], QUESTION = "%QUESTION%";
const KEYSYMS = {" ": "space", ",": "comma", "!": "exclam", "Enter": "Return", "Backspace": "BackSpace"};
const $ = id => document.getElementById(id);
const name = $("name"), notes = $("notes"), list = $("list"), pad = $("pad");
const grid = $("grid").getContext("2d"), ink = pad.getContext("2d");
let state = {name: "", notes: "", keys: [], saved_count: 0, subscribed: false, selected: null, scroll_top: 0,
             cells: {}, right_clicks: 0, double_clicks: 0, drags: [], wheel: {up: 0, down: 0, left: 0, right: 0},
             clipboard: CLIPBOARD, loaded: null, screen: [innerWidth, innerHeight], pasted: [], prompts: 0,
             prompted: null, uploaded: null, loads: 0};
let sending = Promise.resolve(), stroke = null;
const post = (path, body) => { const sent = JSON.stringify(body);
  sending = sending.then(() => fetch(path, {method: "POST", body: sent})).catch(() => {}); };
const save = () => post("/state", state);

function paint(row, col) {
  grid.fillStyle = COLOURS[state.cells[`${row},${col}`] || 0];
  grid.fillRect(col * CELL, row * CELL, CELL, CELL);
  grid.strokeStyle = "#999999";
  grid.strokeRect(col * CELL + 0.5, row * CELL + 0.5, CELL - 1, CELL - 1);
}

function box(el) {
  const r = el.getBoundingClientRect();
  return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)];
}

function layout() {
  const boxes = {};
  for (const key of ["name", "save", "subscribe", "notes", "list", "grid", "pad", "wheel", "field", "ask", "file"])
    boxes[key] = box($(key));
  boxes.list_row_height = Math.round(list.firstChild.getBoundingClientRect().height);
  boxes.markers = MARKERS.map(([x, y, c]) => [boxes.field[0] + x, boxes.field[1] + y, c]);
  boxes.cell = CELL;
  boxes.clipboard = CLIPBOARD;
  post("/layout", boxes);
}

for (let i = 1; i <= 100; i++) {
  const row = document.createElement("div"); row.textContent = `Item ${i}`; list.append(row);
}
const field = $("field").getContext("2d");
field.fillStyle = "#000000";
field.fillRect(0, 0, 48, 48);
for (const [x, y, c] of MARKERS) { field.fillStyle = c; field.fillRect(x, y, 1, 1); }

for (const text of [name, notes]) {
  text.addEventListener("keydown", e => {
    if (e.ctrlKey && e.key === "/") { e.preventDefault(); text.select(); }
    else if (e.key === "Tab" && text === notes) {
      e.preventDefault();
      notes.setRangeText("\\t", notes.selectionStart, notes.selectionEnd, "end");
      state.notes = notes.value; save();
    }
  });
  text.addEventListener("paste", e => {
    state.pasted = [...state.pasted, e.clipboardData ? e.clipboardData.getData("text") : ""].slice(-10); save();
  });
}
name.addEventListener("input", () => { state.name = name.value; save(); });
name.addEventListener("keyup", e => {
  state.keys = [...state.keys, KEYSYMS[e.key] || e.key].slice(-100); state.name = name.value; save();
});
notes.addEventListener("input", () => { state.notes = notes.value; save(); });
$("save").addEventListener("click", () => { state.saved_count += 1; save(); });
$("subscribed").addEventListener("change", e => { state.subscribed = e.target.checked; save(); });
list.addEventListener("scroll", () => { state.scroll_top = list.scrollTop / list.scrollHeight; save(); });
list.addEventListener("mousedown", e => {
  if (e.target.parentNode !== list) return;
  for (const row of list.children) row.classList.toggle("chosen", row === e.target);
  state.selected = e.target.textContent; save();
});
$("grid").addEventListener("mousedown", e => {
  if (e.button !== 0 && e.button !== 2) return;
  const row = Math.floor(e.offsetY / CELL), col = Math.floor(e.offsetX / CELL), key = `${row},${col}`;
  state.cells[key] = ((state.cells[key] || 0) + 1) % COLOURS.length;
  state.right_clicks += e.button === 2 ? 1 : 0;
  paint(row, col); save();
});
$("grid").addEventListener("dblclick", () => { state.double_clicks += 1; save(); });
addEventListener("contextmenu", e => e.preventDefault());

const on = e => {
  const r = pad.getBoundingClientRect(); return [Math.round(e.clientX - r.left), Math.round(e.clientY - r.top)];
};
pad.addEventListener("mousedown", e => { if (e.button === 0) stroke = [on(e)]; });
addEventListener("mousemove", e => {
  if (!stroke) return;
  const [x0, y0] = stroke[stroke.length - 1], [x1, y1] = on(e);
  ink.lineWidth = 3; ink.beginPath(); ink.moveTo(x0, y0); ink.lineTo(x1, y1); ink.stroke();
  stroke.push([x1, y1]);
});
addEventListener("mouseup", e => {
  if (!stroke) return;
  stroke.push(on(e));
  const step = Math.max(1, Math.floor(stroke.length / 50)), last = stroke[stroke.length - 1];
  const path = [...stroke.filter((_, i) => i % step === 0), last];
  const drag = {start: stroke[0], end: last, points: stroke.length, path};
  state.drags = [...state.drags, drag].slice(-20); stroke = null; save();
});

$("wheel").addEventListener("wheel", e => {
  e.preventDefault();
  if (e.deltaY) state.wheel[e.deltaY < 0 ? "up" : "down"] += 1;
  if (e.deltaX) state.wheel[e.deltaX < 0 ? "left" : "right"] += 1;
  save();
}, {passive: false});

$("ask").addEventListener("click", () => {
  const answer = prompt(QUESTION, "");
  state.prompts += 1; state.prompted = answer; save();
});
$("file").addEventListener("change", async e => {
  const chosen = e.target.files[0];
  if (chosen) { state.uploaded = {name: chosen.name, content: await chosen.text()}; save(); }
});

setInterval(async () => {
  try {
    const answer = await fetch("/inbox", {cache: "no-store"});
    if (answer.ok) { const text = await answer.text(); if (text !== state.loaded) { state.loaded = text; save(); } }
  } catch {}
}, 200);

(async () => {
  if (innerWidth < MIN[0] || innerHeight < MIN[1]) {
    state.error = `the screen is ${innerWidth}x${innerHeight}; the fixture needs ${MIN[0]}x${MIN[1]}`;
    return save();
  }
  try { const kept = await (await fetch("/state", {cache: "no-store"})).json(); if (kept) state = kept; } catch {}
  state.loads += 1;
  state.screen = [innerWidth, innerHeight];
  name.value = state.name; notes.value = state.notes; $("subscribed").checked = state.subscribed;
  for (let row = 0; row < GRID; row++) for (let col = 0; col < GRID; col++) paint(row, col);
  save();
  requestAnimationFrame(() => setTimeout(layout, 0));
})();
</script>
</body></html>
""".replace("%QUESTION%", QUESTION)

PAGE_TWO = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{TWO_TITLE}</title></head>
<body style="margin: 0; background: #e6f0ff">
<p style="margin: 40px; font: 24px sans-serif">{TWO_TITLE}</p></body></html>
"""


class Desk(BaseHTTPRequestHandler):
    state_dir: Path

    def do_GET(self) -> None:
        if self.path == "/":
            return self._send(200, DESK.encode(), "text/html; charset=utf-8")
        if self.path == TWO:
            return self._send(200, PAGE_TWO.encode(), "text/html; charset=utf-8")
        if self.path == "/state":
            kept = self.state_dir / "state.json"
            return self._send(200, kept.read_bytes() if kept.exists() else b"null", "application/json")
        message = self.state_dir / "inbox.txt"
        if self.path == "/inbox" and message.exists():
            return self._send(200, message.read_bytes(), "text/plain; charset=utf-8")
        self._send(404, b"", "text/plain")

    def do_POST(self) -> None:
        name = {"/state": "state.json", "/layout": "layout.json"}.get(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if name is None:
            return self._send(404, b"", "text/plain")
        json.loads(body)
        part = self.state_dir / f".{name}.part"
        part.write_bytes(body)
        os.replace(part, self.state_dir / name)
        self._send(204, b"", "text/plain")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def main(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    Desk.state_dir = state_dir
    server = ThreadingHTTPServer(("127.0.0.1", 0), Desk)
    part = state_dir / ".port.part"
    part.write_text(str(server.server_address[1]))
    os.replace(part, state_dir / "port")
    server.serve_forever()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
