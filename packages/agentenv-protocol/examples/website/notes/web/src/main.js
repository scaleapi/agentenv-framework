import "./style.css";

let offsetMs = 0;
let filter = "all";
let renders = 0;
const saving = new Map();
const lastSent = new Map();
const landed = new Map();
const drafts = new Map();
const CLOCK_REFRESH_MS = 5000;

export function universeNow() {
  return new Date(Date.now() + offsetMs);
}

function followUniverse(response) {
  const at = Date.parse(response.headers.get("X-Universe-Now") || "");
  if (!Number.isNaN(at)) offsetMs = at - Date.now();
}

async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  followUniverse(response);
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(data?.error || `${method} ${path}: ${response.status}`);
  return data;
}

const $ = (id) => document.getElementById(id);

function setDraft(id, kind, value, saved) {
  const draft = { ...drafts.get(id) };
  if (value === saved) delete draft[kind];
  else draft[kind] = value;
  if (Object.keys(draft).length) drafts.set(id, draft);
  else drafts.delete(id);
}

function settleDraft(id, sent) {
  const draft = drafts.get(id);
  if (!draft) return;
  for (const kind of Object.keys(sent)) if (draft[kind] === sent[kind]) delete draft[kind];
  if (!Object.keys(draft).length) drafts.delete(id);
}

function queueSave(id, fields) {
  lastSent.set(id, fields);
  const queued = saving.get(id);
  if (queued) {
    queued.next = fields;
    return;
  }
  const state = { next: fields };
  saving.set(id, state);
  state.done = run(async () => {
    try {
      while (state.next) {
        const next = state.next;
        state.next = null;
        await api(`api/notes/${id}`, { method: "PATCH", body: next });
        settleDraft(id, next);
      }
      landed.set(id, renders);
    } catch (error) {
      state.error = error;
      lastSent.set(id, {});
      landed.delete(id);
      throw error;
    } finally {
      saving.delete(id);
    }
  });
}

function showError(error) {
  $("error").textContent = error ? error.message : "";
  $("error").hidden = !error;
}

function formatTime(value) {
  return new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short", timeZone: "UTC" });
}

function tick() {
  $("universe-now").textContent = `${formatTime(universeNow())} UTC`;
  $("universe-now").dateTime = universeNow().toISOString();
}

function noteItem(note) {
  const item = document.createElement("li");
  item.className = `note card${note.done ? " done" : ""}`;
  item.dataset.id = note.id;

  const done = document.createElement("input");
  done.type = "checkbox";
  done.checked = note.done;
  done.setAttribute("aria-label", `Mark "${note.title}" ${note.done ? "not done" : "done"}`);
  done.addEventListener("change", () => run(() => api(`api/notes/${note.id}/toggle`, { method: "POST" })));

  const draft = drafts.get(note.id) ?? {};
  const title = document.createElement("input");
  title.className = "title";
  title.defaultValue = note.title;
  title.value = draft.title ?? note.title;
  title.setAttribute("aria-label", "Title");

  const body = document.createElement("textarea");
  body.className = "body";
  body.rows = 2;
  body.defaultValue = note.body;
  body.value = draft.body ?? note.body;
  body.placeholder = "Details";
  body.setAttribute("aria-label", "Details");

  const save = () => {
    const sent = lastSent.get(note.id) ?? { title: note.title, body: note.body };
    if (title.value === sent.title && body.value === sent.body) return;
    queueSave(note.id, { title: title.value, body: body.value });
  };
  for (const [kind, field] of [["title", title], ["body", body]]) {
    field.addEventListener("input", () => setDraft(note.id, kind, field.value, note[kind]));
    field.addEventListener("change", save);
    field.addEventListener("blur", save);
  }

  const meta = document.createElement("small");
  meta.textContent = `Created ${formatTime(note.created_at)} · updated ${formatTime(note.updated_at)}`;

  const remove = document.createElement("button");
  remove.className = "delete";
  remove.textContent = "Delete";
  remove.addEventListener("click", () => run(async () => {
    await api(`api/notes/${note.id}`, { method: "DELETE" });
    drafts.delete(note.id);
  }));

  const text = document.createElement("div");
  text.className = "text";
  text.append(title, body, meta);
  item.append(done, text, remove);
  return item;
}

function focused() {
  const field = document.activeElement;
  if (!field?.matches?.("#notes .title, #notes .body")) return null;
  return { id: field.closest("li").dataset.id, kind: field.className, start: field.selectionStart,
           end: field.selectionEnd };
}

function refocus(kept) {
  if (!kept) return;
  const field = document.querySelector(`#notes li[data-id="${kept.id}"] .${kept.kind}`);
  if (!field) return;
  field.focus();
  field.setSelectionRange(kept.start, kept.end);
}

async function render() {
  const query = filter === "all" ? "" : `?done=${filter === "done"}`;
  const request = ++renders;
  const notes = await api(`api/notes${query}`);
  if (request !== renders) return;
  for (const [id, after] of landed) {
    if (request > after && !saving.has(id)) {
      lastSent.delete(id);
      landed.delete(id);
    }
  }
  const kept = focused();
  $("notes").replaceChildren(...notes.map(noteItem));
  refocus(kept);
  if (!notes.length) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = "No notes here yet.";
    $("notes").append(empty);
  }
}

async function run(action) {
  try {
    await action();
    showError(null);
    await render();
  } catch (error) {
    showError(error);
  }
}

async function start() {
  const [session, users] = await Promise.all([api("api/session"), api("api/users")]);
  $("user").replaceChildren(...users.map((user) => new Option(user.name || user.id, user.id)));
  let signedInAs = session.user ?? "";
  $("user").value = signedInAs;
  $("user").addEventListener("change", (event) => {
    const user = event.target.value;
    $("user").disabled = true;
    run(async () => {
      try {
        while (saving.size) {
          const pending = [...saving.values()];
          await Promise.all(pending.map((state) => state.done));
          const failed = pending.find((state) => state.error);
          if (failed) throw failed.error;
        }
        await api("api/sign-in", { method: "POST", body: { user } });
        signedInAs = user;
        drafts.clear();
      } finally {
        $("user").value = signedInAs;
        $("user").disabled = false;
      }
    });
  });

  const add = $("new-note").querySelector("button[type=submit]");
  $("new-note").addEventListener("submit", (event) => {
    event.preventDefault();
    if (add.disabled) return;
    const body = { title: $("new-title").value, body: $("new-body").value };
    add.disabled = true;
    run(async () => {
      try {
        await api("api/notes", { method: "POST", body });
      } finally {
        add.disabled = false;
      }
      if ($("new-title").value === body.title) $("new-title").value = "";
      if ($("new-body").value === body.body) $("new-body").value = "";
    });
  });

  for (const button of document.querySelectorAll("[data-filter]")) {
    button.addEventListener("click", () => {
      filter = button.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((b) => b.classList.toggle("active", b === button));
      run(async () => {});
    });
  }

  tick();
  setInterval(tick, 1000);
  setInterval(() => fetch("api/users").then(followUniverse).catch(() => {}), CLOCK_REFRESH_MS);
  await render();
}

start().catch(showError);
