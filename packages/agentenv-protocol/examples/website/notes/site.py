"""Notes: the reference site of ``urn:agentenv:website/v1``, a small notes app on WebsiteEnvironment. Each user keeps
notes they create, edit, mark done and delete; a browser with no session is signed in as the acting user, the seed's
``current_user``, and can sign in as another seeded user. The site keeps its state in the default store, which the data
plane loads, exports and resets, and stamps each note with the universe's time. ``admin`` callers, as the gateway's
``AgentEnv-Role`` names them, can list every user's notes.

Run it with ``python site.py`` after building ``web/`` into ``web/dist``; it serves on ``PORT`` (8000)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from agentenv_protocol import environment_card
from agentenv_protocol.website import (
    ApiContext,
    BadRequest,
    NotFound,
    Unauthorized,
    WebsiteEnvironment,
    api,
    ready_check,
    website,
)
from agentenv_protocol.website.clock import iso

NOTES = "notes"
USERS = "users"


class NewNote(BaseModel):
    title: str
    body: str = ""


class NoteEdit(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    done: Optional[bool] = None


class SignIn(BaseModel):
    user: str


@environment_card(name="notes")
@website(domains=["notes.test"], static="web/dist", health="/healthz", data="data/seed.json")
class Notes(WebsiteEnvironment):
    @ready_check
    def seeded(self) -> bool:
        """The site is usable once its store knows who may sign in."""
        return bool(self.store[USERS])

    @api.get("/api/users", name="list_users")
    def list_users(self) -> list:
        """The users a browser can sign in as."""
        return self.store[USERS]

    @api.post("/api/sign-in", name="sign_in")
    def sign_in(self, body: SignIn, ctx: ApiContext) -> dict:
        """Sign the browser in as one of the site's users."""
        if self.store.get(USERS, body.user) is None:
            raise NotFound(f"no user {body.user!r}")
        ctx.sign_in(body.user)
        return {"user": ctx.user}

    @api.get("/api/notes", name="list_notes")
    def list_notes(self, ctx: ApiContext, done: Optional[bool] = None) -> list:
        """The signed-in user's notes, newest first; with ``done``, only those done or not."""
        match = {"owner": self._user(ctx)} if done is None else {"owner": self._user(ctx), "done": done}
        return sorted(self.store.find(NOTES, **match), key=lambda note: note["created_at"], reverse=True)

    @api.post("/api/notes", name="create_note", status=201)
    def create_note(self, note: NewNote, ctx: ApiContext) -> dict:
        """Add a note for the signed-in user, not done, created and updated now."""
        now = iso(self.now())
        return self.store.insert(NOTES, {"owner": self._user(ctx), "title": _title(note.title), "body": note.body,
                                         "done": False, "created_at": now, "updated_at": now})

    @api.get("/api/notes/{note_id:int}", name="get_note")
    def get_note(self, note_id: int, ctx: ApiContext) -> dict:
        return self._note(note_id, ctx)

    @api.patch("/api/notes/{note_id:int}", name="edit_note")
    def edit_note(self, note_id: int, edit: NoteEdit, ctx: ApiContext) -> dict:
        """Change a note's title, body or done, and its ``updated_at``."""
        note = self._note(note_id, ctx)
        fields = edit.model_dump(exclude_none=True)
        if "title" in fields:
            fields["title"] = _title(fields["title"])
        note.update(fields, updated_at=iso(self.now()))
        return note

    @api.post("/api/notes/{note_id:int}/toggle", name="toggle_note")
    def toggle_note(self, note_id: int, ctx: ApiContext) -> dict:
        """Mark a note done, or not done when it was."""
        note = self._note(note_id, ctx)
        note.update(done=not note["done"], updated_at=iso(self.now()))
        return note

    @api.delete("/api/notes/{note_id:int}", name="delete_note")
    def delete_note(self, note_id: int, ctx: ApiContext) -> dict:
        self.store.delete(NOTES, self._note(note_id, ctx)["id"])
        return {"deleted": note_id}

    @api.get("/api/admin/notes", name="all_notes", roles=["admin"])
    def all_notes(self) -> list:
        """Every user's notes, for an ``admin`` caller only."""
        return self.store[NOTES]

    def _user(self, ctx: ApiContext) -> str:
        if ctx.user is None:
            raise Unauthorized("sign in first")
        return ctx.user

    def _note(self, note_id: int, ctx: ApiContext) -> dict:
        """The signed-in user's note ``note_id``; another user's is as missing as one that doesn't exist."""
        note = self.store.get(NOTES, note_id)
        if note is None or note["owner"] != self._user(ctx):
            raise NotFound(f"no note {note_id}")
        return note


def _title(title: str) -> str:
    if not title.strip():
        raise BadRequest("a note needs a title", code="empty_title")
    return title.strip()


if __name__ == "__main__":
    Notes().serve()
