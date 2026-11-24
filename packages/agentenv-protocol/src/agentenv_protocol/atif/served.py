"""What an atif/1 final read serves beside a task's document, so the task can be replayed offline: ``turn_context``,
the task's TurnContext and how it ended, and ``aux``, its aux log. Both are JSON on the wire, and these models are
their shape; ``RecordedTask.from_served`` turns them, with the task's native records, back into a task ``replay``
takes."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Mapping, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from .builder import Media, TurnContext, timestamp


class _Served(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def wire(self) -> dict[str, Any]:
        """As JSON on the wire: an optional field left at its default is left out, a required one is kept even when
        null."""
        return self.model_dump(mode="json", exclude_defaults=True)


class ServedMedia(_Served):
    """A media file content names by its path, ``media/<sha256>.<ext>``; its bytes are stored apart."""

    path: str
    media_type: str
    info: dict[str, Any]


class MediaItem(_Served):
    """A content item that is a media file."""

    media: ServedMedia


#: A content item: text, an ATIF content part, or a media file.
ContentItem = Annotated[Union[str, MediaItem, dict[str, Any]], Field(union_mode="left_to_right")]


class ServedData(_Served):
    """A final response's diagnostic DataPart, as its last step's ``data`` holds it."""

    data: Any
    metadata: dict[str, Any] | None = None


class ServedError(_Served):
    """A task's error, as its last step's ``error`` holds it."""

    kind: str
    message: str
    code: str | None = None


class AuxEntry(_Served):
    """One record of a task's aux log, in the order it was fed: its aux file, the record, and how many of the task's
    native records were fed before it."""

    name: str
    record: Any
    after: int = Field(ge=0)


class ServedTurn(_Served):
    """A task's TurnContext and how it ended: its state, final response, diagnostic DataParts and error."""

    index: int = Field(ge=0)
    context_id: str
    task_id: str
    message_id: str | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    agent_model: str | None = None
    started_at: str | None = None
    inbound: list[ContentItem] = Field(default_factory=list)
    state: str
    final: list[ContentItem] | None = None
    ended_at: str | None = None
    data: list[ServedData] | None = None
    error: ServedError | None = None

    @classmethod
    def of(cls, context: TurnContext, *, state: str, final: Sequence[Any] | None = None,
           ended_at: str | datetime | None = None, data: Sequence[Mapping[str, Any]] | None = None,
           error: Mapping[str, Any] | None = None) -> ServedTurn:
        """The served form of a task the runner was given ``context`` for and ended as these say."""
        return cls(
            index=context.index, context_id=context.context_id, task_id=context.task_id, message_id=context.message_id,
            agent_name=context.agent_name, agent_version=context.agent_version, agent_model=context.agent_model,
            started_at=timestamp(context.started_at), inbound=_items(context.inbound), state=state,
            final=None if final is None else _items(final), ended_at=timestamp(ended_at),
            data=None if data is None else [ServedData.model_validate(dict(entry)) for entry in data],
            error=None if error is None else ServedError.model_validate(dict(error)))

    def turn_context(self) -> TurnContext:
        return TurnContext(index=self.index, context_id=self.context_id, task_id=self.task_id,
                           message_id=self.message_id, inbound=tuple(_content(self.inbound)),
                           agent_name=self.agent_name, agent_version=self.agent_version, agent_model=self.agent_model,
                           started_at=self.started_at)

    def final_content(self) -> list[Any] | None:
        return None if self.final is None else _content(self.final)


def _items(content: Sequence[Any]) -> list[Any]:
    return [MediaItem(media=ServedMedia(path=item.path, media_type=item.media_type, info=dict(item.info)))
            if isinstance(item, Media) else item for item in content]


def _content(items: Sequence[Any]) -> list[Any]:
    return [Media(item.media.path, item.media.media_type, item.media.info) if isinstance(item, MediaItem) else item
            for item in items]
