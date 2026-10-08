"""Namespaced entity ids, and the object keys, image repositories and filenames derived from them.

An id opening ``@<namespace>/`` names the store it lives in; ``@local/…`` is the only namespace
today. Documents keep an id verbatim. Everything derived from an ``@local`` id uses one bounded
segment, ``local/<slug>-<sha256(id)[:12]>``; an object key or image repository takes a bare id over
200 bytes as ``<slug>-<sha256(id)[:12]>``; every other id passes through byte-identical.
The encoders never raise: they expect ids that ``validate_local_id`` accepts, and still return a
safe segment for any other input. An id may contain spaces, ``&`` and parentheses, so quote one
before it reaches a shell or a URL, and derive object keys, image repositories and filenames with
the helpers here rather than from the raw id.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Optional

LOCAL_NAMESPACE = "local"
LOCAL_PREFIX = f"@{LOCAL_NAMESPACE}/"
MAX_LOCAL_ID_BYTES = 4096
# A filesystem path component holds at most 255 bytes, so a bare id over this many gets a bounded segment.
MAX_BARE_SEGMENT_BYTES = 200

_SLUG_LENGTH = 48
_HASH_LENGTH = 12
_NON_SLUG_RUN = re.compile(r"[^a-z0-9]+")
_ENCODED_LOCAL_ID = re.compile(
    rf"{LOCAL_NAMESPACE}(?:/|-(?:[a-z0-9]+(?:-[a-z0-9]+)*-)?[0-9a-f]{{{_HASH_LENGTH}}}(?:/|$))"
)
_ID_PUNCTUATION = frozenset(" ._-~/@()+,&")
_ID_CATEGORIES = frozenset("LNM")
_INVISIBLE = re.compile(
    "[\u034f\u115f\u1160\u17b4\u17b5\u180b-\u180d\u180f\u3164\ufe00-\ufe0f\uffa0\U000e0100-\U000e01ef]"
)


def parse_namespace(entity_id: str) -> Optional[str]:
    """The namespace an ``@<namespace>/…`` id names, or None for a bare id."""
    if not entity_id.startswith("@"):
        return None
    return entity_id[1:].split("/", 1)[0]


def is_local_id(entity_id: str) -> bool:
    return entity_id.startswith(LOCAL_PREFIX)


def validate_local_id(entity_id: str) -> None:
    """Raise ValueError unless ``entity_id`` is ``@local/`` followed by non-empty path segments,
    none of them ``.`` or ``..`` or opening or closing with a space, made only of letters, digits,
    spaces and ``. _ - ~ / @ ( ) + , &``, and at most 4096 bytes of UTF-8 in all (Linux's PATH_MAX).
    Only the leading ``@`` names a namespace, so one inside a segment is an ordinary character.

    Letters include every script's, with its combining marks, as long as each mark follows a
    letter, digit or mark. The invisible (default-ignorable) letters and marks are refused."""
    if not entity_id.startswith(LOCAL_PREFIX):
        raise ValueError(f"local id {entity_id!r} must start with {LOCAL_PREFIX!r}")
    size = len(entity_id.encode("utf-8", "surrogatepass"))
    if size > MAX_LOCAL_ID_BYTES:
        raise ValueError(f"local id is {size} bytes; the limit is {MAX_LOCAL_ID_BYTES}: {entity_id[:64]!r}…")
    disallowed = sorted({c for c in entity_id[len(LOCAL_PREFIX):] if not _allowed_in_id(c)})
    if disallowed:
        raise ValueError(
            f"local id {entity_id!r} contains {ascii(''.join(disallowed))}; "
            "a local id may contain only letters, digits, spaces and . _ - ~ / @ ( ) + , &"
        )
    segments = entity_id[len(LOCAL_PREFIX):].split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError(f"local id {entity_id!r} has an empty, '.' or '..' path segment")
    if any(segment != segment.strip(" ") for segment in segments):
        raise ValueError(f"local id {entity_id!r} has a path segment that opens or closes with a space")
    if any(_detached_mark(segment) for segment in segments):
        raise ValueError(f"local id {entity_id!r} has a combining mark that does not follow a letter or digit")


def _allowed_in_id(character: str) -> bool:
    if _INVISIBLE.match(character):
        return False
    return character in _ID_PUNCTUATION or unicodedata.category(character)[0] in _ID_CATEGORIES


def _detached_mark(segment: str) -> bool:
    categories = [unicodedata.category(character)[0] for character in segment]
    return any(this == "M" and before not in _ID_CATEGORIES for before, this in zip(["Z", *categories], categories))


def _local_segment(entity_id: str) -> str:
    digest = hashlib.sha256(entity_id.encode("utf-8", "surrogatepass")).hexdigest()[:_HASH_LENGTH]
    decomposed = unicodedata.normalize("NFKD", entity_id.removeprefix(LOCAL_PREFIX))
    folded = "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().replace("~", "")
    slug = _NON_SLUG_RUN.sub("-", folded).strip("-")[:_SLUG_LENGTH].rstrip("-")
    return f"{LOCAL_NAMESPACE}/{slug}-{digest}" if slug else f"{LOCAL_NAMESPACE}/{digest}"


def aliases_local_encoding(entity_id: str) -> bool:
    """Whether a bare id is spelled like an ``@local`` id's encoding: ``local/…``, or the filename form
    ``local-<slug>-<hash>`` alone or opening a path. Its objects, images and files would take that id's."""
    return _ENCODED_LOCAL_ID.match(entity_id) is not None


def key_segment(entity_id: str) -> str:
    """The object-key segment for ``entity_id`` (``artifacts/<type>/<segment>/<version>/…``)."""
    if is_local_id(entity_id):
        return _local_segment(entity_id)
    if len(entity_id.encode("utf-8", "surrogatepass")) > MAX_BARE_SEGMENT_BYTES:
        return _local_segment(entity_id).removeprefix(f"{LOCAL_NAMESPACE}/")
    return entity_id


def image_repository(entity_id: str) -> str:
    """The image repository for ``entity_id``; an ``@local`` id yields a valid OCI repository path."""
    return key_segment(entity_id)


def local_image_repository(entity_id: str) -> str:
    """A repository for ``entity_id`` that is a valid OCI repository path and names no registry, whatever the id:
    ``local/<slug>-<hash>``, the name an image built where it runs is tagged with."""
    return _local_segment(entity_id)


def fs_safe(entity_id: str) -> str:
    """A filename for an ``@local`` id: its key segment with no ``/``. Any other id is returned
    unchanged, so it is only as filename-safe as it already was."""
    return _local_segment(entity_id).replace("/", "-", 1) if is_local_id(entity_id) else entity_id


# The longest chain derive_id makes on an authored skill or CLI id is its per-file artifact,
# ``{id}__files__<16 hex>``, so an authored id leaves room for those 25 bytes. An id built on a derived id
# (an env's ``{env}__cli``) or on a run's id can pass 4096 bytes; validate_local_id refuses it before anything
# reaches a shared store.
MAX_AUTHORED_LOCAL_ID_BYTES = MAX_LOCAL_ID_BYTES - len("__files__") - 16


def derive_id(base: str, suffix: str) -> str:
    """The id of an entity derived from ``base``: a suffix, so the base's namespace carries over."""
    return f"{base}__{suffix}"
