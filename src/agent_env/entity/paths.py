"""Paths to fields of an entity's dict or toml: dotted keys, each with an optional ``[]`` for every element of a list
(``env_ids[]``, ``artifacts[].id``) or ``{}`` for every value of a table (``child_envs{}.sandbox_type``)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

PATH = re.compile(r"[A-Za-z_]\w*(\[\]|\{\})?(\.[A-Za-z_]\w*(\[\]|\{\})?)*")


def field_sites(data: dict, path: str) -> Iterator[tuple[str, dict, str]]:
    """Each place ``path`` holds a value that isn't None in ``data``: where, as a path with indexes, the dict holding
    it, and its key there."""
    *parents, leaf = path.split(".")
    for trail, holder in descend(data, parents, ()):
        if isinstance(holder, dict) and holder.get(leaf) is not None:
            yield ".".join((*trail, leaf)), holder, leaf


def descend(node: Any, segments: list[str], trail: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Each node ``segments`` reach under ``node``, with the path parts that reach it after ``trail``."""
    if not segments:
        yield trail, node
        return
    segment = segments[0]
    key = segment.removesuffix("[]").removesuffix("{}")
    child = node.get(key) if isinstance(node, dict) else None
    if child is None:
        return
    if segment.endswith("[]"):
        if isinstance(child, list):
            for i, item in enumerate(child):
                yield from descend(item, segments[1:], (*trail, f"{key}[{i}]"))
    elif segment.endswith("{}"):
        if isinstance(child, dict):
            for name, item in child.items():
                yield from descend(item, segments[1:], (*trail, key, str(name)))
    else:
        yield from descend(child, segments[1:], (*trail, key))
