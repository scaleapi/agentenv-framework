"""Paths to fields of an entity's dict or toml: dotted keys, each with an optional ``[]`` for every element of a list
(``env_ids[]``, ``artifacts[].id``) or ``{}`` for every value of a table (``child_envs{}.sandbox_type``)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

PATH = re.compile(r"[A-Za-z_]\w*(\[\]|\{\})?(\.[A-Za-z_]\w*(\[\]|\{\})?)*")


def field_sites(data: dict, path: str) -> Iterator[tuple[str, dict | list, str | int]]:
    """Each place ``path`` holds a value that isn't None in ``data``: where, as a path with indexes, what holds it, and
    its key there. A last segment's ``[]`` or ``{}`` is each element of its list or value of its table."""
    *parents, leaf = path.split(".")
    key = leaf.removesuffix("[]").removesuffix("{}")
    for trail, holder in descend(data, parents, ()):
        value = holder.get(key) if isinstance(holder, dict) else None
        if value is None:
            continue
        where = ".".join((*trail, key))
        if leaf.endswith("[]"):
            items = enumerate(value) if isinstance(value, list) else ()
            yield from ((f"{where}[{i}]", value, i) for i, item in items if item is not None)
        elif leaf.endswith("{}"):
            items = value.items() if isinstance(value, dict) else ()
            yield from ((f"{where}.{name}", value, name) for name, item in items if item is not None)
        else:
            yield where, holder, key


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
