"""Walks over JSON-like values: each string in one with its path, and the value under a path of keys."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any


def strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Each string in ``value``, with its path: ``a.b`` under a dict, ``a[0]`` in a list."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from strings(item, f"{path}[{index}]")


def value_at(value: Any, keys: Iterable[str | int]) -> Any:
    """What ``value`` holds under ``keys``: a dict's item by its key, a list's by its index; None where it holds none."""
    for key in keys:
        if isinstance(value, dict) and isinstance(key, str):
            value = value.get(key)
        elif isinstance(value, list) and isinstance(key, int) and 0 <= key < len(value):
            value = value[key]
        else:
            return None
    return value
