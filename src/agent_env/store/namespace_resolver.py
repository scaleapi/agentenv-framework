"""Which configured namespace an entity id is in (``[namespace_resolver]``).

A configured namespace keeps its entities, its tasks' run records and objects in stores of its own
(``agent_env.store.routing``). The resolver says which ids are its: a deployment points it at whatever records
that, such as a service's list of restricted objects. An id derived from one of them (``<id>__…``) is the
resolver's to place too; ``id_bases`` lists what it was derived from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional

from agent_env.store.ids import id_bases


class NamespaceResolver(ABC):
    """Places entity ids in configured namespaces. ``fresh`` asks for an answer read now rather than a cached one:
    the stores ask so before a write, so an id placed a moment ago is never written outside its namespace."""

    @classmethod
    def from_config(cls, **config) -> NamespaceResolver:
        return cls(**config)

    @abstractmethod
    def namespace_of(self, entity_id: str, *, fresh: bool = False) -> Optional[str]:
        """The configured namespace ``entity_id`` is in, or None for the configured stores. Raises when it can't
        tell, so a write is refused rather than placed by a guess."""


class TableNamespaceResolver(NamespaceResolver):
    """Ids listed by namespace in config, ``ids = { <namespace> = [<id>, …] }``, each with every id derived from it."""

    def __init__(self, ids: Optional[Mapping[str, list[str]]] = None) -> None:
        self._placed = {entity_id: namespace for namespace, listed in (ids or {}).items() for entity_id in listed}

    def namespace_of(self, entity_id: str, *, fresh: bool = False) -> Optional[str]:
        return next((self._placed[base] for base in id_bases(entity_id) if base in self._placed), None)
