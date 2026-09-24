"""Universe — abstract base for artifacts that bundle other artifacts together."""

from __future__ import annotations

from typing import ClassVar

from agent_env.artifact.artifact import Artifact


class Universe(Artifact):
    """Abstract base for 'universe' artifacts that bundle other artifacts.

    Cannot be instantiated directly — use a concrete subclass such as
    ``EnvironmentUniverseArtifact`` or ``FileArtifactUniverse``. A universe is
    itself stored in the artifacts collection (so it benefits from the
    existing versioning / query / registry machinery); its payload is a set
    of references (IDs) to other artifacts.
    """

    # Marker flipped to False on any subclass that declares a concrete
    # ``type`` literal default (e.g. ``type: Literal["environment_universe"] = "environment_universe"``).
    __is_abstract__: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # A subclass is concrete if its ``type`` field has a non-empty default.
        type_field = cls.model_fields.get("type")
        if type_field is not None and type_field.default not in (None, ""):
            cls.__is_abstract__ = False

    def __init__(self, **data):
        if self.__class__.__is_abstract__:
            raise TypeError(
                f"{self.__class__.__name__} is abstract and cannot be instantiated; "
                "use a concrete Universe subclass (e.g. EnvironmentUniverseArtifact, "
                "FileArtifactUniverse)."
            )
        super().__init__(**data)
