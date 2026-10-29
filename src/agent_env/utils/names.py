"""The names a gateway deploy writes into its compose file and shell lines: an environment name becomes a compose service,
a container hostname and a schema, an image reference an ``image:`` line, so neither may carry anything else."""

from __future__ import annotations

import re

# Compose's service-name characters, with a letter or digit first so no name reads as a command-line option.
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")

# Docker's reference grammar (distribution/reference): [registry/]path[:tag][@digest], without its bracketed IPv6
# registries, which an unquoted compose value reads as the start of a YAML list.
_DOMAIN_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_DOMAIN = rf"{_DOMAIN_LABEL}(?:\.{_DOMAIN_LABEL})*(?::[0-9]+)?"
_PATH_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_TAG = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"
_DIGEST = r"[A-Za-z][A-Za-z0-9]*(?:[-_+.][A-Za-z][A-Za-z0-9]*)*:[0-9A-Fa-f]{32,}"
_IMAGE_REFERENCE = re.compile(
    rf"(?:{_DOMAIN}/)?{_PATH_COMPONENT}(?:/{_PATH_COMPONENT})*(?::{_TAG})?(?:@{_DIGEST})?"
)


def environment_name_problem(name: str) -> str | None:
    """Why ``name`` can't be an environment name, None when it can."""
    if isinstance(name, str) and _ENVIRONMENT_NAME.fullmatch(name):
        return None
    return (f"environment name {name!r} must start with a letter or digit and hold only letters, digits, "
            f"'_', '.' and '-'")


def image_reference_problem(ref: str) -> str | None:
    """Why ``ref`` isn't an image reference Docker takes, None when it is."""
    if isinstance(ref, str) and _IMAGE_REFERENCE.fullmatch(ref):
        return None
    return f"image name {ref!r} isn't an image reference ([registry/]repository[:tag][@digest])"
