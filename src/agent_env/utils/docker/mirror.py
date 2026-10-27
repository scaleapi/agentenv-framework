"""The Docker Official Images a Dockerfile builds from, and the copy of each mirror.gcr.io serves."""

from __future__ import annotations

import re

from agent_env.utils.docker.dockerfile import instructions, leading_flags

MIRROR = "mirror.gcr.io/library/"
# A Docker Official Image as a FROM names it: in Public ECR's copy of Docker Hub's library, or on Docker Hub.
_OFFICIAL = re.compile(
    r"(?:public\.ecr\.aws/docker/library/|(?:(?:index\.)?docker\.io/)?(?:library/)?)"
    r"(?P<name>[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)(?::(?P<tag>\w[\w.-]{0,127}))?(?:@(?P<digest>sha256:[0-9a-f]{64}))?"
)


def official_bases(dockerfile: str) -> dict[str, str]:
    """Each Docker Official Image a FROM in ``dockerfile`` names, and the same image on mirror.gcr.io, by its digest
    when the FROM pins one; a FROM naming an earlier stage or a build arg names none."""
    bases, stages = {}, set()
    for part in instructions(dockerfile):
        if part.keyword != "FROM" or not (words := leading_flags(part.arguments)[1].split()):
            continue
        image = words[0]
        if image.lower() not in stages and image != "scratch" and (official := _OFFICIAL.fullmatch(image)):
            name, tag, digest = official.group("name", "tag", "digest")
            bases[image] = f"{MIRROR}{name}@{digest}" if digest else f"{MIRROR}{name}{f':{tag}' if tag else ''}"
        if len(words) == 3 and words[1].lower() == "as":
            stages.add(words[2].lower())
    return bases


def with_bases(dockerfile: str, bases: dict[str, str]) -> str:
    """``dockerfile`` with each FROM that names an image in ``bases`` naming the one ``bases`` maps it to."""
    texts = []
    for part in instructions(dockerfile):
        words = leading_flags(part.arguments)[1].split() if part.keyword == "FROM" else []
        if words and (base := bases.get(words[0])):
            texts.append(re.sub(rf"(?<=\s){re.escape(words[0])}(?=\s|$)", lambda _: base, part.text, count=1))
        else:
            texts.append(part.text)
    return "".join(texts)
