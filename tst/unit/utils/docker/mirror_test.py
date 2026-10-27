"""The Docker Official Images a Dockerfile builds from, and the Dockerfile naming mirror.gcr.io's copies instead."""

from agent_env.utils.docker.mirror import official_bases, with_bases

DIGEST = "sha256:" + "a" * 64
DOCKERFILE = f"""# syntax=docker/dockerfile:1
FROM public.ecr.aws/docker/library/node:22-slim@{DIGEST} AS node
FROM --platform=linux/amd64 python:3.11-slim AS python
FROM python
FROM node AS again
FROM ghcr.io/astral-sh/uv:python3.12 AS uv
FROM docker.io/library/alpine:3.22
FROM library/debian
FROM scratch
FROM ${{BASE}}
FROM myorg/app:1
FROM \\
  ubuntu:24.04
RUN echo FROM python
"""


def test_each_official_image_a_stage_builds_from_maps_to_its_mirror_copy_by_digest_when_pinned():
    assert official_bases(DOCKERFILE) == {
        f"public.ecr.aws/docker/library/node:22-slim@{DIGEST}": f"mirror.gcr.io/library/node@{DIGEST}",
        "python:3.11-slim": "mirror.gcr.io/library/python:3.11-slim",
        "docker.io/library/alpine:3.22": "mirror.gcr.io/library/alpine:3.22",
        "library/debian": "mirror.gcr.io/library/debian",
        "ubuntu:24.04": "mirror.gcr.io/library/ubuntu:24.04",
    }


def test_only_the_froms_given_name_their_mirror_copy_and_the_rest_is_unchanged():
    node = f"public.ecr.aws/docker/library/node:22-slim@{DIGEST}"

    rewritten = with_bases(DOCKERFILE, {node: f"mirror.gcr.io/library/node@{DIGEST}",
                                        "ubuntu:24.04": "mirror.gcr.io/library/ubuntu:24.04"})

    assert rewritten == (DOCKERFILE.replace(node, f"mirror.gcr.io/library/node@{DIGEST}")
                         .replace("  ubuntu:24.04", "  mirror.gcr.io/library/ubuntu:24.04"))
    assert with_bases(DOCKERFILE, {}) == DOCKERFILE


def test_a_from_naming_a_stage_that_took_an_images_name_builds_from_that_stage():
    dockerfile = "FROM alpine AS initial\nFROM busybox AS alpine\nRUN touch /stage\nFROM alpine\n"

    bases = official_bases(dockerfile)

    assert bases == {"alpine": "mirror.gcr.io/library/alpine", "busybox": "mirror.gcr.io/library/busybox"}
    assert with_bases(dockerfile, bases) == ("FROM mirror.gcr.io/library/alpine AS initial\n"
                                             "FROM mirror.gcr.io/library/busybox AS alpine\nRUN touch /stage\n"
                                             "FROM alpine\n")
