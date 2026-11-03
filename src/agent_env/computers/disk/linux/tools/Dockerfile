# The disk tools: what makes a Linux computer's disk form from a root filesystem without mounting or booting anything,
# so a build runs them unprivileged on its own platform, for either arch.
FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends e2fsprogs dosfstools mtools fdisk qemu-utils python3 \
    && rm -rf /var/lib/apt/lists/*
COPY disk image-env /usr/local/bin/
