# The QEMU launcher: boots a Linux computer's disk form, mounted or copied at /disk/disk.img, with its settings in a
# NoCloud seed (see launch). It boots either arch's disk, with KVM where /dev/kvm is passed in and the arch is its own,
# behind socat on each forwarded port, and expands a zstd-compressed disk.
FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        qemu-system-x86 qemu-system-arm qemu-utils ovmf qemu-efi-aarch64 dosfstools mtools socat zstd \
    && rm -rf /var/lib/apt/lists/*
COPY launch seed /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/launch"]
