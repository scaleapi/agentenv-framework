# check=skip=InvalidDefaultArgInFrom
# A Linux computer's disk form for one arch: a GPT disk (qcow2) of an EFI system partition holding one UKI, Ubuntu's
# kernel with a busybox initramfs whose init starts the computer, and an ext4 root filesystem that is ROOTFS, a container
# image's root filesystem, plus /etc/agentenv/vm/image.env, its start, from ROOTFS_CONFIG (its OCI config's config
# object, as base64'd JSON). The EFI system partition comes from ESP: the uki stage, on a disk of SIZE_GB, or a disk
# form's image, whose partition and size the disk keeps. Only the uki stage runs the target's programs: the disk itself
# is made on the build's own platform by TOOLS, the disk tools image, so an existing ESP needs no emulation.
ARG ROOTFS
ARG TOOLS
ARG ESP=uki

FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:24737cef17f4e22b1977df7d9a05e77c4e28b7da5b8911f8d8ccda136bf3573c AS kernel
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        linux-image-virtual busybox-static kmod zstd cpio systemd-boot-efi systemd-ukify python3-pefile \
    && rm -rf /var/lib/apt/lists/*
COPY init udhcpc.script uki.sh /build/
RUN sh /build/uki.sh /uki

FROM scratch AS uki
COPY --from=kernel /uki/ /

FROM --platform=$BUILDPLATFORM ${TOOLS} AS start
ARG ROOTFS_CONFIG
RUN printf %s "$ROOTFS_CONFIG" | base64 -d > /config.json && image-env /config.json > /image.env

FROM ${ROOTFS} AS root
COPY --from=start /image.env /etc/agentenv/vm/image.env

FROM ${ESP} AS esp

FROM --platform=$BUILDPLATFORM ${TOOLS} AS disk
ARG TARGETARCH
ARG SIZE_GB=16
RUN --mount=type=bind,from=root,target=/rootfs --mount=type=bind,from=esp,target=/esp \
    disk "$TARGETARCH" /rootfs /esp /out/disk.img "$SIZE_GB"

FROM scratch
COPY --from=disk --chown=107:107 /out/disk.img /disk/disk.img
