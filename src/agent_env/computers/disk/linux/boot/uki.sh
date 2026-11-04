#!/bin/sh
# uki.sh OUT: the one file the disk form's EFI system partition holds, written to OUT under the name UEFI firmware boots
# from removable media: a UKI of Ubuntu's kernel and an initramfs of busybox, init with its seed-env, and the modules a
# virtio guest needs that the kernel doesn't build in, decompressed, since busybox's insmod reads no zstd.
set -eu
out=$1
kver=$(ls /lib/modules | sort -V | tail -1)
case "$(dpkg --print-architecture)" in
    amd64) efi=BOOTX64.EFI stub=linuxx64.efi.stub console=ttyS0 ;;
    arm64) efi=BOOTAA64.EFI stub=linuxaa64.efi.stub console=ttyAMA0 ;;
esac
root=/tmp/initramfs
mkdir -p "$root/bin" "$root/dev" "$root/proc" "$root/sys" "$root/newroot" "$root/seed" "$root/lib/modules" "$out"
cp /bin/busybox "$root/bin/busybox"
cp /build/init /build/seed-env /build/udhcpc.script "$root/"
chmod 755 "$root/init" "$root/udhcpc.script"
: > "$root/modules.order"
for module in virtio_pci virtio_mmio virtio_blk virtio_scsi virtio_net virtio_rng virtio_console ext4 vfat \
              nls_cp437 nls_iso8859-1 nls_utf8 af_packet; do
    modprobe --show-depends -S "$kver" "$module" 2>/dev/null | while read -r kind path _; do
        [ "$kind" = insmod ] || continue
        name=$(basename "$path" .zst)
        grep -qx "/lib/modules/$name" "$root/modules.order" && continue
        case "$path" in
            *.zst) zstd -qdc "$path" > "$root/lib/modules/$name" ;;
            *) cp "$path" "$root/lib/modules/$name" ;;
        esac
        echo "/lib/modules/$name" >> "$root/modules.order"
    done
done
(cd "$root" && find . | sort | cpio -o -H newc --quiet | gzip -9n) > /tmp/initrd.img
# arm64's kernel is a gzipped Image, which the stub can't start; the Image itself is a PE the firmware can.
if [ "$(head -c2 "/boot/vmlinuz-$kver" | od -An -tx1 | tr -d ' ')" = 1f8b ]; then
    zcat "/boot/vmlinuz-$kver" > /tmp/vmlinuz
else
    cp "/boot/vmlinuz-$kver" /tmp/vmlinuz
fi
ukify build --uname "$kver" --linux /tmp/vmlinuz --initrd /tmp/initrd.img \
    --cmdline "console=$console panic=1 quiet loglevel=4" --stub "/usr/lib/systemd/boot/efi/$stub" --output "$out/$efi"
