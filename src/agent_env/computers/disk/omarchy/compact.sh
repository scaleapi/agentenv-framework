# compact.sh: run as root over ssh as an Omarchy disk is about to shut down to be packed, a base or a composed layer. It
# removes what the computer doesn't need, package and pip caches and the journal, then trims the free space, which the
# launcher's disk, attached with discards, passes to its image, so packing it keeps only what Omarchy uses.
rm -rf /var/cache/pacman/pkg/* /root/.cache/pip /home/*/.cache/pip
journalctl --vacuum-size=1M --quiet || true
fstrim -av
