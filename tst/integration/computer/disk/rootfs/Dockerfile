# A small root filesystem, for either arch, that the disk tests make disks of. From mirror.gcr.io, which serves Docker
# Hub's busybox under the same digest for both arches, since the fast tier pulls its bases straight from the registry
# and ECR Public rate-limits the runners' shared IPs.
FROM mirror.gcr.io/library/busybox:1.37@sha256:a7dbc319b1438115a207e11930dc4bb3e643dcb3ec0bdd694bb36cd8fad289ec
ENV GREETING="hello disk"
USER nobody
WORKDIR /tmp
CMD ["sh", "-c", "echo $GREETING"]
