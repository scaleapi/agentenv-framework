# A small root filesystem, for either arch, that the disk tests make disks of.
FROM public.ecr.aws/docker/library/busybox:1.37@sha256:a7dbc319b1438115a207e11930dc4bb3e643dcb3ec0bdd694bb36cd8fad289ec
ENV GREETING="hello disk"
USER nobody
WORKDIR /tmp
CMD ["sh", "-c", "echo $GREETING"]
