# A musl task container, with busybox's shell and no bash or Python.
FROM public.ecr.aws/docker/library/alpine:3.22@sha256:5291449c3df73caf6ed85e649dec1b9e818b39a5d8c871e97afc13e9cd5e8fa8
WORKDIR /workspace/task_repo
RUN echo notes-from-the-task > notes.txt
CMD ["tail", "-f", "/dev/null"]
