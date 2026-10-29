# A task container with no Python, curl or CA certificates, working in a directory of its own.
FROM public.ecr.aws/docker/library/debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587
WORKDIR /workspace/task_repo
RUN echo notes-from-the-task > notes.txt
CMD ["tail", "-f", "/dev/null"]
