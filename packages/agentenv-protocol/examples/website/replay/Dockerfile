ARG REPLAY=packages/agentenv-protocol/examples/website/replay

FROM mirror.gcr.io/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
ARG REPLAY
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY packages/agentenv-protocol /tmp/agentenv-protocol
RUN pip install --no-cache-dir /tmp/agentenv-protocol "uvicorn>=0.30" && rm -rf /tmp/agentenv-protocol
COPY ${REPLAY}/site.py ${REPLAY}/capture.warc.gz /app/
ENV PORT=8000
EXPOSE 8000
RUN useradd --create-home --uid 10001 appuser
USER appuser
CMD ["python", "site.py"]
