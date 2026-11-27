# docker build -t openhands-a2a src/agent_env/agents/agents/openhands
FROM mirror.gcr.io/library/ubuntu:24.04@sha256:562111b88caa79c71391c0c4d3c2114a7d5c35915c20561fd893cb1e07929e6b
WORKDIR /app
# Named sources: agent-env packs the build context install/v1 copies from the COPY sources.
COPY a2a_server.py conversation_run.py install.sh openhands_agent.py openhands_runtime.py requirements.txt start.sh ./
RUN sh install.sh \
    && useradd --create-home agent \
    && mkdir /workspace \
    && chown agent /app /workspace
ENV PATH="/opt/openhands-a2a-venv/bin:${PATH}" OPENHANDS_SUPPRESS_BANNER=1
USER agent
EXPOSE 8000
CMD ["python", "a2a_server.py"]
