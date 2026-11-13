# docker build -t openhands-a2a src/agent_env/agents/agents/openhands
FROM mirror.gcr.io/library/ubuntu:24.04@sha256:2198e4c65076e9cb0b3d695219df667c0e3b0072da64ca3c2648aca17dcecaf5
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
