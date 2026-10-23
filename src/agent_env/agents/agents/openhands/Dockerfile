# docker build -t openhands-a2a src/agent_env/agents/agents/openhands
FROM mirror.gcr.io/library/ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac
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
