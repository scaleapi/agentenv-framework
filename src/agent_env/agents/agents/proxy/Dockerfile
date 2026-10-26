FROM mirror.gcr.io/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
# The tools the steps that act on an agent's sandbox run: collect_artifacts, verify_sandbox, run_code, load_artifact.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates coreutils curl findutils git procps \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/agent
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# install.sh is install/v1's: in the image, so it stays in its build context.
COPY a2a_server.py install.sh ./
RUN useradd --create-home agent && mkdir -p /app && chown agent:agent /app
ENV A2A_PORT=8000 PYTHONPATH=/opt/agent
USER agent
# The actions the driver has it run work where the proxy does: in /app, as agent.
WORKDIR /app
EXPOSE 8000
CMD ["python", "/opt/agent/a2a_server.py"]
