FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
WORKDIR /app
RUN pip install --no-cache-dir starlette pydantic httpx "mcp>=1.25,<2"
COPY agentenv_protocol /app/agentenv_protocol
COPY server.py /app/server.py
COPY seed.json /data/seed.json
ENV PYTHONPATH=/app
# Run as a non-root user rather than root.
RUN useradd --create-home --uid 10001 appuser
USER appuser
CMD ["python", "server.py"]
