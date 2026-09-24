FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install Python dependencies
COPY slack_website/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy database layer (already in tst/data/)
COPY base_mcp/ ./base_mcp/
COPY slack_mcp/ ./slack_mcp/

# Copy backend application code
COPY slack_website/backend/app.py .
COPY slack_website/backend/config.py .
COPY slack_website/backend/db_extensions.py .
COPY slack_website/backend/schemas.py .
COPY slack_website/backend/api/ ./api/

# MCP_SERVER_PATH = dir containing database.py and models.py
ENV MCP_SERVER_PATH=/app/slack_mcp
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE ${PORT}
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
