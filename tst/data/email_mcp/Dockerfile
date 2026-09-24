FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
WORKDIR /app
COPY email_mcp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY base_mcp/ ./base_mcp/
COPY email_mcp/ ./email_mcp/
ENV PYTHONPATH=/app
WORKDIR /app/email_mcp
CMD ["python", "email_service.py"]
