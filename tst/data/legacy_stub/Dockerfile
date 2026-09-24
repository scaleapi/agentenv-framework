FROM public.ecr.aws/docker/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
WORKDIR /app
RUN pip install --no-cache-dir starlette uvicorn
COPY server.py /app/server.py
ENV MCP_PORT=18765
EXPOSE 18765
CMD ["python", "server.py"]
