"""A stand-in for vLLM served with ``--return-tokens-as-token-ids`` and processed logprobs, as OpenEnv's capture proxy
needs its engine: it lists the models it serves and answers chat completions with token ids, a logprob per completion
token and the top two at each, scaled by the temperature as processed logprobs are, or streamed, as an agent's own
client asks for them. ``reply`` makes each answer's message from the request body."""

import json
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

Reply = Callable[[dict[str, Any]], dict[str, Any]]


def says(content: str) -> Reply:
    return lambda body: {"role": "assistant", "content": content}


def shell_then_says(command: str, content: str) -> Reply:
    """Calls the request's shell tool (``bash`` in any case) with ``command`` until a tool result follows the last user
    message, then says ``content``; a request offering no shell gets ``content`` at once, as a harness's side calls
    do."""

    def reply(body: dict[str, Any]) -> dict[str, Any]:
        messages = body["messages"]
        shell = next((tool["function"]["name"] for tool in body.get("tools") or []
                      if tool["function"]["name"].lower() == "bash"), None)
        last_user = max((i for i, message in enumerate(messages) if message.get("role") == "user"), default=-1)
        if shell is None or any(message.get("role") == "tool" for message in messages[last_user + 1:]):
            return {"role": "assistant", "content": content}
        arguments = {"command": command, "description": "Run the command"}
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
             "function": {"name": shell, "arguments": json.dumps(arguments)}}]}

    return reply


def token_ids(message: dict[str, Any]) -> list[int]:
    """A message's tokens: one per character of its role, content and tool calls."""
    text = f"<{message.get('role')}>{json.dumps(message.get('content'))}{json.dumps(message.get('tool_calls') or [])}"
    return [ord(char) % 50000 + 1 for char in text]


@contextmanager
def fake_vllm(model: str, reply: Reply, host: str = "127.0.0.1") -> Iterator[SimpleNamespace]:
    """The engine, listening on ``host``, every interface for a client in a container, at ``.url`` on loopback;
    ``.requests`` are the completion bodies it got, and ``.served`` the models it lists, which a test appends to as vLLM
    lists a LoRA adapter once it's loaded."""
    requests: list[dict[str, Any]] = []
    served = [model]

    async def models(request):
        return JSONResponse({"data": [{"id": name, "object": "model"} for name in served]})

    async def completions(request):
        body = await request.json()
        requests.append(body)
        temperature = body.get("temperature") or 1.0
        message = reply(body)
        prompt = [token for each in body["messages"] for token in token_ids(each)]
        completion = token_ids(message)
        content = [{"token": f"token_id:{token}", "logprob": -0.5 / temperature, "bytes": None,
                    "top_logprobs": [{"token": f"token_id:{token}", "logprob": -0.5 / temperature},
                                     {"token": f"token_id:{token + 1}", "logprob": -3.0 / temperature}]}
                   for token in completion]
        answer = {"id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "model": body["model"]}
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                 "total_tokens": len(prompt) + len(completion)}
        if body.get("stream"):
            return StreamingResponse(_chunks(answer, message, finish, usage), media_type="text/event-stream")
        return JSONResponse({
            **answer, "object": "chat.completion", "prompt_token_ids": prompt,
            "choices": [{"index": 0, "message": message, "finish_reason": finish, "token_ids": completion,
                         "logprobs": {"content": content}}],
            "usage": usage,
        })

    app = Starlette(routes=[Route("/v1/models", models), Route("/v1/chat/completions", completions, methods=["POST"])])
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, name="fake-vllm", daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{port}", requests=requests, served=served)
    finally:
        server.should_exit = True
        thread.join()


async def _chunks(answer: dict[str, Any], message: dict[str, Any], finish: str,
                  usage: dict[str, int]) -> AsyncIterator[str]:
    """The answer as a streamed completion: the whole message in one delta, then its finish reason and usage."""
    delta = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": index, **call} for index, call in enumerate(message["tool_calls"])]
    for choice, extra in (({"index": 0, "delta": delta, "finish_reason": None}, {}),
                          ({"index": 0, "delta": {}, "finish_reason": finish}, {"usage": usage})):
        yield f"data: {json.dumps({**answer, 'object': 'chat.completion.chunk', 'choices': [choice], **extra})}\n\n"
    yield "data: [DONE]\n\n"
