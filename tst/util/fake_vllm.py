"""A stand-in for vLLM served with ``--return-tokens-as-token-ids`` and processed logprobs, as OpenEnv's capture proxy
needs its engine: it lists the models it serves and answers chat completions with token ids, a logprob per completion
token and the top two at each, scaled by the temperature as processed logprobs are. ``reply`` makes each answer's
message from the request body."""

import json
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent_env.training._openenv import Server

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
def fake_vllm(model: str, reply: Reply) -> Iterator[Server]:
    """The engine, served on loopback; ``.requests`` are the completion bodies it got, and ``.served`` the models it
    lists, which a test appends to as vLLM lists a LoRA adapter once it's loaded."""
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
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}", "object": "chat.completion", "model": body["model"],
            "prompt_token_ids": prompt,
            "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls")
                         else "stop", "token_ids": completion, "logprobs": {"content": content}}],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(completion),
                      "total_tokens": len(prompt) + len(completion)},
        })

    server = Server(Starlette(routes=[Route("/v1/models", models),
                                      Route("/v1/chat/completions", completions, methods=["POST"])]))
    server.requests, server.served = requests, served
    try:
        yield server
    finally:
        server.stop()
