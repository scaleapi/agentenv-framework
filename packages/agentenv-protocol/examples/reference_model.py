"""Small OpenAI-compatible model client shared by the runnable examples."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class OpenAICompatibleModel:
    """Call the LiteLLM proxy using its OpenAI-compatible HTTP API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def _request(self, *, api_key: str | None = None) -> tuple[str, dict[str, str]]:
        api_key = api_key or self.api_key or os.getenv("LITELLM_API_KEY")
        if not api_key:
            raise RuntimeError("LITELLM_API_KEY must be set before running the agent")

        base_url = self.base_url or os.getenv("LITELLM_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "LITELLM_BASE_URL must be set before running the agent"
            )
        base_url = base_url.rstrip("/")
        endpoint = (
            f"{base_url}/chat/completions"
            if base_url.endswith("/v1")
            else f"{base_url}/v1/chat/completions"
        )
        return endpoint, {"Authorization": f"Bearer {api_key}"}

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
        api_key: str | None = None,
    ) -> ModelResponse:
        endpoint, headers = self._request(api_key=api_key)
        body: dict[str, Any] = {"model": model, "messages": list(messages)}
        if metadata:
            body["metadata"] = dict(metadata)
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                endpoint,
                headers=headers,
                json=body,
            )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        return ModelResponse(
            text=_message_text(payload["choices"][0]["message"].get("content")),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        endpoint, headers = self._request()
        body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": True,
        }
        if metadata:
            body["metadata"] = dict(metadata)
        async with (
            httpx.AsyncClient(timeout=self.timeout_seconds) as client,
            client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=body,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0]["delta"].get("content")
                if delta:
                    yield delta


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, Mapping)
        )
    return str(content or "")
