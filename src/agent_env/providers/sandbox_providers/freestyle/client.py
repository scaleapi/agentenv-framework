"""The Freestyle v5 HTTP operations used by the sandbox adapter."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


class FreestyleClient:
    def __init__(self, *, api_key: str, api_url: str):
        url = httpx.URL(api_url)
        if (
            url.scheme != "https"
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "Freestyle api_url must be an HTTPS origin without credentials, query or fragment"
            )
        self._api_key = api_key
        self._api_url = str(url).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def request(self, method: str, path: str, **kwargs: Any) -> dict:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._api_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=httpx.Timeout(330, connect=30),
            )
        response = await self._client.request(method, path, **kwargs)
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def vm_path(vm_id: str) -> str:
    return f"/v5/vms/{quote(vm_id, safe='')}"
