import logging

import httpx

logger = logging.getLogger(__name__)


class EmbedderClient:
    def __init__(self, url: str) -> None:
        self._url = url
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(self._url, json={"input": text})
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    async def close(self) -> None:
        await self._client.aclose()
