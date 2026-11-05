"""
IBM Interview Trainer Agent — IBM Granite LLM Client
Uses the watsonx.ai /text/chat API with IBM Granite-4 (ibm/granite-4-h-small).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Current (2025) watsonx.ai chat completions endpoint — same schema as OpenAI chat
WATSONX_CHAT_URL = "{url}/ml/v1/text/chat?version=2023-05-29"
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

# Max retries on 429 rate-limit responses
_MAX_RETRIES = 4
_RETRY_BACKOFF = [2, 5, 10, 20]  # seconds


class GraniteClient:
    """Async client for IBM Granite-4 via the watsonx.ai /text/chat API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._iam_token: str | None = None
        self._http: httpx.AsyncClient = httpx.AsyncClient(timeout=120.0)

    # ── IAM Authentication ──────────────────────────────────────────────────

    async def _get_iam_token(self) -> str:
        """Exchange IBM API key for a short-lived IAM Bearer token."""
        if self._iam_token:
            return self._iam_token

        response = await self._http.post(
            IAM_TOKEN_URL,
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": self.settings.ibm_watsonx_api_key,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        self._iam_token = response.json()["access_token"]
        return self._iam_token  # type: ignore[return-value]

    def invalidate_token(self) -> None:
        """Force token refresh on next call (e.g. after 401)."""
        self._iam_token = None

    # ── Core Generation ─────────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """
        Call IBM Granite via the /text/chat endpoint.

        ``prompt`` is sent as a single ``user`` message.
        The system instruction is prepended as a ``system`` message.
        Returns the assistant's reply text.
        """
        cfg = self.settings

        # Split the prompt into system + user parts if the caller used the
        # legacy <|system|>/<|user|> markers, otherwise send as-is.
        system_text, user_text = _split_prompt(prompt)

        messages: list[dict[str, str]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        payload: dict[str, Any] = {
            "model_id": cfg.ibm_granite_model_id,
            "messages": messages,
            "project_id": cfg.ibm_watsonx_project_id,
            "parameters": {
                "max_tokens": max_new_tokens or cfg.ibm_granite_max_tokens,
                "temperature": temperature if temperature is not None else cfg.ibm_granite_temperature,
                "top_p": cfg.ibm_granite_top_p,
                "repetition_penalty": 1.1,
            },
        }
        if stop_sequences:
            payload["parameters"]["stop_sequences"] = stop_sequences

        url = WATSONX_CHAT_URL.format(url=cfg.ibm_watsonx_url)

        for attempt in range(_MAX_RETRIES):
            token = await self._get_iam_token()
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            try:
                resp = await self._http.post(url, json=payload, headers=headers)

                if resp.status_code == 401:
                    self.invalidate_token()
                    continue  # retry with fresh token

                if resp.status_code == 429:
                    wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                    logger.warning(
                        "watsonx.ai rate limit hit (attempt %d/%d). Retrying in %ds…",
                        attempt + 1, _MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()

            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Granite API error %s: %s",
                    exc.response.status_code,
                    exc.response.text[:500],
                )
                raise
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                logger.error("Unexpected Granite response format: %s", exc)
                raise

        raise RuntimeError(
            f"watsonx.ai rate limit not resolved after {_MAX_RETRIES} retries. "
            "Please wait a moment and try again."
        )

    async def close(self) -> None:
        await self._http.aclose()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _split_prompt(prompt: str) -> tuple[str, str]:
    """
    Extract system and user text from a prompt that may contain
    <|system|> / <|user|> / <|assistant|> markers.
    Returns (system_text, user_text).
    """
    if "<|system|>" not in prompt and "<|user|>" not in prompt:
        # Plain prompt — send entirely as user message
        return "", prompt.strip()

    system_text = ""
    user_text = prompt.strip()

    if "<|system|>" in prompt:
        parts = prompt.split("<|system|>", 1)
        remainder = parts[1]
        if "<|user|>" in remainder:
            sys_part, rest = remainder.split("<|user|>", 1)
            system_text = sys_part.strip()
            user_text = rest
        else:
            system_text = remainder.strip()
            user_text = ""

    # Strip the <|user|> and <|assistant|> markers from user_text
    for marker in ("<|user|>", "<|assistant|>"):
        user_text = user_text.replace(marker, "")
    user_text = user_text.strip()

    return system_text, user_text


# ── Singleton ───────────────────────────────────────────────────────────────
_client: GraniteClient | None = None


def get_granite_client() -> GraniteClient:
    global _client
    if _client is None:
        _client = GraniteClient()
    return _client
