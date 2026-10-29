"""Optional LLM post-processing for already-committed fingerspelled text.

This module never receives images, video, landmarks, or files.  It sends only the
text the user could already see and copy, and it is disabled unless
``ASL_LLM_ENABLED=true`` and ``ASL_LLM_API_KEY`` are set.  The endpoint that uses
it is deliberately non-authoritative: it returns suggestions, not a replacement
transcript and not an ASL translation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import MAX_TEXT_LENGTH
from src.settings import Settings

DISCLAIMER = (
    "Post-processing only. Suggestions are language-model guesses from committed "
    "letters, not ASL translation."
)
SYSTEM_PROMPT = (
    "You post-process ambiguous fingerspelled text from a static ASL handshape "
    "recogniser. Insert likely word boundaries and capitalisation, but do not "
    "invent content and do not translate other languages. Return only a JSON array "
    "of at most three short candidate strings."
)


class LLMError(RuntimeError):
    """Raised when a configured LLM request fails."""


@dataclass(frozen=True)
class SuggestionResult:
    configured: bool
    suggestions: list[str]
    message: str
    model: str | None = None
    sent_fields: tuple[str, ...] = ("text",)


def _extract_suggestions(content: str, *, limit: int = 3) -> list[str]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates: list[str] = []
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        raw = parsed.get("suggestions") or parsed.get("candidates") or []
        if isinstance(raw, str):
            raw = [raw]
        candidates = [str(item) for item in raw]
    elif isinstance(parsed, list):
        candidates = [str(item) for item in parsed]
    else:
        candidates = [line.strip(" -•\t") for line in cleaned.splitlines()]

    result: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = " ".join(candidate.strip().strip("\"'").split())
        if not value or len(value) > MAX_TEXT_LENGTH:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def suggest_text(
    text: str,
    settings: Settings,
    *,
    client: httpx.Client | None = None,
) -> SuggestionResult:
    """Ask an OpenAI-compatible chat endpoint for post-processing suggestions."""

    text = str(text)
    if not settings.llm_configured:
        return SuggestionResult(
            configured=False,
            suggestions=[],
            message=(
                "LLM post-processing is disabled. Set ASL_LLM_ENABLED=true and "
                "ASL_LLM_API_KEY to enable it; the app works without it."
            ),
        )

    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "max_tokens": 160,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }
    endpoint = settings.llm_base_url.rstrip("/") + "/chat/completions"
    owns_client = client is None
    http_client = client or httpx.Client(timeout=settings.llm_timeout_seconds)
    try:
        response = http_client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        content = body["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    finally:
        if owns_client:
            http_client.close()

    suggestions = _extract_suggestions(str(content))
    let = "suggestion" if len(suggestions) == 1 else "suggestions"
    return SuggestionResult(
        configured=True,
        suggestions=suggestions,
        message=f"Received {len(suggestions)} post-processing {let}.",
        model=settings.llm_model,
    )
