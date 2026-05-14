import logging
import threading
import time
from typing import Iterable

from groq import Groq

from config.settings import (
    GROQ_API_KEY,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)

if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY in your environment or .env file before running LLM features.")

client = Groq(api_key=GROQ_API_KEY, timeout=LLM_TIMEOUT_SECONDS)

# ── Quota state ───────────────────────────────────────────────────────────────
# Shared flag so repeated 429s in one session stop multiplying token usage.
_quota_lock = threading.Lock()
_quota_exhausted: bool = False


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "rate_limit" in s or "rate limit" in s or (
        "quota" in s and ("exceed" in s or "exhaust" in s)
    )


def mark_quota_exhausted() -> None:
    global _quota_exhausted
    with _quota_lock:
        if not _quota_exhausted:
            _quota_exhausted = True
            logger.error(
                "Groq quota appears exhausted. LLM calls will fall back immediately "
                "for the remainder of this session."
            )


def is_quota_exhausted() -> bool:
    with _quota_lock:
        return _quota_exhausted


def reset_quota_state() -> None:
    """Call this at the start of a new evaluation run if needed."""
    global _quota_exhausted
    with _quota_lock:
        _quota_exhausted = False


# ── Main call ─────────────────────────────────────────────────────────────────

def complete_chat(
    messages: Iterable[dict],
    *,
    fallback: str | None = None,
    model: str = MODEL_NAME,
    max_retries: int = LLM_MAX_RETRIES,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    """
    Call the configured Groq chat model with retry/fallback behavior.

    - Detects HTTP 429 separately from other errors and uses exponential backoff.
    - After consecutive 429s exhaust retries, marks the session quota as
      exhausted so subsequent calls return the fallback immediately.
    - Returns the fallback string if provided; otherwise raises RuntimeError.
    """
    if is_quota_exhausted():
        logger.warning("Quota exhausted — returning fallback immediately.")
        if fallback is not None:
            return fallback
        raise RuntimeError("Groq quota exhausted.")

    message_list = list(messages)
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    consecutive_429s = 0

    for attempt in range(attempts):
        try:
            kwargs = dict(model=model, messages=message_list, temperature=temperature, timeout=LLM_TIMEOUT_SECONDS)
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
            raise ValueError("LLM response was empty.")

        except Exception as exc:
            last_error = exc
            if _is_rate_limit(exc):
                consecutive_429s += 1
                wait = min(2 ** consecutive_429s * 2, 60)  # exponential: 4, 8, 16 … 60s
                logger.warning(
                    "Rate limit (429) on attempt %d/%d. Waiting %.0fs.",
                    attempt + 1, attempts, wait,
                )
                if attempt >= attempts - 1:
                    mark_quota_exhausted()
                    break
                time.sleep(wait)
            else:
                consecutive_429s = 0
                logger.warning(
                    "Groq chat failed on attempt %d/%d: %s",
                    attempt + 1, attempts, exc,
                )
                if attempt < attempts - 1:
                    time.sleep(1.0 * (attempt + 1))

    if fallback is not None:
        return fallback

    raise RuntimeError("Groq chat completion failed after retries.") from last_error
