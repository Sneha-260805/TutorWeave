import logging
import threading
import time
from typing import Iterable

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from config.settings import (
    GEMINI_API_KEY,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
)

logger = logging.getLogger(__name__)

if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in your environment or .env file before running LLM features.")

genai.configure(api_key=GEMINI_API_KEY)

# ── Quota state ───────────────────────────────────────────────────────────────
_quota_lock = threading.Lock()
_quota_exhausted: bool = False


def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, ResourceExhausted):
        return True
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
                "Gemini quota appears exhausted. LLM calls will fall back immediately "
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
    Call the configured Gemini model with retry/fallback behavior.

    Accepts the same OpenAI-style message list used throughout the codebase:
      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    System messages are extracted and passed as system_instruction to the model.
    Assistant messages are mapped to the "model" role Gemini expects.

    - Detects HTTP 429 / ResourceExhausted separately and uses exponential backoff.
    - After consecutive rate-limit failures, marks quota exhausted so subsequent
      calls return the fallback immediately without hammering the API.
    - Returns fallback string if provided; otherwise raises RuntimeError.
    """
    if is_quota_exhausted():
        logger.warning("Quota exhausted — returning fallback immediately.")
        if fallback is not None:
            return fallback
        raise RuntimeError("Gemini quota exhausted.")

    message_list = list(messages)

    # Separate system instruction from conversation turns
    system_instruction = None
    conversation = []
    for msg in message_list:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        else:
            gemini_role = "model" if role == "assistant" else "user"
            conversation.append({"role": gemini_role, "parts": [content]})

    # Gemini requires at least one user turn
    if not conversation:
        conversation = [{"role": "user", "parts": [system_instruction or ""]}]
        system_instruction = None

    gen_config_kwargs: dict = {"temperature": temperature}
    if max_tokens is not None:
        gen_config_kwargs["max_output_tokens"] = max_tokens

    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    consecutive_429s = 0

    for attempt in range(attempts):
        try:
            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction,
            )
            response = gemini_model.generate_content(
                conversation,
                generation_config=genai.GenerationConfig(**gen_config_kwargs),
                request_options={"timeout": int(LLM_TIMEOUT_SECONDS)},
            )
            content = response.text
            if content and content.strip():
                return content.strip()
            raise ValueError("LLM response was empty.")

        except Exception as exc:
            last_error = exc
            if _is_rate_limit(exc):
                consecutive_429s += 1
                wait = min(2 ** consecutive_429s * 2, 60)  # exponential: 4, 8, 16 … 60s
                logger.warning(
                    "Rate limit on attempt %d/%d. Waiting %.0fs.",
                    attempt + 1, attempts, wait,
                )
                if attempt >= attempts - 1:
                    mark_quota_exhausted()
                    break
                time.sleep(wait)
            else:
                consecutive_429s = 0
                logger.warning(
                    "Gemini chat failed on attempt %d/%d: %s",
                    attempt + 1, attempts, exc,
                )
                if attempt < attempts - 1:
                    time.sleep(1.0 * (attempt + 1))

    if fallback is not None:
        return fallback

    raise RuntimeError("Gemini chat completion failed after retries.") from last_error
