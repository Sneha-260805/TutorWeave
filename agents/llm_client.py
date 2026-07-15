import logging
import threading
import time
from typing import Iterable

from config.settings import (
    GEMINI_API_KEY,
    GROQ_API_KEY,
    LLM_MAX_RETRIES,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

_client_store: dict[str, object] = {}


def _init_groq_client() -> object:
    if not GROQ_API_KEY:
        raise ValueError("Set GROQ_API_KEY in your environment or .env file before running Groq LLM features.")
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError("Install the 'groq' package to use the Groq provider.") from exc
    return Groq(api_key=GROQ_API_KEY)


def _init_openai_client() -> object:
    if not OPENAI_API_KEY:
        raise ValueError("Set OPENAI_API_KEY in your environment or .env file before running OpenAI LLM features.")
    try:
        import openai
    except ImportError as exc:
        raise ImportError("Install the 'openai' package to use the OpenAI provider.") from exc
    openai.api_key = OPENAI_API_KEY
    return openai


def _init_gemini_client() -> object:
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY in your environment or .env file before running Gemini LLM features.")
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError("Install the 'google-generativeai' package to use the Gemini provider.") from exc
    genai.configure(api_key=GEMINI_API_KEY)
    return genai


def _get_client(provider: str) -> object:
    provider = provider.strip().lower()
    if provider in _client_store:
        return _client_store[provider]
    if provider == "groq":
        client = _init_groq_client()
    elif provider == "openai":
        client = _init_openai_client()
    else:
        client = _init_gemini_client()
    _client_store[provider] = client
    return client


# ── Quota state ───────────────────────────────────────────────────────────────
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
                "LLM quota appears exhausted. Calls will fall back immediately "
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


def _build_groq_messages(messages: Iterable[dict]) -> list[dict]:
    return [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in messages]


# ── Main call ─────────────────────────────────────────────────────────────────

def complete_chat(
    messages: Iterable[dict],
    *,
    fallback: str | None = None,
    model: str = MODEL_NAME,
    max_retries: int = LLM_MAX_RETRIES,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    provider: str | None = None,
) -> str:
    """
    Call the configured LLM provider with retry/fallback handling.

    Accepts the same OpenAI-style message list used throughout the codebase:
      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    if is_quota_exhausted():
        logger.warning("Quota exhausted — returning fallback immediately.")
        if fallback is not None:
            return fallback
        raise RuntimeError("LLM quota exhausted.")

    message_list = list(messages)
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    consecutive_429s = 0

    provider = LLM_PROVIDER if provider is None else provider.strip().lower()
    client = _get_client(provider)

    if provider == "groq":
        for attempt in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=_build_groq_messages(message_list),
                    temperature=temperature,
                    max_tokens=max_tokens or 512,
                    timeout=int(LLM_TIMEOUT_SECONDS),
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
                raise ValueError("LLM response was empty.")
            except Exception as exc:
                last_error = exc
                if _is_rate_limit(exc):
                    consecutive_429s += 1
                    wait = min(2 ** consecutive_429s * 2, 60)
                    logger.warning("Rate limit on attempt %d/%d. Waiting %.0fs.", attempt + 1, attempts, wait)
                    if attempt >= attempts - 1:
                        mark_quota_exhausted()
                        break
                    time.sleep(wait)
                else:
                    consecutive_429s = 0
                    logger.warning("Groq chat failed on attempt %d/%d: %s", attempt + 1, attempts, exc)
                    if attempt < attempts - 1:
                        time.sleep(1.0 * (attempt + 1))
    elif provider == "openai":
        for attempt in range(attempts):
            try:
                response = client.ChatCompletion.create(
                    model=model,
                    messages=[{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in message_list],
                    temperature=temperature,
                    max_tokens=max_tokens or 512,
                    request_timeout=int(LLM_TIMEOUT_SECONDS),
                )
                content = response.choices[0].message["content"]
                if content and content.strip():
                    return content.strip()
                raise ValueError("LLM response was empty.")
            except Exception as exc:
                last_error = exc
                if _is_rate_limit(exc):
                    consecutive_429s += 1
                    wait = min(2 ** consecutive_429s * 2, 60)
                    logger.warning("Rate limit on attempt %d/%d. Waiting %.0fs.", attempt + 1, attempts, wait)
                    if attempt >= attempts - 1:
                        mark_quota_exhausted()
                        break
                    time.sleep(wait)
                else:
                    consecutive_429s = 0
                    logger.warning("OpenAI chat failed on attempt %d/%d: %s", attempt + 1, attempts, exc)
                    if attempt < attempts - 1:
                        time.sleep(1.0 * (attempt + 1))
    else:
        # Separate system instruction from conversation turns for Gemini-compatible callers.
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

        if not conversation:
            conversation = [{"role": "user", "parts": [system_instruction or ""]}]
            system_instruction = None

        gen_config_kwargs: dict = {"temperature": temperature}
        if max_tokens is not None:
            gen_config_kwargs["max_output_tokens"] = max_tokens

        for attempt in range(attempts):
            try:
                gemini_model = client.GenerativeModel(
                    model_name=model,
                    system_instruction=system_instruction,
                )
                response = gemini_model.generate_content(
                    conversation,
                    generation_config=client.GenerationConfig(**gen_config_kwargs),
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
                    wait = min(2 ** consecutive_429s * 2, 60)
                    logger.warning("Rate limit on attempt %d/%d. Waiting %.0fs.", attempt + 1, attempts, wait)
                    if attempt >= attempts - 1:
                        mark_quota_exhausted()
                        break
                    time.sleep(wait)
                else:
                    consecutive_429s = 0
                    logger.warning("Gemini chat failed on attempt %d/%d: %s", attempt + 1, attempts, exc)
                    if attempt < attempts - 1:
                        time.sleep(1.0 * (attempt + 1))

    if fallback is not None:
        return fallback

    raise RuntimeError("LLM chat completion failed after retries.") from last_error
