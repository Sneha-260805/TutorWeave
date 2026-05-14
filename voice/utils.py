import logging
import tempfile
from pathlib import Path
from typing import Optional

from .speech_to_text import WhisperTranscriber
from .text_to_speech import PiperTTSEngine

logger = logging.getLogger(__name__)

_transcriber: WhisperTranscriber | None = None
_tts_engine: PiperTTSEngine | None = None


def _get_transcriber() -> WhisperTranscriber | None:
    global _transcriber
    if _transcriber is None:
        try:
            _transcriber = WhisperTranscriber(model_name="base", device="cpu")
        except Exception as exc:
            logger.warning("Whisper transcriber failed to load: %s", exc)
            _transcriber = None
    return _transcriber


def _get_tts_engine() -> PiperTTSEngine | None:
    global _tts_engine
    if _tts_engine is None:
        try:
            _tts_engine = PiperTTSEngine()
        except Exception:
            _tts_engine = None
    return _tts_engine


def transcribe_audio_path(audio_path: str) -> tuple[str, float]:
    if not audio_path:
        return "", 0.0
    transcriber = _get_transcriber()
    if transcriber is None:
        return "", 0.0
    return transcriber.transcribe(audio_path)


def synthesize_text_to_wav(text: str) -> Optional[str]:
    if not text:
        return None

    tts_engine = _get_tts_engine()
    if tts_engine is None:
        return None

    wav_bytes = tts_engine.synthesize(text)
    if not wav_bytes:
        return None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(wav_bytes)
    temp_file.flush()
    temp_file.close()
    return temp_file.name
