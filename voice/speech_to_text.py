import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np

try:
    import whisper
except ModuleNotFoundError:  # pragma: no cover
    whisper = None

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Simple Whisper-based speech-to-text wrapper."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        language: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.language = language
        self.model = None
        self._load_model()

    def _load_model(self):
        if whisper is None:
            raise RuntimeError(
                "Whisper is not installed. Install the openai-whisper package to enable voice transcription."
            )
        logger.info(f"Loading Whisper model '{self.model_name}' on {self.device}")
        self.model = whisper.load_model(self.model_name, device=self.device)

    def _load_audio(
        self,
        audio_source: Union[str, Path, bytes, bytearray],
        sample_rate: int = 16000,
    ) -> np.ndarray:
        if isinstance(audio_source, (str, Path)):
            audio_path = str(audio_source)
            logger.debug(f"Loading audio from file path: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        elif isinstance(audio_source, (bytes, bytearray)):
            audio_bytes = bytes(audio_source)
            logger.debug("Loading audio from raw bytes")
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=sample_rate, mono=True)
        else:
            raise TypeError("Unsupported audio source type for transcription")

        if audio is None or len(audio) == 0:
            raise ValueError("No audio could be loaded for transcription")

        logger.debug(f"Loaded audio: {len(audio)} samples at {sr}Hz")
        return audio

    def _calculate_confidence(self, result: dict) -> float:
        if not result or "segments" not in result:
            return 0.0

        confidences = []
        for segment in result.get("segments", []):
            if segment is None:
                continue
            score = segment.get("avg_logprob")
            if score is not None:
                confidences.append(score)

        if not confidences:
            return 0.0

        avg_logprob = float(sum(confidences) / len(confidences))
        return float(min(max((avg_logprob + 5.0) / 5.0, 0.0), 1.0))

    def transcribe(
        self,
        audio_source: Union[str, Path, bytes, bytearray],
        language: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Tuple[str, float]:
        if whisper is None:
            logger.warning(
                "Whisper is not installed. Returning empty transcription."
            )
            return "", 0.0

        if self.model is None:
            self._load_model()

        try:
            audio = self._load_audio(audio_source)
            result = self.model.transcribe(
                audio,
                language=language or self.language,
                temperature=temperature,
                fp16=self.device != "cpu",
            )
            text = result.get("text", "").strip()
            confidence = self._calculate_confidence(result)
            logger.info(f"Transcribed audio: {text[:80]}")
            return text, confidence
        except Exception as error:
            logger.error(f"Whisper transcription failed: {error}")
            return "", 0.0
