import io
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    import pyttsx3
except ImportError:  # pragma: no cover
    pyttsx3 = None

logger = logging.getLogger(__name__)


class PiperTTSEngine:
    def __init__(
        self,
        voice: str = "alloy",
        speaker_id: Optional[int] = None,
        speed: float = 1.0,
        sample_rate: int = 16000,
    ):
        self.voice = voice
        self.speaker_id = speaker_id
        self.speed = speed
        self.sample_rate = sample_rate
        self.piper_available = self._find_piper()

    def _find_piper(self) -> bool:
        try:
            subprocess.run(["piper", "--help"], capture_output=True, check=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            logger.warning("Piper CLI not available; falling back to local TTS engine")
            return False

    def _pcm_to_wav(self, raw_pcm: bytes) -> bytes:
        audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def _synthesize_with_piper(self, text: str) -> bytes:
        if not text:
            return b""

        cmd = [
            "piper",
            "--voice",
            self.voice,
            "--speed",
            str(self.speed),
            "--input-stream",
            "stdin",
            "--output-raw",
        ]
        if self.speaker_id is not None:
            cmd.extend(["--speaker", str(self.speaker_id)])

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(input=text.encode("utf-8"))

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="ignore")
            logger.error(f"Piper TTS failed: {error_msg}")
            raise RuntimeError("Piper text-to-speech failed")

        return self._pcm_to_wav(stdout)

    def _synthesize_with_pyttsx3(self, text: str) -> bytes:
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed for fallback TTS")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name

        wav_bytes = b""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            with open(temp_path, "rb") as f:
                wav_bytes = f.read()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning(f"Could not remove temporary TTS file: {temp_path}")

        return wav_bytes

    def synthesize(
        self,
        text: str,
        output_format: str = "wav",
    ) -> bytes:
        if not text:
            return b""

        if self.piper_available:
            try:
                return self._synthesize_with_piper(text)
            except Exception as exc:
                logger.warning(f"Piper synthesis failed, falling back: {exc}")

        return self._synthesize_with_pyttsx3(text)
