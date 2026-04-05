# ============================================
# MIND EASE — Speech-to-Text Service (Python)
# ============================================
# Uses faster-whisper (CTranslate2) for efficient offline STT.
# Falls back to stub if model not available.

import asyncio
import logging
import time
import io
import struct
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger("mindease.stt")


class STTService:
    """Offline speech-to-text using faster-whisper."""

    def __init__(self):
        self._model = None
        self._ready = False
        self._model_size = "base.en"

    def load(self) -> bool:
        """Load the whisper model."""
        model_path = settings.resolve_path(settings.whisper_model_path)

        # Try faster-whisper first (fastest option)
        try:
            from faster_whisper import WhisperModel

            # Check if the ggml .bin file exists — faster-whisper uses its own format
            # so we pass the model size string and let it download/cache automatically
            if model_path.exists() and model_path.stat().st_size > 1_000_000:
                logger.info(f"Whisper model file found: {model_path} ({model_path.stat().st_size / 1e6:.1f}MB)")

            # Use model size identifier — faster-whisper will download if not cached
            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
            self._ready = True
            logger.info(f"faster-whisper loaded: {self._model_size}")
            return True
        except ImportError:
            logger.warning("faster-whisper not installed")
        except Exception as e:
            logger.warning(f"faster-whisper load failed: {e}")

        # Try openai-whisper as fallback
        try:
            import whisper as openai_whisper

            self._model = openai_whisper.load_model("base.en")
            self._ready = True
            self._model_size = "openai-whisper"
            logger.info("openai-whisper loaded (fallback)")
            return True
        except ImportError:
            logger.warning("openai-whisper not installed")
        except Exception as e:
            logger.warning(f"openai-whisper load failed: {e}")

        # No STT available — use stub
        logger.warning("No STT model available — using stub mode")
        self._ready = True
        return True

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> dict:
        """
        Transcribe PCM16 audio bytes to text.
        Returns dict with: text, confidence, duration_ms, success
        """
        if not self._ready:
            return {"text": "", "confidence": 0.0, "duration_ms": 0, "success": False}

        start = time.time()

        # Convert PCM16 bytes to float32 numpy array
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Skip very short audio (less than 0.5 seconds)
        if len(audio) < sample_rate * 0.5:
            return {"text": "", "confidence": 0.0, "duration_ms": 0, "success": False}

        try:
            if self._model is None:
                # Stub mode
                duration_ms = int((time.time() - start) * 1000)
                return {
                    "text": f"[STT stub] {len(audio)} samples received",
                    "confidence": 1.0,
                    "duration_ms": duration_ms,
                    "success": True,
                }

            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            text, confidence = await loop.run_in_executor(
                None, self._do_transcribe, audio
            )

            duration_ms = int((time.time() - start) * 1000)

            # Filter out whisper hallucinations / noise artifacts
            text = self._clean_text(text)

            return {
                "text": text,
                "confidence": confidence,
                "duration_ms": duration_ms,
                "success": bool(text.strip()),
            }
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            duration_ms = int((time.time() - start) * 1000)
            return {"text": "", "confidence": 0.0, "duration_ms": duration_ms, "success": False}

    def _do_transcribe(self, audio: np.ndarray) -> tuple:
        """Synchronous transcription (runs in thread pool)."""
        try:
            # faster-whisper path
            from faster_whisper import WhisperModel
            if isinstance(self._model, WhisperModel):
                segments, info = self._model.transcribe(
                    audio,
                    beam_size=1,
                    language="en",
                    initial_prompt="This is a conversation about mental health, emotions, and well-being. Words like depressed, sad, anxious, alone, and struggling may be used.",
                    vad_filter=True,
                    vad_parameters=dict(
                        min_speech_duration_ms=250,
                        min_silence_duration_ms=500,
                    ),
                )
                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text)
                text = " ".join(text_parts).strip()
                return text, info.language_probability if hasattr(info, 'language_probability') else 0.85
        except (ImportError, TypeError):
            pass

        try:
            # openai-whisper path
            import whisper as openai_whisper
            result = openai_whisper.transcribe(
                self._model, 
                audio, 
                language="en", 
                fp16=False,
                initial_prompt="This is a conversation about mental health, emotions, and well-being. Words like depressed, sad, anxious, alone, and struggling may be used."
            )
            return result.get("text", "").strip(), 0.85
        except (ImportError, TypeError):
            pass

        return "", 0.0

    def _clean_text(self, text: str) -> str:
        """Remove common whisper hallucinations and artifacts."""
        if not text:
            return ""

        text = text.strip()

        # Common hallucination patterns from silence/noise
        hallucination_patterns = [
            "thank you",
            "thanks for watching",
            "subscribe",
            "like and subscribe",
            "you",
            "bye",
            "...",
            "[Music]",
            "(music)",
            "[BLANK_AUDIO]",
        ]

        lower = text.lower().strip()
        for pattern in hallucination_patterns:
            if lower == pattern.lower():
                return ""

        # Filter very short results (likely noise)
        if len(text) < 3:
            return ""

        return text

    @property
    def is_ready(self) -> bool:
        return self._ready
