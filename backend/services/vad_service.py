# ============================================
# MIND EASE — Voice Activity Detection (Python)
# ============================================
# Energy-based VAD that accumulates speech audio
# and detects speech boundaries (start/end).

import logging
import math
import struct
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger("mindease.vad")


class VADService:
    """
    Energy-based Voice Activity Detection.
    Accumulates audio, detects speech boundaries, emits complete
    speech segments for downstream processing.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        energy_threshold: float = 0.008,
        min_speech_ms: int = 300,
        min_silence_ms: int = 800,
        max_speech_ms: int = 30000,
    ):
        self._sample_rate = sample_rate
        self._energy_threshold = energy_threshold
        self._min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self._min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self._max_speech_samples = int(sample_rate * max_speech_ms / 1000)

        # State
        self._speech_buffer: list = []
        self._speech_sample_count: int = 0
        self._silence_sample_count: int = 0
        self._is_speaking: bool = False
        self._on_speech_end: Optional[Callable] = None

    def set_speech_end_callback(self, callback: Callable):
        """Set callback invoked with complete speech audio bytes when speech ends."""
        self._on_speech_end = callback

    def feed_audio(self, pcm_bytes: bytes) -> Optional[bytes]:
        """
        Feed PCM16 audio bytes to the VAD.
        Returns complete speech segment bytes if speech ended, else None.
        """
        # Convert to float32 for energy computation
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio) == 0:
            return None

        # Process in 20ms frames (320 samples at 16kHz)
        frame_size = 320
        result = None

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size // 2:
                continue

            energy = self._compute_rms(frame)
            is_speech = energy >= self._energy_threshold

            if is_speech:
                self._silence_sample_count = 0
                self._speech_sample_count += len(frame)

                # Start collecting speech
                if not self._is_speaking and self._speech_sample_count >= self._min_speech_samples:
                    self._is_speaking = True
                    logger.debug("VAD: Speech started")

                # Accumulate audio (keep raw PCM16 bytes)
                start_byte = i * 2  # 2 bytes per sample (int16)
                end_byte = min((i + frame_size) * 2, len(pcm_bytes))
                self._speech_buffer.append(pcm_bytes[start_byte:end_byte])

                # Safety: cap max speech length
                if self._speech_sample_count >= self._max_speech_samples:
                    result = self._emit_speech()

            else:
                self._silence_sample_count += len(frame)

                if self._is_speaking:
                    # Still accumulate during short silence (might be a pause)
                    start_byte = i * 2
                    end_byte = min((i + frame_size) * 2, len(pcm_bytes))
                    self._speech_buffer.append(pcm_bytes[start_byte:end_byte])

                    if self._silence_sample_count >= self._min_silence_samples:
                        # Silence threshold reached -- speech ended
                        result = self._emit_speech()
                else:
                    # Not speaking yet -- reset speech counter if silence continues
                    if self._silence_sample_count > self._min_silence_samples:
                        self._speech_sample_count = 0
                        self._speech_buffer.clear()

        return result

    def _emit_speech(self) -> bytes:
        """Combine buffered speech and reset state."""
        speech_bytes = b"".join(self._speech_buffer)
        logger.debug(
            f"VAD: Speech ended ({len(speech_bytes)} bytes, "
            f"{len(speech_bytes) / 2 / self._sample_rate:.1f}s)"
        )

        self._speech_buffer.clear()
        self._speech_sample_count = 0
        self._silence_sample_count = 0
        self._is_speaking = False

        if self._on_speech_end:
            self._on_speech_end(speech_bytes)

        return speech_bytes

    def _compute_rms(self, samples: np.ndarray) -> float:
        """Compute RMS energy of audio samples."""
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples ** 2)))

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def reset(self):
        """Reset VAD state."""
        self._speech_buffer.clear()
        self._speech_sample_count = 0
        self._silence_sample_count = 0
        self._is_speaking = False
