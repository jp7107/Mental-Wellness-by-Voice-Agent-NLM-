# ============================================
# MIND EASE — Pipeline Orchestrator (Python)
# ============================================
# Coordinates the full inference pipeline:
#   Audio -> VAD -> STT -> Emotion+Mood -> LLM -> TTS
# Manages per-session state and emits results via callback.

import asyncio
import json
import logging
import time
from typing import Optional, Callable, Awaitable

from services.vad_service import VADService
from services.stt_service import STTService
from services.emotion_service import EmotionService, MoodTracker
from services.llm_service import LLMService
from services.tts_service import TTSService
from services.safety_service import SafetyService

logger = logging.getLogger("mindease.pipeline")

# Shared model instances (loaded once, reused across sessions)
_stt: Optional[STTService] = None
_llm: Optional[LLMService] = None
_models_loaded = False
_models_loading = False
_models_lock = asyncio.Lock()


async def ensure_models_loaded():
    """Load shared models (STT, LLM) once across all sessions."""
    global _stt, _llm, _models_loaded, _models_loading

    if _models_loaded:
        return

    async with _models_lock:
        if _models_loaded:
            return

        _models_loading = True
        logger.info("Loading shared models (first session)...")

        start = time.time()

        # Load STT
        _stt = STTService()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _stt.load)

        # Load LLM
        _llm = LLMService()
        await loop.run_in_executor(None, _llm.load)

        elapsed = time.time() - start
        logger.info(f"Shared models loaded in {elapsed:.1f}s")

        _models_loaded = True
        _models_loading = False


class PipelineSession:
    """
    Per-session pipeline state.
    Holds VAD, emotion, mood tracker, safety, and TTS instances.
    STT and LLM are shared across sessions.
    """

    def __init__(self):
        self._vad = VADService(
            energy_threshold=0.008,
            min_speech_ms=300,
            min_silence_ms=800,
        )
        self._emotion = EmotionService()
        self._mood = MoodTracker(window_size=3, safety_threshold=4.0)
        self._tts = TTSService()
        self._safety = SafetyService()
        self._on_message: Optional[Callable[[dict], Awaitable[None]]] = None
        self._processing = False
        self._ready = False

    async def start(self) -> bool:
        """Initialize session and load models if needed."""
        try:
            # Load shared models (no-op if already loaded)
            await ensure_models_loaded()

            # Load per-session services
            self._emotion.load()
            self._tts.load()
            self._safety.load()

            self._ready = True
            logger.info("Pipeline session started")
            return True
        except Exception as e:
            logger.error(f"Pipeline session start failed: {e}")
            return False

    def set_message_handler(self, handler: Callable[[dict], Awaitable[None]]):
        """Set the callback for pipeline output messages."""
        self._on_message = handler

    async def process_audio(self, pcm_bytes: bytes):
        """
        Process incoming PCM16 audio bytes.
        Runs VAD; when speech ends, triggers the full pipeline.
        """
        if not self._ready or self._processing:
            return

        # Feed to VAD
        speech_segment = self._vad.feed_audio(pcm_bytes)

        if speech_segment and len(speech_segment) > 0:
            # Speech segment complete -- run full pipeline
            self._processing = True
            try:
                await self._run_pipeline(speech_segment)
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                if self._on_message:
                    await self._on_message({
                        "type": "error",
                        "code": "PIPELINE_ERROR",
                        "detail": str(e),
                    })
            finally:
                self._processing = False

    async def _run_pipeline(self, speech_bytes: bytes):
        """Execute the full pipeline on a completed speech segment."""
        global _stt, _llm

        pipeline_start = time.time()

        # ── Layer 2: Speech-to-Text ──
        stt_result = await _stt.transcribe(speech_bytes)

        if not stt_result["success"] or not stt_result["text"].strip():
            logger.debug("STT produced no text -- skipping pipeline")
            return

        transcript = stt_result["text"]
        logger.info(f"STT ({stt_result['duration_ms']}ms): {transcript}")

        # Emit transcript
        if self._on_message:
            await self._on_message({
                "type": "transcript",
                "text": transcript,
                "final": True,
                "duration_ms": stt_result["duration_ms"],
            })

        # ── Layer 3: Emotion Classification ──
        emotion_result = self._emotion.classify(transcript)
        emotion_label = emotion_result["label"]
        emotion_confidence = emotion_result["confidence"]

        logger.info(
            f"Emotion ({emotion_result['duration_ms']}ms): "
            f"{emotion_label} ({emotion_confidence * 100:.0f}%)"
        )

        if self._on_message:
            await self._on_message({
                "type": "emotion",
                "label": emotion_label,
                "confidence": emotion_confidence,
                "scores": emotion_result["scores"],
                "duration_ms": emotion_result["duration_ms"],
            })

        # ── Layer 3b: Mood Update ──
        mood_result = self._mood.update(emotion_label)

        logger.info(
            f"Mood: {mood_result['score']}/5 "
            f"(avg: {mood_result['window_average']}, "
            f"safety: {'YES' if mood_result['safety_triggered'] else 'no'})"
        )

        if self._on_message:
            await self._on_message({
                "type": "mood_update",
                "score": mood_result["score"],
                "window_average": mood_result["window_average"],
                "window": mood_result["window"],
                "safety_triggered": mood_result["safety_triggered"],
            })

        # ── Safety Check ──
        if mood_result["safety_triggered"]:
            logger.warning("SAFETY ESCALATION TRIGGERED")
            escalation = self._safety.get_escalation_response()
            if self._on_message:
                await self._on_message({
                    "type": "safety_alert",
                    "message": escalation["message"],
                    "resources": escalation["resources"],
                })
            return  # Skip LLM on safety path

        # ── Layer 4: LLM Response ──
        llm_result = await _llm.generate(
            user_text=transcript,
            emotion=emotion_label,
            confidence=emotion_confidence,
        )

        response_text = llm_result["text"]
        logger.info(f"LLM ({llm_result['duration_ms']}ms): {response_text}")

        if self._on_message:
            await self._on_message({
                "type": "response",
                "text": response_text,
                "tokens_generated": llm_result["tokens_generated"],
                "tokens_per_second": llm_result["tokens_per_second"],
                "duration_ms": llm_result["duration_ms"],
            })

        # ── Layer 5: TTS ──
        if response_text and self._on_message:
            await self._on_message({"type": "tts_start"})

            import base64
            async for chunk in self._tts.synthesize(response_text):
                chunk_b64 = base64.b64encode(chunk).decode("ascii")
                await self._on_message({
                    "type": "tts_chunk",
                    "audio": chunk_b64,
                })

            await self._on_message({"type": "tts_end"})

        total_ms = int((time.time() - pipeline_start) * 1000)
        logger.info(f"Pipeline total: {total_ms}ms")

    async def stop(self):
        """Clean up session resources."""
        self._vad.reset()
        self._mood.reset()
        self._ready = False
        logger.info("Pipeline session stopped")

    @property
    def is_ready(self) -> bool:
        return self._ready
