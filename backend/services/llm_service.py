# ============================================
# MIND EASE — LLM Response Generator (Python)
# ============================================
# Uses llama-cpp-python for local GGUF inference.
# Falls back to empathetic template responses.

import asyncio
import logging
import time
from typing import Optional

from config import settings

logger = logging.getLogger("mindease.llm")

TEMPLATES = {
    "calm": [
        "It's good to hear you're doing well. What's been on your mind?",
        "I'm glad you're feeling calm. Is there anything you'd like to talk about?",
        "That's wonderful. Sometimes it helps to check in even when things are okay.",
    ],
    "sad": [
        "I hear the sadness in your words, and I want you to know that's okay. Would you like to tell me more about what's going on?",
        "It takes courage to share how you're feeling. I'm here to listen.",
        "Feeling sad is a natural response. You don't have to carry this alone.",
    ],
    "anxious": [
        "I can sense you're feeling anxious right now. Let's take a moment together. Can you try taking a slow, deep breath with me?",
        "Anxiety can feel overwhelming, but it does pass. What's weighing on you most?",
        "You're not alone in this. Let's focus on what we can manage right now.",
    ],
    "angry": [
        "I can tell something has really upset you. It's okay to feel angry. Would you like to talk about what happened?",
        "Your feelings are valid. Sometimes naming what we're angry about can help.",
        "I hear your frustration. Let's work through this together.",
    ],
    "fearful": [
        "It sounds like you're going through something frightening. I'm right here with you. What would feel most helpful right now?",
        "Fear can be really difficult to sit with. You're safe in this moment.",
        "I understand you're scared. Let's take this one step at a time.",
    ],
    "distressed": [
        "I can hear how much pain you're in, and I take that seriously. You matter, and support is available.",
        "What you're feeling sounds really intense. Please know you don't have to face this alone.",
        "I'm concerned about how you're feeling. Would it help to talk about what's happening?",
    ],
}

_template_counter = 0


def _get_template_response(emotion: str) -> str:
    global _template_counter
    responses = TEMPLATES.get(emotion, TEMPLATES["calm"])
    idx = _template_counter % len(responses)
    _template_counter += 1
    return responses[idx]


SYSTEM_PROMPT = (
    "You are a compassionate mental wellness companion named Mind Ease. "
    "Respond with short, empathetic messages (1-2 sentences max). "
    "Never give medical diagnosis or prescribe medication. "
    "Acknowledge the user's emotions and offer gentle support. "
    "Current emotion detected: {emotion} (confidence: {confidence}%)"
)


class LLMService:
    """Local LLM inference via llama-cpp-python with template fallback."""

    def __init__(self):
        self._model = None
        self._ready = False
        self._use_templates = True

    def load(self) -> bool:
        model_path = settings.resolve_path(settings.phi3_model_path)

        if not model_path.exists():
            logger.warning(f"LLM model not found at {model_path} -- using templates")
            self._ready = True
            return True

        file_size = model_path.stat().st_size
        if file_size < 100_000_000:
            logger.warning(
                f"LLM model file too small ({file_size / 1e6:.1f}MB) -- "
                f"expected 2GB+ for a real model. Using template responses."
            )
            self._ready = True
            return True

        try:
            from llama_cpp import Llama

            logger.info(f"Loading LLM: {model_path} ({file_size / 1e9:.1f}GB)")
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=0,
                verbose=False,
            )
            self._use_templates = False
            self._ready = True
            logger.info("LLM model loaded successfully")
            return True
        except ImportError:
            logger.warning("llama-cpp-python not installed -- using templates")
            self._ready = True
            return True
        except Exception as e:
            logger.error(f"LLM load error: {e} -- using templates")
            self._ready = True
            return True

    async def generate(
        self, user_text: str, emotion: str = "calm", confidence: float = 0.5,
    ) -> dict:
        start = time.time()

        if not self._ready:
            return {
                "text": "I'm here to listen. Please tell me what's on your mind.",
                "tokens_generated": 0, "tokens_per_second": 0.0,
                "duration_ms": 0, "success": False,
            }

        if self._use_templates or self._model is None:
            text = _get_template_response(emotion)
            duration_ms = int((time.time() - start) * 1000)
            return {
                "text": text, "tokens_generated": 0, "tokens_per_second": 0.0,
                "duration_ms": duration_ms, "success": True,
            }

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._do_generate, user_text, emotion, confidence,
            )
            duration_ms = int((time.time() - start) * 1000)
            tokens = result.get("tokens_generated", 0)
            tps = (tokens * 1000 / duration_ms) if duration_ms > 0 and tokens > 0 else 0.0
            return {
                "text": result["text"], "tokens_generated": tokens,
                "tokens_per_second": round(tps, 1),
                "duration_ms": duration_ms, "success": True,
            }
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            text = _get_template_response(emotion)
            duration_ms = int((time.time() - start) * 1000)
            return {
                "text": text, "tokens_generated": 0, "tokens_per_second": 0.0,
                "duration_ms": duration_ms, "success": True,
            }

    def _do_generate(self, user_text: str, emotion: str, confidence: float) -> dict:
        system_prompt = SYSTEM_PROMPT.format(
            emotion=emotion, confidence=f"{confidence * 100:.0f}",
        )

        # Build Phi-3 instruct prompt using string concatenation to avoid XML-like tag issues
        sys_open = "<" + "|system|" + ">"
        sys_close = "<" + "|end|" + ">"
        usr_open = "<" + "|user|" + ">"
        asst_open = "<" + "|assistant|" + ">"

        prompt = (
            f"{sys_open}\n{system_prompt}\n{sys_close}\n"
            f"{usr_open}\n{user_text}\n{sys_close}\n"
            f"{asst_open}\n"
        )

        try:
            output = self._model(
                prompt,
                max_tokens=80,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=[sys_close, sys_open, usr_open],
                echo=False,
            )

            text = output["choices"][0]["text"].strip() if output["choices"] else ""
            tokens_generated = output.get("usage", {}).get("completion_tokens", 0)

            if not text:
                text = _get_template_response(emotion)

            return {"text": text, "tokens_generated": tokens_generated}
        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            return {"text": _get_template_response(emotion), "tokens_generated": 0}

    @property
    def is_ready(self) -> bool:
        return self._ready

