# ============================================
# MIND EASE — Emotion Classification + Mood Tracking
# ============================================
# Keyword-based heuristic emotion classifier with
# rolling mood window and safety escalation detection.
# Ported from engine/src/emotion.cpp and mood_tracker.cpp

import logging
from collections import deque
from typing import Optional

logger = logging.getLogger("mindease.emotion")

# ── Emotion Labels ──
EMOTION_LABELS = ["calm", "sad", "anxious", "angry", "fearful", "distressed"]

# ── Keyword Sets for Classification ──
EMOTION_KEYWORDS = {
    "sad": {
        "keywords": [
            "sad", "crying", "depressed", "hopeless", "lonely", "alone", "grief",
            "miss", "hurt", "loss", "empty", "tears", "miserable", "terrible", "bad",
            "unhappy", "broken", "worthless", "pain", "sorrow", "struggle", "struggling",
            "heartbroken", "devastated", "down", "blue", "gloomy", "tired", "exhausted",
        ],
        "weight": 0.6,
    },
    "anxious": {
        "keywords": [
            "anxious", "worried", "nervous", "stress", "panic",
            "overwhelmed", "overthinking", "restless", "uneasy",
            "tense", "dread", "can't stop thinking",
            "stressed", "anxiety", "worrying", "freaking out",
            "on edge", "jittery",
        ],
        "weight": 0.6,
    },
    "angry": {
        "keywords": [
            "angry", "furious", "rage", "hate", "frustrated",
            "annoyed", "irritated", "mad", "pissed", "resentful",
            "livid", "enraged", "bitter", "hostile",
        ],
        "weight": 0.55,
    },
    "fearful": {
        "keywords": [
            "afraid", "terrified", "frightened", "scared", "phobia",
            "threat", "danger", "helpless", "vulnerable", "trapped",
            "fear", "horror", "dread", "petrified",
        ],
        "weight": 0.65,
    },
    "distressed": {
        "keywords": [
            "die", "kill", "suicide", "end it", "can't go on",
            "no point", "give up", "self-harm", "cutting", "overdose",
            "want to die", "better off dead", "no reason to live",
            "kill myself", "end my life", "don't want to be here",
        ],
        "weight": 0.85,
    },
    "calm": {
        "keywords": [
            "okay", "fine", "good", "better", "calm", "peaceful",
            "relaxed", "happy", "grateful", "content", "well",
            "great", "wonderful", "amazing", "joy", "cheerful",
        ],
        "weight": 0.4,
    },
}

# ── Mood Score Mapping ──
MOOD_SCORE_MAP = {
    "calm": 1,
    "sad": 2,
    "anxious": 3,
    "angry": 3,
    "fearful": 4,
    "distressed": 5,
}


class EmotionService:
    """Keyword-based emotion classifier with mood tracking."""

    def __init__(self):
        self._ready = False

    def load(self) -> bool:
        self._ready = True
        logger.info("Emotion classifier loaded (keyword-based heuristic)")
        return True

    def classify(self, text: str) -> dict:
        """
        Classify the emotion of a text string.
        Returns dict with: label, confidence, scores, duration_ms
        """
        import time
        start = time.time()

        if not text or not text.strip():
            return {
                "label": "calm",
                "confidence": 0.0,
                "scores": {label: 0.0 for label in EMOTION_LABELS},
                "duration_ms": 0,
            }

        lower_text = text.lower()
        scores = {}
        max_score = 0.0
        max_label = "calm"

        for emotion, config in EMOTION_KEYWORDS.items():
            score = 0.0
            matches = 0

            for kw in config["keywords"]:
                if kw in lower_text:
                    matches += 1
                    score += config["weight"]

            if matches > 0:
                # Normalize and apply diminishing returns
                score = score / matches
                score = min(score * (1.0 + 0.15 * (matches - 1)), 0.95)

            scores[emotion] = score

            if score > max_score:
                max_score = score
                max_label = emotion

        # If no keywords matched, default to calm
        if max_score < 0.01:
            scores["calm"] = 0.5
            max_score = 0.5
            max_label = "calm"

        # Softmax normalization
        import math
        exp_sum = sum(math.exp(s) for s in scores.values())
        if exp_sum > 0:
            normalized = {k: math.exp(v) / exp_sum for k, v in scores.items()}
        else:
            normalized = scores

        duration_ms = int((time.time() - start) * 1000)

        return {
            "label": max_label,
            "confidence": round(max_score, 4),
            "scores": {k: round(v, 4) for k, v in normalized.items()},
            "duration_ms": duration_ms,
        }

    @property
    def is_ready(self) -> bool:
        return self._ready


class MoodTracker:
    """Rolling window mood tracker with safety escalation."""

    def __init__(self, window_size: int = 3, safety_threshold: float = 4.0):
        self._window: deque = deque(maxlen=window_size)
        self._window_size = window_size
        self._safety_threshold = safety_threshold
        self._safety_active = False

    def update(self, emotion_label: str) -> dict:
        """
        Update mood with a new emotion classification.
        Returns dict with: score, window_average, window, safety_triggered
        """
        score = MOOD_SCORE_MAP.get(emotion_label, 1)
        self._window.append(score)

        window_list = list(self._window)
        avg = sum(window_list) / len(window_list) if window_list else 0.0

        # Safety: triggered when ALL scores in full window are >= threshold
        safety_triggered = False
        if len(self._window) >= self._window_size:
            if all(s >= self._safety_threshold for s in self._window):
                safety_triggered = True
                self._safety_active = True

        return {
            "score": score,
            "window_average": round(avg, 2),
            "window": window_list,
            "safety_triggered": safety_triggered,
        }

    @property
    def is_safety_active(self) -> bool:
        return self._safety_active

    def reset(self):
        self._window.clear()
        self._safety_active = False
