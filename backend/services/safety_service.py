import yaml
import logging
from config import settings

logger = logging.getLogger("mindease.safety")


class SafetyService:
    """Deterministic safety escalation — no LLM involved."""

    def __init__(self):
        self._config: dict = {}
        self._response_index: int = 0
        self._cooldown_count: int = 0

    def load(self):
        path = settings.resolve_path(settings.safety_config_path)
        if path.exists():
            with open(path) as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Safety config loaded from {path}")
        else:
            logger.warning(f"Safety config not found at {path}, using defaults")
            self._config = self._default_config()

    def get_escalation_response(self) -> dict:
        """Return the next escalation response (round-robin) with crisis resources."""
        responses = self._config.get("escalation", {}).get("responses", [])
        if not responses:
            responses = self._default_config()["escalation"]["responses"]

        idx = self._response_index % len(responses)
        self._response_index += 1
        self._cooldown_count = self._config.get("escalation", {}).get(
            "cooldown_turns", 5
        )

        resources = self._config.get("escalation", {}).get("crisis_resources", [])

        return {
            "message": responses[idx]["text"].strip(),
            "resources": resources,
        }

    def get_cooldown_message(self) -> str:
        return (
            self._config.get("escalation", {})
            .get(
                "cooldown_message",
                "I'm still here with you. Remember support resources are available anytime.",
            )
            .strip()
        )

    def tick_cooldown(self) -> bool:
        """Decrement cooldown counter. Returns True when cooldown is over."""
        if self._cooldown_count > 0:
            self._cooldown_count -= 1
            return self._cooldown_count == 0
        return True

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown_count > 0

    def _default_config(self) -> dict:
        return {
            "escalation": {
                "cooldown_turns": 5,
                "responses": [
                    {
                        "text": "I hear you, and I want you to know you're "
                        "not alone. Would you like me to share "
                        "some immediate support resources?"
                    },
                    {
                        "text": "What you're feeling sounds really difficult. "
                        "Please know that help is available right now."
                    },
                ],
                "crisis_resources": [
                    {
                        "name": "988 Suicide & Crisis Lifeline",
                        "contact": "Call or text 988",
                        "region": "US",
                    },
                    {"name": "iCall", "contact": "9152987821", "region": "India"},
                    {
                        "name": "Crisis Text Line",
                        "contact": "Text HOME to 741741",
                        "region": "US",
                    },
                ],
                "cooldown_message": "I'm still here with you. Support is "
                "always available.",
            }
        }
