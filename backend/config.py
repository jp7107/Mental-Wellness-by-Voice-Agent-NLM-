# ============================================
# MIND EASE — Backend Configuration
# ============================================
# Pydantic Settings model — reads from .env file
# and environment variables.

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="info", alias="LOG_LEVEL")

    # Model paths
    whisper_model_path: str = Field(
        default="models/whisper-small-q4.bin",
        alias="WHISPER_MODEL_PATH"
    )
    phi3_model_path: str = Field(
        default="models/phi-3-mini-4k-q4.gguf",
        alias="PHI3_MODEL_PATH"
    )
    emotion_model_path: str = Field(
        default="models/qwen2.5-emotion-lora/adapter_model.onnx",
        alias="EMOTION_MODEL_PATH"
    )
    kokoro_model_path: str = Field(
        default="models/kokoro-v0.19/kokoro-v0_19.onnx",
        alias="KOKORO_MODEL_PATH"
    )
    kokoro_voices_path: str = Field(
        default="models/kokoro-v0.19/voices",
        alias="KOKORO_VOICES_PATH"
    )

    # Engine
    engine_binary_path: str = Field(
        default="engine/build/mindease_engine",
        alias="ENGINE_BINARY_PATH"
    )
    engine_log_level: str = Field(default="warn", alias="ENGINE_LOG_LEVEL")

    # Config files
    pipeline_config_path: str = Field(
        default="config/pipeline.yaml",
        alias="PIPELINE_CONFIG_PATH"
    )
    safety_config_path: str = Field(
        default="config/safety_responses.yaml",
        alias="SAFETY_CONFIG_PATH"
    )

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a path relative to the project root."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        env_file_encoding = "utf-8"
        populate_by_name = True


# Singleton instance
settings = Settings()
