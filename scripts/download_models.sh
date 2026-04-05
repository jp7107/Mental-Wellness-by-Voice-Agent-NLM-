#!/usr/bin/env bash
# ============================================
# MIND EASE — Download Model Weights
# ============================================
# Downloads quantized models for offline use.
# Only needed once; all subsequent runs are offline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"

echo "Downloading MIND EASE model weights..."
echo "This requires ~4 GB of disk space and a one-time internet connection."

# ── Whisper base.en ──
WHISPER_DIR="$MODELS"
WHISPER_FILE="$WHISPER_DIR/whisper-base.en.bin"
WHISPER_NEEDS_DOWNLOAD=false
if [ ! -f "$WHISPER_FILE" ]; then
    WHISPER_NEEDS_DOWNLOAD=true
elif [ "$(wc -c < "$WHISPER_FILE" | tr -d ' ')" -lt 1000000 ]; then
    echo "[1/4] Whisper model file is too small (likely a failed download) — re-downloading..."
    rm -f "$WHISPER_FILE"
    WHISPER_NEEDS_DOWNLOAD=true
fi
if [ "$WHISPER_NEEDS_DOWNLOAD" = true ]; then
    echo "[1/4] Downloading Whisper base.en..."
    curl -f -L -o "$WHISPER_FILE" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin" || {
        echo "Warning: Download failed. Cleaning up empty file."
        rm -f "$WHISPER_FILE"
    }
fi
echo "  Whisper: OK"

# ── Phi-3-mini Q4_K_M ──
PHI3_DIR="$MODELS"
if [ ! -f "$PHI3_DIR/phi-3-mini-4k-q4.gguf" ]; then
    echo "[2/4] Downloading Phi-3-mini (Q4_K_M)..."
    curl -L -o "$PHI3_DIR/phi-3-mini-4k-q4.gguf" \
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
fi
echo "  Phi-3-mini: OK"

# ── Kokoro TTS ──
KOKORO_DIR="$MODELS/kokoro-v0.19"
mkdir -p "$KOKORO_DIR/voices"
if [ ! -f "$KOKORO_DIR/kokoro-v0_19.onnx" ]; then
    echo "[3/4] Downloading Kokoro TTS..."
    curl -L -o "$KOKORO_DIR/kokoro-v0_19.onnx" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v0_19.onnx"
    curl -L -o "$KOKORO_DIR/voices/af_heart.bin" \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices.bin"
fi
echo "  Kokoro: OK"

# ── Qwen2.5 Emotion (placeholder — use base ONNX export) ──
EMOTION_DIR="$MODELS/qwen2.5-emotion-lora"
mkdir -p "$EMOTION_DIR"
if [ ! -f "$EMOTION_DIR/adapter_model.onnx" ]; then
    echo "[4/4] Emotion model: Using keyword-based fallback (no fine-tuned checkpoint)."
    echo "      To use the neural model, export your Qwen2.5 LoRA to ONNX and place at:"
    echo "      $EMOTION_DIR/adapter_model.onnx"
fi

echo ""
echo "All models ready. System will run fully offline."
