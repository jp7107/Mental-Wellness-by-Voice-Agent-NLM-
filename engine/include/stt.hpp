#pragma once
// ============================================
// MIND EASE — Speech-to-Text
// ============================================
// Wraps whisper.cpp for fast local transcription.
// Uses greedy decoding with 4-bit quantized model.

#include <string>
#include <vector>
#include <cstdint>

namespace mindease {

struct STTConfig {
    std::string model_path;
    std::string language       = "en";
    int         beam_size      = 1;       // 1 = greedy (fastest)
    int         max_tokens     = 120;
    int         threads        = 4;
    bool        translate      = false;
};

struct TranscriptionResult {
    std::string text;
    float       confidence = 0.0f;
    int64_t     duration_ms = 0;       // Processing time
    bool        success = false;
};

class STT {
public:
    explicit STT(const STTConfig& config);
    ~STT();

    // Non-copyable
    STT(const STT&) = delete;
    STT& operator=(const STT&) = delete;

    /// Load the Whisper model. Returns false on failure.
    bool load_model();

    /// Transcribe audio samples (float32, mono, 16kHz).
    TranscriptionResult transcribe(const std::vector<float>& audio);

    /// Check if model is loaded and ready.
    bool is_ready() const;

private:
    STTConfig config_;
    [[maybe_unused]] void* ctx_ = nullptr;   // whisper_context* (opaque)
    bool      ready_ = false;
};

} // namespace mindease
