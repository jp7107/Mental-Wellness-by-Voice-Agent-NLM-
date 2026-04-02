// ============================================
// MIND EASE — Speech-to-Text (whisper.cpp)
// ============================================
// Wraps whisper.cpp C API for fast offline transcription.
// Falls back to a stub when whisper.cpp is not linked.

#include "stt.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>

#if HAS_WHISPER
#include "whisper.h"
#endif

namespace mindease {

STT::STT(const STTConfig& config) : config_(config) {}

STT::~STT() {
#if HAS_WHISPER
    if (ctx_) {
        whisper_free(static_cast<whisper_context*>(ctx_));
        ctx_ = nullptr;
    }
#endif
}

bool STT::load_model() {
#if HAS_WHISPER
    if (config_.model_path.empty()) {
        std::cerr << "[STT] No model path specified\n";
        return false;
    }

    auto params = whisper_context_default_params();
    params.use_gpu = true;  // Use Metal/CUDA if available

    auto* context = whisper_init_from_file_with_params(
        config_.model_path.c_str(), params
    );

    if (!context) {
        std::cerr << "[STT] Failed to load model: " << config_.model_path << "\n";
        return false;
    }

    ctx_ = context;
    ready_ = true;
    std::cerr << "[STT] Model loaded: " << config_.model_path << "\n";
    return true;
#else
    std::cerr << "[STT] whisper.cpp not available — using stub\n";
    ready_ = true;  // Stub is always "ready"
    return true;
#endif
}

TranscriptionResult STT::transcribe(const std::vector<float>& audio) {
    TranscriptionResult result;

    if (!ready_) {
        result.text = "";
        result.success = false;
        return result;
    }

    auto start = std::chrono::high_resolution_clock::now();

#if HAS_WHISPER
    auto* context = static_cast<whisper_context*>(ctx_);

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.n_threads       = config_.threads;
    params.language        = config_.language.c_str();
    params.translate       = config_.translate;
    params.no_timestamps   = true;
    params.single_segment  = true;
    params.max_tokens      = config_.max_tokens;
    params.print_progress  = false;
    params.print_realtime  = false;
    params.print_special   = false;
    params.print_timestamps = false;
    params.greedy.best_of  = config_.beam_size;

    int ret = whisper_full(context, params, audio.data(), static_cast<int>(audio.size()));

    if (ret != 0) {
        result.text = "";
        result.success = false;
        std::cerr << "[STT] Transcription failed with code: " << ret << "\n";
    } else {
        int n_segments = whisper_full_n_segments(context);
        std::string full_text;
        for (int i = 0; i < n_segments; ++i) {
            const char* seg = whisper_full_get_segment_text(context, i);
            if (seg) full_text += seg;
        }
        // Trim whitespace
        auto ltrim = full_text.find_first_not_of(" \t\n\r");
        auto rtrim = full_text.find_last_not_of(" \t\n\r");
        if (ltrim != std::string::npos) {
            result.text = full_text.substr(ltrim, rtrim - ltrim + 1);
        }
        result.success = !result.text.empty();
        result.confidence = 0.85f; // whisper.cpp doesn't expose per-segment confidence easily
    }
#else
    // Stub: return a placeholder for testing
    result.text = "[STT stub] Audio received (" + std::to_string(audio.size()) + " samples)";
    result.success = true;
    result.confidence = 1.0f;
#endif

    auto end = std::chrono::high_resolution_clock::now();
    result.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return result;
}

bool STT::is_ready() const {
    return ready_;
}

} // namespace mindease
