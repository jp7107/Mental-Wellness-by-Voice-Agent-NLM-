#pragma once
// ============================================
// MIND EASE — LLM Response Generator
// ============================================
// Wraps llama.cpp for local Phi-3-mini-4k-instruct inference.
// Hard-capped at 80 tokens with emotion-aware system prompt.

#include <string>
#include <functional>
#include "emotion.hpp"

namespace mindease {

struct LLMConfig {
    std::string model_path;
    std::string system_prompt;
    int         max_tokens      = 80;
    float       temperature     = 0.7f;
    float       top_p           = 0.9f;
    float       repeat_penalty  = 1.1f;
    int         threads         = 4;
    bool        use_mmap        = true;     // Memory mapping for large models
    int         ctx_size        = 4096;     // Context window size
};

struct LLMResponse {
    std::string text;
    int         tokens_generated = 0;
    int64_t     duration_ms = 0;
    float       tokens_per_second = 0.0f;
    bool        success = false;
    bool        safety_bypassed = false;    // True if response was safety override
};

// Callback for streaming tokens as they're generated
using TokenCallback = std::function<void(const std::string& token)>;

class LLM {
public:
    explicit LLM(const LLMConfig& config);
    ~LLM();

    // Non-copyable
    LLM(const LLM&) = delete;
    LLM& operator=(const LLM&) = delete;

    /// Load the model. Returns false on failure.
    bool load_model();

    /// Generate a response given user text and detected emotion.
    /// Optionally streams tokens via callback as they're produced.
    LLMResponse generate(
        const std::string& user_text,
        EmotionLabel emotion,
        float emotion_confidence,
        TokenCallback on_token = nullptr
    );

    /// Check if model is loaded and ready.
    bool is_ready() const;

    /// Reset conversation context (new session).
    void reset_context();

private:
    LLMConfig config_;
    [[maybe_unused]] void* model_ = nullptr;     // llama_model* (opaque)
    [[maybe_unused]] void* ctx_   = nullptr;     // llama_context* (opaque)
    bool      ready_ = false;

    std::string build_prompt(
        const std::string& user_text,
        EmotionLabel emotion,
        float confidence
    );
};

} // namespace mindease
