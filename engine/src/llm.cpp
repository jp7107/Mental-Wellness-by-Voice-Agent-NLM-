// ============================================
// MIND EASE — LLM Response Generator (llama.cpp)
// ============================================
// Wraps llama.cpp for Phi-3-mini-4k-instruct.
// Falls back to empathetic template responses when unavailable.

#include "llm.hpp"
#include <chrono>
#include <iostream>
#include <sstream>
#include <cstdio>

#if HAS_LLAMA
#include "llama.h"
#endif

namespace mindease {

// ============================================
// Template responses for stub/fallback mode
// ============================================
namespace {

struct TemplateResponse {
    EmotionLabel emotion;
    std::vector<std::string> responses;
};

static const std::vector<TemplateResponse> TEMPLATES = {
    { EmotionLabel::CALM, {
        "It's good to hear you're doing well. What's been on your mind?",
        "I'm glad you're feeling calm. Is there anything you'd like to talk about?",
        "That's wonderful. Sometimes it helps to check in even when things are okay.",
    }},
    { EmotionLabel::SAD, {
        "I hear the sadness in your words, and I want you to know that's okay. "
        "Would you like to tell me more about what's going on?",
        "It takes courage to share how you're feeling. I'm here to listen.",
        "Feeling sad is a natural response. You don't have to carry this alone.",
    }},
    { EmotionLabel::ANXIOUS, {
        "I can sense you're feeling anxious right now. Let's take a moment together. "
        "Can you try taking a slow, deep breath with me?",
        "Anxiety can feel overwhelming, but it does pass. What's weighing on you most?",
        "You're not alone in this. Let's focus on what we can manage right now.",
    }},
    { EmotionLabel::ANGRY, {
        "I can tell something has really upset you. It's okay to feel angry. "
        "Would you like to talk about what happened?",
        "Your feelings are valid. Sometimes naming what we're angry about can help.",
        "I hear your frustration. Let's work through this together.",
    }},
    { EmotionLabel::FEARFUL, {
        "It sounds like you're going through something frightening. "
        "I'm right here with you. What would feel most helpful right now?",
        "Fear can be really difficult to sit with. You're safe in this moment.",
        "I understand you're scared. Let's take this one step at a time.",
    }},
    { EmotionLabel::DISTRESSED, {
        "I can hear how much pain you're in, and I take that seriously. "
        "You matter, and support is available.",
        "What you're feeling sounds really intense. Please know you don't have to face this alone.",
        "I'm concerned about how you're feeling. Would it help to talk about what's happening?",
    }},
};

static int response_counter = 0;

std::string get_template_response(EmotionLabel emotion) {
    for (const auto& tmpl : TEMPLATES) {
        if (tmpl.emotion == emotion) {
            int idx = response_counter % static_cast<int>(tmpl.responses.size());
            response_counter++;
            return tmpl.responses[idx];
        }
    }
    return "I'm here and listening. Tell me more about how you're feeling.";
}

} // anonymous namespace

// ============================================
// LLM Implementation
// ============================================
LLM::LLM(const LLMConfig& config) : config_(config) {}

LLM::~LLM() {
#if HAS_LLAMA
    if (ctx_) {
        llama_free(static_cast<llama_context*>(ctx_));
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(static_cast<llama_model*>(model_));
        model_ = nullptr;
    }
#endif
}

bool LLM::load_model() {
#if HAS_LLAMA
    if (config_.model_path.empty()) {
        std::cerr << "[LLM] No model path — using template responses\n";
        ready_ = true;
        return true;
    }

    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.use_mmap = config_.use_mmap;

    auto* model = llama_model_load_from_file(config_.model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "[LLM] Failed to load model: " << config_.model_path
                  << " — falling back to templates\n";
        ready_ = true;
        return true;
    }
    model_ = model;

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config_.ctx_size;
    ctx_params.n_threads = config_.threads;
    ctx_params.n_threads_batch = config_.threads;

    auto* ctx = llama_init_from_model(static_cast<llama_model*>(model_), ctx_params);
    if (!ctx) {
        std::cerr << "[LLM] Failed to create context — falling back to templates\n";
        llama_model_free(static_cast<llama_model*>(model_));
        model_ = nullptr;
        ready_ = true;
        return true;
    }
    ctx_ = ctx;

    ready_ = true;
    std::cerr << "[LLM] Model loaded: " << config_.model_path << "\n";
    return true;
#else
    std::cerr << "[LLM] llama.cpp not available — using template responses\n";
    ready_ = true;
    return true;
#endif
}

LLMResponse LLM::generate(
    const std::string& user_text,
    EmotionLabel emotion,
    float emotion_confidence,
    TokenCallback on_token
) {
    LLMResponse response;

    if (!ready_) {
        response.text = "I'm here to listen. Please tell me what's on your mind.";
        response.success = false;
        return response;
    }

    auto start = std::chrono::high_resolution_clock::now();

    bool used_llama = false;

#if HAS_LLAMA
    if (ctx_ && model_) {
        std::string prompt = build_prompt(user_text, emotion, emotion_confidence);
        auto* model_ptr = static_cast<llama_model*>(model_);
        auto* ctx_ptr   = static_cast<llama_context*>(ctx_);

        const int max_input = 1024;
        std::vector<llama_token> tokens(max_input);
        auto vocab = llama_model_get_vocab(model_ptr);
        int n_prompt_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int>(prompt.size()),
            tokens.data(),
            max_input,
            true,
            false
        );

        if (n_prompt_tokens < 0) {
            std::cerr << "[LLM] Tokenization failed\n";
        } else {
            tokens.resize(n_prompt_tokens);
            used_llama = true;

            llama_memory_clear(llama_get_memory(ctx_ptr), true);

            llama_batch batch = llama_batch_init(max_input, 0, 1);
            for (int i = 0; i < n_prompt_tokens; i++) {
                batch.token[batch.n_tokens] = tokens[i];
                batch.pos[batch.n_tokens] = i;
                batch.n_seq_id[batch.n_tokens] = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens] = (i == n_prompt_tokens - 1);
                batch.n_tokens++;
            }

            if (llama_decode(ctx_ptr, batch) != 0) {
                std::cerr << "[LLM] Prompt decode failed\n";
                used_llama = false;
            } else {
                std::string generated_text;
                int n_cur = n_prompt_tokens;
                int n_gen = 0;

                auto* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(config_.temperature));
                llama_sampler_chain_add(smpl, llama_sampler_init_top_p(config_.top_p, 1));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
                
                auto* vocab = llama_model_get_vocab(model_ptr);

                while (n_gen < config_.max_tokens) {
                    llama_token new_token = llama_sampler_sample(smpl, ctx_ptr, -1);

                    if (llama_token_is_eog(vocab, new_token)) break;

                    char buf[256];
                    int piece_len = llama_token_to_piece(
                        vocab, new_token, buf, sizeof(buf), 0, true
                    );
                    if (piece_len > 0) {
                        std::string piece(buf, piece_len);
                        generated_text += piece;
                        n_gen++;
                        if (on_token) on_token(piece);
                    }

                    batch.n_tokens = 0; // clear batch
                    batch.token[batch.n_tokens] = new_token;
                    batch.pos[batch.n_tokens] = n_cur;
                    batch.n_seq_id[batch.n_tokens] = 1;
                    batch.seq_id[batch.n_tokens][0] = 0;
                    batch.logits[batch.n_tokens] = true;
                    batch.n_tokens++;
                    
                    n_cur++;

                    if (llama_decode(ctx_ptr, batch) != 0) {
                        std::cerr << "[LLM] Decode step failed at token " << n_gen << "\n";
                        break;
                    }
                }

                llama_sampler_free(smpl);
                llama_batch_free(batch);

                response.text = generated_text;
                response.tokens_generated = n_gen;
                response.success = !generated_text.empty();
            }
        }
    }
#endif

    if (!used_llama) {
        response.text = get_template_response(emotion);
        response.tokens_generated = 0;
        response.success = true;
        if (on_token) on_token(response.text);
    }

    auto end = std::chrono::high_resolution_clock::now();
    response.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    if (response.duration_ms > 0 && response.tokens_generated > 0) {
        response.tokens_per_second = static_cast<float>(response.tokens_generated) * 1000.0f
                                     / static_cast<float>(response.duration_ms);
    }

    return response;
}

bool LLM::is_ready() const {
    return ready_;
}

void LLM::reset_context() {
#if HAS_LLAMA
    if (ctx_) {
        llama_memory_clear(llama_get_memory(static_cast<llama_context*>(ctx_)), true);
    }
#endif
}

std::string LLM::build_prompt(
    const std::string& user_text,
    EmotionLabel emotion,
    float confidence
) {
    // Phi-3-mini instruct chat template
    std::ostringstream oss;

    // System message
    oss << "<|system|>\n";

    // Replace placeholders in system prompt
    std::string sys = config_.system_prompt;
    std::string emotion_str = EmotionClassifier::label_to_string(emotion);

    auto pos = sys.find("{emotion_label}");
    if (pos != std::string::npos) {
        sys.replace(pos, 15, emotion_str);
    }

    pos = sys.find("{confidence}");
    if (pos != std::string::npos) {
        char conf_buf[16];
        snprintf(conf_buf, sizeof(conf_buf), "%.0f", confidence * 100.0f);
        sys.replace(pos, 12, conf_buf);
    }

    oss << sys << "\n<|end|>\n";

    // User message
    oss << "<|user|>\n"
         << user_text
         << "\n<|end|>\n";

    // Assistant turn prompt
    oss << "<|assistant|>\n";

    return oss.str();
}

} // namespace mindease
