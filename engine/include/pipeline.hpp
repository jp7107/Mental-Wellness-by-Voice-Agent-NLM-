#pragma once
// ============================================
// MIND EASE — Pipeline Orchestrator
// ============================================
// Coordinates the 5-layer inference pipeline:
//   Audio → VAD → STT → Emotion+Mood (parallel) → LLM
// Communicates with Python backend via stdin/stdout IPC.

#include <string>
#include <memory>
#include <functional>
#include <atomic>
#include "vad.hpp"
#include "stt.hpp"
#include "emotion.hpp"
#include "mood_tracker.hpp"
#include "llm.hpp"

namespace mindease {

// IPC message types (JSON "type" field)
namespace ipc {
    constexpr const char* MSG_AUDIO        = "audio";
    constexpr const char* MSG_TRANSCRIPT   = "transcript";
    constexpr const char* MSG_EMOTION      = "emotion";
    constexpr const char* MSG_MOOD         = "mood_update";
    constexpr const char* MSG_RESPONSE     = "response";
    constexpr const char* MSG_SAFETY       = "safety_alert";
    constexpr const char* MSG_ERROR        = "error";
    constexpr const char* MSG_STATUS       = "status";
    constexpr const char* MSG_SHUTDOWN     = "shutdown";
}

struct PipelineConfig {
    // Model paths
    std::string whisper_model_path;
    std::string emotion_model_path;
    std::string llm_model_path;

    // Sub-configs
    VADConfig      vad;
    STTConfig      stt;
    EmotionConfig  emotion;
    MoodConfig     mood;
    LLMConfig      llm;
};

struct PipelineResult {
    TranscriptionResult  transcription;
    EmotionResult        emotion;
    MoodUpdate           mood;
    LLMResponse          response;
    int64_t              total_duration_ms = 0;
};

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    ~Pipeline();

    // Non-copyable
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /// Initialize all models. Returns false if any critical model fails.
    bool init();

    /// Process an audio chunk (PCM float32, mono, 16kHz).
    /// Results are sent via the output callback.
    void process_audio(const std::vector<float>& audio);

    /// Set the callback for pipeline results.
    /// Called asynchronously when a complete pipeline pass finishes.
    void set_result_callback(std::function<void(const PipelineResult&)> callback);

    /// Run the IPC event loop (reads from stdin, writes to stdout).
    /// This blocks until shutdown message received.
    void run_ipc_loop();

    /// Request graceful shutdown.
    void shutdown();

    /// Check if all models are loaded and ready.
    bool is_ready() const;

private:
    PipelineConfig config_;

    std::unique_ptr<VAD>               vad_;
    std::unique_ptr<STT>               stt_;
    std::unique_ptr<EmotionClassifier>  emotion_;
    std::unique_ptr<MoodTracker>        mood_;
    std::unique_ptr<LLM>               llm_;

    std::function<void(const PipelineResult&)> result_callback_;
    std::atomic<bool> running_{false};

    /// Execute the full pipeline on a completed speech segment.
    PipelineResult execute_pipeline(const std::vector<float>& speech_audio);

    /// IPC helpers
    void send_ipc_message(const std::string& json);
    std::string read_ipc_message();
    void handle_ipc_message(const std::string& json);

    /// Convert pipeline result to IPC JSON messages.
    void emit_results(const PipelineResult& result);
};

} // namespace mindease
