#pragma once
// ============================================
// MIND EASE — Voice Activity Detection
// ============================================
// Wraps Silero VAD for real-time speech boundary detection.
// Uses a rolling energy gate + neural VAD for robust
// speech/silence discrimination.

#include <vector>
#include <cstdint>
#include <functional>

namespace mindease {

struct VADConfig {
    float    threshold             = 0.5f;
    int      min_speech_duration_ms = 250;
    int      min_silence_duration_ms = 600;
    int      window_size_samples   = 512;
    int      sample_rate           = 16000;
    float    energy_threshold      = 0.01f;
};

enum class VADEvent {
    SPEECH_START,
    SPEECH_CONTINUE,
    SPEECH_END,
    SILENCE
};

using VADCallback = std::function<void(VADEvent, const std::vector<float>&)>;

class VAD {
public:
    explicit VAD(const VADConfig& config = {});
    ~VAD();

    // Non-copyable
    VAD(const VAD&) = delete;
    VAD& operator=(const VAD&) = delete;

    /// Initialize the VAD model. Returns false on failure.
    bool init(const std::string& model_path = "");

    /// Process a chunk of audio samples (float32, mono, 16kHz).
    /// Calls the registered callback when speech events are detected.
    void process(const std::vector<float>& samples);

    /// Set the callback for VAD events.
    void set_callback(VADCallback callback);

    /// Reset internal state (between sessions).
    void reset();

    /// Get the accumulated speech buffer since last SPEECH_START.
    const std::vector<float>& get_speech_buffer() const;

    /// Check if currently in speech.
    bool is_speaking() const;

private:
    VADConfig           config_;
    VADCallback         callback_;
    std::vector<float>  speech_buffer_;
    bool                speaking_ = false;
    int                 silence_samples_ = 0;
    int                 speech_samples_  = 0;

    float compute_energy(const std::vector<float>& samples) const;
    float compute_vad_probability(const std::vector<float>& window) const;
};

} // namespace mindease
