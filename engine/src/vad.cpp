// ============================================
// MIND EASE — Voice Activity Detection
// ============================================
// Energy-based VAD with optional neural model support.
// Falls back to energy-only detection when Silero model unavailable.

#include "vad.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace mindease {

VAD::VAD(const VADConfig& config) : config_(config) {}

VAD::~VAD() = default;

bool VAD::init(const std::string& model_path) {
    // Silero VAD model loading would go here when available.
    // For now, we use energy-based detection which is
    // fast (~0.1ms) and sufficiently accurate for clear speech.
    if (!model_path.empty()) {
        std::cerr << "[VAD] Neural model path provided but not yet loaded. "
                  << "Using energy-based detection.\n";
    }
    reset();
    return true;
}

void VAD::process(const std::vector<float>& samples) {
    if (samples.empty() || !callback_) return;

    // Process in windows
    const size_t window = static_cast<size_t>(config_.window_size_samples);

    for (size_t i = 0; i < samples.size(); i += window) {
        size_t end = std::min(i + window, samples.size());
        std::vector<float> win(samples.begin() + i, samples.begin() + end);

        float energy = compute_energy(win);
        float prob   = compute_vad_probability(win);

        bool is_speech = (prob >= config_.threshold) && (energy >= config_.energy_threshold);

        if (is_speech) {
            silence_samples_ = 0;
            speech_samples_ += static_cast<int>(win.size());

            int speech_ms = (speech_samples_ * 1000) / config_.sample_rate;

            if (!speaking_ && speech_ms >= config_.min_speech_duration_ms) {
                speaking_ = true;
                callback_(VADEvent::SPEECH_START, speech_buffer_);
            }

            // Accumulate speech audio
            speech_buffer_.insert(speech_buffer_.end(), win.begin(), win.end());

            if (speaking_) {
                callback_(VADEvent::SPEECH_CONTINUE, win);
            }
        } else {
            silence_samples_ += static_cast<int>(win.size());
            int silence_ms = (silence_samples_ * 1000) / config_.sample_rate;

            if (speaking_ && silence_ms >= config_.min_silence_duration_ms) {
                // Speech ended — fire event with accumulated buffer
                speaking_ = false;
                callback_(VADEvent::SPEECH_END, speech_buffer_);

                // Reset for next utterance
                speech_buffer_.clear();
                speech_samples_ = 0;
                silence_samples_ = 0;
            } else if (!speaking_) {
                callback_(VADEvent::SILENCE, win);
            }
        }
    }
}

void VAD::set_callback(VADCallback callback) {
    callback_ = std::move(callback);
}

void VAD::reset() {
    speech_buffer_.clear();
    speaking_ = false;
    silence_samples_ = 0;
    speech_samples_  = 0;
}

const std::vector<float>& VAD::get_speech_buffer() const {
    return speech_buffer_;
}

bool VAD::is_speaking() const {
    return speaking_;
}

float VAD::compute_energy(const std::vector<float>& samples) const {
    if (samples.empty()) return 0.0f;

    float sum_sq = 0.0f;
    for (float s : samples) {
        sum_sq += s * s;
    }
    return std::sqrt(sum_sq / static_cast<float>(samples.size()));
}

float VAD::compute_vad_probability(const std::vector<float>& window) const {
    // Energy-based probability estimation.
    // Maps RMS energy to a [0, 1] probability using a sigmoid-like curve.
    // Tuned so that config_.energy_threshold maps to ~0.5 probability.
    float energy = compute_energy(window);

    if (energy < 1e-7f) return 0.0f;

    // Sigmoid mapping: p = 1 / (1 + exp(-k * (energy - threshold)))
    float k = 20.0f / config_.energy_threshold;  // Steepness
    float x = energy - config_.energy_threshold;
    float p = 1.0f / (1.0f + std::exp(-k * x));

    return p;
}

} // namespace mindease
