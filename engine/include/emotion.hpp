#pragma once
// ============================================
// MIND EASE — Emotion Classifier
// ============================================
// Runs Qwen2.5-1.5B (LoRA fine-tuned, exported to ONNX)
// for real-time emotion classification from text.

#include <string>
#include <vector>
#include <array>
#include <unordered_map>

namespace mindease {

// Fixed emotion classes matching config/pipeline.yaml
enum class EmotionLabel : int {
    CALM       = 0,
    SAD        = 1,
    ANXIOUS    = 2,
    ANGRY      = 3,
    FEARFUL    = 4,
    DISTRESSED = 5,
    UNKNOWN    = -1
};

struct EmotionResult {
    EmotionLabel                         primary_emotion = EmotionLabel::UNKNOWN;
    float                                confidence = 0.0f;
    std::array<float, 6>                 scores = {};   // Indexed by EmotionLabel
    int64_t                              duration_ms = 0;
    bool                                 success = false;
};

struct EmotionConfig {
    std::string model_path;
    float       confidence_threshold = 0.3f;
    int         threads             = 2;
};

class EmotionClassifier {
public:
    explicit EmotionClassifier(const EmotionConfig& config);
    ~EmotionClassifier();

    // Non-copyable
    EmotionClassifier(const EmotionClassifier&) = delete;
    EmotionClassifier& operator=(const EmotionClassifier&) = delete;

    /// Load the ONNX model. Returns false on failure.
    bool load_model();

    /// Classify the emotion of a text transcription.
    EmotionResult classify(const std::string& text);

    /// Check if model is loaded and ready.
    bool is_ready() const;

    /// Get the string name of an emotion label.
    static std::string label_to_string(EmotionLabel label);

    /// Get the EmotionLabel from a string name.
    static EmotionLabel string_to_label(const std::string& name);

private:
    EmotionConfig config_;
    void*         session_ = nullptr;  // OrtSession* (opaque)
    bool          ready_ = false;

    std::vector<float> tokenize_and_encode(const std::string& text);
};

} // namespace mindease
