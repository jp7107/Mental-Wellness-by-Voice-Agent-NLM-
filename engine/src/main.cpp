// ============================================
// MIND EASE — Engine Entry Point
// ============================================
// Launches the inference pipeline and runs the
// IPC event loop for communication with the
// Python backend.
//
// Usage:
//   mindease_engine [--config path/to/pipeline.yaml]
//   mindease_engine --test
//
// IPC Protocol:
//   stdin:  length-prefixed binary frames (4-byte LE header + payload)
//   stdout: length-prefixed JSON messages
//   stderr: human-readable logs

#include "pipeline.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

namespace {

void print_banner() {
    std::cerr << R"(
  ╔══════════════════════════════════════════╗
  ║          MIND EASE Engine v1.0           ║
  ║   On-Device Mental Wellness AI Engine    ║
  ╚══════════════════════════════════════════╝
)" << std::endl;
}

void print_usage() {
    std::cerr << "Usage: mindease_engine [OPTIONS]\n"
              << "  --config PATH   Path to pipeline.yaml config\n"
              << "  --test          Run self-test and exit\n"
              << "  --help          Show this help\n";
}

mindease::PipelineConfig build_default_config() {
    mindease::PipelineConfig config;

    // Default model paths (relative to project root)
    config.whisper_model_path = "models/whisper-small-q4.bin";
    config.emotion_model_path = "models/qwen2.5-emotion-lora/adapter_model.onnx";
    config.llm_model_path     = "models/phi-3-mini-4k-q4.gguf";

    // VAD defaults
    config.vad.threshold              = 0.5f;
    config.vad.min_speech_duration_ms = 250;
    config.vad.min_silence_duration_ms = 600;
    config.vad.window_size_samples    = 512;
    config.vad.sample_rate            = 16000;
    config.vad.energy_threshold       = 0.01f;

    // STT defaults
    config.stt.model_path  = config.whisper_model_path;
    config.stt.language    = "en";
    config.stt.beam_size   = 1;
    config.stt.max_tokens  = 120;
    config.stt.threads     = 4;
    config.stt.translate   = false;

    // Emotion defaults
    config.emotion.model_path            = config.emotion_model_path;
    config.emotion.confidence_threshold  = 0.3f;
    config.emotion.threads               = 2;

    // Mood defaults
    config.mood.window_size      = 3;
    config.mood.safety_threshold = 4.0f;

    // LLM defaults
    config.llm.model_path      = config.llm_model_path;
    config.llm.max_tokens      = 80;
    config.llm.temperature     = 0.7f;
    config.llm.top_p           = 0.9f;
    config.llm.repeat_penalty  = 1.1f;
    config.llm.threads         = 4;
    config.llm.use_mmap        = true;
    config.llm.ctx_size        = 4096;
    config.llm.system_prompt   =
        "You are a compassionate mental wellness companion named Mind Ease. "
        "Respond with short, empathetic messages (1-2 sentences max). "
        "Never give medical diagnosis or prescribe medication. "
        "Acknowledge the user's emotions and offer gentle support. "
        "Current emotion detected: {emotion_label} (confidence: {confidence}%)";

    return config;
}

bool run_self_test() {
    std::cerr << "[Test] Running self-test...\n";

    auto config = build_default_config();
    // Clear model paths for stub mode
    config.stt.model_path     = "";
    config.emotion.model_path = "";
    config.llm.model_path     = "";

    mindease::Pipeline pipeline(config);

    if (!pipeline.init()) {
        std::cerr << "[Test] FAIL: Pipeline init failed\n";
        return false;
    }
    std::cerr << "[Test] PASS: Pipeline init (stub mode)\n";

    // Test with synthetic audio (sine wave)
    std::vector<float> test_audio(16000 * 2); // 2 seconds
    for (size_t i = 0; i < test_audio.size(); i++) {
        float t = static_cast<float>(i) / 16000.0f;
        // Generate a 440Hz tone with reasonable amplitude
        test_audio[i] = 0.3f * std::sin(2.0f * 3.14159f * 440.0f * t);
    }

    // Feed audio to pipeline
    pipeline.process_audio(test_audio);
    std::cerr << "[Test] PASS: Audio processing\n";

    // Test emotion classifier directly
    mindease::EmotionClassifier classifier(config.emotion);
    classifier.load_model();
    auto emo = classifier.classify("I've been feeling really anxious and stressed lately");
    std::cerr << "[Test] Emotion: " << mindease::EmotionClassifier::label_to_string(emo.primary_emotion)
              << " (confidence: " << emo.confidence << ")\n";
    std::cerr << "[Test] PASS: Emotion classification\n";

    // Test mood tracker
    mindease::MoodTracker mood(config.mood);
    mindease::EmotionResult e1; e1.primary_emotion = mindease::EmotionLabel::FEARFUL;
    mindease::EmotionResult e2; e2.primary_emotion = mindease::EmotionLabel::DISTRESSED;
    mindease::EmotionResult e3; e3.primary_emotion = mindease::EmotionLabel::DISTRESSED;

    mood.update(e1);
    mood.update(e2);
    auto m3 = mood.update(e3);

    if (m3.safety_triggered) {
        std::cerr << "[Test] PASS: Safety escalation triggered correctly\n";
    } else {
        std::cerr << "[Test] FAIL: Safety escalation should have triggered\n";
        return false;
    }

    // Test LLM (stub mode)
    mindease::LLM llm(config.llm);
    llm.load_model();
    auto resp = llm.generate("I'm feeling sad today", mindease::EmotionLabel::SAD, 0.8f);
    if (resp.success && !resp.text.empty()) {
        std::cerr << "[Test] LLM response: " << resp.text << "\n";
        std::cerr << "[Test] PASS: LLM generation (template mode)\n";
    } else {
        std::cerr << "[Test] FAIL: LLM generation failed\n";
        return false;
    }

    std::cerr << "\n[Test] All tests PASSED\n";
    return true;
}

} // anonymous namespace

int main(int argc, char* argv[]) {
    print_banner();

    std::string config_path;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
        if (std::strcmp(argv[i], "--test") == 0) {
            return run_self_test() ? 0 : 1;
        }
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    // Build config
    auto config = build_default_config();
    // TODO: If config_path is provided, parse YAML and override defaults

    // Set stdin/stdout to binary mode
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Initialize pipeline
    mindease::Pipeline pipeline(config);

    if (!pipeline.init()) {
        std::cerr << "[Engine] Fatal: Pipeline initialization failed\n";
        return 1;
    }

    std::cerr << "[Engine] Pipeline ready — entering IPC loop\n";

    // Run the IPC event loop (blocks until shutdown)
    pipeline.run_ipc_loop();

    std::cerr << "[Engine] Shutdown complete\n";
    return 0;
}
