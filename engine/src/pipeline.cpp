// ============================================
// MIND EASE — Pipeline Orchestrator
// ============================================
// Coordinates the full inference pipeline with
// parallel emotion + mood processing via std::async.
// Communicates with Python backend via stdin/stdout IPC.

#include "pipeline.hpp"
#include <chrono>
#include <iostream>
#include <sstream>
#include <future>
#include <cstring>

// Minimal JSON helpers (avoids external dependency)
namespace json_util {

std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

std::string to_json_obj(std::initializer_list<std::pair<std::string, std::string>> fields) {
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (auto& [k, v] : fields) {
        if (!first) oss << ",";
        oss << "\"" << k << "\":" << v;
        first = false;
    }
    oss << "}";
    return oss.str();
}

std::string str_val(const std::string& s) { return "\"" + escape(s) + "\""; }
std::string num_val(float v)   { char b[32]; snprintf(b, sizeof(b), "%.4f", v); return b; }
std::string num_val(int v)     { return std::to_string(v); }
std::string num_val(int64_t v) { return std::to_string(v); }
std::string bool_val(bool v)   { return v ? "true" : "false"; }

std::string arr_val(const std::vector<int>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) oss << ",";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

std::string arr_val(const std::array<float, 6>& v) {
    std::ostringstream oss;
    oss << "{";
    const char* labels[] = {"calm", "sad", "anxious", "angry", "fearful", "distressed"};
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) oss << ",";
        char buf[32];
        snprintf(buf, sizeof(buf), "%.4f", v[i]);
        oss << "\"" << labels[i] << "\":" << buf;
    }
    oss << "}";
    return oss.str();
}

// Simple JSON value extractor (for "type" field only)
std::string get_type(const std::string& json) {
    auto pos = json.find("\"type\"");
    if (pos == std::string::npos) return "";
    pos = json.find("\"", pos + 6);
    if (pos == std::string::npos) return "";
    auto end = json.find("\"", pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

} // namespace json_util

namespace mindease {

Pipeline::Pipeline(const PipelineConfig& config) : config_(config) {}

Pipeline::~Pipeline() {
    shutdown();
}

bool Pipeline::init() {
    std::cerr << "[Pipeline] Initializing models...\n";
    auto start = std::chrono::high_resolution_clock::now();

    // Initialize VAD
    vad_ = std::make_unique<VAD>(config_.vad);
    if (!vad_->init()) {
        std::cerr << "[Pipeline] VAD init failed\n";
        return false;
    }

    // Initialize STT
    stt_ = std::make_unique<STT>(config_.stt);
    if (!stt_->load_model()) {
        std::cerr << "[Pipeline] STT model load failed\n";
        return false;
    }

    // Initialize Emotion Classifier
    emotion_ = std::make_unique<EmotionClassifier>(config_.emotion);
    if (!emotion_->load_model()) {
        std::cerr << "[Pipeline] Emotion model load failed\n";
        return false;
    }

    // Initialize Mood Tracker
    mood_ = std::make_unique<MoodTracker>(config_.mood);

    // Initialize LLM
    llm_ = std::make_unique<LLM>(config_.llm);
    if (!llm_->load_model()) {
        std::cerr << "[Pipeline] LLM model load failed\n";
        return false;
    }

    // Set up VAD callback
    vad_->set_callback([this](VADEvent event, const std::vector<float>& audio) {
        if (event == VADEvent::SPEECH_END) {
            // Speech ended — run the full pipeline
            auto result = execute_pipeline(audio);
            emit_results(result);
        }
    });

    auto end = std::chrono::high_resolution_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cerr << "[Pipeline] All models loaded in " << load_ms << "ms\n";

    running_ = true;
    return true;
}

void Pipeline::process_audio(const std::vector<float>& audio) {
    if (!running_ || !vad_) return;
    vad_->process(audio);
}

void Pipeline::set_result_callback(std::function<void(const PipelineResult&)> callback) {
    result_callback_ = std::move(callback);
}

PipelineResult Pipeline::execute_pipeline(const std::vector<float>& speech_audio) {
    PipelineResult result;
    auto pipeline_start = std::chrono::high_resolution_clock::now();

    // ── Layer 2: Speech-to-Text ──
    result.transcription = stt_->transcribe(speech_audio);

    if (!result.transcription.success || result.transcription.text.empty()) {
        std::cerr << "[Pipeline] STT produced no text — skipping\n";
        return result;
    }

    std::cerr << "[Pipeline] STT (" << result.transcription.duration_ms << "ms): "
              << result.transcription.text << "\n";

    // ── Layer 3: Emotion Classification (async) ──
    auto emotion_future = std::async(std::launch::async, [&]() {
        return emotion_->classify(result.transcription.text);
    });

    // Wait for emotion result
    result.emotion = emotion_future.get();

    // Update mood with actual emotion result
    result.mood = mood_->update(result.emotion);

    std::cerr << "[Pipeline] Emotion (" << result.emotion.duration_ms << "ms): "
              << EmotionClassifier::label_to_string(result.emotion.primary_emotion)
              << " (" << (result.emotion.confidence * 100.0f) << "%)\n";
    std::cerr << "[Pipeline] Mood: " << result.mood.current_score
              << " (avg: " << result.mood.window_average
              << ", safety: " << (result.mood.safety_triggered ? "YES" : "no") << ")\n";

    // ── Safety Check ──
    if (result.mood.safety_triggered) {
        std::cerr << "[Pipeline] SAFETY ESCALATION TRIGGERED\n";
        result.response.safety_bypassed = true;
        result.response.success = true;
        // Response text will be filled by Python backend from safety_responses.yaml
        result.response.text = "__SAFETY_ESCALATION__";
    } else {
        // ── Layer 4: LLM Response Generation ──
        result.response = llm_->generate(
            result.transcription.text,
            result.emotion.primary_emotion,
            result.emotion.confidence
        );
        std::cerr << "[Pipeline] LLM (" << result.response.duration_ms << "ms): "
                  << result.response.text << "\n";
    }

    auto pipeline_end = std::chrono::high_resolution_clock::now();
    result.total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        pipeline_end - pipeline_start
    ).count();

    std::cerr << "[Pipeline] Total: " << result.total_duration_ms << "ms\n";

    return result;
}

// ============================================
// IPC — stdin/stdout communication
// ============================================
void Pipeline::run_ipc_loop() {
    std::cerr << "[Pipeline] IPC loop started\n";

    // Send ready status
    send_ipc_message(json_util::to_json_obj({
        {"type", json_util::str_val("status")},
        {"status", json_util::str_val("ready")}
    }));

    while (running_) {
        std::string msg = read_ipc_message();
        if (msg.empty()) {
            // EOF or error
            std::cerr << "[Pipeline] IPC EOF — shutting down\n";
            break;
        }
        handle_ipc_message(msg);
    }

    std::cerr << "[Pipeline] IPC loop ended\n";
}

void Pipeline::send_ipc_message(const std::string& json) {
    // Length-prefixed protocol: 4 bytes LE length + JSON
    uint32_t len = static_cast<uint32_t>(json.size());
    char header[4];
    header[0] = static_cast<char>(len & 0xFF);
    header[1] = static_cast<char>((len >> 8) & 0xFF);
    header[2] = static_cast<char>((len >> 16) & 0xFF);
    header[3] = static_cast<char>((len >> 24) & 0xFF);

    std::cout.write(header, 4);
    std::cout.write(json.data(), json.size());
    std::cout.flush();
}

std::string Pipeline::read_ipc_message() {
    // Read 4-byte length header
    char header[4];
    if (!std::cin.read(header, 4)) return "";

    uint32_t len = static_cast<uint32_t>(
        (static_cast<unsigned char>(header[0]))       |
        (static_cast<unsigned char>(header[1]) << 8)  |
        (static_cast<unsigned char>(header[2]) << 16) |
        (static_cast<unsigned char>(header[3]) << 24)
    );

    if (len == 0 || len > 10 * 1024 * 1024) {
        std::cerr << "[Pipeline] Invalid IPC message length: " << len << "\n";
        return "";
    }

    std::string data(len, '\0');
    if (!std::cin.read(&data[0], len)) return "";

    return data;
}

void Pipeline::handle_ipc_message(const std::string& json) {
    std::string msg_type = json_util::get_type(json);

    if (msg_type == ipc::MSG_AUDIO) {
        // Audio data follows as binary after the JSON header
        // For IPC, audio is sent as a separate binary message
        // The JSON just signals "next message is audio"
        std::string audio_msg = read_ipc_message();
        if (audio_msg.empty()) return;

        // Convert raw PCM16 bytes to float32
        size_t n_samples = audio_msg.size() / 2;
        std::vector<float> samples(n_samples);
        const int16_t* pcm = reinterpret_cast<const int16_t*>(audio_msg.data());
        for (size_t i = 0; i < n_samples; i++) {
            samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
        }

        process_audio(samples);
    }
    else if (msg_type == ipc::MSG_SHUTDOWN) {
        shutdown();
    }
    else {
        std::cerr << "[Pipeline] Unknown message type: " << msg_type << "\n";
    }
}

void Pipeline::emit_results(const PipelineResult& result) {
    // Emit transcript
    if (result.transcription.success) {
        send_ipc_message(json_util::to_json_obj({
            {"type",       json_util::str_val("transcript")},
            {"text",       json_util::str_val(result.transcription.text)},
            {"final",      json_util::bool_val(true)},
            {"duration_ms", json_util::num_val(result.transcription.duration_ms)}
        }));
    }

    // Emit emotion
    if (result.emotion.success) {
        send_ipc_message(json_util::to_json_obj({
            {"type",       json_util::str_val("emotion")},
            {"label",      json_util::str_val(EmotionClassifier::label_to_string(result.emotion.primary_emotion))},
            {"confidence", json_util::num_val(result.emotion.confidence)},
            {"scores",     json_util::arr_val(result.emotion.scores)},
            {"duration_ms", json_util::num_val(result.emotion.duration_ms)}
        }));
    }

    // Emit mood update
    send_ipc_message(json_util::to_json_obj({
        {"type",            json_util::str_val("mood_update")},
        {"score",           json_util::num_val(result.mood.current_score)},
        {"window_average",  json_util::num_val(result.mood.window_average)},
        {"window",          json_util::arr_val(result.mood.window)},
        {"safety_triggered", json_util::bool_val(result.mood.safety_triggered)}
    }));

    // Emit safety alert or response
    if (result.response.safety_bypassed) {
        send_ipc_message(json_util::to_json_obj({
            {"type",    json_util::str_val("safety_alert")},
            {"message", json_util::str_val("__SAFETY_ESCALATION__")}
        }));
    } else if (result.response.success) {
        send_ipc_message(json_util::to_json_obj({
            {"type",              json_util::str_val("response")},
            {"text",              json_util::str_val(result.response.text)},
            {"tokens_generated",  json_util::num_val(result.response.tokens_generated)},
            {"tokens_per_second", json_util::num_val(result.response.tokens_per_second)},
            {"duration_ms",       json_util::num_val(result.response.duration_ms)}
        }));
    }

    // Also invoke the result callback if set
    if (result_callback_) {
        result_callback_(result);
    }
}

void Pipeline::shutdown() {
    running_ = false;
}

bool Pipeline::is_ready() const {
    return running_ &&
           stt_ && stt_->is_ready() &&
           emotion_ && emotion_->is_ready() &&
           llm_ && llm_->is_ready();
}

} // namespace mindease
