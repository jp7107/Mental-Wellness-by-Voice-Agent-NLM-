// ============================================
// MIND EASE — Mood Tracker
// ============================================
// Stateful rolling window that maps emotion classifications
// to mood scores and triggers safety escalation when
// sustained distress is detected.

#include "mood_tracker.hpp"
#include <numeric>
#include <algorithm>

namespace mindease {

MoodTracker::MoodTracker(const MoodConfig& config) : config_(config) {}

MoodUpdate MoodTracker::update(const EmotionResult& emotion) {
    MoodUpdate update;

    // Map emotion to mood score
    int score = emotion_to_score(emotion.primary_emotion);
    update.current_score = score;

    // Add to rolling window
    window_.push_back(score);
    while (static_cast<int>(window_.size()) > config_.window_size) {
        window_.pop_front();
    }

    // Compute window state
    update.window_average = compute_average();
    update.window = get_window();

    // Count consecutive turns at or above threshold
    int consecutive = 0;
    for (auto it = window_.rbegin(); it != window_.rend(); ++it) {
        if (static_cast<float>(*it) >= config_.safety_threshold) {
            consecutive++;
        } else {
            break;
        }
    }
    update.consecutive_high = consecutive;

    // Check safety condition
    update.safety_triggered = check_safety_condition();
    if (update.safety_triggered) {
        safety_active_ = true;
    }

    return update;
}

int MoodTracker::current_score() const {
    if (window_.empty()) return 0;
    return window_.back();
}

std::vector<int> MoodTracker::get_window() const {
    return std::vector<int>(window_.begin(), window_.end());
}

bool MoodTracker::is_safety_active() const {
    return safety_active_;
}

void MoodTracker::reset() {
    window_.clear();
    safety_active_ = false;
}

int MoodTracker::emotion_to_score(EmotionLabel label) {
    // Mapping from spec:
    // calm=1, sad=2, anxious=3, angry=3, fearful=4, distressed=5
    switch (label) {
        case EmotionLabel::CALM:       return 1;
        case EmotionLabel::SAD:        return 2;
        case EmotionLabel::ANXIOUS:    return 3;
        case EmotionLabel::ANGRY:      return 3;
        case EmotionLabel::FEARFUL:    return 4;
        case EmotionLabel::DISTRESSED: return 5;
        default:                       return 1;
    }
}

float MoodTracker::compute_average() const {
    if (window_.empty()) return 0.0f;
    float sum = std::accumulate(window_.begin(), window_.end(), 0.0f);
    return sum / static_cast<float>(window_.size());
}

bool MoodTracker::check_safety_condition() const {
    // Trigger when: ALL of the last `window_size` scores are >= threshold
    if (static_cast<int>(window_.size()) < config_.window_size) {
        return false;
    }

    // Check all entries in the window
    for (int score : window_) {
        if (static_cast<float>(score) < config_.safety_threshold) {
            return false;
        }
    }

    return true;
}

} // namespace mindease
