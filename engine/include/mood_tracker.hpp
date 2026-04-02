#pragma once
// ============================================
// MIND EASE — Mood Tracker
// ============================================
// Maintains a rolling window of mood scores derived
// from emotion classifications. Triggers safety
// escalation when sustained high-risk state detected.

#include <vector>
#include <deque>
#include <string>
#include "emotion.hpp"

namespace mindease {

struct MoodConfig {
    int   window_size       = 3;     // Number of turns in rolling window
    float safety_threshold  = 4.0f;  // Average score triggering escalation
};

struct MoodUpdate {
    int                  current_score = 0;    // 1-5
    float                window_average = 0.0f;
    std::vector<int>     window;               // Recent scores
    bool                 safety_triggered = false;
    int                  consecutive_high = 0; // Consecutive turns at/above threshold
};

class MoodTracker {
public:
    explicit MoodTracker(const MoodConfig& config = {});

    /// Update mood with a new emotion classification result.
    /// Returns the current mood state and whether safety was triggered.
    MoodUpdate update(const EmotionResult& emotion);

    /// Get current mood score without updating.
    int current_score() const;

    /// Get the rolling window of scores.
    std::vector<int> get_window() const;

    /// Check if safety escalation is currently active.
    bool is_safety_active() const;

    /// Reset mood tracker state (new session).
    void reset();

    /// Map an emotion label to a mood score (1-5).
    static int emotion_to_score(EmotionLabel label);

private:
    MoodConfig     config_;
    std::deque<int> window_;
    bool           safety_active_ = false;

    float compute_average() const;
    bool  check_safety_condition() const;
};

} // namespace mindease
