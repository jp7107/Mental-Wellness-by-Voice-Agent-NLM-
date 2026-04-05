'use client'
import React from 'react'
import { EMOTION_COLORS, EMOTION_LABELS } from '@/lib/constants'

interface MoodRingProps {
  emotion: string
  confidence: number
  score: number
  scores?: Record<string, number>
}

export default function MoodRing({ emotion, confidence, score, scores = {} }: MoodRingProps) {
  const color = EMOTION_COLORS[emotion] ?? '#7c6af7'
  const label = EMOTION_LABELS[emotion] ?? emotion

  return (
    <div className="glass" style={{ padding: '20px 24px', minWidth: 220 }}>
      <p style={{ fontSize: 11, letterSpacing: 2, textTransform: 'uppercase',
        color: 'var(--text-muted)', marginBottom: 12 }}>Emotional State</p>

      {/* Primary emotion */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div style={{
          width: 44, height: 44, borderRadius: '50%',
          background: `radial-gradient(circle, ${color}, ${color}88)`,
          boxShadow: `0 0 20px ${color}66`,
          transition: 'all 0.5s ease',
          flexShrink: 0,
        }} />
        <div>
          <div style={{ fontSize: 20, fontWeight: 700, fontFamily: 'Outfit, sans-serif',
            color, transition: 'color 0.4s ease' }}>{label}</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {(confidence * 100).toFixed(0)}% confidence
          </div>
        </div>
      </div>

      {/* Mood score bar */}
      <div style={{ marginBottom: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between',
          fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
          <span>Mood Level</span>
          <span>{score} / 5</span>
        </div>
        <div style={{ height: 6, background: 'rgba(255,255,255,0.08)', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${(score / 5) * 100}%`,
            background: `linear-gradient(90deg, ${color}, ${color}aa)`,
            borderRadius: 4,
            transition: 'width 0.6s cubic-bezier(0.4,0,0.2,1)',
            boxShadow: `0 0 8px ${color}66`,
          }} />
        </div>
      </div>

      {/* Score bars */}
      {Object.entries(scores).length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          {Object.entries(scores).map(([emo, val]) => (
            <div key={emo} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', width: 65 }}>{EMOTION_LABELS[emo] ?? emo}</span>
              <div style={{ flex: 1, height: 3, background: 'rgba(255,255,255,0.06)', borderRadius: 3 }}>
                <div style={{
                  height: '100%',
                  width: `${Math.min(val * 100, 100)}%`,
                  background: EMOTION_COLORS[emo] ?? '#7c6af7',
                  borderRadius: 3,
                  transition: 'width 0.5s ease',
                }} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
