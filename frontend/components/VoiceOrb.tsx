'use client'
import React from 'react'

interface VoiceOrbProps {
  isListening: boolean
  isProcessing: boolean
  onToggle: () => void
  emotion?: string
}

const EMOTION_COLORS: Record<string, [string, string]> = {
  calm:       ['#6af79a', '#3ddb6a'],
  sad:        ['#5eb8ff', '#2d8fcc'],
  anxious:    ['#f7c26a', '#d4983a'],
  angry:      ['#f77b6a', '#d44a37'],
  fearful:    ['#d06af7', '#a03acc'],
  distressed: ['#f76a8a', '#d43360'],
}

export default function VoiceOrb({ isListening, isProcessing, onToggle, emotion = 'calm' }: VoiceOrbProps) {
  const [c1, c2] = EMOTION_COLORS[emotion] ?? EMOTION_COLORS.calm

  return (
    <div style={{ position: 'relative', width: 200, height: 200, cursor: 'pointer' }} onClick={onToggle}>
      {/* Outer glow rings */}
      {isListening && (
        <>
          <div style={{
            position: 'absolute', inset: -20,
            borderRadius: '50%',
            border: `2px solid ${c1}`,
            opacity: 0.3,
            animation: 'pulse-ring 2s ease-out infinite',
          }} />
          <div style={{
            position: 'absolute', inset: -40,
            borderRadius: '50%',
            border: `1px solid ${c1}`,
            opacity: 0.15,
            animation: 'pulse-ring 2s ease-out infinite 0.5s',
          }} />
        </>
      )}

      {/* Main orb */}
      <div style={{
        position: 'absolute', inset: 0,
        borderRadius: '50%',
        background: isListening
          ? `radial-gradient(circle at 35% 35%, ${c1}, ${c2})`
          : 'radial-gradient(circle at 35% 35%, rgba(124,106,247,0.6), rgba(94,184,255,0.3))',
        boxShadow: isListening
          ? `0 0 60px ${c1}66, 0 0 120px ${c1}22, inset 0 1px 0 rgba(255,255,255,0.3)`
          : '0 0 40px rgba(124,106,247,0.3), inset 0 1px 0 rgba(255,255,255,0.2)',
        transition: 'all 0.5s cubic-bezier(0.4,0,0.2,1)',
        animation: isListening ? 'orb-breathe 3s ease-in-out infinite' : 'none',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {/* Icon */}
        <svg width="52" height="52" viewBox="0 0 24 24" fill="none">
          {isProcessing ? (
            <circle cx="12" cy="12" r="6" stroke="white" strokeWidth="2"
              strokeDasharray="25" strokeDashoffset="0"
              style={{ animation: 'spin 1s linear infinite', transformOrigin: '12px 12px' }} />
          ) : (
            <>
              <rect x="9" y="2" width="6" height="13" rx="3" fill="white" opacity={isListening ? 1 : 0.7} />
              <path d="M5 10.5a7 7 0 0014 0" stroke="white" strokeWidth="2" strokeLinecap="round" fill="none" opacity={isListening ? 1 : 0.7} />
              <line x1="12" y1="17.5" x2="12" y2="21" stroke="white" strokeWidth="2" strokeLinecap="round" />
              <line x1="9" y1="21" x2="15" y2="21" stroke="white" strokeWidth="2" strokeLinecap="round" />
            </>
          )}
        </svg>
      </div>

      <style>{`
        @keyframes pulse-ring {
          0%   { transform: scale(1);   opacity: 0.3; }
          100% { transform: scale(1.5); opacity: 0; }
        }
        @keyframes orb-breathe {
          0%,100% { transform: scale(1); }
          50%      { transform: scale(1.05); }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
