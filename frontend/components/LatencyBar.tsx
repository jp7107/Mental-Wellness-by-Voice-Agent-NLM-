'use client'
import React from 'react'

interface LatencyBarProps {
  layers: Array<{ name: string; ms: number; target: number }>
}

export default function LatencyBar({ layers }: LatencyBarProps) {
  if (layers.length === 0) return null
  return (
    <div className="glass" style={{ padding: '14px 18px' }}>
      <p style={{ fontSize: 10, letterSpacing: 2, textTransform: 'uppercase',
        color: 'var(--text-muted)', marginBottom: 10 }}>Latency</p>
      {layers.map(l => {
        const ok = l.ms <= l.target
        const pct = Math.min((l.ms / l.target) * 100, 100)
        return (
          <div key={l.name} style={{ marginBottom: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between',
              fontSize: 10, color: 'var(--text-muted)', marginBottom: 3 }}>
              <span>{l.name}</span>
              <span style={{ color: ok ? '#6af79a' : '#f76a8a' }}>{l.ms}ms</span>
            </div>
            <div style={{ height: 3, background: 'rgba(255,255,255,0.06)', borderRadius: 3 }}>
              <div style={{
                height: '100%', width: `${pct}%`,
                background: ok ? '#6af79a' : '#f76a8a',
                borderRadius: 3, transition: 'width 0.6s ease',
              }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}
