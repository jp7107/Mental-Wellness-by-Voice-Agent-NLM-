'use client'
import React from 'react'

interface Resource { name: string; contact: string; region?: string }

interface SafetyBannerProps {
  message: string
  resources: Resource[]
  onDismiss?: () => void
}

export default function SafetyBanner({ message, resources, onDismiss }: SafetyBannerProps) {
  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(247,106,138,0.15), rgba(208,106,247,0.1))',
      border: '1px solid rgba(247,106,138,0.4)',
      borderRadius: 16, padding: '20px 24px',
      animation: 'banner-in 0.4s ease',
      position: 'relative',
    }}>
      {onDismiss && (
        <button onClick={onDismiss} style={{
          position: 'absolute', top: 12, right: 12,
          background: 'none', border: 'none', cursor: 'pointer',
          color: 'var(--text-muted)', fontSize: 18, lineHeight: 1,
        }}>×</button>
      )}

      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', marginBottom: 14 }}>
        <span style={{ fontSize: 22, flexShrink: 0 }}>🤝</span>
        <p style={{ fontSize: 14, lineHeight: 1.6, color: 'var(--text-primary)' }}>{message}</p>
      </div>

      {resources.length > 0 && (
        <div>
          <p style={{ fontSize: 11, letterSpacing: 1.5, textTransform: 'uppercase',
            color: 'rgba(247,106,138,0.8)', marginBottom: 10 }}>Support Resources</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {resources.map((r, i) => (
              <div key={i} style={{
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                background: 'rgba(255,255,255,0.04)', borderRadius: 10,
                padding: '10px 14px',
              }}>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{r.name}</div>
                  {r.region && <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{r.region}</div>}
                </div>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#f76a8a', fontFamily: 'monospace' }}>
                  {r.contact}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <style>{`
        @keyframes banner-in {
          from { opacity: 0; transform: translateY(-8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}
